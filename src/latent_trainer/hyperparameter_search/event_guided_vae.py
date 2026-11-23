"""
Hyperparameter search, confirmation and final training of the event model.

Three stages, each a separate command so every long run starts deliberately:

1. **search**: Ray Tune + Optuna (multivariate TPE) with ASHA pruning, on a
   fixed seeded subset of the training split and capped epochs.
2. **confirm**: the best ``confirm_top_k`` search configs, each with
   ``confirm_seeds`` seeds on the full training split; the config with the
   best mean ``best_val_objective`` wins (``best_config.json``).
3. **final**: the winning config with ``final_seeds`` seeds and checkpoints;
   the seed with the median objective is the served model
   (``final_results.json``, MLflow tag ``served``).

The test split is never loaded. Every trial loads the cache itself, so trials
are independent processes; several share the GPU (``gpus_per_trial``).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import mlflow
import optuna
import ray
from lightning.pytorch.callbacks import Callback
from mlflow import MlflowClient
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from latent_trainer.configs.experiment_config import ExperimentConfig
from latent_trainer.configs.search_space import (
    event_params_from_flat,
    get_event_guided_vae_search_space,
)
from latent_trainer.features.data_utils import load_event_cache
from latent_trainer.models.train_event_guided import (
    search_subset,
    train_event_guided_run,
)
from latent_trainer.settings import DATA_DIR, OUTPUT_DIR, SEED
from latent_trainer.tracking.mlflow_utils import file_sha256, start_parent_run

logger = logging.getLogger(__name__)


def stage_dir(config: ExperimentConfig) -> Path:
    """
    Return the folder holding the stage hand-off files of an experiment.

    Parameters
    ----------
    config : ExperimentConfig
        Run configuration.

    Returns
    -------
    Path
        ``output/hpo/<experiment_name>``.
    """
    return OUTPUT_DIR / "hpo" / config.experiment_name


def search_study_name(config: ExperimentConfig) -> str:
    """
    Return the Optuna study name of an experiment's search stage.

    Parameters
    ----------
    config : ExperimentConfig
        Run configuration.

    Returns
    -------
    str
        ``<experiment_name>_search``.
    """
    return f"{config.experiment_name}_search"


class TuneReportCallback(Callback):
    """
    Report ``val_objective`` to Ray Tune after every validation epoch.

    Each report is one ``training_iteration``, which ASHA uses as time.
    """

    def on_validation_end(self, trainer, pl_module) -> None:
        """
        Send the epoch's objective and accuracy to Ray Tune.

        Parameters
        ----------
        trainer : lightning.Trainer
            The trainer.
        pl_module : lightning.LightningModule
            The module.
        """
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if "val_objective" not in metrics:
            return
        tune.report(
            {
                "objective": float(metrics["val_objective"]),
                "val_acc": float(metrics.get("val_acc", 0.0)),
                "epoch": trainer.current_epoch,
            }
        )


def _prepare(config: ExperimentConfig) -> Path:
    """
    Resolve the cache path, hash it once and start Ray.

    Parameters
    ----------
    config : ExperimentConfig
        Run configuration; ``cache_sha256`` is filled in.

    Returns
    -------
    Path
        The cache path.
    """
    cache_path = (DATA_DIR / config.dataset_filename).resolve()
    if not config.cache_sha256:
        config.cache_sha256 = file_sha256(cache_path)
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=False,
            _temp_dir=os.path.join(os.getcwd(), "ray_tmp"),
        )
    return cache_path


def _resources(config: ExperimentConfig) -> dict:
    """
    Return the Ray resources of one trial.

    Parameters
    ----------
    config : ExperimentConfig
        Run configuration.

    Returns
    -------
    dict
        CPU and GPU share per trial.
    """
    return {"cpu": config.cpus_per_trial, "gpu": config.gpus_per_trial}


def max_concurrent(config: ExperimentConfig) -> int:
    """
    Return how many trials may share the GPU at once.

    Parameters
    ----------
    config : ExperimentConfig
        Run configuration.

    Returns
    -------
    int
        ``floor(1 / gpus_per_trial)`` (at least 1; 1 without a GPU share).
    """
    if config.gpus_per_trial <= 0:
        return 1
    return max(1, int(1.0 / config.gpus_per_trial + 1e-9))


def run_event_search(config: ExperimentConfig) -> optuna.Study:
    """
    Run the search stage.

    Parameters
    ----------
    config : ExperimentConfig
        Run configuration (``n_trials``, ``search_*``, ``asha_*``).

    Returns
    -------
    optuna.Study
        The search study (stored in ``config.optuna_db``).
    """
    cache_path = _prepare(config)

    with start_parent_run(
        experiment_name=config.experiment_name,
        run_name="event_search",
        tracking_uri=config.mlflow_tracking_uri,
        params={
            "stage": "search",
            "n_trials": config.n_trials,
            "search_train_fraction": config.search_train_fraction,
            "search_max_epochs": config.search_max_epochs,
            "sampler": "TPE(multivariate)",
            "scheduler": "ASHA",
        },
    ) as parent_run:
        parent_run_id = parent_run.info.run_id

        def trainable(ray_config: dict) -> None:
            """
            Train one search trial on the training subset.

            Parameters
            ----------
            ray_config : dict
                Sampled configuration.
            """
            bundle = load_event_cache(cache_path)
            subset = search_subset(
                len(bundle.train.labels), config.search_train_fraction, SEED
            )
            trial_id = tune.get_context().get_trial_id()
            metrics = train_event_guided_run(
                bundle=bundle,
                config=config,
                params=ray_config,
                seed=SEED,
                run_name=f"search_{trial_id}",
                mode="search",
                cache_path=cache_path,
                parent_run_id=parent_run_id,
                train_indices=subset,
                extra_callbacks=[TuneReportCallback()],
            )
            tune.report(
                {
                    "objective": metrics["best_val_objective"],
                    "val_acc": metrics.get("val_acc", 0.0),
                    "epoch": metrics["epochs"],
                    "done": True,
                }
            )

        search = OptunaSearch(
            space=get_event_guided_vae_search_space,
            metric="objective",
            mode="min",
            sampler=optuna.samplers.TPESampler(multivariate=True, seed=SEED),
            storage=optuna.storages.RDBStorage(url=config.optuna_db),
            study_name=search_study_name(config),
        )
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=config.search_max_epochs,
            grace_period=config.asha_grace_period,
            reduction_factor=config.asha_reduction_factor,
        )
        tuner = tune.Tuner(
            tune.with_resources(trainable, resources=_resources(config)),
            tune_config=tune.TuneConfig(
                search_alg=search,
                scheduler=scheduler,
                num_samples=config.n_trials,
                max_concurrent_trials=max_concurrent(config),
                metric="objective",
                mode="min",
                trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
            ),
            run_config=tune.RunConfig(
                name=f"{config.experiment_name}_search",
                storage_path=str(OUTPUT_DIR / "ray_results"),
            ),
        )
        results = tuner.fit()
        best = results.get_best_result(metric="objective", mode="min", scope="all")
        mlflow.log_metric("best_objective", float(best.metrics["objective"]))
        logger.info(
            "Search done: best objective %.4f, config %s",
            best.metrics["objective"],
            best.config,
        )
    return search._ot_study


def top_search_configs(config: ExperimentConfig) -> list[dict]:
    """
    Read the best completed search configs from the Optuna study.

    Parameters
    ----------
    config : ExperimentConfig
        Run configuration (``optuna_db``, ``confirm_top_k``).

    Returns
    -------
    list[dict]
        Up to ``confirm_top_k`` full configs, best first, each with its
        ``search_objective`` and ``search_trial``.

    Raises
    ------
    ValueError
        If the study has no completed trial.
    """
    study = optuna.load_study(
        study_name=search_study_name(config), storage=config.optuna_db
    )
    done = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not done:
        raise ValueError(f"study {search_study_name(config)} has no completed trial")
    done.sort(key=lambda t: t.value)
    return [
        {
            **event_params_from_flat(t.params),
            "search_objective": t.value,
            "search_trial": t.number,
        }
        for t in done[: config.confirm_top_k]
    ]


def _grid_runs(
    config: ExperimentConfig,
    cache_path: Path,
    runs: list[dict],
    mode: str,
    parent_run_id: str,
    stage_name: str,
) -> list[dict]:
    """
    Train a fixed list of (config, seed) runs in parallel through Ray.

    Parameters
    ----------
    config : ExperimentConfig
        Run configuration.
    cache_path : Path
        Events cache.
    runs : list[dict]
        Each ``{"params": ..., "seed": ..., "name": ...}``.
    mode : str
        ``"confirm"`` or ``"final"``.
    parent_run_id : str
        MLflow parent run.
    stage_name : str
        Ray experiment name suffix.

    Returns
    -------
    list[dict]
        The metrics of every run, in the order of ``runs``.
    """

    def trainable(ray_config: dict) -> None:
        """
        Train one (config, seed) run on the full training split.

        Parameters
        ----------
        ray_config : dict
            ``{"index": i}`` into ``runs``.
        """
        run = runs[ray_config["index"]]
        bundle = load_event_cache(cache_path)
        metrics = train_event_guided_run(
            bundle=bundle,
            config=config,
            params=run["params"],
            seed=run["seed"],
            run_name=run["name"],
            mode=mode,
            cache_path=cache_path,
            parent_run_id=parent_run_id,
        )
        tune.report({"index": ray_config["index"], "result": json.dumps(metrics)})

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources=_resources(config)),
        param_space={"index": tune.grid_search(list(range(len(runs))))},
        tune_config=tune.TuneConfig(
            max_concurrent_trials=max_concurrent(config),
            trial_dirname_creator=lambda trial: f"run_{trial.trial_id}",
        ),
        run_config=tune.RunConfig(
            name=f"{config.experiment_name}_{stage_name}",
            storage_path=str(OUTPUT_DIR / "ray_results"),
        ),
    )
    results = tuner.fit()
    by_index: dict[int, dict] = {}
    for result in results:
        if result.error is not None:
            raise RuntimeError(f"{stage_name} run failed: {result.error}")
        by_index[int(result.metrics["index"])] = json.loads(result.metrics["result"])
    return [by_index[i] for i in range(len(runs))]


def run_event_confirm(config: ExperimentConfig) -> dict:
    """
    Run the confirm stage and write ``best_config.json``.

    Parameters
    ----------
    config : ExperimentConfig
        Run configuration (``confirm_top_k``, ``confirm_seeds``).

    Returns
    -------
    dict
        The winning config with its mean and std objective.
    """
    cache_path = _prepare(config)
    candidates = top_search_configs(config)
    seeds = [SEED + i for i in range(config.confirm_seeds)]
    runs = [
        {"params": params, "seed": seed, "name": f"confirm_c{rank}_s{seed}"}
        for rank, params in enumerate(candidates)
        for seed in seeds
    ]
    with start_parent_run(
        experiment_name=config.experiment_name,
        run_name="event_confirm",
        tracking_uri=config.mlflow_tracking_uri,
        params={"stage": "confirm", "top_k": len(candidates), "seeds": str(seeds)},
    ) as parent_run:
        results = _grid_runs(
            config, cache_path, runs, "confirm", parent_run.info.run_id, "confirm"
        )
        summary = []
        for rank, params in enumerate(candidates):
            scores = [
                r["best_val_objective"]
                for run, r in zip(runs, results, strict=True)
                if run["name"].startswith(f"confirm_c{rank}_")
            ]
            mean = sum(scores) / len(scores)
            std = (
                sum((s - mean) ** 2 for s in scores) / max(len(scores) - 1, 1)
            ) ** 0.5
            summary.append(
                {
                    "rank": rank,
                    "params": params,
                    "scores": scores,
                    "mean": mean,
                    "std": std,
                }
            )
            mlflow.log_metric(f"confirm_c{rank}_mean", mean)
            mlflow.log_metric(f"confirm_c{rank}_std", std)
        winner = min(summary, key=lambda entry: entry["mean"])

        out = stage_dir(config)
        out.mkdir(parents=True, exist_ok=True)
        (out / "confirm_results.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        best = {
            "params": winner["params"],
            "mean": winner["mean"],
            "std": winner["std"],
            "created": datetime.now().isoformat(timespec="seconds"),
            "cache_sha256": config.cache_sha256,
        }
        (out / "best_config.json").write_text(
            json.dumps(best, indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(out / "best_config.json"))
        mlflow.log_metric("winner_mean", winner["mean"])
        logger.info(
            "Confirm done: config %d wins, objective %.4f +- %.4f",
            winner["rank"],
            winner["mean"],
            winner["std"],
        )
    return best


def run_event_final(config: ExperimentConfig) -> dict:
    """
    Run the final stage and mark the median seed as the served model.

    Parameters
    ----------
    config : ExperimentConfig
        Run configuration (``final_seeds``).

    Returns
    -------
    dict
        Per-seed results and the served run (``final_results.json``).

    Raises
    ------
    FileNotFoundError
        If the confirm stage has not written ``best_config.json``.
    """
    cache_path = _prepare(config)
    best_path = stage_dir(config) / "best_config.json"
    if not best_path.exists():
        raise FileNotFoundError(f"{best_path} missing; run the confirm stage first")
    params = json.loads(best_path.read_text(encoding="utf-8"))["params"]
    params = {
        k: v for k, v in params.items() if k not in ("search_objective", "search_trial")
    }
    seeds = [SEED + i for i in range(config.final_seeds)]
    runs = [{"params": params, "seed": s, "name": f"final_s{s}"} for s in seeds]

    with start_parent_run(
        experiment_name=config.experiment_name,
        run_name="event_final",
        tracking_uri=config.mlflow_tracking_uri,
        params={"stage": "final", "seeds": str(seeds)},
    ) as parent_run:
        results = _grid_runs(
            config, cache_path, runs, "final", parent_run.info.run_id, "final"
        )
        ordered = sorted(
            zip(seeds, results, strict=True),
            key=lambda item: item[1]["best_val_objective"],
        )
        served_seed, served = ordered[(len(ordered) - 1) // 2]
        MlflowClient(tracking_uri=config.mlflow_tracking_uri).set_tag(
            served["run_id"], "served", "true"
        )
        report = {
            "params": params,
            "runs": [
                {
                    "seed": s,
                    **{k: r[k] for k in ("best_val_objective", "run_id", "checkpoint")},
                }
                for s, r in zip(seeds, results, strict=True)
            ],
            "served": {
                "seed": served_seed,
                "run_id": served["run_id"],
                "checkpoint": served["checkpoint"],
                "rule": "median best_val_objective of the seeds",
            },
        }
        out = stage_dir(config)
        out.mkdir(parents=True, exist_ok=True)
        (out / "final_results.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(out / "final_results.json"))
        logger.info(
            "Final done: served seed %d (%s)", served_seed, served["checkpoint"]
        )
    return report
