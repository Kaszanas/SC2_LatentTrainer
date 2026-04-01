"""Two-stage pipeline HPO and best-trial retraining.

Provides three entry-points for the two-stage (VAE → Classifier) pipeline:

* :func:`run_two_stage_pipeline` — runs one complete trial given a params dict.
* :func:`run_hyperparameter_search` — Ray Tune + Optuna distributed sweep.
* :func:`run_two_stage_best` — loads the best Optuna trial and retrains.
"""

from __future__ import annotations

import logging
import os
import uuid

import lightning as pl
import mlflow
import optuna
import ray
import torch
from ray import tune
from ray.tune.search.optuna import OptunaSearch

from latent_trainer.configs.experiment_config import ExperimentConfig
from latent_trainer.configs.search_space import (
    get_two_stage_search_space,
    reconstruct_hidden_dims,
)
from latent_trainer.features.data_utils import load_and_normalize
from latent_trainer.models.train_two_stage import train_two_stage_pipeline
from latent_trainer.settings import DATA_DIR, OUTPUT_DIR, SEED
from latent_trainer.tracking.mlflow_utils import (
    start_parent_run,
)

logger = logging.getLogger(__name__)


def run_two_stage_hyperparameter_search(config: ExperimentConfig) -> optuna.Study:
    """Launch a Ray Tune sweep with Optuna as the search backend.

    Each trial runs the full two-stage pipeline and reports ``val_acc``
    back to Optuna/Ray for pruning and selection.  All trials are
    nested under a parent MLFlow run for grouped UI display.

    Returns the completed :class:`optuna.Study` for further analysis.
    """
    data = load_and_normalize(cache_path=DATA_DIR / config.dataset_filename)
    train_X = data.train_X
    train_y = data.train_y
    val_X = data.val_X
    val_y = data.val_y
    input_dim = train_X.shape[-1]

    train_X_ref = ray.put(train_X)
    train_y_ref = ray.put(train_y)
    val_X_ref = ray.put(val_X)
    val_y_ref = ray.put(val_y)

    with start_parent_run(
        experiment_name=config.experiment_name,
        run_name="hparam_search",
        tracking_uri=config.mlflow_tracking_uri,
        params={
            "n_trials": config.n_trials,
            "search_algorithm": "OptunaSearch (TPE)",
        },
    ) as parent_run:
        parent_run_id = parent_run.info.run_id

        def trainable(ray_config: dict) -> dict:
            """Ray trainable: run one two-stage pipeline trial."""
            _train_X = ray.get(train_X_ref)
            _train_y = ray.get(train_y_ref)
            _val_X = ray.get(val_X_ref)
            _val_y = ray.get(val_y_ref)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pl.seed_everything(SEED)

            trial_tag = uuid.uuid4().hex[:6]

            # Create a trial-level run nested under the sweep parent so that
            # the VAE and classifier child runs are grouped together in MLflow.
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.experiment_name)
            trial_run = mlflow.start_run(run_name=f"trial_{trial_tag}")
            trial_run_id = trial_run.info.run_id
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)
            mlflow.log_params({
                k: str(v) if isinstance(v, list) else v
                for k, v in ray_config.items()
            })
            mlflow.end_run()

            acc = train_two_stage_pipeline(
                train_X=_train_X,
                train_y=_train_y,
                val_X=_val_X,
                val_y=_val_y,
                input_dim=input_dim,
                device=device,
                config=config,
                params=ray_config,
                trial_num=trial_tag,
                parent_run_id=trial_run_id,
                sweep_mode=True,
            )

            with mlflow.start_run(run_id=trial_run_id):
                mlflow.log_metric("val_acc", acc)

            return {"val_acc": acc}

        optuna_storage = optuna.storages.RDBStorage(url=config.optuna_db)
        optuna_search = OptunaSearch(
            space=get_two_stage_search_space,
            metric="val_acc",
            mode="max",
            storage=optuna_storage,
            study_name=config.study_name,
        )

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                log_to_driver=False,
                _temp_dir=os.path.join(os.getcwd(), "ray_tmp"),
            )

        tuner = tune.Tuner(
            tune.with_resources(
                trainable=trainable,
                resources={
                    "cpu": config.cpus_per_trial,
                    "gpu": config.gpus_per_trial,
                },
            ),
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
                num_samples=config.n_trials,
                metric="val_acc",
                mode="max",
                trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
            ),
            run_config=tune.RunConfig(
                name=config.study_name,
                storage_path=str(OUTPUT_DIR / "ray_results"),
            ),
        )

        results = tuner.fit()

        best = results.get_best_result(metric="val_acc", mode="max")
        logger.info("Best trial config: %s", best.config)
        logger.info("Best val_acc: %.4f", best.metrics["val_acc"])

        mlflow.log_metric("best_val_acc", best.metrics["val_acc"])
        mlflow.log_params({f"best_{k}": v for k, v in best.config.items()})

    return optuna_search._ot_study


def run_two_stage_best(config: ExperimentConfig) -> float:
    """Load the best trial from the Optuna study and run a full two-stage training.

    Reconstructs ``vae_hidden_dims`` / ``cls_hidden_dims`` from the flat
    per-layer Optuna params before calling :func:`run_two_stage_pipeline`.
    Writes ``output/two_stage_model.pth`` so ``paths/main.py`` can load it.

    Returns
    -------
    float
        Best validation accuracy achieved in this run.
    """
    study = optuna.load_study(study_name=config.study_name, storage=config.optuna_db)
    flat_params = study.best_trial.params
    logger.info("Loaded best trial #%d: %s", study.best_trial.number, flat_params)

    params = {
        **flat_params,
        "vae_hidden_dims": reconstruct_hidden_dims(flat_params, "vae"),
        "cls_hidden_dims": reconstruct_hidden_dims(flat_params, "cls"),
    }

    data = load_and_normalize(config.dataset_filename)
    input_dim = data.train_X.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(SEED)

    acc = train_two_stage_pipeline(
        train_X=data.train_X,
        train_y=data.train_y,
        val_X=data.val_X,
        val_y=data.val_y,
        input_dim=input_dim,
        device=device,
        config=config,
        params=params,
        norm_mean=data.mean,
        norm_std=data.std,
    )
    logger.info("Best-params two-stage run complete.  val_acc=%.4f", acc)
    return acc
