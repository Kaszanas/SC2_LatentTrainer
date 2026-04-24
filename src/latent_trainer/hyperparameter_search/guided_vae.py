"""Guided-VAE pipeline HPO and best-trial retraining.

Provides two entry-points for the supervised Guided-VAE pipeline:

* :func:`run_guided_vae_hyperparameter_search` — Ray Tune + Optuna parallel sweep.
* :func:`run_guided_vae_best` — loads the best Optuna trial and retrains.
"""

from __future__ import annotations

import logging
import os
import uuid

import lightning as pl
import mlflow
import optuna
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch

from latent_trainer.configs.experiment_config import ExperimentConfig
from latent_trainer.configs.hyperparam_settings import VAE_HIDDEN_DIM_CHOICES
from latent_trainer.configs.search_space import (
    get_guided_vae_search_space,
    reconstruct_guided_vae_nz,
    reconstruct_guided_vae_supervised_dim,
    reconstruct_hidden_dims,
)
from latent_trainer.features.data_utils import load_and_normalize
from latent_trainer.models.train_guided import train_guided_pipeline
from latent_trainer.settings import DATA_DIR, OUTPUT_DIR, SEED
from latent_trainer.tracking.mlflow_utils import (
    start_parent_run,
)

logger = logging.getLogger(__name__)


def run_guided_vae_hyperparameter_search(config: ExperimentConfig) -> optuna.Study:
    """Run a Ray Tune + Optuna parallel HPO sweep for the Guided-VAE pipeline.

    Raw tensors are pre-placed in the Ray object store so that each worker
    can build its own DataLoaders with the trial-specific batch size.
    All trial runs are nested under a parent MLFlow run.

    Returns
    -------
    optuna.Study
        Completed study for further analysis / logging.
    """
    data = load_and_normalize(DATA_DIR / config.dataset_filename)
    input_dim = data.train_X.shape[-1]

    # Pre-place tensors in Ray object store — batch_size varies per trial
    # so DataLoaders must be created inside each worker.
    train_X_ref = ray.put(data.train_X)
    train_y_ref = ray.put(data.train_y)
    val_X_ref = ray.put(data.val_X)
    val_y_ref = ray.put(data.val_y)

    with start_parent_run(
        experiment_name=config.experiment_name,
        run_name="guided_vae_hpo",
        tracking_uri=config.mlflow_tracking_uri,
        params={"n_trials": config.n_trials, "search_algorithm": "OptunaSearch (TPE)"},
    ) as parent_run:
        parent_run_id = parent_run.info.run_id

        def trainable(ray_config: dict) -> dict:
            """Ray trainable: run one Guided-VAE screening trial."""
            train_X = ray.get(train_X_ref)
            train_y = ray.get(train_y_ref)
            val_X = ray.get(val_X_ref)
            val_y = ray.get(val_y_ref)

            pl.seed_everything(SEED)
            # Extract the Optuna trial number (added by get_guided_vae_search_space)
            # and strip it so the model never receives it as a hyperparameter.
            trial_num = ray_config.pop("_trial_number", None)
            trial_tag = (
                str(trial_num) if trial_num is not None else uuid.uuid4().hex[:6]
            )

            metrics = train_guided_pipeline(
                train_X=train_X,
                train_y=train_y,
                val_X=val_X,
                val_y=val_y,
                input_dim=input_dim,
                config=config,
                params=ray_config,
                parent_run_id=parent_run_id,
                trial_tag=trial_tag,
                sweep_mode=True,
            )
            objective = sum(
                w * metrics[m] for m, w in config.hpo_objective_weights.items()
            )
            return {**metrics, "objective": objective}

        optuna_storage = optuna.storages.RDBStorage(url=config.optuna_db)
        optuna_search = OptunaSearch(
            space=get_guided_vae_search_space,
            metric="objective",
            mode="min",
            storage=optuna_storage,
            study_name=config.experiment_name,
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
                metric="objective",
                mode="min",
                trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
            ),
            run_config=tune.RunConfig(
                name=config.experiment_name,
                storage_path=str(OUTPUT_DIR / "ray_results"),
            ),
        )

        results = tuner.fit()
        best = results.get_best_result(metric="objective", mode="min")
        logger.info(
            "Best guided-VAE trial  objective=%.4f  val_vae_loss=%.4f  val_cls_loss=%.4f  config=%s",
            best.metrics["objective"],
            best.metrics["val_vae_loss"],
            best.metrics["val_cls_loss"],
            best.config,
        )

        mlflow.log_metric("best_objective",    best.metrics["objective"])
        mlflow.log_metric("best_val_vae_loss", best.metrics["val_vae_loss"])
        mlflow.log_metric("best_val_cls_loss", best.metrics["val_cls_loss"])
        mlflow.log_metric("best_val_acc",      best.metrics["val_acc"])
        mlflow.log_params({f"best_{k}": v for k, v in best.config.items()})

    return optuna_search._ot_study


def run_guided_vae_best(config: ExperimentConfig) -> None:
    """Load the best Optuna trial and run a full Guided-VAE training.

    Uses all training epochs (``config.guided_vae_epochs``) and writes
    checkpoints + MLFlow artifacts via :func:`train_guided_pipeline`.
    """
    pl.seed_everything(SEED)

    study = optuna.load_study(
        study_name=config.experiment_name,
        storage=config.optuna_db,
    )
    best = study.best_trial
    flat_params = best.params
    logger.info(
        "Loaded best guided-VAE trial %d  objective=%.4f",
        best.number,
        best.value,
    )

    encoder_hidden_dims = reconstruct_hidden_dims(
        flat_params,
        prefix="enc",
        width_choices=VAE_HIDDEN_DIM_CHOICES,
    )

    nz = reconstruct_guided_vae_nz(flat_params, encoder_hidden_dims)
    supervised_dim = reconstruct_guided_vae_supervised_dim(flat_params, nz)

    params = {
        **flat_params,
        "encoder_hidden_dims": encoder_hidden_dims,
        "nz": nz,
        "supervised_dim": supervised_dim,
    }

    data = load_and_normalize(DATA_DIR / config.dataset_filename)

    train_guided_pipeline(
        train_X=data.train_X,
        train_y=data.train_y,
        val_X=data.val_X,
        val_y=data.val_y,
        input_dim=data.train_X.shape[-1],
        config=config,
        params=params,
        norm_mean=data.mean,
        norm_std=data.std,
        sweep_mode=False,
    )
    logger.info("Guided-VAE best-params run complete.")
