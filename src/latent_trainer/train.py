"""Unified training entrypoint for the SC2 Latent Trainer.

Supports both sweep (Ray Tune + Optuna HPO) and best (retrain with best
Optuna trial params) modes for two pipelines, with MLFlow experiment tracking.

Workflow
~~~~~~~~
1. Run ``--mode sweep`` to search hyperparameters via Optuna.
2. Run ``--mode best`` to load ``study.best_trial.params`` and retrain
   a full model for use by ``paths/main.py``.

All hyperparameters come exclusively from the Optuna study — never from
CLI flags.

MLFlow tracking
~~~~~~~~~~~~~~~
All runs are tracked via MLFlow using a local SQLite database by default
(``sqlite:///mlflow.db``).  Override with ``--mlflow-uri`` to point at a
remote tracking server.  HPO trials are nested under a parent run for
grouped display in the MLFlow UI.
"""

from __future__ import annotations

import logging

import click
import pytorch_lightning as pl

from latent_trainer.configs.experiment_config import ExperimentConfig
from latent_trainer.hyperparameter_search.guided_vae import (
    run_guided_vae_best,
    run_guided_vae_hyperparameter_search,
)
from latent_trainer.hyperparameter_search.two_stage import (
    run_two_stage_best,
    run_two_stage_hyperparameter_search,
)
from latent_trainer.settings import DEFAULT_MLFLOW_URI, LOGGING_FORMAT, SEED
from latent_trainer.tracking.mlflow_utils import log_best_trial, setup_mlflow

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--pipeline",
    type=click.Choice(["two_stage", "guided_vae"]),
    default="two_stage",
    show_default=True,
    help="Training pipeline to use.",
)
@click.option(
    "--dataset_filename",
    default="cached_dataset_rich.pt",
    show_default=True,
    help="Filename of the cached dataset.  See 'features/main.py' to generate it.",
)
@click.option(
    "--sweep",
    is_flag=True,
    default=False,
    show_default=True,
    help="--sweep runs Ray+Optuna HPO; --best (default) retrains using the best Optuna trial.",
)
@click.option(
    "--n_trials",
    type=int,
    default=20,
    show_default=True,
    help="Optuna trials for sweep mode.",
)
@click.option(
    "--experiment_name",
    help="MLFlow experiment name.",
    required=True,
)
@click.option(
    "--mlflow_uri",
    default=DEFAULT_MLFLOW_URI,
    show_default=True,
    help="MLFlow tracking URI. Defaults to sqlite:///mlflow.db.",
)
@click.option(
    "--gpus_per_trial",
    type=float,
    default=0.1,
    show_default=True,
    help="Fractional GPU per Ray trial.",
)
@click.option(
    "--cpus_per_trial",
    type=int,
    default=2,
    show_default=True,
    help="CPUs per Ray trial.",
)
@click.option(
    "--optuna_db",
    default="sqlite:///optuna_study.db",
    show_default=True,
    help="Optuna storage URL.",
)
def main(
    pipeline: str,
    dataset_filename: str,
    sweep: bool,
    n_trials: int,
    experiment_name: str,
    mlflow_uri: str,
    gpus_per_trial: float,
    cpus_per_trial: int,
    optuna_db: str,
) -> None:
    """SC2 Latent Trainer — unified training & HPO entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOGGING_FORMAT,
    )

    pl.seed_everything(SEED)

    if mlflow_uri is None:
        mlflow_uri = DEFAULT_MLFLOW_URI

    config = ExperimentConfig(
        pipeline=pipeline,
        dataset_filename=dataset_filename,
        sweep=sweep,
        experiment_name=experiment_name,
        mlflow_tracking_uri=mlflow_uri,
        n_trials=n_trials,
        gpus_per_trial=gpus_per_trial,
        cpus_per_trial=cpus_per_trial,
        optuna_db=optuna_db,
    )

    setup_mlflow(
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        experiment_name=config.experiment_name,
    )

    match pipeline:
        case "two_stage":
            _train_two_stage(config=config)
        case "guided_vae":
            _train_guided_vae(config=config)
        case _:
            raise click.BadParameter(f"Unknown pipeline: {pipeline}")


def _train_two_stage(config: ExperimentConfig) -> None:
    """Dispatch between sweep and best for the two-stage pipeline."""
    if config.sweep:
        logger.info(f"Starting Ray Tune + Optuna sweep ({config.n_trials} trials)...")
        study = run_two_stage_hyperparameter_search(config=config)
        log_best_trial(study=study, config=config)
        logger.info(f"Sweep complete. Best trial: {study.best_trial.params}")
        return

    logger.info("Retraining two-stage pipeline with best Optuna params...")
    acc = run_two_stage_best(config=config)
    logger.info(f"Best-params run complete. val_acc={acc:.4f}")


def _train_guided_vae(config: ExperimentConfig) -> None:
    """Dispatch between sweep and best for the Guided-VAE pipeline."""
    if config.sweep:
        logger.info(f"Starting Guided-VAE Optuna sweep ({config.n_trials} trials)...")
        study = run_guided_vae_hyperparameter_search(config=config)
        log_best_trial(study=study, config=config)
        logger.info(f"Guided-VAE sweep complete. Best trial: {study.best_trial.params}")
        return

    logger.info("Retraining Guided-VAE pipeline with best Optuna params…")
    run_guided_vae_best(config=config)
    logger.info("Guided-VAE best-params run complete.")


if __name__ == "__main__":
    main()
