"""Unified training entrypoint for the SC2 Latent Trainer.

Supports both single-run and HPO (Ray Tune + Optuna) modes, with MLFlow
experiment tracking and flexible architecture search.

Examples
--------
Single run (two-stage pipeline)::

    uv run python -m latent_trainer.train \\
        --pipeline two_stage \\
        --cache data/cached_dataset_rich.pt \\
        --mode single

Guided VAE pipeline::

    uv run python -m latent_trainer.train \\
        --pipeline guided_vae \\
        --cache data/cached_dataset_rich.pt

Hyperparameter sweep::

    uv run python -m latent_trainer.train \\
        --pipeline two_stage \\
        --cache data/cached_dataset_rich.pt \\
        --mode sweep \\
        --n-trials 50 \\
        --experiment-name "SC2_TwoStage_ArchSearch" \\
        --mlflow-uri http://localhost:5000
"""

from __future__ import annotations

import logging

import click
import lightning as L
import torch

from latent_trainer.configs.experiment_config import ExperimentConfig
from latent_trainer.data_utils import load_and_normalize
from latent_trainer.hpo.ray_optuna_search import (
    run_hpo,
    run_two_stage_pipeline,
)
from latent_trainer.models.train_model import _load_cached_data, train_guided
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
    "--cache",
    default="data/cached_dataset_rich.pt",
    show_default=True,
    help="Path to the cached .pt dataset.",
)
@click.option(
    "--mode",
    type=click.Choice(["single", "sweep"]),
    default="single",
    show_default=True,
    help="'single' for a default-param run, 'sweep' for Ray+Optuna HPO.",
)
@click.option("--n-trials", type=int, default=20, show_default=True, help="Optuna trials for sweep mode.")
@click.option("--experiment-name", default="SC2_Latent_TwoStage", show_default=True, help="MLFlow experiment name.")
@click.option("--mlflow-uri", default="mlruns", show_default=True, help="MLFlow tracking URI (e.g. http://localhost:5000).")
@click.option("--vae-epochs", type=int, default=200, show_default=True, help="Max VAE epochs.")
@click.option("--cls-epochs", type=int, default=100, show_default=True, help="Max classifier epochs.")
@click.option("--batch-size", type=int, default=256, show_default=True, help="Default batch size.")
@click.option("--latent-dim", type=int, default=32, show_default=True, help="Default latent dimensionality.")
@click.option("--vae-lr", type=float, default=1e-3, show_default=True, help="Default VAE learning rate.")
@click.option("--cls-lr", type=float, default=1e-3, show_default=True, help="Default classifier learning rate.")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed.")
@click.option("--gpus-per-trial", type=float, default=1.0, show_default=True, help="Fractional GPU per Ray trial.")
@click.option("--cpus-per-trial", type=int, default=2, show_default=True, help="CPUs per Ray trial.")
@click.option("--optuna-db", default="sqlite:///optuna_study.db", show_default=True, help="Optuna storage URL.")
@click.option("--study-name", default="latent_trainer_hpo", show_default=True, help="Optuna study name.")
def main(
    pipeline: str,
    cache: str,
    mode: str,
    n_trials: int,
    experiment_name: str,
    mlflow_uri: str,
    vae_epochs: int,
    cls_epochs: int,
    batch_size: int,
    latent_dim: int,
    vae_lr: float,
    cls_lr: float,
    seed: int,
    gpus_per_trial: float,
    cpus_per_trial: int,
    optuna_db: str,
    study_name: str,
) -> None:
    """SC2 Latent Trainer — unified training & HPO entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = ExperimentConfig(
        pipeline=pipeline,
        cache_path=cache,
        mode=mode,
        experiment_name=experiment_name,
        mlflow_tracking_uri=mlflow_uri,
        n_trials=n_trials,
        vae_epochs=vae_epochs,
        cls_epochs=cls_epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        vae_lr=vae_lr,
        cls_lr=cls_lr,
        seed=seed,
        gpus_per_trial=gpus_per_trial,
        cpus_per_trial=cpus_per_trial,
        optuna_db=optuna_db,
        study_name=study_name,
    )

    L.seed_everything(config.seed)
    setup_mlflow(config)

    if pipeline == "two_stage":
        _run_two_stage(config)
    elif pipeline == "guided_vae":
        _run_guided_vae(config)
    else:
        raise click.BadParameter(f"Unknown pipeline: {pipeline}")


def _run_two_stage(config: ExperimentConfig) -> None:
    """Dispatch between single-run and sweep for the two-stage pipeline."""
    if config.mode == "sweep":
        logger.info(
            "Starting Ray Tune + Optuna sweep (%d trials)…", config.n_trials
        )
        study = run_hpo(config)
        log_best_trial(study, config)
        logger.info("Sweep complete.  Best trial: %s", study.best_trial.params)

    else:
        # Single run with default parameters
        logger.info("Running single two-stage training run…")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_X, train_y, val_X, val_y, _mean, _std = load_and_normalize(
            config.cache_path
        )
        input_dim = train_X.shape[-1]

        params = {
            "latent_dim": config.latent_dim,
            "vae_lr": config.vae_lr,
            "cls_lr": config.cls_lr,
            "batch_size": config.batch_size,
            "dropout": 0.3,
        }

        acc = run_two_stage_pipeline(
            train_X, train_y, val_X, val_y,
            input_dim, device, config, params,
        )
        logger.info("Single run complete.  Best val accuracy: %.2f%%", acc)

def _run_guided_vae(config: ExperimentConfig) -> None:
    """Run the guided-VAE pipeline with cached data."""
    logger.info("Running guided-VAE pipeline…")

    train_loader, val_loader, input_dim = _load_cached_data(
        config.cache_path, config.batch_size,
    )

    train_guided(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        output_dir="output",
        epochs=config.guided_vae_epochs,
        nz=config.latent_dim,
        lr=config.vae_lr,
    )
    logger.info("Guided-VAE pipeline complete.")


if __name__ == "__main__":
    main()
