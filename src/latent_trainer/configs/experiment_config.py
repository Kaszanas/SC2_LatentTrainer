"""Experiment configuration dataclass.

Centralises all runtime parameters so they can be passed between modules
without ad-hoc dictionaries.  The unified ``train.py`` CLI populates
this dataclass from command-line arguments and passes it to all
downstream functions.

Usage::

    config = ExperimentConfig(
        pipeline="two_stage",
        cache_path="data/cached_dataset_rich.pt",
        mode="sweep",
        n_trials=30,
    )
    setup_mlflow(config)
"""

from __future__ import annotations

from dataclasses import dataclass

from latent_trainer.settings import DEFAULT_MLFLOW_URI


@dataclass
class ExperimentConfig:
    """Holds all runtime parameters for a training / HPO run.

    Parameters
    ----------
    pipeline:
        Which training pipeline to use (``"two_stage"`` or ``"guided_vae"``).
    cache_path:
        Path to the pre-processed ``.pt`` dataset cache.
    mode:
        ``"single"`` for a one-shot run or ``"sweep"`` for Ray+Optuna HPO.
    experiment_name:
        MLFlow experiment name.
    mlflow_tracking_uri:
        URI for the MLFlow tracking server.  Defaults to a local SQLite
        database (``sqlite:///mlflow.db``) for robust, query-able storage.
        Override via ``--mlflow-uri`` CLI flag or by editing this default.
    n_trials:
        Number of Optuna trials when running a sweep.
    vae_epochs:
        Max epochs for the VAE stage (two-stage pipeline).
    cls_epochs:
        Max epochs for the classifier stage (two-stage pipeline).
    guided_vae_epochs:
        Max epochs for the GuidedVAE pipeline.
    batch_size:
        Default batch size (overridden during sweep).
    latent_dim:
        Default latent dimensionality (overridden during sweep).
    vae_lr:
        Default VAE learning rate (overridden during sweep).
    cls_lr:
        Default classifier learning rate (overridden during sweep).
    seed:
        Random seed for reproducibility.
    gpus_per_trial:
        Fractional GPU allocation per Ray trial (e.g. 0.5 for 2 concurrent
        trials on 1 GPU).
    cpus_per_trial:
        CPU allocation per Ray trial.
    optuna_db:
        Optuna storage URL.  SQLite by default for persistence across
        restarts and for the Optuna dashboard.
    study_name:
        Optuna study name (used for persistence / resumption).
    """

    # Pipeline selection
    pipeline: str = "two_stage"
    dataset_filename: str = "cached_dataset_rich.pt"
    mode: str = "sweep"

    # MLFlow tracking
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_URI
    experiment_name: str = "SC2_Latent_TwoStage"

    # Sweep configuration
    n_trials: int = 20

    # Training defaults (overridden per-trial during sweeps)
    vae_epochs: int = 200
    cls_epochs: int = 100
    guided_vae_epochs: int = 10

    # Ray resource allocation
    gpus_per_trial: float = 1.0
    cpus_per_trial: int = 2

    # Optuna persistence
    optuna_db: str = "sqlite:///optuna_study.db"
    study_name: str = "latent_trainer_hpo"
