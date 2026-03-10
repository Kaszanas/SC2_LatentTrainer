"""Experiment configuration dataclass.

Centralises all runtime parameters so they can be passed between modules
without ad-hoc dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from latent_trainer.config import DEFAULT_MLFLOW_URI


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
        Optuna storage URL for the dashboard.
    study_name:
        Optuna study name (used for persistence / resumption).
    """

    pipeline: str = "two_stage"
    cache_path: str = "data/cached_dataset_rich.pt"
    mode: str = "single"

    # MLFlow
    experiment_name: str = "SC2_Latent_TwoStage"
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_URI

    # Sweep
    n_trials: int = 20

    # Training defaults
    vae_epochs: int = 200
    cls_epochs: int = 100
    guided_vae_epochs: int = 10
    batch_size: int = 256
    latent_dim: int = 32
    vae_lr: float = 1e-3
    cls_lr: float = 1e-3
    seed: int = 42

    # Ray resources
    gpus_per_trial: float = 1.0
    cpus_per_trial: int = 2

    # Optuna persistence
    optuna_db: str = "sqlite:///optuna_study.db"
    study_name: str = "latent_trainer_hpo"
