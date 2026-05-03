"""Experiment configuration dataclass.

Centralises all runtime parameters so they can be passed between modules
without ad-hoc dictionaries.  The unified ``train.py`` CLI populates
this dataclass from command-line arguments and passes it to all
downstream functions.

"""

from __future__ import annotations

from dataclasses import dataclass, field

from latent_trainer.settings import DEFAULT_MLFLOW_URI


@dataclass
class ExperimentConfig:
    """Holds all runtime parameters for a training / HPO run.

    Parameters
    ----------
    pipeline:
        Which training pipeline to use (``"guided_vae"``).
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
    """

    # Pipeline selection
    sweep: bool
    pipeline: str = "guided_vae"
    dataset_filename: str = "cached_dataset_rich.pt"

    # MLFlow tracking
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_URI
    experiment_name: str | None = None

    # Sweep configuration
    n_trials: int = 20

    # Training defaults (overridden per-trial during sweeps)
    # Two Stage:
    vae_epochs: int = 200
    cls_epochs: int = 100

    # Guided VAE Max Epochs (both sweep and final training)
    guided_vae_epochs: int = 100

    # Ray resource allocation
    # Runs 10 jobs in parallel:
    gpus_per_trial: float = 0.1
    # uses 2 CPUs per trial:
    cpus_per_trial: int = 2

    # Optuna persistence
    optuna_db: str = "sqlite:///optuna_study.db"

    # Early-stopping patience for HPO screening trials.
    # Kept separate from the full-training patience (which is hardcoded in train_guided)
    # because screening trials need more patience to warm up without wasting time.
    hpo_early_stopping_patience: int = 15

    # MLflow source for "best" mode param loading.
    # None → uses experiment_name as source.
    mlflow_source_experiment: str | None = None
    # None → finds the most recent best_trial_summary run (tag source=optuna_best_trial).
    # str  → reads params from any named run in the source experiment.
    mlflow_source_run: str | None = None

    # Explicit MLflow run name for non-sweep retraining.
    # None → auto-generated as "{source_label}_{timestamp}".
    run_name: str | None = None

    # HPO objective: weighted sum of validation metrics.
    # Keys must match metric names logged by LitGuidedVAE:
    #   val_loss, val_vae_loss, val_cls_loss, val_acc
    # Use positive weights to minimise, negative to maximise (e.g. val_acc).
    # Default: minimise val_vae_loss only (backward-compatible).
    hpo_objective_weights: dict[str, float] = field(
        default_factory=lambda: {"val_vae_loss": 0.5, "val_cls_loss": 0.5}
    )
