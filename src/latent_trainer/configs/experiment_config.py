"""
Experiment configuration dataclass.

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
    """
    All runtime parameters of a training or HPO run.

    Parameters
    ----------
    sweep : bool
        True for a Ray Tune + Optuna sweep, False for a single/best run.
    pipeline : str
        Training pipeline, by default ``"guided_vae"``.
    dataset_filename : str
        Cached dataset file name inside ``DATA_DIR``.
    mlflow_tracking_uri : str
        MLflow tracking URI, by default the local SQLite database.
    experiment_name : str | None
        MLflow experiment name, by default None.
    n_trials : int
        Optuna trials in a sweep, by default 20.
    vae_epochs : int
        Legacy two-stage VAE epochs (unused by the guided VAE), by default 200.
    cls_epochs : int
        Legacy two-stage classifier epochs (unused by the guided VAE), by default 100.
    guided_vae_epochs : int
        Maximum guided VAE epochs, for sweeps and final training, by default 100.
    gpus_per_trial : float
        Fractional GPU per Ray trial, by default 0.1.
    cpus_per_trial : int
        CPUs per Ray trial, by default 2.
    optuna_db : str
        Optuna storage URL, by default ``sqlite:///optuna_study.db``.
    hpo_early_stopping_patience : int
        Early-stopping patience of sweep trials, by default 15.
    mlflow_source_experiment : str | None
        Experiment to load "best" parameters from; None uses
        ``experiment_name``.
    mlflow_source_run : str | None
        Run to load "best" parameters from; None uses the latest
        best-trial summary run.
    run_name : str | None
        MLflow run name for retraining; None auto-generates one.
    max_input_dim : int | None
        Keep only the first N feature columns (e.g. the first k granular bins);
        None keeps all.
    hpo_objective_weights : dict[str, float]
        Weights of the validation metrics summed into the HPO objective, by
        default ``val_vae_loss`` and ``val_cls_loss`` at 0.5 each.
    """

    # Pipeline selection
    sweep: bool
    pipeline: str = "guided_vae"
    dataset_filename: str = "cached_dataset_rich_sc2egset.pt"

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

    # Truncate the loaded feature tensor to its first N columns before
    # training (per-feature z-score normalization is slice-invariant, so
    # this is equivalent to having cached only those columns). Used to sweep
    # over a single wide cache (e.g. the "granular" transform's cumulative
    # 5%-of-game bins) without reprocessing the dataset per sweep point.
    # None → use the full cached feature width.
    max_input_dim: int | None = None

    # HPO objective: weighted sum of validation metrics.
    # Keys must match metric names logged by LitGuidedVAE:
    #   val_loss, val_vae_loss, val_cls_loss, val_acc
    # Use positive weights to minimise, negative to maximise (e.g. val_acc).
    # Default: minimise val_vae_loss only (backward-compatible).
    hpo_objective_weights: dict[str, float] = field(
        default_factory=lambda: {"val_vae_loss": 0.5, "val_cls_loss": 0.5}
    )
