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
    slice_ticks : int
        Event model: PlayerStats ticks per slice, by default 1.
    drop_tail_frac : float
        Event model: share of each game's slices dropped from the end, by
        default 0.1.
    val_cutoffs_min : tuple[float, ...]
        Event model: fixed validation cutoffs in game minutes, validated next
        to the whole game, by default (5, 10, 15).
    kl_warmup_epochs : float
        Event model: epochs over which the KL weight ramps up, by default 2.0.
    precision : str
        Lightning precision for the event model, by default ``"bf16-mixed"``.
    num_workers : int
        DataLoader workers for the event model, by default 0.
    compile_model : bool
        Compile the event model's heavy submodules with ``torch.compile``
        (needs Triton: ``triton`` on Linux, ``triton-windows`` on Windows, both
        installed by the ``cuda`` extra), by default False.
    resume : bool
        Confirm/final: reuse runs that already finished with the same setup
        and train only the missing ones, by default False.
    search_train_fraction : float
        Share of the training split used by search trials, by default 0.3.
    search_max_epochs : int
        Epoch cap of search trials, by default 30.
    asha_grace_period : int
        Epochs before ASHA may stop a trial, by default 3.
    asha_reduction_factor : int
        ASHA reduction factor, by default 3.
    confirm_top_k : int
        Best search configs retrained in the confirm stage, by default 5.
    confirm_seeds : int
        Seeds per config in the confirm stage, by default 3.
    final_seeds : int
        Seeds of the chosen config in the final stage, by default 5.
    early_stopping_patience : int
        Patience of confirm and final runs, by default 10.
    cache_sha256 : str
        SHA-256 of the cache file, filled once by the driver for provenance.
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

    # Event model data settings (fixed during the search):
    slice_ticks: int = 1
    drop_tail_frac: float = 0.1
    val_cutoffs_min: tuple[float, ...] = (5.0, 10.0, 15.0)
    kl_warmup_epochs: float = 2.0
    precision: str = "bf16-mixed"
    num_workers: int = 0
    compile_model: bool = False
    resume: bool = False

    # Event model search, confirm and final stages:
    search_train_fraction: float = 0.3
    search_max_epochs: int = 30
    asha_grace_period: int = 3
    asha_reduction_factor: int = 3
    confirm_top_k: int = 5
    confirm_seeds: int = 3
    final_seeds: int = 5
    early_stopping_patience: int = 10
    cache_sha256: str = ""
