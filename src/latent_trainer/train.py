"""
Unified training entrypoint for the SC2 Latent Trainer.

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
from latent_trainer.configs.hyperparam_settings import (
    VAE_HIDDEN_DIM_CHOICES,
)
from latent_trainer.configs.search_space import (
    reconstruct_guided_vae_nz,
    reconstruct_hidden_dims,
)
from latent_trainer.hyperparameter_search.guided_vae import (
    run_guided_vae_best,
    run_guided_vae_hyperparameter_search,
)
from latent_trainer.settings import DEFAULT_MLFLOW_URI, LOGGING_FORMAT, SEED
from latent_trainer.tracking.mlflow_utils import log_best_trial, setup_mlflow

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--pipeline",
    type=click.Choice(["guided_vae", "event_guided_vae"]),
    default="guided_vae",
    show_default=True,
    help="Training pipeline to use.",
)
@click.option(
    "--dataset_filename",
    default=None,
    help=(
        "Filename of the cached dataset in data/. Defaults to "
        "cached_dataset_rich_sc2egset.pt (guided_vae) or "
        "cached_dataset_events.pt (event_guided_vae)."
    ),
)
@click.option(
    "--sweep",
    is_flag=True,
    default=False,
    show_default=True,
    help="--sweep runs Ray+Optuna hyperparameter search; Otherwise trains using the best Optuna trial.",
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
@click.option(
    "--source_experiment",
    default=None,
    help=(
        "MLflow experiment to load params from in 'best' mode. "
        "Defaults to --experiment_name."
    ),
)
@click.option(
    "--source_run",
    default=None,
    help=(
        "MLflow run name to load params from in 'best' mode. "
        "If omitted, uses the most recent run tagged source=optuna_best_trial."
    ),
)
@click.option(
    "--run_name",
    default=None,
    help=(
        "Explicit MLflow run name for non-sweep retraining. "
        "If omitted, auto-generated as '<source>_<timestamp>'."
    ),
)
@click.option(
    "--confirm",
    is_flag=True,
    default=False,
    help="event_guided_vae: retrain the best search configs with several seeds.",
)
@click.option(
    "--final",
    is_flag=True,
    default=False,
    help="event_guided_vae: train the confirmed config with several seeds.",
)
@click.option(
    "--max_epochs",
    type=int,
    default=None,
    help="Epoch cap of training runs (confirm/final; guided_vae runs).",
)
@click.option(
    "--search_max_epochs",
    type=int,
    default=None,
    help="event_guided_vae: epoch cap of search trials.",
)
@click.option(
    "--search_train_fraction",
    type=float,
    default=None,
    help="event_guided_vae: share of the training split used by search trials.",
)
@click.option(
    "--confirm_top_k",
    type=int,
    default=None,
    help="event_guided_vae: search configs retrained in the confirm stage.",
)
@click.option(
    "--confirm_seeds",
    type=int,
    default=None,
    help="event_guided_vae: seeds per config in the confirm stage.",
)
@click.option(
    "--final_seeds",
    type=int,
    default=None,
    help="event_guided_vae: seeds of the final stage.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help=(
        "event_guided_vae --confirm/--final: reuse runs that already finished "
        "with the same setup and train only the missing ones."
    ),
)
@click.option(
    "--num_workers",
    type=int,
    default=None,
    help="event_guided_vae: DataLoader workers preparing training batches.",
)
@click.option(
    "--compile_model",
    is_flag=True,
    default=False,
    help="event_guided_vae: compile the model with torch.compile (needs Triton).",
)
@click.option(
    "--max_input_dim",
    type=int,
    default=None,
    help=(
        "Truncate the loaded feature tensor to its first N columns before "
        "training ('best' mode only). Per-feature normalization is "
        "slice-invariant, so this is equivalent to caching only those "
        "columns. Useful for sweeping over a single wide cache (e.g. the "
        "'granular' transform's cumulative 5%-of-game bins) without "
        "reprocessing per sweep point."
    ),
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
    source_experiment: str | None,
    source_run: str | None,
    run_name: str | None,
    confirm: bool,
    final: bool,
    max_epochs: int | None,
    search_max_epochs: int | None,
    search_train_fraction: float | None,
    confirm_top_k: int | None,
    confirm_seeds: int | None,
    final_seeds: int | None,
    resume: bool,
    num_workers: int | None,
    compile_model: bool,
    max_input_dim: int | None,
) -> None:
    """
    Train the guided VAE or run its hyperparameter sweep.

    Parameters
    ----------
    pipeline : str
        Training pipeline (``guided_vae`` or ``event_guided_vae``).
    dataset_filename : str | None
        Cached dataset file name inside ``DATA_DIR``; None picks the
        pipeline's default cache.
    sweep : bool
        Run the Ray + Optuna sweep; otherwise retrain from the best trial.
    n_trials : int
        Optuna trials in sweep mode.
    experiment_name : str
        MLflow experiment name.
    mlflow_uri : str
        MLflow tracking URI; None uses the local SQLite database.
    gpus_per_trial : float
        Fractional GPU per Ray trial.
    cpus_per_trial : int
        CPUs per Ray trial.
    optuna_db : str
        Optuna storage URL.
    source_experiment : str | None
        Experiment to load best parameters from; None uses ``experiment_name``.
    source_run : str | None
        Run to load parameters from; None uses the latest best-trial summary.
    run_name : str | None
        MLflow run name for retraining; None auto-generates one.
    confirm : bool
        Event pipeline: run the confirm stage.
    final : bool
        Event pipeline: run the final stage.
    max_epochs : int | None
        Epoch cap of training runs; None keeps the config default.
    search_max_epochs : int | None
        Epoch cap of search trials; None keeps the config default.
    search_train_fraction : float | None
        Share of the training split used by search trials.
    confirm_top_k : int | None
        Search configs retrained in the confirm stage.
    confirm_seeds : int | None
        Seeds per config in the confirm stage.
    final_seeds : int | None
        Seeds of the final stage.
    resume : bool
        Event pipeline, confirm/final: reuse identical finished runs.
    num_workers : int | None
        Event pipeline: DataLoader workers for training batches.
    compile_model : bool
        Event pipeline: compile the model with ``torch.compile``.
    max_input_dim : int | None
        Keep only the first N feature columns (best mode only).

    Raises
    ------
    click.BadParameter
        If the pipeline is unknown or more than one stage flag is given.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=LOGGING_FORMAT,
    )

    pl.seed_everything(SEED)

    if mlflow_uri is None:
        mlflow_uri = DEFAULT_MLFLOW_URI

    if sum([sweep, confirm, final]) > 1:
        raise click.BadParameter("use at most one of --sweep, --confirm, --final")
    if resume and not (confirm or final):
        raise click.BadParameter("--resume works with --confirm or --final")
    if dataset_filename is None:
        dataset_filename = (
            "cached_dataset_events.pt"
            if pipeline == "event_guided_vae"
            else "cached_dataset_rich_sc2egset.pt"
        )

    overrides = {
        "guided_vae_epochs": max_epochs,
        "search_max_epochs": search_max_epochs,
        "search_train_fraction": search_train_fraction,
        "confirm_top_k": confirm_top_k,
        "confirm_seeds": confirm_seeds,
        "final_seeds": final_seeds,
        "num_workers": num_workers,
        "compile_model": compile_model or None,
        "resume": resume or None,
    }
    config = ExperimentConfig(
        **{key: value for key, value in overrides.items() if value is not None},
        pipeline=pipeline,
        dataset_filename=dataset_filename,
        sweep=sweep,
        experiment_name=experiment_name,
        mlflow_tracking_uri=mlflow_uri,
        n_trials=n_trials,
        gpus_per_trial=gpus_per_trial,
        cpus_per_trial=cpus_per_trial,
        optuna_db=optuna_db,
        mlflow_source_experiment=source_experiment,
        mlflow_source_run=source_run,
        run_name=run_name,
        max_input_dim=max_input_dim,
    )

    setup_mlflow(
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        experiment_name=config.experiment_name,
    )

    match pipeline:
        case "guided_vae":
            if confirm or final:
                raise click.BadParameter("--confirm/--final are for event_guided_vae")
            _train_guided_vae(config=config)
        case "event_guided_vae":
            _train_event_guided_vae(config=config, confirm=confirm, final=final)
        case _:
            raise click.BadParameter(f"Unknown pipeline: {pipeline}")


def _train_guided_vae(config: ExperimentConfig) -> None:
    """
    Dispatch the guided VAE pipeline to sweep or best-trial retraining.

    Parameters
    ----------
    config : ExperimentConfig
        Run configuration; ``config.sweep`` selects the mode.
    """
    if config.sweep:
        logger.info(f"Starting Guided-VAE Optuna sweep ({config.n_trials} trials)...")
        study = run_guided_vae_hyperparameter_search(config=config)
        flat_params = study.best_trial.params
        encoder_hidden_dims = reconstruct_hidden_dims(
            flat_params,
            prefix="enc",
            width_choices=VAE_HIDDEN_DIM_CHOICES,
        )
        nz = reconstruct_guided_vae_nz(flat_params, encoder_hidden_dims)
        supervised_dim = flat_params["supervised_dim"]
        log_best_trial(
            study=study,
            config=config,
            additional_params={
                "encoder_hidden_dims": str(encoder_hidden_dims),
                "latent_dim": nz,
                "supervised_dim": supervised_dim,
            },
        )
        logger.info(f"Guided-VAE sweep complete. Best trial: {study.best_trial.params}")
        return

    logger.info("Retraining Guided-VAE pipeline with best Optuna params…")
    run_guided_vae_best(config=config)
    logger.info("Guided-VAE best-params run complete.")


def _train_event_guided_vae(
    config: ExperimentConfig, confirm: bool, final: bool
) -> None:
    """
    Dispatch the event pipeline to its search, confirm or final stage.

    Parameters
    ----------
    config : ExperimentConfig
        Run configuration; ``config.sweep`` selects the search stage.
    confirm : bool
        Run the confirm stage.
    final : bool
        Run the final stage.

    Raises
    ------
    click.BadParameter
        If no stage is selected.
    """
    from latent_trainer.hyperparameter_search.event_guided_vae import (
        run_event_confirm,
        run_event_final,
        run_event_search,
    )

    if config.sweep:
        run_event_search(config)
    elif confirm:
        run_event_confirm(config)
    elif final:
        run_event_final(config)
    else:
        raise click.BadParameter(
            "event_guided_vae needs one stage: --sweep, --confirm or --final"
        )


if __name__ == "__main__":
    main()
