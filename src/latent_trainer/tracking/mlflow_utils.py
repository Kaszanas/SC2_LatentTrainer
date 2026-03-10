"""MLFlow helper utilities.

Provides thin wrappers around the MLFlow client so every module uses
consistent tracking URIs, experiment naming, artifact logging, and
parent–child run nesting.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
from lightning.pytorch.loggers import MLFlowLogger

from latent_trainer.config import DEFAULT_MLFLOW_URI

if TYPE_CHECKING:
    import optuna

    from latent_trainer.configs.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


def setup_mlflow(config: ExperimentConfig) -> str:
    """Configure the global MLFlow tracking URI and create the experiment.

    Returns
    -------
    str
        The experiment ID (useful for downstream queries).
    """
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    experiment = mlflow.set_experiment(config.experiment_name)
    logger.info(
        "MLFlow: tracking_uri=%s  experiment=%s (id=%s)",
        config.mlflow_tracking_uri,
        config.experiment_name,
        experiment.experiment_id,
    )
    return experiment.experiment_id


def create_mlflow_logger(
    experiment_name: str,
    run_name: str,
    tracking_uri: str = DEFAULT_MLFLOW_URI,
    *,
    run_id: str | None = None,
) -> MLFlowLogger:
    """Create a Lightning ``MLFlowLogger`` instance.

    Parameters
    ----------
    experiment_name:
        MLFlow experiment name.
    run_name:
        Human-readable name for this run (e.g. ``"trial_3_stage1_vae"``).
    tracking_uri:
        MLFlow tracking URI.  Defaults to the project-wide SQLite URI.
    run_id:
        If provided, attach the logger to an *existing* MLFlow run
        (e.g. a parent run opened with ``mlflow.start_run()``).
    """
    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        run_id=run_id,
    )


# ------------------------------------------------------------------
# Parent–child run nesting
# ------------------------------------------------------------------


def start_parent_run(
    experiment_name: str,
    run_name: str,
    tracking_uri: str = DEFAULT_MLFLOW_URI,
    params: dict | None = None,
    tags: dict | None = None,
) -> mlflow.ActiveRun:
    """Open a parent MLFlow run that child trial runs can nest under.

    The caller **must** use this as a context manager::

        with start_parent_run(...) as parent_run:
            parent_run_id = parent_run.info.run_id
            ...

    Parameters
    ----------
    experiment_name:
        MLFlow experiment name.
    run_name:
        Human-readable name (e.g. ``"hparam_search"``).
    tracking_uri:
        MLFlow tracking URI.
    params:
        Optional dict of params to log on the parent run immediately.
    tags:
        Optional dict of tags.

    Returns
    -------
    mlflow.ActiveRun
        Context-managed ``ActiveRun`` object.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    active_run = mlflow.start_run(run_name=run_name)
    if params:
        mlflow.log_params(params)
    if tags:
        mlflow.set_tags(tags)
    return active_run


def create_child_mlflow_logger(
    experiment_name: str,
    run_name: str,
    parent_run_id: str,
    tracking_uri: str = DEFAULT_MLFLOW_URI,
    params: dict | None = None,
) -> MLFlowLogger:
    """Create a child MLFlow run tagged with its parent's run ID.

    This opens a new run and tags it with ``parent_run_id`` so it
    appears grouped in the MLFlow UI.

    Parameters
    ----------
    experiment_name:
        MLFlow experiment name.
    run_name:
        Human-readable name for this child run.
    parent_run_id:
        The ``run_id`` of the parent run opened via :func:`start_parent_run`.
    tracking_uri:
        MLFlow tracking URI.
    params:
        Optional dict of params to log on this child run.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    child_run = mlflow.start_run(run_name=run_name, nested=True)
    child_run_id = child_run.info.run_id
    mlflow.set_tag("mlflow.parentRunId", parent_run_id)
    if params:
        mlflow.log_params(params)
    mlflow.end_run()  # Close the manual run; Lightning logger re-opens via run_id

    return MLFlowLogger(
        experiment_name=experiment_name,
        run_id=child_run_id,
        tracking_uri=tracking_uri,
    )


# ------------------------------------------------------------------
# Artifact logging
# ------------------------------------------------------------------


def log_checkpoint_artifacts(
    checkpoint_dir: str,
    tracking_uri: str = DEFAULT_MLFLOW_URI,
) -> None:
    """Log a checkpoint directory as MLFlow artifacts on the active run.

    Must be called inside an active ``mlflow.start_run()`` context.

    Parameters
    ----------
    checkpoint_dir:
        Path to the directory containing checkpoint files.
    tracking_uri:
        MLFlow tracking URI (set before logging).
    """
    mlflow.set_tracking_uri(tracking_uri)
    ckpt_path = Path(checkpoint_dir)
    if ckpt_path.exists():
        mlflow.log_artifacts(str(ckpt_path), artifact_path="checkpoints")
        logger.info("MLFlow: logged checkpoints from %s", checkpoint_dir)
    else:
        logger.warning("MLFlow: checkpoint dir %s does not exist, skipping", checkpoint_dir)


def log_best_trial(study: optuna.Study, config: ExperimentConfig) -> None:
    """Log the best Optuna trial as a dedicated MLFlow run.

    This creates a summary run containing the best hyperparameters and
    the objective value, making it easy to find in the MLFlow UI.
    """
    best = study.best_trial
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name="best_trial_summary"):
        mlflow.log_params(best.params)
        mlflow.log_metric("best_val_metric", best.value)
        mlflow.log_metric("best_trial_number", best.number)
        mlflow.set_tag("source", "optuna_best_trial")

    logger.info(
        "MLFlow: logged best trial #%d (value=%.4f) as summary run",
        best.number,
        best.value,
    )
