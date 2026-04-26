"""MLFlow helper utilities.

Provides thin wrappers around the MLFlow client so every module uses
consistent tracking URIs, experiment naming, artifact logging, and
parent-child run nesting.

Three key patterns are implemented:

1. **SQLite-backed tracking** -- all modules default to
   ``sqlite:///mlflow.db`` via :data:`~latent_trainer.config.DEFAULT_MLFLOW_URI`.

2. **Parent-child run nesting** -- HPO sweeps create a single parent run
   and nest each trial underneath it.  Trials are tagged with
   ``mlflow.parentRunId`` so they appear grouped in the MLFlow UI.

3. **Artifact logging** -- checkpoint directories and model files are
   logged as MLFlow artifacts for easy retrieval from the UI.

Usage::

    from latent_trainer.tracking.mlflow_utils import (
        create_mlflow_logger,
        start_parent_run,
        create_child_mlflow_logger,
        log_checkpoint_artifacts,
    )

    # Simple single run:
    mlf_logger = create_mlflow_logger("my_experiment", "run_1")

    # Nested HPO sweep:
    with start_parent_run("my_experiment", "hparam_search") as parent:
        for i in range(n_trials):
            child_logger = create_child_mlflow_logger(
                "my_experiment", f"trial_{i}", parent.info.run_id,
            )
"""

from __future__ import annotations

import ast
import getpass
import logging
from pathlib import Path
from typing import Any

import mlflow
import optuna
from lightning.pytorch.loggers import MLFlowLogger
from mlflow import MlflowClient
from mlflow.utils.mlflow_tags import (
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_RUN_NAME,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_USER,
)

from latent_trainer.configs.experiment_config import ExperimentConfig
from latent_trainer.settings import DEFAULT_MLFLOW_URI, OUTPUT_DIR

logger = logging.getLogger(__name__)


def setup_mlflow(mlflow_tracking_uri: str, experiment_name: str) -> str:
    """Configure the global MLFlow tracking URI and create the experiment.

    Returns
    -------
    str
        The experiment ID (useful for downstream queries).
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    logger.info(
        f"MLFlow: tracking_uri={mlflow_tracking_uri}  experiment={experiment_name} (id={experiment.experiment_id})"
    )
    return experiment.experiment_id


# ------------------------------------------------------------------
# Parent-child run nesting
#
# During HPO sweeps, a "parent" run is opened and all individual trial
# runs are created as children.  This groups trials in the MLFlow UI
# and lets you log aggregate results (best params, best metric) on
# the parent run itself.
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
        artifact_location=str(OUTPUT_DIR / "mlruns"),
    )


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
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    tags = {
        MLFLOW_PARENT_RUN_ID: parent_run_id,
        MLFLOW_RUN_NAME: run_name,
        MLFLOW_USER: getpass.getuser(),
        MLFLOW_SOURCE_TYPE: "LOCAL",
        MLFLOW_SOURCE_NAME: "train.py",
    }
    run = client.create_run(experiment_id=experiment_id, run_name=run_name, tags=tags)
    child_run_id = run.info.run_id

    if params:
        client.log_batch(
            child_run_id,
            params=[mlflow.entities.Param(k, str(v)) for k, v in params.items()],
        )

    return MLFlowLogger(
        experiment_name=experiment_name,
        run_id=child_run_id,
        tracking_uri=tracking_uri,
        artifact_location=str(OUTPUT_DIR / "mlruns"),
    )


# ------------------------------------------------------------------
# Artifact logging
#
# After training, checkpoint files and final model weights can be
# uploaded to MLFlow as artifacts.  This makes them browsable and
# downloadable from the MLFlow UI without needing filesystem access.
# ------------------------------------------------------------------
def log_checkpoint_artifacts(
    checkpoint_dir: Path,
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
    if checkpoint_dir.exists():
        mlflow.log_artifacts(local_dir=str(checkpoint_dir))
        logger.info(f"MLFlow: logged checkpoints from {checkpoint_dir}")
    else:
        logger.warning(
            f"MLFlow: checkpoint dir {checkpoint_dir} does not exist, skipping"
        )


def log_best_trial(
    study: optuna.Study,
    config: ExperimentConfig,
    additional_params: dict | None = None,
) -> None:
    """Log the best Optuna trial as a dedicated MLFlow run.

    This creates a summary run containing the best hyperparameters and
    the objective value, making it easy to find in the MLFlow UI.
    """
    best = study.best_trial
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name="best_trial_summary"):
        mlflow.log_params(best.params)
        if additional_params:
            mlflow.log_params(additional_params)

        mlflow.log_metric("best_val_metric", best.value)
        mlflow.log_metric("best_trial_number", best.number)
        mlflow.set_tag("source", "optuna_best_trial")

    logger.info(
        f"MLFlow: logged best trial #{best.number} (value={best.value:.4f}) as summary run"
    )


def log_artifact(
    checkpoint_dir: Path,
    mlflow_logger: MLFlowLogger,
    model_path: Path,
):

    if not mlflow_logger.run_id:
        raise ValueError("MLFlow logger must have an active run_id to log artifacts")

    # Log checkpoints as MLFlow artifacts
    mlflow.set_tracking_uri(mlflow_logger._tracking_uri)
    with mlflow.start_run(run_id=mlflow_logger.run_id):
        log_checkpoint_artifacts(
            checkpoint_dir=checkpoint_dir,
            tracking_uri=mlflow_logger._tracking_uri,
        )
        mlflow.log_artifact(local_path=model_path)

    logger.info(f"Training complete.  Model saved to {str(model_path)}")


# ---------------------------------------------------------------------------
# MLflow-based param loading for guided-VAE retraining
# ---------------------------------------------------------------------------

_GUIDED_VAE_PARAM_TYPES: dict[str, type] = {
    "latent_dim":            int,
    "supervised_dim":        int,
    "batch_size":            int,
    "encoder_hidden_dims":   list,
    "classification_weight": float,
    "learning_rate":         float,
    "learning_rate_cls":     float,
    "weight_decay":          float,
    "weight_decay_cls":      float,
}


def _coerce_guided_vae_params(raw: dict[str, str]) -> dict[str, Any]:
    """Coerce string-valued MLflow params to Python types."""
    result: dict[str, Any] = {}
    for key, raw_val in raw.items():
        t = _GUIDED_VAE_PARAM_TYPES[key]
        result[key] = ast.literal_eval(raw_val) if t is list else t(raw_val)
    return result


def load_guided_vae_params_from_mlflow(
    experiment_name: str,
    run_name: str | None = None,
    tracking_uri: str = DEFAULT_MLFLOW_URI,
) -> dict[str, Any]:
    """Load GuidedVAE training params from MLflow.

    Parameters
    ----------
    experiment_name:
        MLflow experiment to search in.
    run_name:
        ``None`` → finds the most recent run tagged
        ``source=optuna_best_trial`` (created by :func:`log_best_trial`).
        ``str`` → finds any run with that ``mlflow.runName`` tag.
    tracking_uri:
        MLflow tracking URI.

    Returns
    -------
    dict[str, Any]
        Params coerced to Python types, ready to pass to
        :func:`~latent_trainer.models.train_guided.train_guided_pipeline`.

    Raises
    ------
    ValueError
        If no matching run is found or required param keys are missing.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"MLflow experiment '{experiment_name}' not found.")

    if run_name is None:
        # Strategy A: find the best_trial_summary run (log_best_trial tags it)
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.source = 'optuna_best_trial'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise ValueError(
                f"No 'optuna_best_trial' run found in experiment '{experiment_name}'. "
                "Run a sweep first, or supply --source_run."
            )
    else:
        # Strategy B: find any named run
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.`mlflow.runName` = '{run_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise ValueError(
                f"No run named '{run_name}' found in experiment '{experiment_name}'."
            )

    raw = runs[0].data.params
    filtered = {k: v for k, v in raw.items() if k in _GUIDED_VAE_PARAM_TYPES}

    missing = set(_GUIDED_VAE_PARAM_TYPES) - set(filtered)
    if missing:
        raise ValueError(
            f"Run '{runs[0].info.run_name}' is missing required params: {missing}. "
            "Is this a compatible guided-VAE run?"
        )

    return _coerce_guided_vae_params(filtered)
