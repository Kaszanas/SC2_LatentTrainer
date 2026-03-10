"""MLFlow helper utilities.

Provides thin wrappers around the MLFlow client so every module uses
consistent tracking URIs, experiment naming, and artifact logging.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mlflow
from lightning.pytorch.loggers import MLFlowLogger

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
    tracking_uri: str = "mlruns",
) -> MLFlowLogger:
    """Create a Lightning ``MLFlowLogger`` instance.

    Parameters
    ----------
    experiment_name:
        MLFlow experiment name.
    run_name:
        Human-readable name for this run (e.g. ``"trial_3_stage1_vae"``).
    tracking_uri:
        MLFlow tracking URI.
    """
    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
    )


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
