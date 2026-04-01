"""Ray Tune + Optuna hyperparameter search.

Orchestrates distributed HPO by:
1. Defining a *trainable* function that runs a full pipeline for one trial.
2. Using ``OptunaSearch`` as the Ray Tune search algorithm.
3. Nesting every trial under a parent MLFlow run for grouped UI display.
4. Logging checkpoint artifacts after the best trial.

Usage (from ``train.py`` entrypoint)::

    from latent_trainer.hyperparameter_search.ray_optuna_search import run_hpo
    run_hpo(config)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import lightning as pl
import mlflow
import optuna
import ray
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from torch.utils.data import DataLoader, TensorDataset

from latent_trainer.configs.experiment_config import ExperimentConfig
from latent_trainer.configs.search_space import get_two_stage_search_space
from latent_trainer.data_utils import extract_latents, load_and_normalize
from latent_trainer.models.lightning.lit_classifier import LatentClassifier
from latent_trainer.models.lightning.lit_vae import LitVAE
from latent_trainer.tracking.mlflow_utils import (
    create_child_mlflow_logger,
    create_mlflow_logger,
    log_checkpoint_artifacts,
    start_parent_run,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Two-stage pipeline (single trial)
# ------------------------------------------------------------------


def run_two_stage_pipeline(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    input_dim: int,
    device: torch.device,
    config: ExperimentConfig,
    params: dict[str, Any],
    norm_mean: torch.Tensor | None = None,
    norm_std: torch.Tensor | None = None,
    trial_num: int | None = None,
    parent_run_id: str | None = None,
) -> float:
    """Execute the two-stage training pipeline and return best val accuracy."""
    latent_dim = params["latent_dim"]
    vae_lr = params["vae_lr"]
    cls_lr = params["cls_lr"]
    batch_size = params["batch_size"]
    dropout = params.get("dropout", 0.3)
    vae_epochs = params.get("vae_epochs", config.vae_epochs)
    cls_epochs = params.get("cls_epochs", config.cls_epochs)
    vae_hidden_dims = params.get("vae_hidden_dims", None)
    cls_hidden_dims = params.get("cls_hidden_dims", None)

    run_prefix = f"trial_{trial_num}" if trial_num is not None else "run"

    # ---- Stage 1: VAE ------------------------------------------------
    train_flat = train_X.reshape(-1, input_dim)
    val_flat = val_X.reshape(-1, input_dim)

    vae_train_dl = DataLoader(
        TensorDataset(train_flat),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    vae_val_dl = DataLoader(
        TensorDataset(val_flat),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    vae = LitVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=vae_hidden_dims,
        lr=vae_lr,
    )

    # Use child nesting if we have a parent run
    if parent_run_id:
        mlf_vae = create_child_mlflow_logger(
            config.experiment_name,
            f"{run_prefix}_stage1_vae",
            parent_run_id=parent_run_id,
            tracking_uri=config.mlflow_tracking_uri,
            params={"stage": "vae", "trial": trial_num, **params},
        )
    else:
        mlf_vae = create_mlflow_logger(
            config.experiment_name,
            f"{run_prefix}_stage1_vae",
            config.mlflow_tracking_uri,
        )

    ckpt_dir_vae = f"output/checkpoints_vae/{run_prefix}"
    ckpt_vae = ModelCheckpoint(
        dirpath=ckpt_dir_vae,
        filename="vae-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    es_vae = EarlyStopping(monitor="val_loss", patience=20, mode="min")

    trainer_vae = pl.Trainer(
        max_epochs=vae_epochs,
        accelerator="auto",
        devices=1,
        logger=mlf_vae,
        callbacks=[ckpt_vae, es_vae],
        enable_progress_bar=True,
    )
    trainer_vae.fit(vae, vae_train_dl, vae_val_dl)

    # Log VAE checkpoints as artifacts
    if mlf_vae.run_id:
        with mlflow.start_run(run_id=mlf_vae.run_id):
            log_checkpoint_artifacts(ckpt_dir_vae, config.mlflow_tracking_uri)

    best_vae = LitVAE.load_from_checkpoint(ckpt_vae.best_model_path)
    best_vae.eval()
    best_vae.to(device)

    # ---- Extract latents ----------------------------------------------
    train_z = extract_latents(best_vae, train_X, device, batch_size)
    val_z = extract_latents(best_vae, val_X, device, batch_size)

    # ---- Stage 2: Classifier ------------------------------------------
    cls_train_dl = DataLoader(
        TensorDataset(train_z, train_y.unsqueeze(1)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    cls_val_dl = DataLoader(
        TensorDataset(val_z, val_y.unsqueeze(1)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    cls_model = LatentClassifier(
        latent_dim=latent_dim,
        hidden_dims=cls_hidden_dims,
        lr=cls_lr,
        dropout=dropout,
    )

    if parent_run_id:
        mlf_cls = create_child_mlflow_logger(
            config.experiment_name,
            f"{run_prefix}_stage2_classifier",
            parent_run_id=parent_run_id,
            tracking_uri=config.mlflow_tracking_uri,
            params={"stage": "classifier", "trial": trial_num},
        )
    else:
        mlf_cls = create_mlflow_logger(
            config.experiment_name,
            f"{run_prefix}_stage2_classifier",
            config.mlflow_tracking_uri,
        )

    ckpt_dir_cls = f"output/checkpoints_cls/{run_prefix}"
    ckpt_cls = ModelCheckpoint(
        dirpath=ckpt_dir_cls,
        filename="cls-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        monitor="val_acc",
        mode="max",
    )
    es_cls = EarlyStopping(monitor="val_loss", patience=15, mode="min")

    trainer_cls = pl.Trainer(
        max_epochs=cls_epochs,
        accelerator="auto",
        devices=1,
        logger=mlf_cls,
        callbacks=[ckpt_cls, es_cls],
        enable_progress_bar=True,
    )
    trainer_cls.fit(cls_model, cls_train_dl, cls_val_dl)

    # Log classifier checkpoints as artifacts
    if mlf_cls.run_id:
        with mlflow.start_run(run_id=mlf_cls.run_id):
            log_checkpoint_artifacts(ckpt_dir_cls, config.mlflow_tracking_uri)

    # Save a combined pointer file for the path-charting CLI.
    # Only written on single runs (trial_num is None) to avoid overwriting
    # the file on every HPO trial.
    if trial_num is None and norm_mean is not None and norm_std is not None:
        os.makedirs("output", exist_ok=True)
        torch.save(
            {
                "vae_ckpt_path": ckpt_vae.best_model_path,
                "cls_ckpt_path": ckpt_cls.best_model_path,
                "latent_dim": latent_dim,
                "input_dim": input_dim,
                "normalization": {"mean": norm_mean, "std": norm_std},
            },
            "output/two_stage_model.pth",
        )

    return ckpt_cls.best_model_score.item()


# ------------------------------------------------------------------
# Ray Tune HPO
# ------------------------------------------------------------------


def run_hpo(config: ExperimentConfig) -> optuna.Study:
    """Launch a Ray Tune sweep with Optuna as the search backend.

    Each trial runs the full two-stage pipeline and reports ``val_acc``
    back to Optuna/Ray for pruning and selection.  All trials are
    nested under a parent MLFlow run for grouped UI display.

    Returns the completed :class:`optuna.Study` for further analysis.
    """
    # Pre-load data once (will be shipped to workers via Ray object store)
    train_X, train_y, val_X, val_y, _mean, _std = load_and_normalize(config.cache_path)
    input_dim = train_X.shape[-1]

    # Put tensors in Ray object store for efficient sharing
    train_X_ref = ray.put(train_X)
    train_y_ref = ray.put(train_y)
    val_X_ref = ray.put(val_X)
    val_y_ref = ray.put(val_y)

    # Open a parent run that all trials nest under
    with start_parent_run(
        experiment_name=config.experiment_name,
        run_name="hparam_search",
        tracking_uri=config.mlflow_tracking_uri,
        params={
            "n_trials": config.n_trials,
            "search_algorithm": "OptunaSearch (TPE)",
        },
    ) as parent_run:
        parent_run_id = parent_run.info.run_id

        def trainable(ray_config: dict) -> dict:
            """Ray trainable: run one two-stage pipeline trial."""
            _train_X = ray.get(train_X_ref)
            _train_y = ray.get(train_y_ref)
            _val_X = ray.get(val_X_ref)
            _val_y = ray.get(val_y_ref)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pl.seed_everything(config.seed)

            acc = run_two_stage_pipeline(
                _train_X,
                _train_y,
                _val_X,
                _val_y,
                input_dim,
                device,
                config,
                params=ray_config,
                trial_num=ray_config.get("__trial_index"),
                parent_run_id=parent_run_id,
            )
            return {"val_acc": acc}

        # Create Optuna search algorithm for Ray Tune
        optuna_search = OptunaSearch(
            space=get_two_stage_search_space,
            metric="val_acc",
            mode="max",
            storage=config.optuna_db,
            study_name=config.study_name,
            load_if_exists=True,
        )

        # Initialise Ray (idempotent)
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                log_to_driver=False,
                _temp_dir=os.path.join(os.getcwd(), "ray_tmp"),
            )

        tuner = tune.Tuner(
            tune.with_resources(
                trainable,
                resources={"cpu": config.cpus_per_trial, "gpu": config.gpus_per_trial},
            ),
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
                num_samples=config.n_trials,
                metric="val_acc",
                mode="max",
            ),
            run_config=ray.train.RunConfig(
                name=config.study_name,
                storage_path="output/ray_results",
            ),
        )

        results = tuner.fit()

        best = results.get_best_result(metric="val_acc", mode="max")
        logger.info("Best trial config: %s", best.config)
        logger.info("Best val_acc: %.4f", best.metrics["val_acc"])

        # Log best result on the parent run
        mlflow.log_metric("best_val_acc", best.metrics["val_acc"])
        mlflow.log_params({f"best_{k}": v for k, v in best.config.items()})

    # Return the underlying Optuna study for further analysis / logging
    return optuna_search._ot_study
