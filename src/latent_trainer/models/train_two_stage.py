from typing import Any

import lightning as pl
import mlflow
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from latent_trainer.configs.experiment_config import ExperimentConfig
from latent_trainer.features.data_utils import extract_latents
from latent_trainer.models.lightning.lit_classifier import LatentClassifier
from latent_trainer.models.lightning.lit_vae import LitVAE
from latent_trainer.settings import DATA_DIR, OUTPUT_DIR
from latent_trainer.tracking.mlflow_utils import (
    create_child_mlflow_logger,
    create_mlflow_logger,
    log_checkpoint_artifacts,
)


def train_two_stage_pipeline(
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
    sweep_mode: bool = False,
) -> float:
    """Execute the two-stage training pipeline and return best val accuracy."""
    latent_dim = params["latent_dim"]
    vae_lr = params["vae_lr"]
    cls_lr = params["cls_lr"]
    batch_size = params["batch_size"]
    dropout = params["dropout"]
    vae_epochs = config.vae_epochs
    cls_epochs = config.cls_epochs
    vae_hidden_dims = params.get("vae_hidden_dims", None)
    cls_hidden_dims = params.get("cls_hidden_dims", None)

    if not vae_hidden_dims or not cls_hidden_dims:
        raise ValueError(
            "Hidden dims must be provided in params for both VAE and Classifier."
        )

    run_prefix = f"trial_{trial_num}" if trial_num is not None else "run"
    is_sweep = sweep_mode

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

    ckpt_dir_vae = OUTPUT_DIR / f"checkpoints_vae/{run_prefix}"
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
        enable_progress_bar=not is_sweep,
    )
    trainer_vae.fit(
        model=vae,
        train_dataloaders=vae_train_dl,
        val_dataloaders=vae_val_dl,
    )

    if mlf_vae.run_id:
        with mlflow.start_run(run_id=mlf_vae.run_id):
            log_checkpoint_artifacts(
                checkpoint_dir=ckpt_dir_vae,
                tracking_uri=config.mlflow_tracking_uri,
            )

    best_vae = LitVAE.load_from_checkpoint(ckpt_vae.best_model_path)
    best_vae.eval()
    best_vae.to(device)

    # ---- Extract latents ---------------------------------------------
    train_z = extract_latents(
        encoder=best_vae,
        data=train_X,
        device=device,
        batch_size=batch_size,
    )
    val_z = extract_latents(
        encoder=best_vae,
        data=val_X,
        device=device,
        batch_size=batch_size,
    )

    # ---- Stage 2: Classifier -----------------------------------------
    cls_train_dl = DataLoader(
        dataset=TensorDataset(train_z, train_y.unsqueeze(1)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=config.drop_last,  # BatchNorm1d requires >1 sample per batch
    )
    cls_val_dl = DataLoader(
        dataset=TensorDataset(val_z, val_y.unsqueeze(1)),
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
            experiment_name=config.experiment_name,
            run_name=f"{run_prefix}_stage2_classifier",
            parent_run_id=parent_run_id,
            tracking_uri=config.mlflow_tracking_uri,
            params={"stage": "classifier", "trial": trial_num},
        )
    else:
        mlf_cls = create_mlflow_logger(
            experiment_name=config.experiment_name,
            run_name=f"{run_prefix}_stage2_classifier",
            tracking_uri=config.mlflow_tracking_uri,
        )

    ckpt_dir_cls = DATA_DIR / f"checkpoints_cls/{run_prefix}"
    ckpt_cls = ModelCheckpoint(
        dirpath=ckpt_dir_cls,
        filename="cls-{epoch:02d}-{val_acc:.2f}",
        save_top_k=5,
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
        enable_progress_bar=not is_sweep,
    )
    trainer_cls.fit(
        model=cls_model,
        train_dataloaders=cls_train_dl,
        val_dataloaders=cls_val_dl,
    )

    if mlf_cls.run_id:
        with mlflow.start_run(run_id=mlf_cls.run_id):
            log_checkpoint_artifacts(ckpt_dir_cls, config.mlflow_tracking_uri)

    # Save pointer file for paths/main.py (only on non-HPO runs)
    if trial_num is None and norm_mean is not None and norm_std is not None:
        torch.save(
            {
                "vae_ckpt_path": ckpt_vae.best_model_path,
                "cls_ckpt_path": ckpt_cls.best_model_path,
                "latent_dim": latent_dim,
                "input_dim": input_dim,
                "normalization": {"mean": norm_mean, "std": norm_std},
            },
            OUTPUT_DIR / "two_stage_model.pth",
        )

    return ckpt_cls.best_model_score.item()
