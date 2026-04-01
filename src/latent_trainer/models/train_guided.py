# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
import logging
from pathlib import Path

import lightning as pl
import mlflow
import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE
from latent_trainer.settings import DEFAULT_MLFLOW_URI, OUTPUT_DIR
from latent_trainer.tracking.mlflow_utils import (
    create_child_mlflow_logger,
    create_mlflow_logger,
    log_checkpoint_artifacts,
)

logger = logging.getLogger(__name__)


def train_guided(
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    output_dir: Path | str = OUTPUT_DIR,
    epochs: int = 10,
    nz: int = 16,
    w_cls: float = 200.0,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    lr_c: float = 1e-4,
    weight_decay_c: float = 1e-4,
    encoder_hidden_dims: list[int] | None = None,
    test_interval: int = 1,
    mlflow_uri: str = DEFAULT_MLFLOW_URI,
    experiment_name: str = "SC2_GuidedVAE",
    run_name: str | None = None,
    parent_run_id: str | None = None,
) -> LitGuidedVAE:
    """Run a single guided-VAE training run and return the trained model."""

    model = LitGuidedVAE(
        n_vae_dis=nz,
        lr=lr,
        weight_decay=weight_decay,
        lr_c=lr_c,
        weight_decay_c=weight_decay_c,
        w_cls=w_cls,
        input_dim=input_dim,
        encoder_hidden_dims=encoder_hidden_dims,
    )

    checkpoints_path = output_dir / "checkpoints"
    filename_pattern = "model-{epoch:02d}-{val_vae_loss:.4f}"

    checkpoint_cb = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoints_path,
        filename=filename_pattern,
        monitor="val_vae_loss",
        mode="min",
        save_last=True,
        save_top_k=3,
    )
    early_stop = pl.pytorch.callbacks.EarlyStopping(
        monitor="val_vae_loss",
        patience=5,
        mode="min",
    )
    tb_logger = pl.pytorch.loggers.TensorBoardLogger(
        save_dir=output_dir,
        name="tensorboard_logs",
    )
    # Use child nesting if we have a parent run
    if parent_run_id:
        mlf_logger = create_child_mlflow_logger(
            experiment_name=experiment_name,
            run_name=run_name or "guided_vae_train",
            parent_run_id=parent_run_id,
            tracking_uri=mlflow_uri,
        )
    else:
        mlf_logger = create_mlflow_logger(
            experiment_name=experiment_name,
            run_name=run_name or "guided_vae_train",
            tracking_uri=mlflow_uri,
        )

    trainer = Trainer(
        max_epochs=epochs,
        logger=[tb_logger, mlf_logger],
        enable_progress_bar=True,
        callbacks=[checkpoint_cb, early_stop],
        accelerator="auto",
        devices=1,
        check_val_every_n_epoch=test_interval,
        log_every_n_steps=10,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Save final model in PyTorch format
    final_model_path = output_dir / "final_model.pth"
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.model.state_dict(),
            "classifier_state_dict": model.classifier.state_dict(),
            "loss": trainer.callback_metrics.get("train_vae_loss", float("inf")).item(),
        },
        final_model_path,
    )

    # Log checkpoints as MLFlow artifacts
    ckpt_dir = output_dir / "checkpoints"
    if mlf_logger.run_id:
        mlflow.set_tracking_uri(mlflow_uri)
        with mlflow.start_run(run_id=mlf_logger.run_id):
            log_checkpoint_artifacts(checkpoint_dir=ckpt_dir, tracking_uri=mlflow_uri)
            mlflow.log_artifact(local_path=final_model_path)

    logger.info("Training complete.  Model saved to %s", final_model_path)

    return model
