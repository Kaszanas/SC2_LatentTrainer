import logging
from pathlib import Path

from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE
from latent_trainer.settings import CHECKPOINTS_DIR, DEFAULT_MLFLOW_URI, OUTPUT_DIR
from latent_trainer.tracking.mlflow_utils import (
    create_child_mlflow_logger,
    create_mlflow_logger,
    log_artifact,
)

logger = logging.getLogger(__name__)


def train_guided(
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    output_dir: Path | str = OUTPUT_DIR,
    supervised_dim: int,
    epochs: int,
    vae_latent_dim: int,
    classification_weight: float,
    encoder_hidden_dims: list[int],
    experiment_name: str,
    run_name: str,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    learning_rate_classifier: float = 1e-4,
    weight_decay_c: float = 1e-4,
    test_interval: int = 1,
    mlflow_uri: str = DEFAULT_MLFLOW_URI,
    parent_run_id: str | None = None,
) -> LitGuidedVAE:
    """Run a single guided-VAE training run and return the trained model."""

    guided_vae_model = LitGuidedVAE(
        vae_latent_dim=vae_latent_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        supervised_dim=supervised_dim,
        learning_rate_classifier=learning_rate_classifier,
        weight_decay_c=weight_decay_c,
        classification_weight=classification_weight,
        input_dim=input_dim,
        encoder_hidden_dims=encoder_hidden_dims,
    )

    filename_pattern = "model-{epoch:02d}-{val_vae_loss:.4f}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINTS_DIR,
        filename=filename_pattern,
        monitor="val_vae_loss",
        mode="min",
        save_last=True,
        save_top_k=3,
    )
    early_stopping = EarlyStopping(
        monitor="val_vae_loss",
        patience=5,
        mode="min",
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=output_dir,
        name="tensorboard_logs",
    )
    # Use child nesting if we have a parent run
    if parent_run_id:
        mlf_logger = create_child_mlflow_logger(
            experiment_name=experiment_name,
            run_name=run_name,
            parent_run_id=parent_run_id,
            tracking_uri=mlflow_uri,
        )
    else:
        mlf_logger = create_mlflow_logger(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=mlflow_uri,
        )

    trainer = Trainer(
        max_epochs=epochs,
        logger=[tensorboard_logger, mlf_logger],
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",
        devices=1,
        check_val_every_n_epoch=test_interval,
        log_every_n_steps=10,
    )
    trainer.fit(
        model=guided_vae_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Save Lighting Model with best hyperparameters (for easy loading later)
    log_artifact(
        checkpoint_dir=CHECKPOINTS_DIR,
        mlflow_logger=mlf_logger,
        model_path=Path(checkpoint_callback.best_model_path),
    )

    return guided_vae_model
