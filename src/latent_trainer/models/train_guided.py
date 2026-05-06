import logging
from pathlib import Path
from typing import Any

import click
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset

from latent_trainer.configs.experiment_config import ExperimentConfig
from latent_trainer.features.data_utils import load_and_normalize
from latent_trainer.features.type import NormalizedDataloaders
from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE
from latent_trainer.settings import (
    CHECKPOINTS_DIR,
    DATA_DIR,
    DEFAULT_MLFLOW_URI,
    OUTPUT_DIR,
)
from latent_trainer.tracking.mlflow_utils import (
    create_child_mlflow_logger,
    create_mlflow_logger,
    log_artifact,
)

logger = logging.getLogger(__name__)


def train_guided(
    *,
    normalized_dataloaders: NormalizedDataloaders,
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
        input_dim=normalized_dataloaders.input_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        mean=normalized_dataloaders.mean,
        std=normalized_dataloaders.std,
    )

    filename_pattern = "guided-vae-{epoch:02d}-{val_vae_loss:.4f}"

    run_checkpoint_dir = CHECKPOINTS_DIR / experiment_name / run_name

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_checkpoint_dir,
        filename=filename_pattern,
        monitor="val_vae_loss",
        mode="min",
        save_last=True,
        save_top_k=5,
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
        train_dataloaders=normalized_dataloaders.train_loader,
        val_dataloaders=normalized_dataloaders.val_loader,
    )

    # Save the best model under a custom name for easier retrieval later:
    best_model_path = run_checkpoint_dir / "best.ckpt"
    guided_vae_model = LitGuidedVAE.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    trainer.save_checkpoint(best_model_path)

    # Save Lighting Model with best hyperparameters (for easy loading later)
    log_artifact(
        checkpoint_dir=run_checkpoint_dir,
        mlflow_logger=mlf_logger,
        model_path=best_model_path,
    )

    return guided_vae_model


def train_guided_pipeline(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    input_dim: int,
    config: ExperimentConfig,
    params: dict[str, Any],
    norm_mean: torch.Tensor | None = None,
    norm_std: torch.Tensor | None = None,
    parent_run_id: str | None = None,
    trial_tag: str | None = None,
    sweep_mode: bool = False,
) -> dict[str, float] | None:
    """Run one Guided-VAE trial and return all validation metrics (sweep) or None.

    Used by both the Ray Tune HPO trainable (``sweep_mode=True``) and the
    final best-params retraining run (``sweep_mode=False``).

    Parameters
    ----------
    train_X, train_y, val_X, val_y:
        Pre-normalised feature and label tensors.
    input_dim:
        Number of features per sample.
    config:
        Experiment configuration (experiment name, tracking URI, epochs …).
    params:
        Flat hyperparameter dict from the search space; expected keys:
        ``nz``, ``batch_size``, ``cls``, ``lr``, ``weight_decay``,
        ``lr_c``, ``weight_decay_c``, ``supervised_dim``,
        ``encoder_hidden_dims``.
    norm_mean, norm_std:
        Normalisation statistics to embed in the model for inference-time
        de-normalisation.  Pass ``None`` during HPO screening.
    parent_run_id:
        MLFlow parent run ID for nested child logging.
    trial_tag:
        Short identifier appended to the run name.  ``None`` → ``"guided_vae_best"``.
    sweep_mode:
        ``True``: fast screening — no checkpoints, no TensorBoard, no
        artifact upload, progress bar suppressed.
        ``False``: full training — ModelCheckpoint, TensorBoardLogger,
        and artifact logging enabled.
    """
    run_name = f"guided_vae_{trial_tag}" if trial_tag else "guided_vae_best"
    batch_size: int = params["batch_size"]

    mlflow_params = {
        key: (str(value) if isinstance(value, list) else value)
        for key, value in params.items()
    }

    train_loader = DataLoader(
        TensorDataset(train_X, train_y),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_X, val_y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    if parent_run_id:
        mlf_logger = create_child_mlflow_logger(
            experiment_name=config.experiment_name,
            run_name=run_name,
            parent_run_id=parent_run_id,
            tracking_uri=config.mlflow_tracking_uri,
            params=mlflow_params,
        )
    else:
        mlf_logger = create_mlflow_logger(
            experiment_name=config.experiment_name,
            run_name=run_name,
            tracking_uri=config.mlflow_tracking_uri,
        )

    model = LitGuidedVAE(
        input_dim=input_dim,
        encoder_hidden_dims=params["encoder_hidden_dims"],
        supervised_dim=params["supervised_dim"],
        latent_dim=params["latent_dim"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        learning_rate_cls=params["learning_rate_cls"],
        weight_decay_cls=params["weight_decay_cls"],
        classification_weight=params["classification_weight"],
        mean=norm_mean,
        std=norm_std,
    )

    early_stopping = EarlyStopping(
        monitor="val_vae_loss",
        patience=config.hpo_early_stopping_patience,
        mode="min",
    )

    if sweep_mode:
        trainer = Trainer(
            max_epochs=config.guided_vae_epochs,
            accelerator="auto",
            devices=1,
            logger=mlf_logger,
            callbacks=[early_stopping],
            enable_progress_bar=False,
            enable_checkpointing=False,
            log_every_n_steps=10,
        )
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        cb = trainer.callback_metrics
        return {
            "val_loss": cb.get("val_loss", float("inf")).item(),
            "val_vae_loss": cb.get("val_vae_loss", float("inf")).item(),
            "val_cls_loss": cb.get("val_cls_loss", float("inf")).item(),
            "val_acc": cb.get("val_acc", 0.0).item(),
        }

    # Full training: checkpoints, TensorBoard, artifact upload
    run_checkpoint_dir = CHECKPOINTS_DIR / config.experiment_name / run_name
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_checkpoint_dir,
        filename="guided-vae-{epoch:02d}-{val_vae_loss:.4f}",
        monitor="val_vae_loss",
        mode="min",
        save_last=True,
        save_top_k=5,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=OUTPUT_DIR,
        name="tensorboard_logs",
    )
    trainer = Trainer(
        max_epochs=config.guided_vae_epochs,
        accelerator="auto",
        devices=1,
        logger=[tensorboard_logger, mlf_logger],
        callbacks=[checkpoint_callback, early_stopping],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    best_model_path = run_checkpoint_dir / "best.ckpt"
    trainer.save_checkpoint(best_model_path)
    log_artifact(
        checkpoint_dir=run_checkpoint_dir,
        mlflow_logger=mlf_logger,
        model_path=best_model_path,
    )

    return None


@click.command(help="Retrain the Guided VAE with fixed best-known hyperparameters.")
@click.option(
    "--dataset_filename",
    default="cached_dataset_rich_sc2egset.pt",
    show_default=True,
    help="Cached dataset filename inside DATA_DIR.",
)
@click.option("--experiment_name", required=True, help="MLFlow experiment name.")
@click.option(
    "--run_name", default="guided_vae_best", show_default=True, help="MLFlow run name."
)
@click.option(
    "--latent_dim",
    type=int,
    default=32,
    show_default=True,
    help="VAE latent dimensionality.",
)
@click.option(
    "--encoder_hidden_dims",
    default="256,128,64",
    show_default=True,
    help="Encoder hidden layer sizes, comma-separated (e.g. '256,128,64').",
)
@click.option(
    "--supervised_dim",
    type=int,
    default=2,
    show_default=True,
    help="Supervised head output dimension.",
)
@click.option("--batch_size", type=int, default=64, show_default=True)
@click.option(
    "--learning_rate",
    type=float,
    default=1e-4,
    show_default=True,
    help="VAE learning rate.",
)
@click.option(
    "--weight_decay",
    type=float,
    default=1e-5,
    show_default=True,
    help="VAE weight decay.",
)
@click.option(
    "--learning_rate_cls",
    type=float,
    default=1e-4,
    show_default=True,
    help="Classifier learning rate.",
)
@click.option(
    "--weight_decay_cls",
    type=float,
    default=1e-4,
    show_default=True,
    help="Classifier weight decay.",
)
@click.option(
    "--classification_weight",
    type=float,
    default=0.5,
    show_default=True,
    help="Weight of classification loss relative to reconstruction loss.",
)
@click.option(
    "--epochs",
    type=int,
    default=100,
    show_default=True,
    help="Maximum training epochs.",
)
@click.option(
    "--mlflow_uri",
    default=DEFAULT_MLFLOW_URI,
    show_default=True,
    help="MLFlow tracking URI.",
)
def main(
    dataset_filename: str,
    experiment_name: str,
    run_name: str,
    latent_dim: int,
    encoder_hidden_dims: str,
    supervised_dim: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    learning_rate_cls: float,
    weight_decay_cls: float,
    classification_weight: float,
    epochs: int,
    mlflow_uri: str,
) -> None:

    hidden_dims = [int(d) for d in encoder_hidden_dims.split(",")]

    dataset_path = DATA_DIR / dataset_filename
    data = load_and_normalize(cached_dataset_filepath=dataset_path)

    config = ExperimentConfig(
        sweep=False,
        pipeline="guided_vae",
        dataset_filename=dataset_filename,
        experiment_name=experiment_name,
        mlflow_tracking_uri=mlflow_uri,
        guided_vae_epochs=epochs,
        run_name=run_name,
    )

    params = {
        "latent_dim": latent_dim,
        "encoder_hidden_dims": hidden_dims,
        "supervised_dim": supervised_dim,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "learning_rate_cls": learning_rate_cls,
        "weight_decay_cls": weight_decay_cls,
        "classification_weight": classification_weight,
    }

    train_guided_pipeline(
        train_X=data.train_X,
        train_y=data.train_y,
        val_X=data.val_X,
        val_y=data.val_y,
        input_dim=data.train_X.shape[-1],
        config=config,
        params=params,
        norm_mean=data.mean,
        norm_std=data.std,
        sweep_mode=False,
    )


if __name__ == "__main__":
    main()
