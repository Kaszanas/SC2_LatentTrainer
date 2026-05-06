"""Standalone training script for the OT-Flow Matching model.

Loads a pre-trained GuidedVAE checkpoint, encodes all training and validation
data into latent space, builds (z_losing, z_winning) pairs per match, and
trains a :class:`~latent_trainer.paths.flow.LitOTFlowMatching` model with
Lightning + MLFlow logging.

Usage::

    uv run python -m latent_trainer.paths.train \\
        --model_path output/checkpoints/my_exp/best.ckpt \\
        --experiment_name latent_flow \\
        --epochs 100
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import lightning as L
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset

from latent_trainer.features.data_utils import load_and_normalize
from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE
from latent_trainer.paths.data import (
    compute_loss_latents,
    compute_win_latents,
    encode_player,
    get_supervised_dim,
)
from latent_trainer.paths.flow import LitOTFlowMatching
from latent_trainer.settings import (
    CHECKPOINTS_DIR,
    DATA_DIR,
    DEFAULT_MLFLOW_URI,
    LOGGING_FORMAT,
    OUTPUT_DIR,
    SEED,
)
from latent_trainer.tracking.mlflow_utils import (
    create_mlflow_logger,
    log_artifact,
    setup_mlflow,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--dataset_filename",
    default="cached_dataset_rich_sc2egset.pt",
    show_default=True,
    help="Filename of the cached dataset placed in DATA_DIR.",
)
@click.option(
    "--model_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    help="Path to the trained GuidedVAE checkpoint (.ckpt).",
)
@click.option(
    "--experiment_name",
    default="latent_flow",
    show_default=True,
    help="MLFlow experiment name.",
)
@click.option(
    "--run_name",
    default="ot_flow_matching",
    show_default=True,
    help="MLFlow run name.",
)
@click.option(
    "--hidden_dim",
    type=int,
    default=256,
    show_default=True,
    help="Hidden layer width for the velocity network.",
)
@click.option(
    "--epochs",
    type=int,
    default=100,
    show_default=True,
    help="Maximum training epochs.",
)
@click.option(
    "--batch_size",
    type=int,
    default=512,
    show_default=True,
    help="Mini-batch size.",
)
@click.option(
    "--learning_rate",
    type=float,
    default=1e-3,
    show_default=True,
    help="Adam learning rate.",
)
@click.option(
    "--mlflow_uri",
    default=DEFAULT_MLFLOW_URI,
    show_default=True,
    help="MLFlow tracking URI.",
)
@click.option(
    "--output_dir",
    default=str(OUTPUT_DIR),
    show_default=True,
    type=click.Path(path_type=Path, resolve_path=True),
    help="Base directory for checkpoints and TensorBoard logs.",
)
def main(
    dataset_filename: str,
    model_path: Path,
    experiment_name: str,
    run_name: str,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mlflow_uri: str,
    output_dir: Path,
) -> None:
    """Train an OT-Flow Matching model to map losing → winning latents."""
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
    L.seed_everything(SEED)

    setup_mlflow(mlflow_tracking_uri=mlflow_uri, experiment_name=experiment_name)

    logger.info("Loading GuidedVAE checkpoint...")
    guided_vae = LitGuidedVAE.load_from_checkpoint(model_path)
    guided_vae.eval()

    logger.info("Loading and normalizing dataset...")
    data = load_and_normalize(cached_dataset_filepath=DATA_DIR / dataset_filename)

    def _make_pairs(
        X: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z0 = encode_player(vae=guided_vae.model, data=X[:, 0, :])
        z1 = encode_player(vae=guided_vae.model, data=X[:, 1, :])
        z_loss = compute_loss_latents(y, z0, z1)
        z_win = compute_win_latents(y, z0, z1)
        return z_loss, z_win

    logger.info("Encoding training split...")
    z_loss_train, z_win_train = _make_pairs(data.train_X, data.train_y)
    logger.info("Encoding validation split...")
    z_loss_val, z_win_val = _make_pairs(data.val_X, data.val_y)

    sup_dim = get_supervised_dim(guided_vae)
    z_loss_train = z_loss_train[:, :sup_dim]
    z_win_train = z_win_train[:, :sup_dim]
    z_loss_val = z_loss_val[:, :sup_dim]
    z_win_val = z_win_val[:, :sup_dim]
    latent_dim = sup_dim
    logger.info(
        f"supervised_dim={sup_dim}  full_latent_dim={guided_vae.model.n_vae_dis}  "
        f"train_pairs={len(z_loss_train)}  val_pairs={len(z_loss_val)}"
    )

    train_loader = DataLoader(
        TensorDataset(z_loss_train, z_win_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(z_loss_val, z_win_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    flow_model = LitOTFlowMatching(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
    )

    run_checkpoint_dir = CHECKPOINTS_DIR / experiment_name / run_name
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_checkpoint_dir,
        filename="flow-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=3,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    tensorboard_logger = TensorBoardLogger(save_dir=output_dir, name="tensorboard_logs")
    mlf_logger = create_mlflow_logger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=mlflow_uri,
    )

    trainer = Trainer(
        max_epochs=epochs,
        logger=[tensorboard_logger, mlf_logger],
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    trainer.fit(
        model=flow_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    best_model_path = run_checkpoint_dir / "best.ckpt"
    trainer.save_checkpoint(best_model_path)
    log_artifact(
        checkpoint_dir=run_checkpoint_dir,
        mlflow_logger=mlf_logger,
        model_path=best_model_path,
    )
    logger.info(f"Flow model saved to {best_model_path}")


if __name__ == "__main__":
    main()
