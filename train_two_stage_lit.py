import logging
import argparse
from typing import Any

import lightning as L
import mlflow
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from optuna.integration.mlflow import MLflowCallback
from torch.utils.data import DataLoader, TensorDataset

from latent_trainer.data_utils import extract_latents, load_and_normalize

logger = logging.getLogger(__name__)


# --- VAE Lightning Module ---
class LitVAE(L.LightningModule):
    def __init__(
        self,
        input_dim: int = 203,
        latent_dim: int = 32,
        hidden_dims: list[int] | None = None,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr

        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU(True))
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU(True))
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def _common_step(
        self, batch: tuple[torch.Tensor], batch_idx: int, stage: str
    ) -> torch.Tensor:
        # Batch is just x because dataset is TensorDataset(x)
        (x,) = batch
        recon_x, mu, logvar = self(x)

        # Loss calculation
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss

        # Normalize loss for logging (avg per batch item)
        batch_size = x.size(0)
        self.log(f"{stage}_loss", loss / batch_size, prog_bar=True)
        self.log(f"{stage}_recon_loss", recon_loss / batch_size, prog_bar=False)
        self.log(f"{stage}_kl_loss", kl_loss / batch_size, prog_bar=False)

        return loss

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> None:
        self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


# --- Classifier Lightning Module ---
class LitClassifier(L.LightningModule):
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: list[int] | None = None,
        lr: float = 1e-3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.lr = lr
        self.dropout = dropout

        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Takes latent vectors from BOTH players: [batch, 2*latent_dim]
        layers = []
        in_dim = latent_dim * 2
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.BCELoss()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def _common_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str
    ) -> torch.Tensor:
        z_batch, y_batch = batch
        pred = self(z_batch)
        loss = self.loss_fn(pred, y_batch)

        # Accuracy
        preds = (pred > 0.5).float()
        acc = (preds == y_batch).float().mean() * 100

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)

        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        self._common_step(batch, batch_idx, "val")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


# --- Helpers ---
# load_and_normalize and extract_latents are imported from latent_trainer.data_utils


def run_pipeline(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    input_dim: int,
    device: torch.device,
    args: argparse.Namespace,
    params: dict[str, Any],
    trial_num: int | None = None,
) -> float:
    """Executes the two-stage training pipeline with given parameters."""
    latent_dim = params["latent_dim"]
    vae_lr = params["vae_lr"]
    cls_lr = params["cls_lr"]
    batch_size = params["batch_size"]
    dropout = params.get("dropout", 0.3)
    vae_epochs = params.get("vae_epochs", args.vae_epochs)
    cls_epochs = params.get("cls_epochs", args.cls_epochs)

    vae_hidden_dims = params.get("vae_hidden_dims", None)
    cls_hidden_dims = params.get("cls_hidden_dims", None)

    run_prefix = f"trial_{trial_num}" if trial_num is not None else "run"

    # --- Stage 1: VAE Training ---
    logger.info("\n" + "=" * 60)
    logger.info(f"STAGE 1: VAE Training ({run_prefix})")
    logger.info("=" * 60)

    # Flatten inputs for VAE: [N, 2, F] -> [N*2, F]
    train_flat = train_X.reshape(-1, input_dim)
    val_flat = val_X.reshape(-1, input_dim)

    vae_train_loader = DataLoader(
        TensorDataset(train_flat),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    vae_val_loader = DataLoader(
        TensorDataset(val_flat),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    vae_model = LitVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=vae_hidden_dims,
        lr=vae_lr,
    )

    # Loggers & Callbacks for VAE
    mlflow_logger_vae = MLFlowLogger(
        experiment_name=args.experiment_name, run_name=f"{run_prefix}_stage1_vae"
    )
    checkpoint_callback_vae = ModelCheckpoint(
        dirpath=f"output/checkpoints_vae/{run_prefix}",
        filename="vae-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stop_vae = EarlyStopping(monitor="val_loss", patience=20, mode="min")

    trainer_vae = L.Trainer(
        max_epochs=vae_epochs,
        accelerator="auto",
        devices=1,
        logger=mlflow_logger_vae,
        callbacks=[checkpoint_callback_vae, early_stop_vae],
        enable_progress_bar=True,
    )

    trainer_vae.fit(vae_model, vae_train_loader, vae_val_loader)

    # Load best VAE model
    logger.info(f"Loading best VAE checkpoint: {checkpoint_callback_vae.best_model_path}")
    best_vae = LitVAE.load_from_checkpoint(checkpoint_callback_vae.best_model_path)
    best_vae.eval()
    best_vae.to(device)

    # --- Intermediate: Extract Latents ---
    logger.info("\nExtracting latents...")
    train_z = extract_latents(best_vae, train_X, device, batch_size)
    val_z = extract_latents(best_vae, val_X, device, batch_size)
    logger.info(f"Latents shape - Train: {train_z.shape}, Val: {val_z.shape}")

    # --- Stage 2: Classifier Training ---
    logger.info("\n" + "=" * 60)
    logger.info(f"STAGE 2: Classifier Training ({run_prefix})")
    logger.info("=" * 60)

    # Use unsqueeze(1) to match classifier output shape [N, 1]
    cls_train_loader = DataLoader(
        TensorDataset(train_z, train_y.unsqueeze(1)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    cls_val_loader = DataLoader(
        TensorDataset(val_z, val_y.unsqueeze(1)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    cls_model = LitClassifier(
        latent_dim=latent_dim,
        hidden_dims=cls_hidden_dims,
        lr=cls_lr,
        dropout=dropout,
    )

    # Loggers & Callbacks for Classifier
    mlflow_logger_cls = MLFlowLogger(
        experiment_name=args.experiment_name, run_name=f"{run_prefix}_stage2_classifier"
    )
    checkpoint_callback_cls = ModelCheckpoint(
        dirpath=f"output/checkpoints_cls/{run_prefix}",
        filename="cls-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        monitor="val_acc",
        mode="max",
    )
    early_stop_cls = EarlyStopping(monitor="val_loss", patience=15, mode="min")

    trainer_cls = L.Trainer(
        max_epochs=cls_epochs,
        accelerator="auto",
        devices=1,
        logger=mlflow_logger_cls,
        callbacks=[checkpoint_callback_cls, early_stop_cls],
        enable_progress_bar=True,
    )

    trainer_cls.fit(cls_model, cls_train_loader, cls_val_loader)

    logger.info(
        f"\nBest Classifier Accuracy: {checkpoint_callback_cls.best_model_score:.2f}%"
    )

    # Return best accuracy as a float
    return checkpoint_callback_cls.best_model_score.item()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage VAE training (Lightning + MLFlow)"
    )
    parser.add_argument(
        "--cache", default="data/cached_dataset_rich.pt", help="Path to cached dataset"
    )
    parser.add_argument(
        "--latent-dim", type=int, default=32, help="VAE latent dimension"
    )
    parser.add_argument(
        "--vae-epochs", type=int, default=200, help="Max epochs for VAE stage"
    )
    parser.add_argument(
        "--cls-epochs", type=int, default=100, help="Max epochs for classifier stage"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--vae-lr", type=float, default=1e-3, help="VAE learning rate")
    parser.add_argument(
        "--cls-lr", type=float, default=1e-3, help="Classifier learning rate"
    )
    parser.add_argument(
        "--experiment-name",
        default="SC2_Latent_TwoStage",
        help="MLFlow experiment name",
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Run hyperparameter sweep with Optuna"
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Number of trials for sweep"
    )
    args = parser.parse_args()

    # --- Setup ---
    L.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Data ---
    train_X, train_y, val_X, val_y, mean, std = load_and_normalize(args.cache)
    input_dim = train_X.shape[-1]

    if args.sweep:
        logger.info("Starting Optuna Hyperparameter Sweep...")

        def objective(trial):
            # Define architecture search space
            vae_arch_name = trial.suggest_categorical(
                "vae_arch", ["small", "medium", "large"]
            )
            vae_archs = {
                "small": [128, 64],
                "medium": [256, 128],
                "large": [512, 256, 128],
            }

            cls_arch_name = trial.suggest_categorical("cls_arch", ["shallow", "deep"])
            cls_archs = {
                "shallow": [64],
                "deep": [128, 64],
            }

            params = {
                "latent_dim": trial.suggest_int("latent_dim", 16, 64),
                "vae_lr": trial.suggest_float("vae_lr", 1e-4, 1e-2, log=True),
                "cls_lr": trial.suggest_float("cls_lr", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "vae_hidden_dims": vae_archs[vae_arch_name],
                "cls_hidden_dims": cls_archs[cls_arch_name],
            }

            acc = run_pipeline(
                train_X,
                train_y,
                val_X,
                val_y,
                input_dim,
                device,
                args,
                params,
                trial_num=trial.number,
            )
            return acc

        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name="val_acc",
            mlflow_kwargs={"experiment_name": args.experiment_name},
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials, callbacks=[mlflow_callback])

        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best params: {study.best_trial.params}")

    else:
        # Standard Single Run
        params = {
            "latent_dim": args.latent_dim,
            "vae_lr": args.vae_lr,
            "cls_lr": args.cls_lr,
            "batch_size": args.batch_size,
            "dropout": 0.3,  # Default
        }
        run_pipeline(train_X, train_y, val_X, val_y, input_dim, device, args, params)
        logger.info("Two-stage training complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
