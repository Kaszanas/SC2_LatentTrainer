"""Train models module."""

from __future__ import print_function
import click
import optuna
import logging
from pathlib import Path
import os

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
import sys

# Add the project root to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from latent_trainer.models.guided_vae import Classifier, suGuidedVAE
from latent_trainer.models.losses import loss_supervised

from latent_trainer.config import LOGGING_FORMAT

from sc2_datasets.lightning.sc2_egset_datamodule import SC2EGSetDataModule

from sc2_datasets.available_replaypacks import SC2EGSET_DATASET_REPLAYPACKS
from sc2_datasets.transforms.mmr_vs_result import mmr_vs_result
from sc2_datasets.transforms.pytorch.economy_vs_outcome import (
    economy_average_vs_outcome,
)

from sc2_datasets.transforms.utils import select_outcome_1v1
import lightning as L
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, Dataset


class CachedSC2Dataset(Dataset):
    """Loads pre-processed tensors from a .pt cache file for instant access."""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SafeDataset(Dataset):
    """Wraps a dataset to catch exceptions in __getitem__ and return None instead."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            return self.dataset[idx]
        except Exception:
            return None


def collate_fn_filter_none(batch):
    """Custom collate function that filters out None samples."""
    batch = [
        item for item in batch if item is not None and all(x is not None for x in item)
    ]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def wrap_dataloader(dataloader, collate_fn=collate_fn_filter_none):
    """Re-create a DataLoader with SafeDataset and custom collate_fn."""
    return DataLoader(
        dataset=SafeDataset(dataloader.dataset),
        batch_size=dataloader.batch_size,
        num_workers=dataloader.num_workers,
        collate_fn=collate_fn,
        shuffle=isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler),
    )


def load_cached_dataloaders(cache_path, batch_size):
    """Load pre-processed dataset from cache, normalize features, and return DataLoaders."""
    logging.info(f"Loading cached dataset from {cache_path}...")
    cached = torch.load(cache_path, weights_only=True)

    train_feats = cached["train_features"].float()  # [N, 2, 39]
    val_feats = cached["val_features"].float()

    # Normalize features: fit on train, apply to both
    # Reshape to [N*2, 39] for per-feature normalization across all players
    orig_shape = train_feats.shape
    train_flat = train_feats.reshape(-1, orig_shape[-1])  # [N*2, 39]
    val_flat = val_feats.reshape(-1, orig_shape[-1])

    # Compute mean and std from training data
    mean = train_flat.mean(dim=0, keepdim=True)  # [1, 39]
    std = train_flat.std(dim=0, keepdim=True) + 1e-8  # avoid division by zero

    # Apply normalization
    train_flat = (train_flat - mean) / std
    val_flat = (val_flat - mean) / std

    train_feats = train_flat.reshape(orig_shape)
    val_feats = val_flat.reshape(val_feats.shape)

    logging.info(
        f"  Feature normalization applied (mean={mean.mean():.2f}, std={std.mean():.2f})"
    )
    logging.info(f"  Train samples: {len(train_feats)}, Val samples: {len(val_feats)}")
    logging.info(f"  Feature shape: {train_feats.shape}")

    train_dataset = CachedSC2Dataset(train_feats, cached["train_labels"])
    val_dataset = CachedSC2Dataset(val_feats, cached["val_labels"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_feats.shape[-1]


class LitGuidedVAE(L.LightningModule):
    """Lightning module for supervised Guided VAE training with adversarial classifier."""

    def __init__(
        self,
        n_vae_dis=16,
        lr=1e-4,
        weight_decay=1e-5,
        lr_c=1e-4,
        weight_decay_c=1e-4,
        w_cls=50000.0,
        input_dim=2,
    ):
        """Initialize the LitGuidedVAE module.
        Args:
            n_vae_dis (int): Size of the VAE latent distribution.
            lr (float): Learning rate for the VAE optimizer.
            weight_decay (float): Weight decay for the VAE optimizer.
            lr_c (float): Learning rate for the classifier optimizer.
            weight_decay_c (float): Weight decay for the classifier optimizer.
            w_cls (float): Weight for the classification loss.
            input_dim (int): Input feature dimension (depends on transform).
        """

        super().__init__()
        self.save_hyperparameters()

        # Set manual optimization flag to use multiple optimizers
        self.automatic_optimization = False

        # Main VAE model
        self.model = suGuidedVAE(n_vae_dis=n_vae_dis, input_dim=input_dim)

        # Classifier for adversarial training
        self.classifier = Classifier(n_vae_dis=n_vae_dis)

        # Store hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_c = lr_c
        self.weight_decay_c = weight_decay_c
        self.w_cls = w_cls

        # Track metrics
        self.total_valid_samples = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        # Get optimizers for manual optimization
        opt_vae, opt_cls, opt_adv = self.optimizers()

        data, label = batch

        # Process labels
        if label.dtype == torch.int8:
            label = label.float()

        if len(label.shape) == 1:
            label = label.unsqueeze(1)

        # Expand labels to match data dimensions if data is 3D [batch, num_players, features]
        if len(data.shape) == 3 and len(label.shape) == 2:
            # Expand label from [batch, 1] to [batch, num_players, 1]
            label = label.unsqueeze(1).expand(-1, data.shape[1], -1)

        # Ensure labels are float after all transformations
        if label.dtype != torch.float32:
            label = label.float()

        # Filter out invalid labels (-1)
        # For 3D data [batch, num_players, features], we need to filter carefully
        if len(data.shape) == 3 and len(label.shape) == 3:
            # Filter out samples where ANY player has invalid label
            valid_indices = (label != -1).all(dim=1).all(dim=1)  # [batch]
            if valid_indices.sum() == 0:
                self.log("train_skip_batch", 1)
                return None
            valid_data = data[valid_indices]  # [valid_batch, num_players, features]
            valid_label = label[valid_indices]  # [valid_batch, num_players, 1]
        else:
            # For 2D data, use simple filtering
            valid_indices = (label != -1).squeeze()
            if valid_indices.sum() == 0:
                self.log("train_skip_batch", 1)
                return None
            valid_data = data[valid_indices]
            valid_label = label[valid_indices]

        # Ensure labels are in valid range [0,1]
        valid_label = torch.clamp(valid_label, 0, 1)

        # Track valid samples
        batch_valid_samples = valid_indices.sum().item()
        self.total_valid_samples += batch_valid_samples

        # Step 1: VAE step
        # Clear gradients for optimizer 1
        opt_vae.zero_grad()

        recon_batch, mu, logvar, re = self.model(valid_data)
        loss_list = loss_supervised(recon_batch, valid_data, mu, logvar)
        vae_loss = loss_list[0]
        loss_cls = F.binary_cross_entropy(re, valid_label, reduction="sum")
        vae_total_loss = vae_loss + loss_cls * self.w_cls

        # Calculate accuracy
        pred = (re > 0.5).float()
        acc = pred.eq(valid_label).sum().item() / valid_label.numel() * 100

        # Log metrics with on_step=True and on_epoch=True for better TensorBoard tracking
        self.log("train_vae_loss", vae_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_cls_loss", loss_cls, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_vae_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_valid_samples", batch_valid_samples, on_epoch=True)

        # Manually backpropagate and optimize
        self.manual_backward(vae_total_loss)
        opt_vae.step()

        # Step 2: Classifier step
        # Clear gradients for optimizer 2
        opt_cls.zero_grad()

        mu, logvar = self.model.encode(valid_data)
        z = self.model.reparameterize(mu, logvar).detach()
        # Slice latent dimensions (last dimension), excluding first latent dim used for classification
        if len(z.shape) == 3:
            z_cls = z[:, :, 1:]  # [batch, num_players, latent_dim-1]
        else:
            z_cls = z[:, 1:]  # [batch, latent_dim-1]
        cls1 = self.classifier(z_cls)
        cls_loss = F.binary_cross_entropy(cls1, valid_label, reduction="sum")
        cls_loss *= self.w_cls

        # Calculate accuracy
        pred = (cls1 > 0.5).float()
        acc = pred.eq(valid_label).sum().item() / valid_label.numel() * 100

        # Log metrics with on_step=True and on_epoch=True for better TensorBoard tracking
        self.log("train_c_loss", cls_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_c_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        # Manually backpropagate and optimize
        self.manual_backward(cls_loss)
        opt_cls.step()

        # Step 3: Adversarial step
        # Clear gradients for optimizer 3
        opt_adv.zero_grad()

        mu, logvar = self.model.encode(valid_data)
        z = self.model.reparameterize(mu, logvar)
        # Slice latent dimensions (last dimension), excluding first latent dim used for classification
        if len(z.shape) == 3:
            z_cls = z[:, :, 1:]  # [batch, num_players, latent_dim-1]
        else:
            z_cls = z[:, 1:]  # [batch, latent_dim-1]
        cls2 = self.classifier(z_cls)
        label1 = torch.empty_like(valid_label).fill_(0.5)
        adv_loss = F.binary_cross_entropy(cls2, label1, reduction="sum")
        adv_loss *= 0.0  # Disabled: adversarial step was cancelling classifier learning

        # Log metrics with on_step=True and on_epoch=True for better TensorBoard tracking
        self.log("train_adv_loss", adv_loss, prog_bar=True, on_step=True, on_epoch=True)

        # Manually backpropagate and optimize
        self.manual_backward(adv_loss)
        opt_adv.step()

        # Return combined loss for logging purposes only
        return vae_total_loss + cls_loss + adv_loss

    def on_train_epoch_end(self):
        self.log("epoch_valid_samples", self.total_valid_samples)
        self.total_valid_samples = 0  # Reset for next epoch

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None
        data, label = batch

        # Process labels
        if label.dtype == torch.int8:
            label = label.float()

        if len(label.shape) == 1:
            label = label.unsqueeze(1)

        # Expand labels to match data dimensions if data is 3D [batch, num_players, features]
        if len(data.shape) == 3 and len(label.shape) == 2:
            # Expand label from [batch, 1] to [batch, num_players, 1]
            label = label.unsqueeze(1).expand(-1, data.shape[1], -1)

        # Ensure labels are float after all transformations
        if label.dtype != torch.float32:
            label = label.float()

        # Filter out invalid labels (-1)
        # For 3D data [batch, num_players, features], we need to filter carefully
        if len(data.shape) == 3 and len(label.shape) == 3:
            # Filter out samples where ANY player has invalid label
            valid_indices = (label != -1).all(dim=1).all(dim=1)  # [batch]
            if valid_indices.sum() == 0:
                return None
            valid_data = data[valid_indices]  # [valid_batch, num_players, features]
            valid_label = label[valid_indices]  # [valid_batch, num_players, 1]
        else:
            # For 2D data, use simple filtering
            valid_indices = (label != -1).squeeze()
            if valid_indices.sum() == 0:
                return None
            valid_data = data[valid_indices]
            valid_label = label[valid_indices]

        # Ensure labels are in valid range [0,1]
        valid_label = torch.clamp(valid_label, 0, 1)

        # Run forward pass
        recon_batch, mu, logvar, re = self.model(valid_data)

        # Calculate losses
        loss_list = loss_supervised(recon_batch, valid_data, mu, logvar)
        vae_loss = loss_list[0]
        loss_cls = F.binary_cross_entropy(re, valid_label, reduction="sum")

        # Calculate accuracy
        pred = (re > 0.5).float()
        acc = pred.eq(valid_label).sum().item() / valid_label.numel() * 100

        # Log metrics with enhanced settings for better visualization
        self.log("val_vae_loss", vae_loss, prog_bar=True, sync_dist=True)
        self.log("val_cls_loss", loss_cls, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)

        return vae_loss

    def configure_optimizers(self):
        # Optimizer for the VAE
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Optimizer for the Classifier
        optimizer_c = optim.Adam(
            self.classifier.parameters(), lr=self.lr_c, weight_decay=self.weight_decay_c
        )

        # Adversarial optimizer (uses VAE params but trained adversarially)
        optimizer_adv = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        return [optimizer, optimizer_c, optimizer_adv], []  # No schedulers


# Training is now handled in the LitGuidedVAE class training_step method

# The objective function is now defined inside the main function when using Optuna


# CLICK command line interface
@click.command()
@click.option(
    "--batch-size",
    "-b",
    default=128,
    help="input batch size for training (default: 128)",
    type=int,
)
@click.option("--output", default="output", help="output directory for results")
@click.option(
    "--epochs", default=10, help="number of epochs to train (default: 10)", type=int
)
@click.option("--nz", default=16, help="bottleneck size", type=int)
@click.option(
    "--cls",
    default=200.0,
    help="classification error weight for supervised Guided-VAE",
    type=float,
)
@click.option(
    "--num_workers", default=0, help="number of workers for dataloader", type=int
)
@click.option("--test_interval", default=1, help="interval for testing", type=int)
@click.option("--lr", default=1e-4, help="learning rate", type=float)
@click.option("--weight_decay", default=1e-5, help="weight decay", type=float)
@click.option(
    "--lr_c",
    default=1e-4,
    help="classifier learning rate(in supervised version)",
    type=float,
)
@click.option(
    "--weight_decay_c",
    default=1e-4,
    help="classifier weight decay(in supervised version)",
    type=float,
)
@click.option(
    "--transform",
    default="mmr_vs_result",
    type=click.Choice(["mmr_vs_result", "economy_average_vs_outcome"]),
    help="which transform to use",
)
@click.option(
    "--optuna",
    "use_optuna",
    is_flag=True,
    help="use Optuna for hyperparameter optimization",
)
@click.option(
    "--n_trials",
    default=20,
    help="number of Optuna trials (only used with --optuna)",
    type=int,
)
@click.option(
    "--optuna_epochs",
    default=3,
    help="number of epochs per trial for Optuna (only used with --optuna)",
    type=int,
)
@click.option(
    "--optuna_db",
    default="sqlite:///optuna_study.db",
    help="Optuna database URL for dashboard (only used with --optuna)",
    type=str,
)
@click.option(
    "--study_name",
    default="vae_optimization",
    help="Optuna study name (only used with --optuna)",
    type=str,
)
@click.option(
    "--cached",
    "cache_path",
    default=None,
    type=str,
    help="Path to cached dataset .pt file (from preprocess_dataset.py). Skips slow SC2 data loading.",
)
def main(
    batch_size,
    output,
    epochs,
    nz,
    cls,
    num_workers,
    test_interval,
    lr,
    weight_decay,
    lr_c,
    weight_decay_c,
    transform,
    use_optuna,
    n_trials,
    optuna_epochs,
    optuna_db,
    study_name,
    cache_path,
):
    """Main function to parse arguments and start training.
    Args:
        batch_size (int): Batch size for training.
        output (str): Output directory for results.
        epochs (int): Number of epochs to train.
        nz (int): Bottleneck size.
        cls (float): Classification error weight for supervised Guided-VAE.
        num_workers (int): Number of workers for dataloader.
        test_interval (int): Interval for testing.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        lr_c (float): Classifier learning rate (in supervised version).
        weight_decay_c (float): Classifier weight decay (in supervised version).
        transform (str): Which transform to use.
        use_optuna (bool): Whether to use Optuna for hyperparameter optimization.
        n_trials (int): Number of Optuna trials (only used with --optuna).
        optuna_epochs (int): Number of epochs per trial for Optuna (only used with --optuna).
        optuna_db (str): Optuna database URL for dashboard (only used with --optuna).
        study_name (str): Optuna study name (only used with --optuna).
    """
    # Set up more verbose logging to help diagnose issues
    logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT)
    logging.info("Starting model training...")

    download_path = Path("./data/download").resolve().as_posix()
    unpack_path = Path("./data/unpack").resolve().as_posix()

    logging.info(f"{download_path=}")
    logging.info(f"{unpack_path=}")

    if not os.path.exists(output):
        os.mkdir(output)

    torch.manual_seed(1024)

    # Select the appropriate transform
    transform_map = {
        "mmr_vs_result": mmr_vs_result,
        "economy_average_vs_outcome": economy_average_vs_outcome,
        "select_outcome_1v1": select_outcome_1v1,
        # 'average_player_stats': average_player_stats,  # Add this when available
    }
    selected_transform = transform_map.get(transform, mmr_vs_result)

    # Determine input dimension based on transform
    input_dim_map = {
        "mmr_vs_result": 2,
        "economy_average_vs_outcome": 39,
        "select_outcome_1v1": 2,
    }
    input_dim = input_dim_map.get(transform, 2)

    # Load data: either from cache or from SC2EGSet datamodule
    use_cache = cache_path is not None and os.path.exists(cache_path)

    if use_cache:
        logging.info(f"Using cached dataset from: {cache_path}")
        train_loader, val_loader, cached_input_dim = load_cached_dataloaders(
            cache_path, batch_size
        )
        input_dim = cached_input_dim
    else:
        logging.info(f"Using live SC2EGSet dataset with transform: {transform}")
        sc2_egset_datamodule = SC2EGSetDataModule(
            unpack_dir="./data/unpack",
            download_dir="./data/download",
            download=True,
            replaypacks=SC2EGSET_DATASET_REPLAYPACKS,
            transform=selected_transform,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        sc2_egset_datamodule.prepare_data()
        sc2_egset_datamodule.setup()

    logging.info(f"Input dimension: {input_dim}")

    # Set up callbacks
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(output, "checkpoints"),
        filename="model-{epoch:02d}-{val_vae_loss:.4f}",
        monitor="val_vae_loss",
        mode="min",
        save_last=True,
        save_top_k=3,
    )

    early_stopping = L.pytorch.callbacks.EarlyStopping(
        monitor="val_vae_loss", patience=5, mode="min"
    )

    # TensorBoard logger
    tb_logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=output,
        name="tensorboard_logs",
        version=None,  # Use root directory directly for cleaner access
    )

    # Check if Optuna optimization is requested
    if use_optuna:
        logging.info("Starting Optuna hyperparameter optimization...")
        logging.info(f"Running {n_trials} trials with {optuna_epochs} epochs each")
        logging.info(f"Study will be saved to: {optuna_db}")
        logging.info(f"Study name: {study_name}")
        logging.info(f"To view Optuna Dashboard, run: optuna-dashboard {optuna_db}")

        def objective(trial):
            # Suggest hyperparameters
            nz_trial = trial.suggest_int("nz", 8, 64)
            cls_trial = trial.suggest_float("cls", 0.1, 10.0)
            lr_trial = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            weight_decay_trial = trial.suggest_float(
                "weight_decay", 1e-6, 1e-3, log=True
            )
            lr_c_trial = trial.suggest_float("lr_c", 1e-5, 1e-3, log=True)
            weight_decay_c_trial = trial.suggest_float(
                "weight_decay_c", 1e-6, 1e-3, log=True
            )

            # Create Lightning model with suggested hyperparameters
            model = LitGuidedVAE(
                n_vae_dis=nz_trial,
                lr=lr_trial,
                weight_decay=weight_decay_trial,
                lr_c=lr_c_trial,
                weight_decay_c=weight_decay_c_trial,
                w_cls=cls_trial,
                input_dim=input_dim,
            )

            # Pruning callback for Optuna
            pruning_callback = optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor="val_vae_loss"
            )

            # Logger for this trial
            trial_logger = L.pytorch.loggers.TensorBoardLogger(
                save_dir=os.path.join(output, "tensorboard_logs", "optuna_trials"),
                name=f"trial_{trial.number}",
                version=None,  # Use root directory directly for cleaner access
            )

            # Create trainer with fewer epochs for trial
            trainer = Trainer(
                max_epochs=optuna_epochs,
                logger=trial_logger,
                enable_progress_bar=True,
                callbacks=[pruning_callback],
                accelerator="auto",
                devices=1,
                enable_checkpointing=False,
                log_every_n_steps=10,
            )

            # Get the dataloaders directly to avoid the prepare_data issue
            train_loader = wrap_dataloader(sc2_egset_datamodule.train_dataloader())
            val_loader = wrap_dataloader(sc2_egset_datamodule.val_dataloader())

            # Train model with explicit dataloaders instead of the datamodule
            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )

            # Return the final validation loss
            return trainer.callback_metrics["val_vae_loss"].item()

        # Create Optuna study with database storage
        study = optuna.create_study(
            study_name=study_name,
            storage=optuna_db,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            load_if_exists=True,  # Allow resuming if study exists
        )

        # Run optimization
        study.optimize(objective, n_trials=n_trials)

        # Print results and log hyperparameters to TensorBoard
        hparam_dict = study.best_trial.params
        metric_dict = {"val_vae_loss": study.best_trial.value}
        tb_logger.log_hyperparams(hparam_dict, metric_dict)

        logging.info("Optuna optimization completed!")
        logging.info(f"Best trial: {study.best_trial.number}")
        logging.info(f"Best value (loss): {study.best_trial.value}")
        logging.info("Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            logging.info(f"  {key}: {value}")

        logging.info(f"\n{'=' * 60}")
        logging.info("To view the Optuna Dashboard with all trials, run:")
        logging.info(f"  optuna-dashboard {optuna_db}")
        logging.info(f"{'=' * 60}\n")

        # Save best hyperparameters
        best_params_path = os.path.join(output, "best_hyperparameters.txt")
        with open(best_params_path, "w") as f:
            f.write(f"Optuna Study: {study_name}\n")
            f.write(f"Database: {optuna_db}\n")
            f.write(f"Total trials: {len(study.trials)}\n")
            f.write(f"Best trial: {study.best_trial.number}\n")
            f.write(f"Best value (loss): {study.best_trial.value}\n")
            f.write("\nBest hyperparameters:\n")
            for key, value in study.best_trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nTo view dashboard: optuna-dashboard {optuna_db}\n")
        logging.info(f"Best hyperparameters saved to {best_params_path}")

        # Train final model with best hyperparameters
        logging.info("Training final model with best hyperparameters...")
        best_params = study.best_trial.params

        # Create model with best parameters
        model = LitGuidedVAE(
            n_vae_dis=best_params["nz"],
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
            lr_c=best_params["lr_c"],
            weight_decay_c=best_params["weight_decay_c"],
            w_cls=best_params["cls"],
            input_dim=input_dim,
        )

        # Train with full epochs
        final_trainer = Trainer(
            max_epochs=epochs,
            logger=tb_logger,
            enable_progress_bar=True,
            callbacks=[checkpoint_callback, early_stopping],
            accelerator="auto",
            devices=1,
            log_every_n_steps=10,
        )

        # Get the dataloaders directly
        train_loader = wrap_dataloader(sc2_egset_datamodule.train_dataloader())
        val_loader = wrap_dataloader(sc2_egset_datamodule.val_dataloader())

        # Train with explicit dataloaders
        final_trainer.fit(
            model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        # Save final model in PyTorch format for compatibility
        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": model.model.state_dict(),
                "classifier_state_dict": model.classifier.state_dict(),
                "hyperparameters": best_params,
            },
            f"{output}/final_model.pth",
        )

        print(
            f"Training completed with Optuna! Final model saved to {output}/final_model.pth"
        )
        print(f"Best checkpoints saved in: {os.path.join(output, 'checkpoints')}")

    else:
        # Normal training without Optuna
        logging.info("Starting normal training (without Optuna)...")
        logging.info(
            f"TensorBoard logs will be saved to: {os.path.join(output, 'tensorboard_logs')}"
        )
        logging.info(
            f"To view logs, run: tensorboard --logdir={os.path.join(output, 'tensorboard_logs')}"
        )

        # Create model
        model = LitGuidedVAE(
            n_vae_dis=nz,
            lr=lr,
            weight_decay=weight_decay,
            lr_c=lr_c,
            weight_decay_c=weight_decay_c,
            w_cls=cls,
            input_dim=input_dim,
        )

        # Create trainer
        trainer = Trainer(
            max_epochs=epochs,
            logger=tb_logger,
            enable_progress_bar=True,
            callbacks=[checkpoint_callback, early_stopping],
            accelerator="auto",
            devices=1,
            check_val_every_n_epoch=test_interval,
            log_every_n_steps=10,
        )

        # Get dataloaders
        if not use_cache:
            train_loader = wrap_dataloader(sc2_egset_datamodule.train_dataloader())
            val_loader = wrap_dataloader(sc2_egset_datamodule.val_dataloader())

        # Train the model
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Save final model in PyTorch format for compatibility
        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": model.model.state_dict(),
                "classifier_state_dict": model.classifier.state_dict(),
                "loss": trainer.callback_metrics.get(
                    "train_vae_loss", float("inf")
                ).item(),
            },
            f"{output}/final_model.pth",
        )

        print(f"Training completed! Final model saved to {output}/final_model.pth")
        print(f"Best checkpoints saved in: {os.path.join(output, 'checkpoints')}")

    """Entry point for the script.
    This function is decorated with Click to handle command line arguments.    
    """


if __name__ == "__main__":
    # The Click decorator will parse command line arguments and pass them to main
    main()
