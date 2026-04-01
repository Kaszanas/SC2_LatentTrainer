"""Train the supervised Guided VAE (``suGuidedVAE``).

Provides both cached (``.pt``) and live (``SC2EGSetDataModule``) data
loading, with optional Optuna hyperparameter search.

MLFlow integration
~~~~~~~~~~~~~~~~~~
Every training run is logged to MLFlow (dual TensorBoard + MLFlow loggers).
When running Optuna HPO, trial runs are nested under a parent run (see
:func:`run_optuna_search`).  Model checkpoints and the final ``.pth`` file
are logged as MLFlow artifacts for easy retrieval.

Usage examples::

    # Cached dataset (fast):
    uv run python -m latent_trainer.models.train_model --cached data/cached_dataset_rich.pt

    # Live SC2EGSet dataset:
    uv run python -m latent_trainer.models.train_model --transform economy_average_vs_outcome

    # Optuna HPO:
    uv run python -m latent_trainer.models.train_model \\
        --cached data/cached_dataset_rich.pt --optuna --n_trials 20
"""

from __future__ import annotations

import logging
import os

import click
import lightning as pl
import mlflow
import optuna
import torch
from lightning.pytorch import Trainer
from sc2_datasets.available_replaypacks import SC2EGSET_DATASET_REPLAYPACKS
from sc2_datasets.lightning.sc2_egset_datamodule import SC2EGSetDataModule
from sc2_datasets.transforms.mmr_vs_result import mmr_vs_result
from sc2_datasets.transforms.pytorch.economy_vs_outcome import (
    economy_average_vs_outcome,
)
from sc2_datasets.transforms.utils import select_outcome_1v1
from torch.utils.data import DataLoader, Dataset

from latent_trainer.config import DEFAULT_MLFLOW_URI, LOGGING_FORMAT
from latent_trainer.data_utils import load_cached_dataloaders
from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE
from latent_trainer.tracking.mlflow_utils import (
    create_child_mlflow_logger,
    create_mlflow_logger,
    log_checkpoint_artifacts,
    start_parent_run,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------
class SafeDataset(Dataset):
    """Wraps a dataset to catch exceptions in ``__getitem__`` and return ``None``."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        try:
            return self.dataset[idx]
        except Exception:
            return None


def collate_fn_filter_none(batch: list) -> list | None:
    """Custom collate function that filters out ``None`` samples."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def wrap_dataloader(
    dataloader: DataLoader,
    collate_fn=collate_fn_filter_none,
) -> DataLoader:
    """Re-create a DataLoader with :class:`SafeDataset` and custom collate."""
    return DataLoader(
        SafeDataset(dataloader.dataset),
        batch_size=dataloader.batch_size,
        shuffle=isinstance(dataloader.sampler, torch.utils.data.RandomSampler),
        num_workers=dataloader.num_workers,
        collate_fn=collate_fn,
    )


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
# Maps transform name → (transform function, input dimension)
_TRANSFORM_REGISTRY: dict[str, tuple] = {
    "mmr_vs_result": (mmr_vs_result, 2),
    "economy_average_vs_outcome": (economy_average_vs_outcome, 39),
    "select_outcome_1v1": (select_outcome_1v1, 2),
}


def _load_live_data(
    transform: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, int]:
    """Load data via the live ``SC2EGSetDataModule``."""
    selected_transform, input_dim = _TRANSFORM_REGISTRY.get(
        transform,
        (mmr_vs_result, 2),
    )
    sc2_dm = SC2EGSetDataModule(
        unpack_dir="./data/unpack",
        download_dir="./data/download",
        download=True,
        replaypacks=SC2EGSET_DATASET_REPLAYPACKS,
        transform=selected_transform,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    sc2_dm.prepare_data()
    sc2_dm.setup()
    train_dl = wrap_dataloader(sc2_dm.train_dataloader())
    val_dl = wrap_dataloader(sc2_dm.val_dataloader())
    return train_dl, val_dl, input_dim


def _load_cached_data(
    cache_path: str,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, int]:
    """Load data from a pre-processed ``.pt`` cache file."""
    train_dl, val_dl, input_dim = load_cached_dataloaders(
        cache_path=cache_path,
        batch_size=batch_size,
    )
    return train_dl, val_dl, input_dim


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def train_guided(
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    output_dir: str = "output",
    epochs: int = 10,
    nz: int = 16,
    w_cls: float = 200.0,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    lr_c: float = 1e-4,
    weight_decay_c: float = 1e-4,
    test_interval: int = 1,
    mlflow_uri: str = DEFAULT_MLFLOW_URI,
    experiment_name: str = "SC2_GuidedVAE",
    run_name: str | None = None,
    parent_run_id: str | None = None,
) -> LitGuidedVAE:
    """Run a single guided-VAE training run and return the trained model."""
    os.makedirs(output_dir, exist_ok=True)

    model = LitGuidedVAE(
        n_vae_dis=nz,
        lr=lr,
        weight_decay=weight_decay,
        lr_c=lr_c,
        weight_decay_c=weight_decay_c,
        w_cls=w_cls,
        input_dim=input_dim,
    )

    checkpoint_cb = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="model-{epoch:02d}-{val_vae_loss:.4f}",
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
    final_model_path = os.path.join(output_dir, "final_model.pth")
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
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    if mlf_logger.run_id:
        mlflow.set_tracking_uri(mlflow_uri)
        with mlflow.start_run(run_id=mlf_logger.run_id):
            log_checkpoint_artifacts(ckpt_dir, mlflow_uri)
            mlflow.log_artifact(final_model_path)

    logger.info("Training complete.  Model saved to %s", final_model_path)

    return model


def run_optuna_search(
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    output_dir: str = "output",
    n_trials: int = 20,
    optuna_epochs: int = 3,
    optuna_db: str = "sqlite:///optuna_study.db",
    study_name: str = "vae_optimization",
    full_epochs: int = 10,
    mlflow_uri: str = DEFAULT_MLFLOW_URI,
    experiment_name: str = "SC2_GuidedVAE",
) -> optuna.Study:
    """Run Optuna HPO for the guided VAE and retrain with best params.

    All trials are nested under a parent MLFlow run for grouped UI display.
    """
    os.makedirs(output_dir, exist_ok=True)

    tb_logger = pl.pytorch.loggers.TensorBoardLogger(
        save_dir=output_dir,
        name="tensorboard_logs",
    )

    # Open a parent run that all Optuna trials nest under
    with start_parent_run(
        experiment_name=experiment_name,
        run_name="guided_vae_search",
        tracking_uri=mlflow_uri,
        params={"n_trials": n_trials, "optuna_epochs": optuna_epochs},
    ) as parent_run:
        parent_run_id = parent_run.info.run_id

        def objective(trial: optuna.Trial) -> float:
            model = LitGuidedVAE(
                n_vae_dis=trial.suggest_int("nz", 8, 64),
                lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                lr_c=trial.suggest_float("lr_c", 1e-5, 1e-3, log=True),
                weight_decay_c=trial.suggest_float(
                    "weight_decay_c", 1e-6, 1e-3, log=True
                ),
                w_cls=trial.suggest_float("cls", 0.1, 10.0),
                input_dim=input_dim,
            )
            pruning_cb = optuna.integration.PyTorchLightningPruningCallback(
                trial,
                monitor="val_vae_loss",
            )
            trial_tb = pl.pytorch.loggers.TensorBoardLogger(
                save_dir=os.path.join(output_dir, "tensorboard_logs", "optuna_trials"),
                name=f"trial_{trial.number}",
            )
            trial_mlf = create_child_mlflow_logger(
                experiment_name=experiment_name,
                run_name=f"guided_vae_trial_{trial.number}",
                parent_run_id=parent_run_id,
                tracking_uri=mlflow_uri,
                params=trial.params,
            )
            trainer = Trainer(
                max_epochs=optuna_epochs,
                logger=[trial_tb, trial_mlf],
                enable_progress_bar=True,
                callbacks=[pruning_cb],
                accelerator="auto",
                devices=1,
                enable_checkpointing=False,
                log_every_n_steps=10,
            )
            trainer.fit(
                model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )
            return trainer.callback_metrics["val_vae_loss"].item()

        study = optuna.create_study(
            study_name=study_name,
            storage=optuna_db,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=n_trials)

        # Log best trial to TensorBoard + parent MLFlow run
        best = study.best_trial
        tb_logger.log_hyperparams(best.params, {"val_vae_loss": best.value})
        logger.info("Best trial #%d  val_vae_loss=%.4f", best.number, best.value)
        for k, v in best.params.items():
            logger.info("  %s: %s", k, v)

        # Log best results to the parent run
        mlflow.log_metric("best_val_vae_loss", best.value)
        mlflow.log_metric("best_trial_number", best.number)
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.set_tag("source", "optuna_best_trial")

    # Retrain with best params (outside parent run — gets its own run)
    logger.info("Retraining with best hyperparameters\u2026")
    train_guided(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        output_dir=output_dir,
        epochs=full_epochs,
        nz=best.params["nz"],
        w_cls=best.params["cls"],
        lr=best.params["lr"],
        weight_decay=best.params["weight_decay"],
        lr_c=best.params["lr_c"],
        weight_decay_c=best.params["weight_decay_c"],
        mlflow_uri=mlflow_uri,
        experiment_name=experiment_name,
        run_name="guided_vae_best_retrain",
    )

    return study


@click.command()
@click.option(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    show_default=True,
    help="Training batch size.",
)
@click.option(
    "--output",
    default="output",
    show_default=True,
    help="Output directory for results.",
)
@click.option(
    "--epochs",
    default=10,
    type=int,
    show_default=True,
    help="Number of training epochs.",
)
@click.option(
    "--nz", default=16, type=int, show_default=True, help="Bottleneck (latent) size."
)
@click.option(
    "--cls",
    default=200.0,
    type=float,
    show_default=True,
    help="Classification error weight.",
)
@click.option(
    "--num-workers", default=0, type=int, show_default=True, help="DataLoader workers."
)
@click.option(
    "--test-interval",
    default=1,
    type=int,
    show_default=True,
    help="Validation every N epochs.",
)
@click.option(
    "--lr", default=1e-4, type=float, show_default=True, help="VAE learning rate."
)
@click.option(
    "--weight-decay",
    default=1e-5,
    type=float,
    show_default=True,
    help="VAE weight decay.",
)
@click.option(
    "--lr-c",
    default=1e-4,
    type=float,
    show_default=True,
    help="Classifier learning rate.",
)
@click.option(
    "--weight-decay-c",
    default=1e-4,
    type=float,
    show_default=True,
    help="Classifier weight decay.",
)
@click.option(
    "--transform",
    default="economy_average_vs_outcome",
    type=click.Choice(list(_TRANSFORM_REGISTRY)),
    show_default=True,
    help="SC2 transform.",
)
@click.option("--optuna", "use_optuna", is_flag=True, help="Enable Optuna HPO.")
@click.option(
    "--n-trials", default=20, type=int, show_default=True, help="Optuna trial count."
)
@click.option(
    "--optuna-epochs", default=3, type=int, show_default=True, help="Epochs per trial."
)
@click.option(
    "--optuna-db",
    default="sqlite:///optuna_study.db",
    show_default=True,
    help="Optuna DB URL.",
)
@click.option(
    "--study-name",
    default="vae_optimization",
    show_default=True,
    help="Optuna study name.",
)
@click.option(
    "--cached", "cache_path", default=None, type=str, help="Path to cached .pt dataset."
)
@click.option(
    "--mlflow-uri", default="mlruns", show_default=True, help="MLFlow tracking URI."
)
@click.option(
    "--experiment-name",
    default="SC2_GuidedVAE",
    show_default=True,
    help="MLFlow experiment name.",
)
def main(
    batch_size: int,
    output: str,
    epochs: int,
    nz: int,
    cls: float,
    num_workers: int,
    test_interval: int,
    lr: float,
    weight_decay: float,
    lr_c: float,
    weight_decay_c: float,
    transform: str,
    use_optuna: bool,
    n_trials: int,
    optuna_epochs: int,
    optuna_db: str,
    study_name: str,
    cache_path: str | None,
    mlflow_uri: str,
    experiment_name: str,
) -> None:
    """Train the supervised Guided VAE with optional Optuna HPO."""

    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)

    # Load data
    if cache_path and os.path.exists(cache_path):
        logger.info("Using cached dataset: %s", cache_path)
        train_loader, val_loader, input_dim = _load_cached_data(cache_path, batch_size)
    else:
        logger.info("Using live SC2EGSet dataset (transform=%s)", transform)
        train_loader, val_loader, input_dim = _load_live_data(
            transform,
            batch_size,
            num_workers,
        )

    logger.info("Input dimension: %d", input_dim)

    if use_optuna:
        run_optuna_search(
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            output_dir=output,
            n_trials=n_trials,
            optuna_epochs=optuna_epochs,
            optuna_db=optuna_db,
            study_name=study_name,
            full_epochs=epochs,
            mlflow_uri=mlflow_uri,
            experiment_name=experiment_name,
        )
    else:
        train_guided(
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            output_dir=output,
            epochs=epochs,
            nz=nz,
            w_cls=cls,
            lr=lr,
            weight_decay=weight_decay,
            lr_c=lr_c,
            weight_decay_c=weight_decay_c,
            test_interval=test_interval,
            mlflow_uri=mlflow_uri,
            experiment_name=experiment_name,
        )


if __name__ == "__main__":
    main()
