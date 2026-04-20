"""Guided-VAE pipeline HPO and best-trial retraining.

Provides two entry-points for the supervised Guided-VAE pipeline:

* :func:`run_guided_vae_hpo` — Optuna sweep (single-process, no Ray).
* :func:`run_guided_vae_best` — loads the best Optuna trial and retrains.
"""

from __future__ import annotations

import logging

import lightning as pl
import mlflow
import optuna
from lightning.pytorch.callbacks import EarlyStopping

from latent_trainer.configs.experiment_config import ExperimentConfig
from latent_trainer.configs.search_space import (
    get_guided_vae_search_space,
    reconstruct_hidden_dims,
)
from latent_trainer.features.data_utils import (
    load_and_normalize,
    load_cached_dataloaders,
)
from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE
from latent_trainer.models.train_guided import train_guided
from latent_trainer.settings import DATA_DIR, OUTPUT_DIR, SEED
from latent_trainer.tracking.mlflow_utils import (
    create_child_mlflow_logger,
    start_parent_run,
)

logger = logging.getLogger(__name__)


def run_guided_vae_hyperparameter_search(config: ExperimentConfig) -> optuna.Study:
    """Run an Optuna HPO sweep for the Guided-VAE pipeline.

    Each trial creates its own DataLoaders (batch size is a search param),
    trains for a fixed number of screening epochs, and reports
    ``val_vae_loss`` to Optuna.  All trials are nested under a parent
    MLFlow run.

    Returns
    -------
    optuna.Study
        Completed study for further analysis / logging.
    """
    data = load_and_normalize(DATA_DIR / config.dataset_filename)
    input_dim = data.train_X.shape[-1]

    with start_parent_run(
        experiment_name=config.experiment_name,
        run_name="guided_vae_hpo",
        tracking_uri=config.mlflow_tracking_uri,
        params={"n_trials": config.n_trials, "search_algorithm": "Optuna (TPE)"},
    ) as parent_run:
        parent_run_id = parent_run.info.run_id

        def objective(trial: optuna.Trial) -> float:
            pl.seed_everything(SEED)
            params = get_guided_vae_search_space(trial=trial)
            batch_size: int = params["batch_size"]

            train_loader, val_loader, _ = load_cached_dataloaders(
                cache_path=DATA_DIR / config.dataset_filename,
                batch_size=batch_size,
            )

            mlf_trial = create_child_mlflow_logger(
                experiment_name=config.experiment_name,
                run_name=f"guided_vae_trial_{trial.number}",
                parent_run_id=parent_run_id,
                tracking_uri=config.mlflow_tracking_uri,
                params=params,
            )

            model = LitGuidedVAE(
                input_dim=input_dim,
                encoder_hidden_dims=params["encoder_hidden_dims"],
                supervised_dim=params["supervised_dim"],
                vae_latent_dim=params["nz"],
                learning_rate=params["lr"],
                weight_decay=params["weight_decay"],
                learning_rate_classifier=params["lr_c"],
                weight_decay_c=params["weight_decay_c"],
                classification_weight=params["cls"],
            )
            early_stopping = EarlyStopping(
                monitor="val_vae_loss",
                patience=7,
                mode="min",
            )

            trainer = pl.Trainer(
                max_epochs=config.guided_vae_epochs,
                accelerator="auto",
                devices=1,
                logger=mlf_trial,
                callbacks=[early_stopping],
                enable_progress_bar=True,
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
            study_name=config.experiment_name,
            storage=config.optuna_db,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            load_if_exists=True,
        )
        study.optimize(func=objective, n_trials=config.n_trials)

        mlflow.log_metric("best_val_vae_loss", study.best_trial.value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_trial.params.items()})

    logger.info(
        f"Guided-VAE HPO complete.  Best trial {study.best_trial.number}  val_vae_loss={study.best_trial.value:.4f}",
    )
    return study


def run_guided_vae_best(config: ExperimentConfig) -> None:
    """Load the best Optuna trial and run a full Guided-VAE training.

    Uses all training epochs (``config.guided_vae_epochs``) and writes
    checkpoints + MLFlow artifacts via :func:`train_guided`.
    """
    study = optuna.load_study(
        study_name=config.experiment_name,
        storage=config.optuna_db,
    )
    best = study.best_trial
    flat_params = best.params
    logger.info(
        f"Loaded best guided-VAE trial {best.number}  val_vae_loss={best.value}",
    )

    params = {
        **flat_params,
        "encoder_hidden_dims": reconstruct_hidden_dims(
            params=flat_params,
            prefix="enc",
        ),
    }

    batch_size: int = params["batch_size"]
    train_loader, val_loader, input_dim = load_cached_dataloaders(
        cache_path=DATA_DIR / config.dataset_filename,
        batch_size=batch_size,
    )
    pl.seed_everything(SEED)

    train_guided(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        output_dir=OUTPUT_DIR,
        epochs=config.guided_vae_epochs,
        supervised_dim=params["supervised_dim"],
        vae_latent_dim=params["nz"],
        classification_weight=params["cls"],
        learning_rate=params["lr"],
        weight_decay=params["weight_decay"],
        learning_rate_classifier=params["lr_c"],
        weight_decay_c=params["weight_decay_c"],
        encoder_hidden_dims=params["encoder_hidden_dims"],
        mlflow_uri=config.mlflow_tracking_uri,
        experiment_name=config.experiment_name,
        run_name="guided_vae_best",
    )
    logger.info("Guided-VAE best-params run complete.")
