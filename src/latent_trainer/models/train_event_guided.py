"""
One training run of the event-stream guided VAE.

Used by every stage (search, confirm, final). Training draws random prefix
crops; validation runs the validation split once per setting, the whole game
and each fixed cutoff, so ``val_objective`` is deterministic. The test split
is never passed in.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, Subset

from latent_trainer.configs.experiment_config import ExperimentConfig
from latent_trainer.configs.search_space import EVENT_MODEL_KEYS
from latent_trainer.features.data_utils import collate_event_sequences
from latent_trainer.features.type import EventDataBundle, EventSequenceDataset
from latent_trainer.models.lightning.lit_event_guided_vae import LitEventGuidedVAE
from latent_trainer.settings import CHECKPOINTS_DIR
from latent_trainer.tracking.mlflow_utils import (
    create_child_mlflow_logger,
    create_mlflow_logger,
    run_provenance,
)

logger = logging.getLogger(__name__)

MODES = ("search", "confirm", "final")


def validation_tags(cutoffs_min: Sequence[float]) -> list[str]:
    """
    Name the validation settings: the whole game, then each cutoff.

    Parameters
    ----------
    cutoffs_min : Sequence[float]
        Fixed cutoffs in game minutes.

    Returns
    -------
    list[str]
        E.g. ``["full", "5min", "10min", "15min"]``.
    """
    return ["full", *[f"{cutoff:g}min" for cutoff in cutoffs_min]]


def search_subset(n_train: int, fraction: float, seed: int) -> list[int]:
    """
    Choose the fixed, seeded subset of training replays used by search trials.

    Parameters
    ----------
    n_train : int
        Training replays.
    fraction : float
        Share to keep, in ``(0, 1]``.
    seed : int
        Seed; every trial of a search gets the same subset.

    Returns
    -------
    list[int]
        Sorted replay indices.

    Raises
    ------
    ValueError
        If ``fraction`` is not in ``(0, 1]``.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    generator = torch.Generator().manual_seed(seed)
    keep = max(1, round(n_train * fraction))
    return sorted(torch.randperm(n_train, generator=generator)[:keep].tolist())


def make_loaders(
    bundle: EventDataBundle,
    params: dict,
    config: ExperimentConfig,
    seed: int,
    train_indices: list[int] | None = None,
) -> tuple[DataLoader, list[DataLoader], list[str]]:
    """
    Build the training loader and one validation loader per setting.

    Only ``bundle.train`` and ``bundle.val`` are read.

    Parameters
    ----------
    bundle : EventDataBundle
        Loaded events cache.
    params : dict
        Trial configuration (``batch_size``, ``min_crop_s``,
        ``max_events_per_slice``).
    config : ExperimentConfig
        Run configuration (slices, tail, cutoffs, workers).
    seed : int
        Seed of the shuffling order.
    train_indices : list[int] | None
        Restrict training to these replays (search subset), by default None.

    Returns
    -------
    tuple[DataLoader, list[DataLoader], list[str]]
        Training loader, validation loaders and their tags.
    """
    collate = functools.partial(
        collate_event_sequences, max_events_per_slice=params["max_events_per_slice"]
    )
    common = {
        "slice_ticks": config.slice_ticks,
        "drop_tail_frac": config.drop_tail_frac,
    }
    train_set: Dataset = EventSequenceDataset(
        bundle.train, crop="random", min_crop_s=params["min_crop_s"], **common
    )
    if train_indices is not None:
        train_set = Subset(train_set, train_indices)
    workers = {
        "num_workers": config.num_workers,
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=params["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate,
        **workers,
    )
    val_sets = [EventSequenceDataset(bundle.val, crop=None, **common)] + [
        EventSequenceDataset(bundle.val, crop=cutoff * 60.0, **common)
        for cutoff in config.val_cutoffs_min
    ]
    val_loaders = [
        DataLoader(
            dataset,
            batch_size=max(params["batch_size"], 64),
            shuffle=False,
            collate_fn=collate,
            **workers,
        )
        for dataset in val_sets
    ]
    return train_loader, val_loaders, validation_tags(config.val_cutoffs_min)


def build_model(
    bundle: EventDataBundle,
    params: dict,
    config: ExperimentConfig,
    steps_per_epoch: int,
) -> LitEventGuidedVAE:
    """
    Create the Lightning module and fit its input scaling on the training split.

    Parameters
    ----------
    bundle : EventDataBundle
        Loaded events cache.
    params : dict
        Trial configuration.
    config : ExperimentConfig
        Run configuration.
    steps_per_epoch : int
        Training batches per epoch, for the KL warm-up length.

    Returns
    -------
    LitEventGuidedVAE
        The module, ready to train.
    """
    model = LitEventGuidedVAE(
        channel_names=bundle.stat_channel_names,
        unit_vocab=bundle.unit_vocab,
        event_type_names=bundle.event_type_names,
        race_names=bundle.race_names,
        **{key: params[key] for key in EVENT_MODEL_KEYS},
        kl_warmup_steps=int(steps_per_epoch * config.kl_warmup_epochs),
        data_config={
            "slice_ticks": config.slice_ticks,
            "drop_tail_frac": config.drop_tail_frac,
            "min_crop_s": params["min_crop_s"],
            "max_events_per_slice": params["max_events_per_slice"],
            "val_cutoffs_min": list(config.val_cutoffs_min),
            "batch_size": params["batch_size"],
        },
        objective_weights=dict(config.hpo_objective_weights),
    )
    return model.fit_input_transform(bundle.train.stat_values)


def train_event_guided_run(
    bundle: EventDataBundle,
    config: ExperimentConfig,
    params: dict,
    seed: int,
    run_name: str,
    mode: str,
    cache_path: Path,
    parent_run_id: str | None = None,
    train_indices: list[int] | None = None,
    extra_callbacks: Sequence[Callback] = (),
) -> dict:
    """
    Train one event-model configuration with one seed.

    Parameters
    ----------
    bundle : EventDataBundle
        Loaded events cache (only train and val are used).
    config : ExperimentConfig
        Run configuration.
    params : dict
        Trial configuration (model and data keys of the search space).
    seed : int
        Seed of initialisation, crops and shuffling.
    run_name : str
        MLflow run name (and checkpoint folder name in ``final`` mode).
    mode : str
        ``"search"`` (capped epochs, search patience, no checkpoints),
        ``"confirm"`` (full epochs, no checkpoints) or ``"final"`` (full
        epochs, checkpoints).
    cache_path : Path
        Events cache file, recorded for provenance.
    parent_run_id : str | None
        MLflow parent run to nest under, by default None.
    train_indices : list[int] | None
        Restrict training to these replays, by default None.
    extra_callbacks : Sequence[Callback]
        Additional callbacks (e.g. reporting to Ray Tune), by default none.

    Returns
    -------
    dict
        Last-epoch ``val_*`` metrics, ``best_val_objective``, ``epochs``,
        ``run_id`` and, in ``final`` mode, ``checkpoint`` (best checkpoint path).

    Raises
    ------
    ValueError
        If ``mode`` is unknown.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    pl.seed_everything(seed, workers=True)

    train_loader, val_loaders, tags = make_loaders(
        bundle, params, config, seed, train_indices
    )
    model = build_model(bundle, params, config, steps_per_epoch=len(train_loader))
    model.val_tags = tags

    logged = {
        **params,
        **run_provenance(seed, bundle.splits_sha256, config.cache_sha256, cache_path),
        "mode": mode,
        "n_train_replays": len(train_loader.dataset),
    }
    if parent_run_id:
        mlf_logger = create_child_mlflow_logger(
            experiment_name=config.experiment_name,
            run_name=run_name,
            parent_run_id=parent_run_id,
            tracking_uri=config.mlflow_tracking_uri,
            params=logged,
        )
    else:
        mlf_logger = create_mlflow_logger(
            experiment_name=config.experiment_name,
            run_name=run_name,
            tracking_uri=config.mlflow_tracking_uri,
        )
        mlf_logger.log_hyperparams(logged)

    search = mode == "search"
    early_stopping = EarlyStopping(
        monitor="val_objective",
        mode="min",
        patience=config.hpo_early_stopping_patience
        if search
        else config.early_stopping_patience,
    )
    callbacks: list[Callback] = [early_stopping, *extra_callbacks]
    checkpoint = None
    if mode == "final":
        checkpoint = ModelCheckpoint(
            dirpath=CHECKPOINTS_DIR / config.experiment_name / run_name,
            filename="event-guided-vae-{epoch:02d}-{val_objective:.4f}",
            monitor="val_objective",
            mode="min",
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(checkpoint)

    trainer = pl.Trainer(
        max_epochs=config.search_max_epochs if search else config.guided_vae_epochs,
        accelerator="auto",
        devices=1,
        precision=config.precision,
        logger=mlf_logger,
        callbacks=callbacks,
        enable_progress_bar=mode == "final",
        enable_checkpointing=checkpoint is not None,
        enable_model_summary=False,
        num_sanity_val_steps=0 if search else 2,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loaders)

    metrics = {
        name: float(value)
        for name, value in trainer.callback_metrics.items()
        if name.startswith("val_")
    }
    best = early_stopping.best_score
    metrics["best_val_objective"] = float(best) if best is not None else float("inf")
    metrics["epochs"] = trainer.current_epoch
    metrics["run_id"] = mlf_logger.run_id
    if checkpoint is not None:
        metrics["checkpoint"] = checkpoint.best_model_path
    mlf_logger.log_metrics({"best_val_objective": metrics["best_val_objective"]})
    return metrics
