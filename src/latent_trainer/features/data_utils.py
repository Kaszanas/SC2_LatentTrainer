"""Shared data loading and normalisation utilities.

This module provides the **single canonical implementations** of dataset
loading, normalisation, and latent extraction.  All training scripts,
analysis tools, and the hyperparameter search system import from here
instead of maintaining their own copies.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from latent_trainer.features.type import (
    CachedDatasetFileSpec,
    CachedSC2Dataset,
    Encoder,
    EventCacheFileSpec,
    EventDataBundle,
    EventSplit,
    NormalizedData,
    NormalizedDataloaders,
    NormalizedDataWithLabels,
)

logger = logging.getLogger(__name__)


def normalize(
    train_X: torch.Tensor,
    val_X: torch.Tensor,
    test_X: torch.Tensor,
) -> NormalizedData:
    """Per-feature z-score normalisation.  Fit on *train*, apply to both.

    Parameters
    ----------
    train_X:
        Training features, shape ``[N, ...]`` where the last dim is features.
    val_X:
        Validation features with the same trailing feature dimension.

    Returns
    -------
    NormalizedData
        Normalised tensors and normalisation parameters.
    """
    shape = train_X.shape
    train_flat = train_X.reshape(-1, shape[-1])
    val_flat = val_X.reshape(-1, shape[-1])
    test_flat = test_X.reshape(-1, shape[-1])

    mean = train_flat.mean(dim=0, keepdim=True)
    # A plain `+ 1e-8` epsilon is not a real floor for near-constant features
    # (e.g. a rare stat that's ~always 0): true std there can itself be
    # ~1e-8, so dividing by ~2e-8 still blows up any nonzero outlier value
    # into an enormous z-score (confirmed: a single training game with one
    # nonzero occurrence of an otherwise-always-zero stat produced a z-score
    # of ~1.5e10, which propagated into an astronomically large encoded
    # latent vector and poisoned downstream latent-space centroids/means).
    # These are raw count/aggregate features with typical scale in the
    # hundreds-to-thousands, so a std floor of 1.0 is a real floor, not a
    # no-op, while being negligible for every normally-scaled feature.
    std = train_flat.std(dim=0, keepdim=True).clamp(min=1.0)

    train_flat = (train_flat - mean) / std
    val_flat = (val_flat - mean) / std
    test_flat = (test_flat - mean) / std

    normalized_data = NormalizedData(
        train_X=train_flat.reshape(shape),
        val_X=val_flat.reshape(val_X.shape),
        test_X=test_flat.reshape(test_X.shape),
        mean=mean,
        std=std,
    )

    return normalized_data


# Dataset loading
def load_and_normalize(
    cached_dataset_filepath: Path,
) -> NormalizedDataWithLabels:
    """
    Load a cached ``.pt`` dataset and normalise features.

    The cache is expected to contain at least::

        {"train_features": ..., "train_labels": ...,
         "val_features": ...,   "val_labels": ...}

    Parameters
    ----------
    cache_path : Path
        Path to the ``.pt`` file produced by ``preprocess_dataset.py``.

    Returns
    -------
    NormalizedDataWithLabels
        Normalised tensors and normalisation parameters.
    """
    logger.info(f"Loading data from {cached_dataset_filepath}")
    cached: dict[str, torch.Tensor] = torch.load(
        f=str(cached_dataset_filepath),
        weights_only=True,
    )

    cached_data_spec = CachedDatasetFileSpec(**cached)

    train_y = cached_data_spec.train_labels.float()
    val_y = cached_data_spec.val_labels.float()
    test_y = cached_data_spec.test_labels.float()

    normalized_data = normalize(
        train_X=cached_data_spec.train_features.float(),
        val_X=cached_data_spec.val_features.float(),
        test_X=cached_data_spec.test_features.float(),
    )

    logger.info(
        f"Feature shape: {list(normalized_data.train_X.shape)}  |  Train: {len(normalized_data.train_X)}  Val: {len(normalized_data.val_X)}"
    )

    return NormalizedDataWithLabels(
        train_X=normalized_data.train_X,
        train_y=train_y,
        val_X=normalized_data.val_X,
        val_y=val_y,
        test_X=normalized_data.test_X,
        test_y=test_y,
        mean=normalized_data.mean,
        std=normalized_data.std,
    )


def load_cached_dataloaders(
    cache_path: Path,
    batch_size: int,
) -> NormalizedDataloaders:
    """
    Load cached dataset, normalise, and return ready-to-use DataLoaders.

    This is a convenience wrapper around :func:`load_and_normalize` for
    training loops that need DataLoaders rather than raw tensors.

    Parameters
    ----------
    cache_path : Path
        Path to the cached dataset file.
    batch_size : int
        Batch size for the DataLoaders.

    Returns
    -------
    NormalizedData
        Contains the DataLoaders and normalisation parameters.
    """

    normalized_data = load_and_normalize(cached_dataset_filepath=cache_path)
    input_dim = normalized_data.train_X.shape[-1]

    train_loader = DataLoader(
        CachedSC2Dataset(
            features=normalized_data.train_X,
            labels=normalized_data.train_y,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        CachedSC2Dataset(
            features=normalized_data.val_X,
            labels=normalized_data.val_y,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return NormalizedDataloaders(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        mean=normalized_data.mean,
        std=normalized_data.std,
    )


# ------------------------------------------------------------------
# Event-sequence pipeline
# ------------------------------------------------------------------
def _pad_stack(
    streams: list[torch.Tensor],
    n_players: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad a flat list of ``B * n_players`` streams and fold back to ``[B, P, L, ...]``."""

    max_len = max((stream.shape[0] for stream in streams), default=0)
    trailing = streams[0].shape[1:] if streams else ()

    padded = torch.full(
        (len(streams), max_len, *trailing),
        pad_value,
        dtype=streams[0].dtype,
    )
    for index, stream in enumerate(streams):
        padded[index, : stream.shape[0]] = stream

    return padded.reshape(-1, n_players, max_len, *trailing)


def collate_event_sequences(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad a batch of ragged event samples to *that batch's* maximum lengths.

    Padding per batch rather than to a global maximum matters here because the
    event stream varies by well over an order of magnitude between a fast all-in
    and a long macro game (Phase 0: p50 ~2 events/tick, p99 13.5k tokens per
    player).  Padding everything to the global maximum would spend most of the
    attention budget on masked-out positions.

    Both players are padded together, so the returned tensors are
    ``[B, 2, L, ...]`` and the masks mark real positions with ``True``.
    """
    n_players = len(batch[0]["players"])
    streams = [player for sample in batch for player in sample["players"]]

    stat_lengths = torch.tensor(
        [stream["stat_values"].shape[0] for stream in streams], dtype=torch.long
    )
    evt_lengths = torch.tensor(
        [stream["evt_type"].shape[0] for stream in streams], dtype=torch.long
    )

    collated = {
        "stat_values": _pad_stack(
            [stream["stat_values"] for stream in streams], n_players
        ),
        "stat_loops": _pad_stack(
            [stream["stat_loops"].long() for stream in streams], n_players
        ),
        "stat_slice": _pad_stack(
            [stream["stat_slice"].long() for stream in streams], n_players
        ),
        "evt_type": _pad_stack(
            [stream["evt_type"].long() for stream in streams], n_players
        ),
        "evt_unit": _pad_stack(
            [stream["evt_unit"].long() for stream in streams], n_players
        ),
        "evt_loops": _pad_stack(
            [stream["evt_loops"].long() for stream in streams], n_players
        ),
        "evt_slice": _pad_stack(
            [stream["evt_slice"].long() for stream in streams], n_players
        ),
        "label": torch.stack([sample["label"] for sample in batch]),
    }

    stat_max = collated["stat_values"].shape[2]
    evt_max = collated["evt_type"].shape[2]
    positions_stat = torch.arange(stat_max).unsqueeze(0)
    positions_evt = torch.arange(evt_max).unsqueeze(0)

    collated["stat_mask"] = (positions_stat < stat_lengths.unsqueeze(1)).reshape(
        -1, n_players, stat_max
    )
    collated["evt_mask"] = (positions_evt < evt_lengths.unsqueeze(1)).reshape(
        -1, n_players, evt_max
    )

    # Number of level-2 slices per stream: the stat stream defines the horizon,
    # since a player can fall event-silent late without the game being over.
    n_slices = torch.tensor(
        [
            int(stream["stat_slice"][-1]) + 1 if stream["stat_slice"].numel() else 0
            for stream in streams
        ],
        dtype=torch.long,
    )
    collated["n_slices"] = n_slices.reshape(-1, n_players)

    return collated


def _split_from_spec(spec: EventCacheFileSpec, prefix: str) -> EventSplit:
    """Pull one split's ten ragged tensors off the cache spec."""
    return EventSplit(
        stat_values=getattr(spec, f"{prefix}_stat_values").float(),
        stat_loops=getattr(spec, f"{prefix}_stat_loops"),
        stat_tick=getattr(spec, f"{prefix}_stat_tick"),
        stat_offsets=getattr(spec, f"{prefix}_stat_offsets"),
        evt_type=getattr(spec, f"{prefix}_evt_type"),
        evt_unit=getattr(spec, f"{prefix}_evt_unit"),
        evt_loops=getattr(spec, f"{prefix}_evt_loops"),
        evt_tick=getattr(spec, f"{prefix}_evt_tick"),
        evt_offsets=getattr(spec, f"{prefix}_evt_offsets"),
        labels=getattr(spec, f"{prefix}_labels"),
    )


def load_event_cache(cached_dataset_filepath: Path) -> EventDataBundle:
    """Load an ``events`` cache and z-score the stat channels.

    Only ``stat_values`` is normalised, per channel, fitted on train.  The event
    stream is categorical ids and game loops: those are embedded and positionally
    encoded, never standardised.

    No masking is needed when fitting the statistics, which is the quiet payoff
    of the ragged layout — the flat stat array contains only real ticks, so
    ``mean``/``std`` over it are already exactly the masked statistics.  A padded
    layout would have biased both toward zero in proportion to game length.
    """
    logger.info(f"Loading event data from {cached_dataset_filepath}")
    cached = torch.load(f=str(cached_dataset_filepath), weights_only=False)
    spec = EventCacheFileSpec(**cached)

    train = _split_from_spec(spec, "train")
    val = _split_from_spec(spec, "val")
    test = _split_from_spec(spec, "test")

    mean = train.stat_values.mean(dim=0, keepdim=True)
    # Same std floor, and for the same reason, as `normalize` above: these are
    # raw counts in the hundreds-to-thousands, and a near-constant channel whose
    # true std is ~1e-8 would otherwise turn a single nonzero game into a
    # ~1e10 z-score that poisons every downstream latent centroid.
    std = train.stat_values.std(dim=0, keepdim=True).clamp(min=1.0)

    for split in (train, val, test):
        if split.stat_values.numel():
            split.stat_values = (split.stat_values - mean) / std

    logger.info(
        f"Event cache: train {len(train.labels)} / val {len(val.labels)} / "
        f"test {len(test.labels)} replays  |  "
        f"{len(spec.stat_channel_names)} stat channels, "
        f"{len(spec.unit_vocab)} unit types"
    )

    return EventDataBundle(
        train=train,
        val=val,
        test=test,
        mean=mean,
        std=std,
        stat_channel_names=spec.stat_channel_names,
        unit_vocab=spec.unit_vocab,
        event_type_names=spec.event_type_names,
        tick_interval_loops=spec.tick_interval_loops,
    )


# Latent extraction
def extract_latents(
    encoder: Encoder,
    data: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Extract deterministic latent representations using a frozen encoder.

    Processes each player's features separately through the encoder and
    concatenates the resulting mu vectors.

    Parameters
    ----------
    encoder:
        Any model with an ``encode(x) → (mu, logvar)`` method.
    data:
        Feature tensor of shape ``[N, 2, F]`` (two players).
    device:
        Device to run inference on.
    batch_size:
        Batch size for inference.

    Returns
    -------
    torch.Tensor
        Concatenated latent means, shape ``[N, 2 * latent_dim]``.
    """
    encoder.eval()  # type: ignore[union-attr]
    encoder.to(device)  # type: ignore[union-attr]

    all_latents: list[torch.Tensor] = []
    for player_idx in range(2):
        player_data = data[:, player_idx, :]
        latents: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(player_data), batch_size):
                batch = player_data[i : i + batch_size].to(device)
                mu, _ = encoder.encode(batch)
                latents.append(mu.cpu())
        all_latents.append(torch.cat(latents, dim=0))

    return torch.cat(all_latents, dim=1)
