"""
Shared data loading and normalisation utilities.

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
    """
    Per-feature z-score normalisation, fitted on train and applied to all splits.

    The std is floored at 1.0: a near-constant feature (true std ~1e-8) would
    otherwise turn a single nonzero value into a ~1e10 z-score.

    Parameters
    ----------
    train_X : torch.Tensor
        Training features, ``[N, ...]`` with features in the last dimension.
    val_X : torch.Tensor
        Validation features with the same feature dimension.
    test_X : torch.Tensor
        Test features with the same feature dimension.

    Returns
    -------
    NormalizedData
        Normalised tensors and the normalisation parameters.
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
    Load a dense ``.pt`` cache and normalise its features.

    Parameters
    ----------
    cached_dataset_filepath : Path
        Cache produced by ``python -m latent_trainer.features``.

    Returns
    -------
    NormalizedDataWithLabels
        Normalised features, float labels and the normalisation parameters.
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


# Event-sequence pipeline:
def _pad_stack(
    streams: list[torch.Tensor],
    n_players: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad a flat list of ``B * n_players`` streams and fold to ``[B, P, L, ...]``.

    Parameters
    ----------
    streams : list[torch.Tensor]
        Streams ordered sample-major, player-minor; each ``[L_i, ...]``.
    n_players : int
        Players per sample.
    pad_value : float
        Fill value for padded positions, by default 0.0.

    Returns
    -------
    torch.Tensor
        Padded tensor ``[B, n_players, max(L_i), ...]``.
    """
    max_len = max((stream.shape[0] for stream in streams), default=0)
    trailing = streams[0].shape[1:] if streams else ()

    padded = torch.full(
        (len(streams), max_len, *trailing),
        pad_value,
        dtype=streams[0].dtype,
    )
    for index, stream in enumerate(streams):
        padded[index, : stream.shape[0]] = stream

    # Explicit batch size: with max_len == 0 a -1 would be ambiguous.
    return padded.reshape(len(streams) // n_players, n_players, max_len, *trailing)


def _group_events(
    evt_slice: torch.Tensor,
    stat_slice: torch.Tensor,
    stat_loops: torch.Tensor,
    max_events_per_slice: int,
) -> tuple[torch.Tensor, ...]:
    """
    Index one stream's events by slice, capped at ``K`` per slice.

    Events arrive sorted by loop, so each slice's events are one contiguous run.
    A slice with more than ``K`` events keeps ``K`` of them evenly spaced over
    the run (deterministic, and spread over the whole interval rather than its
    start); the true count is returned so the cap stays visible.

    Parameters
    ----------
    evt_slice : torch.Tensor
        Slice of each event, ``[E]``, non-decreasing.
    stat_slice : torch.Tensor
        Slice of each stat tick, ``[T]``, non-decreasing.
    stat_loops : torch.Tensor
        Loop of each stat tick, ``[T]``.
    max_events_per_slice : int
        The cap ``K``.

    Returns
    -------
    tuple[torch.Tensor, ...]
        ``group`` ``[S, K]`` event positions (0 where masked), ``mask``
        ``[S, K]``, ``count`` ``[S]`` true events per slice, ``slice_loops``
        ``[S]`` the loop of each slice's last stat tick, ``last_row``
        ``[S]`` that tick's row in the stream (forward-filled over gaps), and
        ``slot`` ``[E]`` the inverse of ``group``: each event's flat position
        ``slice * K + k``, or -1 for an event dropped by the cap, with
        ``S = last stat slice + 1``.

    Raises
    ------
    ValueError
        If an event lies in a slice after the last stat slice.
    """
    k = max_events_per_slice
    n_slices = int(stat_slice[-1]) + 1 if stat_slice.numel() else 0
    evt_slice = evt_slice.long()
    if evt_slice.numel() and int(evt_slice[-1]) >= n_slices:
        raise ValueError(
            f"event in slice {int(evt_slice[-1])} after the last stat slice "
            f"{n_slices - 1}"
        )

    count = torch.bincount(evt_slice, minlength=n_slices)
    start = torch.cumsum(count, dim=0) - count
    take = count.clamp(max=k)
    position = torch.arange(k).unsqueeze(0)
    offset = position * count.unsqueeze(1) // take.clamp(min=1).unsqueeze(1)
    mask = position < take.unsqueeze(1)
    group = torch.where(mask, start.unsqueeze(1) + offset, 0)

    slice_loops = torch.zeros(n_slices, dtype=torch.long).scatter_reduce(
        0, stat_slice.long(), stat_loops.long(), reduce="amax", include_self=False
    )
    # Row of each slice's last stat tick; a slice without a tick (a rare gap in
    # the grid) reuses the previous one, which stays causal.
    last_row = torch.full((n_slices,), -1, dtype=torch.long).scatter_reduce(
        0,
        stat_slice.long(),
        torch.arange(len(stat_slice)),
        reduce="amax",
        include_self=True,
    )
    if n_slices:
        last_row = torch.cummax(last_row, dim=0).values.clamp(min=0)

    # Inverse map: kept events are unique in ``group``, so each gets one slot.
    slot = torch.full((evt_slice.numel(),), -1, dtype=torch.long)
    slot[group[mask]] = torch.arange(n_slices * k).reshape(n_slices, k)[mask]
    return group, mask, count, slice_loops, last_row, slot


def collate_event_sequences(
    batch: list[dict], max_events_per_slice: int | None = None
) -> dict[str, torch.Tensor]:
    """
    Pad a batch of ragged event samples to that batch's maximum lengths.

    Padding per batch rather than to a global maximum matters because event
    streams vary by over an order of magnitude between games. Both players are
    padded together; masks mark real positions with True. Padded ids are 0
    (``<PAD>``), but padded loops/slices also read 0, so always use the masks.
    Use ``functools.partial`` to pass ``max_events_per_slice`` to a DataLoader.

    Parameters
    ----------
    batch : list[dict]
        Samples from :class:`EventSequenceDataset`.
    max_events_per_slice : int | None
        Cap ``K`` on events per slice in ``evt_group``. None (the default)
        keeps every event: ``K`` is the largest slice in the batch.

    Returns
    -------
    dict[str, torch.Tensor]
        ``stat_values`` ``[B, 2, Ls, C]``; ``stat_loops``/``stat_slice``
        ``[B, 2, Ls]``; ``evt_type``/``evt_unit``/``evt_loops``/``evt_slice``
        ``[B, 2, Le]``; ``stat_mask``/``evt_mask`` (bool); ``n_slices``
        ``[B, 2]``; ``slice_loops`` ``[B, 2, S]`` (loop of each slice's last
        tick) with ``slice_mask``; ``slice_stat_index`` ``[B, 2, S]`` (row of
        that tick in ``stat_values``); ``evt_group`` ``[B, 2, S, K]`` positions into
        the event dimension with ``evt_group_mask``; ``evt_count`` ``[B, 2, S]``
        true events per slice before the cap; ``race`` ``[B, 2]``;
        ``crop_loop`` ``[B]``; ``label`` ``[B]``; ``evt_slot`` ``[B, 2, Le]``
        each event's flat slot ``slice * K + k`` in its stream (``S * K`` for
        padded and capped-out events); ``slice_valid_index`` ``[Nv]`` flat
        indices of the real slices in ``slice_mask``.

    Raises
    ------
    ValueError
        If ``max_events_per_slice < 1``.
    """
    if max_events_per_slice is not None and max_events_per_slice < 1:
        raise ValueError(
            f"max_events_per_slice must be >= 1, got {max_events_per_slice}"
        )
    n_players = len(batch[0]["players"])
    streams = [player for sample in batch for player in sample["players"]]
    if max_events_per_slice is None:
        max_events_per_slice = max(
            [
                int(torch.bincount(stream["evt_slice"].long()).max())
                for stream in streams
                if stream["evt_slice"].numel()
            ],
            default=1,
        )

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
        "race": torch.stack([sample["race"] for sample in batch]),
        "crop_loop": torch.stack([sample["crop_loop"] for sample in batch]),
    }

    grouped = [
        _group_events(
            stream["evt_slice"],
            stream["stat_slice"],
            stream["stat_loops"],
            max_events_per_slice,
        )
        for stream in streams
    ]
    collated["evt_group"] = _pad_stack([g[0] for g in grouped], n_players)
    collated["evt_group_mask"] = _pad_stack(
        [g[1] for g in grouped], n_players, pad_value=False
    )
    collated["evt_count"] = _pad_stack([g[2] for g in grouped], n_players)
    collated["slice_loops"] = _pad_stack([g[3] for g in grouped], n_players)
    collated["slice_stat_index"] = _pad_stack([g[4] for g in grouped], n_players)

    stat_max = collated["stat_values"].shape[2]
    evt_max = collated["evt_type"].shape[2]
    positions_stat = torch.arange(stat_max).unsqueeze(0)
    positions_evt = torch.arange(evt_max).unsqueeze(0)

    n_batch = len(batch)
    collated["stat_mask"] = (positions_stat < stat_lengths.unsqueeze(1)).reshape(
        n_batch, n_players, stat_max
    )
    collated["evt_mask"] = (positions_evt < evt_lengths.unsqueeze(1)).reshape(
        n_batch, n_players, evt_max
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
    collated["n_slices"] = n_slices.reshape(n_batch, n_players)
    slice_max = collated["slice_loops"].shape[2]
    collated["slice_mask"] = (
        torch.arange(slice_max).unsqueeze(0) < n_slices.unsqueeze(1)
    ).reshape(n_batch, n_players, slice_max)

    # Index forms computed here on the CPU, so the model never has to turn a
    # boolean mask into indices on the GPU (each such call waits for the GPU).
    k = collated["evt_group"].shape[-1]
    dummy = slice_max * k  # one spare slot for padded and capped-out events
    slots = [torch.where(g[5] >= 0, g[5], dummy) for g in grouped]
    collated["evt_slot"] = _pad_stack(slots, n_players, pad_value=dummy)
    collated["slice_valid_index"] = collated["slice_mask"].reshape(-1).nonzero()[:, 0]
    return collated


def _split_from_spec(spec: EventCacheFileSpec, prefix: str) -> EventSplit:
    """
    Pull one split's eleven tensors off the cache spec.

    Parameters
    ----------
    spec : EventCacheFileSpec
        Loaded cache.
    prefix : str
        Split name: ``train``, ``val`` or ``test``.

    Returns
    -------
    EventSplit
        The split, with ``stat_values`` cast to float32.
    """
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
        race=getattr(spec, f"{prefix}_race"),
    )


def _check_split_layout(split: EventSplit, name: str, ids: list[str]) -> None:
    """
    Verify that a split's CSR layout matches two streams per replay.

    ``EventSequenceDataset`` reads replay ``i`` as streams ``2i`` and ``2i + 1``
    and its label as ``labels[i]``. A replay with any other number of streams
    would shift every later replay onto its neighbour's data and label without
    raising, so the layout is checked once here.

    Parameters
    ----------
    split : EventSplit
        Split as loaded from the cache.
    name : str
        Split name, for error messages.
    ids : list[str]
        Replay ids of the split; empty in caches built before the split manifest.

    Raises
    ------
    ValueError
        If either offsets array does not have ``2N + 1`` entries, does not start
        at 0, decreases, or does not end at the length of its value arrays, if
        ``race`` is not one per stream, or if ``ids`` is non-empty and not one
        per replay.
    """
    n_replays = int(split.labels.numel())
    layouts = {
        "stat": (
            split.stat_offsets,
            [split.stat_values, split.stat_loops, split.stat_tick],
        ),
        "evt": (
            split.evt_offsets,
            [split.evt_type, split.evt_unit, split.evt_loops, split.evt_tick],
        ),
    }
    for stream, (offsets, arrays) in layouts.items():
        where = f"{name} {stream}_offsets"
        if offsets.numel() != 2 * n_replays + 1:
            raise ValueError(
                f"{where} has {offsets.numel()} entries, expected "
                f"2 * {n_replays} replays + 1 = {2 * n_replays + 1}"
            )
        if int(offsets[0]) != 0:
            raise ValueError(f"{where} starts at {int(offsets[0])}, expected 0")
        if bool((offsets[1:] < offsets[:-1]).any()):
            raise ValueError(f"{where} decreases")
        for array in arrays:
            if array.shape[0] != int(offsets[-1]):
                raise ValueError(
                    f"{where} ends at {int(offsets[-1])}, but a {stream} array "
                    f"has {array.shape[0]} rows"
                )
    if split.race.numel() != 2 * n_replays:
        raise ValueError(
            f"{name} has {split.race.numel()} race ids for {2 * n_replays} streams"
        )
    if ids and len(ids) != n_replays:
        raise ValueError(f"{name} has {len(ids)} replay ids for {n_replays} labels")


def load_event_cache(cached_dataset_filepath: Path) -> EventDataBundle:
    """
    Load an ``events`` cache with raw stat channels.

    Stat values stay in raw units: the event model owns its input scaling
    (``ChannelTransform``, fitted on ``bundle.train.stat_values``), so the
    checkpoint carries it and serving cannot normalise differently.

    Parameters
    ----------
    cached_dataset_filepath : Path
        Cache produced by ``--transform events``.

    Returns
    -------
    EventDataBundle
        Raw splits and cache metadata.

    Raises
    ------
    ValueError
        If the cache predates race and PAC channels, or a split's stream layout
        is malformed (see ``_check_split_layout``).
    """
    logger.info(f"Loading event data from {cached_dataset_filepath}")
    cached = torch.load(f=str(cached_dataset_filepath), weights_only=False)
    if "train_race" not in cached:
        raise ValueError(
            f"{cached_dataset_filepath} predates the race and PAC channels; "
            "rebuild it with --transform events"
        )
    spec = EventCacheFileSpec(**cached)

    train = _split_from_spec(spec, "train")
    val = _split_from_spec(spec, "val")
    test = _split_from_spec(spec, "test")
    for name, split in (("train", train), ("val", val), ("test", test)):
        _check_split_layout(split, name, getattr(spec, f"{name}_ids"))

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
        stat_channel_names=spec.stat_channel_names,
        race_names=spec.race_names,
        unit_vocab=spec.unit_vocab,
        event_type_names=spec.event_type_names,
        tick_interval_loops=spec.tick_interval_loops,
        split_ids={"train": spec.train_ids, "val": spec.val_ids, "test": spec.test_ids},
        splits_sha256=spec.splits_sha256,
    )


# Latent extraction
def extract_latents(
    encoder: Encoder,
    data: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Extract deterministic latent representations using a frozen encoder.

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
