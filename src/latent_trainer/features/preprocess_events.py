"""
Build the ``events`` cache: ragged per-player event streams for the sequence model.

Split membership comes from ``splits.json`` via ``extract_with_manifest``, which
also guarantees index-ordered rows.  The unit vocabulary is a sorted union built
after extraction, so ids never depend on worker scheduling.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from latent_trainer.features.event_transform import (
    EVENT_TYPE_NAMES,
    RESERVED_UNIT_TOKENS,
    TICK_INTERVAL_LOOPS,
    ReplayEventSample,
    event_transform,
)
from latent_trainer.features.extraction import (
    extract_with_manifest,
    load_source_dataset,
)
from latent_trainer.features.rich_transform import SORTED_PLAYERSTATS_KEYS
from latent_trainer.features.splits import SPLIT_NAMES
from latent_trainer.features.type import EventCacheFileSpec
from latent_trainer.settings import DATA_DIR

logger = logging.getLogger(__name__)

TRANSFORM_NAME = "events"


def build_unit_vocab(sample_groups: list[list[ReplayEventSample]]) -> list[str]:
    """
    Union every replay's local vocabulary into one sorted global vocabulary.

    Sorting makes ids independent of worker scheduling and sample order; the
    reserved tokens stay first so id 0 is always padding.

    Parameters
    ----------
    sample_groups : list[list[ReplayEventSample]]
        Extracted samples, grouped (e.g. by split).

    Returns
    -------
    list[str]
        ``["<PAD>", "<UNK>", *sorted(names)]``.
    """
    names: set[str] = set()
    for group in sample_groups:
        for sample in group:
            names.update(sample.local_vocab)

    return [*RESERVED_UNIT_TOKENS, *sorted(names)]


def _flatten_split(
    samples: list[ReplayEventSample],
    unit_vocab: list[str],
) -> dict[str, torch.Tensor]:
    """
    Concatenate a split's streams into flat arrays plus CSR-style offsets.

    Parameters
    ----------
    samples : list[ReplayEventSample]
        The split's samples, in manifest order.
    unit_vocab : list[str]
        Global vocabulary used to remap each replay's local unit ids.

    Returns
    -------
    dict[str, torch.Tensor]
        The ten per-split fields of :class:`EventCacheFileSpec`, unprefixed.
    """
    global_id = {name: index for index, name in enumerate(unit_vocab)}

    stat_values: list[np.ndarray] = []
    stat_loops: list[np.ndarray] = []
    stat_tick: list[np.ndarray] = []
    stat_offsets: list[int] = [0]

    evt_type: list[np.ndarray] = []
    evt_unit: list[np.ndarray] = []
    evt_loops: list[np.ndarray] = []
    evt_tick: list[np.ndarray] = []
    evt_offsets: list[int] = [0]

    labels: list[int] = []

    for sample in samples:
        # Local -> global id remap, one small lookup table per replay.
        local_to_global = np.asarray(
            [global_id[name] for name in sample.local_vocab], dtype=np.int64
        )

        for player in sample.players:
            stat_values.append(player.stat_values)
            stat_loops.append(player.stat_loops)
            stat_tick.append(player.stat_tick)
            stat_offsets.append(stat_offsets[-1] + player.stat_values.shape[0])

            evt_type.append(player.evt_type)
            evt_unit.append(
                local_to_global[player.evt_unit_local]
                if player.evt_unit_local.size
                else np.empty(0, dtype=np.int64)
            )
            evt_loops.append(player.evt_loops)
            evt_tick.append(player.evt_tick)
            evt_offsets.append(evt_offsets[-1] + player.evt_type.shape[0])

        labels.append(sample.label)

    def _cat(arrays: list[np.ndarray], dtype: torch.dtype) -> torch.Tensor:
        if not arrays:
            return torch.empty(0, dtype=dtype)
        return torch.from_numpy(np.concatenate(arrays)).to(dtype)

    empty_stats = torch.empty((0, len(SORTED_PLAYERSTATS_KEYS)), dtype=torch.float32)

    return {
        "stat_values": (
            torch.from_numpy(np.concatenate(stat_values)).float()
            if stat_values
            else empty_stats
        ),
        "stat_loops": _cat(stat_loops, torch.int32),
        "stat_tick": _cat(stat_tick, torch.int32),
        "stat_offsets": torch.tensor(stat_offsets, dtype=torch.int64),
        "evt_type": _cat(evt_type, torch.int8),
        "evt_unit": _cat(evt_unit, torch.int16),
        "evt_loops": _cat(evt_loops, torch.int32),
        "evt_tick": _cat(evt_tick, torch.int32),
        "evt_offsets": torch.tensor(evt_offsets, dtype=torch.int64),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def preprocess_event_dataset(
    single_json_dataset_path: Path | str,
    splits_path: Path | str,
    create_splits: bool = False,
    output_directory: Path | str = DATA_DIR,
    n_workers: int = 8,
    chunk_size: int = 64,
    n_samples: int = 0,
    seed: int = 42,
) -> None:
    """
    Extract event streams for every manifest replay and cache them ragged.

    Writes ``cached_dataset_events[_<n>].pt`` to ``output_directory``.

    Parameters
    ----------
    single_json_dataset_path : Path | str
        Path to the single-JSON SC2EGSet file.
    splits_path : Path | str
        Location of ``splits.json``.
    create_splits : bool
        Create the manifest from the replays this run keeps, by default False.
    output_directory : Path | str
        Directory for the cache file, by default ``DATA_DIR``.
    n_workers : int
        Worker processes, by default 8.
    chunk_size : int
        Replays per worker task, by default 64.
    n_samples : int
        If > 0, extract a seeded subset, by default 0.
    seed : int
        Seed for split creation and sampling, by default 42.
    """
    logger.info(
        f"SC2EGSet event-stream extraction: n_workers={n_workers}, "
        f"chunk_size={chunk_size}, n_samples={n_samples}"
    )

    result = extract_with_manifest(
        load_source_dataset(single_json_dataset_path),
        event_transform,
        transform_name=TRANSFORM_NAME,
        dataset_file=Path(single_json_dataset_path).name,
        splits_path=Path(splits_path),
        create_splits=create_splits,
        n_samples=n_samples,
        seed=seed,
        n_workers=n_workers,
        chunk_size=chunk_size,
    )
    samples: dict[str, list[ReplayEventSample]] = result.outputs

    unit_vocab = build_unit_vocab([samples[split] for split in SPLIT_NAMES])
    logger.info(f"  Unit vocabulary: {len(unit_vocab)} entries (incl. 2 reserved)")

    flattened = {
        split: _flatten_split(samples[split], unit_vocab) for split in SPLIT_NAMES
    }
    spec_fields = {
        f"{split}_{name}": tensor
        for split, streams in flattened.items()
        for name, tensor in streams.items()
    }

    manifest = result.manifest
    file_spec = EventCacheFileSpec(
        **spec_fields,
        stat_channel_names=list(SORTED_PLAYERSTATS_KEYS),
        unit_vocab=unit_vocab,
        event_type_names=list(EVENT_TYPE_NAMES),
        tick_interval_loops=TICK_INTERVAL_LOOPS,
        transform=TRANSFORM_NAME,
        train_ids=manifest.ids("train"),
        val_ids=manifest.ids("val"),
        test_ids=manifest.ids("test"),
        splits_sha256=manifest.sha256(),
    )

    os.makedirs(output_directory, exist_ok=True)
    suffix = f"_{n_samples}" if n_samples > 0 else ""
    path_to_save = (
        Path(output_directory) / f"cached_dataset_{TRANSFORM_NAME}{suffix}.pt"
    )
    torch.save(asdict(file_spec), path_to_save)

    file_size_mb = os.path.getsize(path_to_save) / (1024 * 1024)
    n_ticks = int(flattened["train"]["stat_offsets"][-1])
    n_events = int(flattened["train"]["evt_offsets"][-1])

    logger.info("=" * 60)
    logger.info("Event-stream pre-processing complete!")
    for split in SPLIT_NAMES:
        logger.info(f"  {split:<6} replays:   {len(samples[split])}")
    logger.info(f"  Skipped / errors: {result.n_skipped} / {result.n_errors}")
    logger.info(
        f"  Train stat ticks: {n_ticks:,} over {2 * len(samples['train']):,} streams"
    )
    logger.info(f"  Train events:     {n_events:,}")
    logger.info(f"  Unit vocabulary:  {len(unit_vocab)}")
    logger.info(f"  Splits sha256:    {manifest.sha256()}")
    logger.info(f"  Cache file:       {path_to_save} ({file_size_mb:.1f} MB)")
    logger.info("=" * 60)
