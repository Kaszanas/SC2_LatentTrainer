"""Build the ``events`` cache: ragged event streams for the sequence pipeline.

This is a parallel entrypoint to ``preprocess_dataset.py``, not a replacement.
It exists because the event transform's output is ragged — each replay yields a
variable number of stat ticks and events per player — so the
``torch.stack(features)`` assembly the dense pipeline relies on does not apply.
Everything about the existing ``rich`` / ``granular`` / ``averaged_economy``
caches is untouched.

Two ordering guarantees, both of which the dense path currently lacks:

- **Chunk results are reassembled in submission order**, not completion order.
  ``preprocess_dataset.py`` accumulates under ``wait(..., FIRST_COMPLETED)``, so
  its cached row order varies between runs even with a fixed seed.
- **The unit vocabulary is a sorted union**, built once after extraction.  Each
  worker interns type names locally and independently; global ids are assigned
  here, so they never depend on which worker saw which replay first.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sc2_datasets.lightning.sc2_egset_datamodule import SC2EGSetDataModuleSingleJSON
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from latent_trainer.features.event_transform import (
    EVENT_TYPE_NAMES,
    RESERVED_UNIT_TOKENS,
    TICK_INTERVAL_LOOPS,
    ReplayEventSample,
    event_transform,
)
from latent_trainer.features.preprocess_dataset import (
    _subsample_indices,
    check_split,
)
from latent_trainer.features.rich_transform import SORTED_PLAYERSTATS_KEYS
from latent_trainer.features.type import EventCacheFileSpec
from latent_trainer.settings import DATA_DIR

logger = logging.getLogger(__name__)

TRANSFORM_NAME = "events"


def _extract_index_chunk(
    dataset_object: DataLoader,
    indices: list[int],
) -> tuple[list[ReplayEventSample], int, list[str]]:
    """Extract one contiguous chunk of replays inside a single worker call."""

    samples: list[ReplayEventSample] = []
    skipped = 0
    errors: list[str] = []

    for index in indices:
        try:
            sample = event_transform(sc2_replay=dataset_object[index])
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            errors.append(f"{type(exc).__name__}: {exc}"[:200])
            continue

        if sample is None:
            skipped += 1
        else:
            samples.append(sample)

    return samples, skipped, errors


def extract_split(
    dataset_object: DataLoader,
    executor: type[ProcessPoolExecutor] | type[ThreadPoolExecutor],
    n_workers: int = 8,
    chunk_size: int = 64,
    set_name: str = "train",
) -> tuple[list[ReplayEventSample], int, list[str]]:
    """Extract every replay in a split, preserving submission order.

    ``executor.map`` with a chunked index list gives back results in the order
    the chunks were submitted regardless of the order they finish in, so the
    cached row order is reproducible for a fixed seed.  That is the property
    ``process_set_chunked_single_pool`` loses by consuming ``FIRST_COMPLETED``.
    """
    total_size = len(dataset_object)
    chunks: list[list[int]] = [
        list(range(start, min(start + chunk_size, total_size)))
        for start in range(0, total_size, chunk_size)
    ]

    samples: list[ReplayEventSample] = []
    skipped = 0
    errors: list[str] = []

    with executor(max_workers=n_workers) as pool:
        results = pool.map(
            _extract_index_chunk,
            [dataset_object] * len(chunks),
            chunks,
        )

        with tqdm(total=total_size, desc=f"{set_name} (events)") as progress:
            for chunk, (chunk_samples, chunk_skipped, chunk_errors) in zip(
                chunks, results
            ):
                samples.extend(chunk_samples)
                skipped += chunk_skipped
                errors.extend(chunk_errors)
                progress.update(len(chunk))

    return samples, skipped, errors


def build_unit_vocab(sample_groups: list[list[ReplayEventSample]]) -> list[str]:
    """Union every replay's local vocabulary into one sorted global vocabulary.

    Sorting is what makes the ids stable: the same dataset produces the same
    vocabulary regardless of worker scheduling, sample order, or interpreter
    run.  The two reserved tokens stay at the head so id 0 is always padding.
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
    """Concatenate a split's streams into flat arrays plus CSR-style offsets."""

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


def _rebalance_splits(
    train: list[ReplayEventSample],
    val: list[ReplayEventSample],
    test: list[ReplayEventSample],
    total: int,
) -> tuple[list[ReplayEventSample], list[ReplayEventSample], list[ReplayEventSample]]:
    """Top train/val back up to 80/10 from test, matching the dense pipeline.

    Skipped replays (draws, sub-3-minute games, missing stats) thin each split
    unevenly.  ``preprocess_dataset.py`` restores the ratios by moving samples
    out of test, which is the least training-critical split; this mirrors that
    so the two pipelines stay comparable.  Moving whole samples before
    flattening keeps the offsets consistent by construction.
    """
    if len(train) < 0.8 * total:
        needed = int(0.8 * total) - len(train)
        logger.warning(
            f"Training set has {len(train)} samples (< 80% of {total}); "
            f"moving {needed} from the test set to restore the split ratios."
        )
        train = train + test[:needed]
        test = test[needed:]

    if len(val) < 0.1 * total:
        needed = int(0.1 * total) - len(val)
        logger.warning(
            f"Validation set has {len(val)} samples (< 10% of {total}); "
            f"moving {needed} from the test set to restore the split ratios."
        )
        val = val + test[:needed]
        test = test[needed:]

    return train, val, test


def preprocess_event_dataset(
    single_json_dataset_path: Path | str,
    output_directory: Path | str = DATA_DIR,
    n_workers: int = 8,
    chunk_size: int = 64,
    n_samples: int = 0,
    seed: int = 42,
) -> None:
    """Extract raw event streams for every replay and cache them ragged."""

    logger.info(
        f"SC2EGSet event-stream extraction: n_workers={n_workers}, "
        f"chunk_size={chunk_size}, n_samples={n_samples}"
    )

    datamodule = SC2EGSetDataModuleSingleJSON(
        json_path=single_json_dataset_path,
        download=False,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    train_dataset = datamodule.train_dataset
    test_dataset = datamodule.test_dataset
    val_dataset = datamodule.val_dataset

    train_dataset.indices = sorted(train_dataset.indices)
    test_dataset.indices = sorted(test_dataset.indices)
    val_dataset.indices = sorted(val_dataset.indices)

    total = check_split(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
    )

    if n_samples > 0:
        generator = torch.Generator()
        generator.manual_seed(seed)
        n_train = round(n_samples * len(train_dataset.indices) / total)
        n_test = round(n_samples * len(test_dataset.indices) / total)
        n_val = n_samples - n_train - n_test
        train_dataset.indices = _subsample_indices(
            train_dataset.indices, n_train, generator
        )
        test_dataset.indices = _subsample_indices(
            test_dataset.indices, n_test, generator
        )
        val_dataset.indices = _subsample_indices(val_dataset.indices, n_val, generator)
        total = (
            len(train_dataset.indices)
            + len(test_dataset.indices)
            + len(val_dataset.indices)
        )
        logger.info(
            f"  Sampled {len(train_dataset.indices)} train / "
            f"{len(test_dataset.indices)} test / "
            f"{len(val_dataset.indices)} val (seed={seed})"
        )

    executor = ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor

    extracted: dict[str, list[ReplayEventSample]] = {}
    skipped_total = 0
    all_errors: list[str] = []

    for set_name, dataset in (
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ):
        samples, skipped, errors = extract_split(
            dataset_object=dataset,
            executor=executor,
            n_workers=n_workers,
            chunk_size=chunk_size,
            set_name=set_name,
        )
        extracted[set_name] = samples
        skipped_total += skipped
        all_errors.extend(errors)
        logger.info(
            f"  {set_name}: {len(samples)} valid, {skipped} skipped, "
            f"{len(errors)} errors"
        )

    train_samples, val_samples, test_samples = _rebalance_splits(
        train=extracted["train"],
        val=extracted["val"],
        test=extracted["test"],
        total=total,
    )

    unit_vocab = build_unit_vocab([train_samples, val_samples, test_samples])
    logger.info(f"  Unit vocabulary: {len(unit_vocab)} entries (incl. 2 reserved)")

    flattened = {
        "train": _flatten_split(train_samples, unit_vocab),
        "val": _flatten_split(val_samples, unit_vocab),
        "test": _flatten_split(test_samples, unit_vocab),
    }

    spec_fields = {
        f"{split}_{name}": tensor
        for split, streams in flattened.items()
        for name, tensor in streams.items()
    }

    file_spec = EventCacheFileSpec(
        **spec_fields,
        stat_channel_names=list(SORTED_PLAYERSTATS_KEYS),
        unit_vocab=unit_vocab,
        event_type_names=list(EVENT_TYPE_NAMES),
        tick_interval_loops=TICK_INTERVAL_LOOPS,
        transform=TRANSFORM_NAME,
    )

    os.makedirs(output_directory, exist_ok=True)
    suffix = f"_{n_samples}" if n_samples > 0 else ""
    path_to_save = Path(output_directory) / f"cached_dataset_{TRANSFORM_NAME}{suffix}.pt"
    torch.save(asdict(file_spec), path_to_save)

    file_size_mb = os.path.getsize(path_to_save) / (1024 * 1024)
    n_ticks = int(flattened["train"]["stat_offsets"][-1])
    n_events = int(flattened["train"]["evt_offsets"][-1])

    if all_errors:
        logger.warning(f"  {len(all_errors)} replays failed; top causes:")
        for cause, count in Counter(all_errors).most_common(5):
            logger.warning(f"    {count:>5}x  {cause}")

    logger.info(f"\n{'=' * 60}")
    logger.info("Event-stream pre-processing complete!")
    logger.info(f"  Transform:        {TRANSFORM_NAME}")
    logger.info(f"  Valid train:      {len(train_samples)}")
    logger.info(f"  Valid val:        {len(val_samples)}")
    logger.info(f"  Valid test:       {len(test_samples)}")
    logger.info(f"  Skipped (None):   {skipped_total}")
    logger.info(f"  Errors:           {len(all_errors)}")
    logger.info(
        f"  Train stat ticks: {n_ticks:,} over {2 * len(train_samples):,} streams"
    )
    logger.info(f"  Train events:     {n_events:,}")
    logger.info(f"  Unit vocabulary:  {len(unit_vocab)}")
    logger.info(f"  Cache file:       {path_to_save} ({file_size_mb:.1f} MB)")
    logger.info(f"{'=' * 60}")
