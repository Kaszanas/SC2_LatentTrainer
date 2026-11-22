"""
Dense feature caches (rich / granular / averaged_economy) for the guided VAE.

Split membership comes from ``splits.json`` (see ``features/splits.py``); run via
``python -m latent_trainer.features``.
"""

import functools
import logging
import os
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import click
import torch
from sc2_datasets.lightning.sc2_egset_datamodule import (
    SC2EGSetDataModuleSingleJSON,
)
from sc2_datasets.replay_data.sc2_replay_data import SC2ReplayData
from sc2_datasets.transforms.pytorch.economy_vs_outcome import (
    economy_average_vs_outcome,
)
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from latent_trainer.features.event_transform import event_transform
from latent_trainer.features.extraction import (
    extract_with_manifest,
    load_source_dataset,
)
from latent_trainer.features.rich_transform import (
    rich_transform,
    rich_transform_granular,
)
from latent_trainer.features.splits import SPLIT_NAMES
from latent_trainer.features.type import CachedDatasetFileSpec
from latent_trainer.settings import DATA_DIR


def _transform_single_object(
    dataset_object: DataLoader,
    index: int,
    transform_fn: Callable,
) -> tuple[torch.Tensor, torch.Tensor] | None | Exception:
    """
    Load one replay and run it through :func:`process_replay`, capturing errors.

    Parameters
    ----------
    dataset_object : DataLoader
        Indexable dataset returning ``SC2ReplayData``.
    index : int
        Index into ``dataset_object``.
    transform_fn : Callable
        Dense transform returning ``(features, label)`` or None.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] | None | Exception
        ``(features, label)``, None if skipped, or the raised exception.
    """
    try:
        replay = dataset_object[index]

        result = process_replay(
            replay=replay,
            transform_fn=transform_fn,
        )

        if result is None:
            return None
    except Exception as e:
        return e

    return result


def process_replay(
    replay: SC2ReplayData,
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Apply a dense transform, skipping replays without a decided outcome.

    Parameters
    ----------
    replay : SC2ReplayData
        Replay to transform.
    transform_fn : Callable[[SC2ReplayData], tuple[torch.Tensor, int]]
        Transform returning ``(features, label)`` or None.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] | None
        ``(features, label)``, or None if the transform skipped the replay or the
        label is -1 (draw/tie/undecided).
    """
    # Rich transform takes the raw replay directly
    result = transform_fn(replay)

    if result is None:
        return None

    features, label = result

    if features is None or label is None:
        return None

    if label == -1:  # Undecided/Draw/Tie
        return None

    return features, label


class TransformEnumFunction(click.Choice):
    """Custom Click Choice type that returns the actual transform function instead of the string name."""

    _TRANSFORM_NAMES: dict[Callable, str] = {
        rich_transform: "rich",
        economy_average_vs_outcome: "averaged_economy",
        rich_transform_granular: "granular",
        event_transform: "events",
    }

    def convert(self, value, param, ctx):
        """
        Map a ``--transform`` choice to its transform function.

        Parameters
        ----------
        value : str
            Choice name.
        param : click.Parameter | None
            The option being converted.
        ctx : click.Context | None
            Current click context.

        Returns
        -------
        Callable
            The transform function.

        Raises
        ------
        click.BadParameter
            If ``value`` is not a known transform.
        """
        match value:
            case "rich":
                return rich_transform
            case "averaged_economy":
                return economy_average_vs_outcome
            case "granular":
                return rich_transform_granular
            case "events":
                # Ragged output; dispatched to its own entrypoint by main().
                return event_transform
            case _:
                raise click.BadParameter(f"Invalid transform choice: {value}")


def _transform_index_chunk(
    dataset_object: DataLoader,
    indices: list[int],
    transform_fn: Callable,
) -> tuple[list[torch.Tensor], list[torch.Tensor], int, int]:
    """
    Process a contiguous chunk of replay indices in one worker call.

    Parameters
    ----------
    dataset_object : DataLoader
        Indexable dataset returning ``SC2ReplayData``.
    indices : list[int]
        Indices of the chunk.
    transform_fn : Callable
        Dense transform returning ``(features, label)`` or None.

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor], int, int]
        Features, labels, number skipped and number of errors.
    """
    set_features: list[torch.Tensor] = []
    set_labels: list[torch.Tensor] = []
    skipped = 0
    errors = 0

    for index in indices:
        result = _transform_single_object(
            dataset_object=dataset_object,
            index=index,
            transform_fn=transform_fn,
        )

        if isinstance(result, Exception):
            errors += 1
            logging.error(f"Error processing replay at index {index}: {result}")
        elif result is None:
            skipped += 1
        else:
            features, label = result
            set_features.append(
                features
                if isinstance(features, torch.Tensor)
                else torch.tensor(features, dtype=torch.float32)
            )
            set_labels.append(label)

    return set_features, set_labels, skipped, errors


def process_set_chunked_single_pool(
    dataset_object: DataLoader,
    executor: ProcessPoolExecutor | ThreadPoolExecutor,
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]] = None,
    n_workers: int = 8,
    chunk_size: int = 256,
    max_inflight_tasks: int | None = None,
    set_name: str = "train",
) -> tuple[list[torch.Tensor], list[torch.Tensor], int, int]:
    """
    Transform one split with chunked tasks on a single persistent process pool.

    Results are collected in completion order, so row order is not
    reproducible; only the ``--test_only`` path uses this.

    Parameters
    ----------
    dataset_object : DataLoader
        Indexable dataset returning ``SC2ReplayData``.
    executor : ProcessPoolExecutor | ThreadPoolExecutor
        Executor class to instantiate.
    transform_fn : Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]]
        Dense transform, by default None.
    n_workers : int
        Worker processes, by default 8.
    chunk_size : int
        Indices per task, by default 256.
    max_inflight_tasks : int | None
        Cap on submitted-but-unfinished tasks; None uses ``1.1 * n_workers``.
    set_name : str
        Progress bar label, by default "train".

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor], int, int]
        Features, labels, number skipped and number of errors.
    """
    if max_inflight_tasks is None:
        max_inflight_tasks = max(2, int(n_workers * 1.1))

    total_size = len(dataset_object)
    chunks: list[list[int]] = [
        list(range(start, min(start + chunk_size, total_size)))
        for start in range(0, total_size, chunk_size)
    ]

    set_features: list[torch.Tensor] = []
    set_labels: list[torch.Tensor] = []
    skipped = 0
    errors = 0

    pending_futures = set()
    next_chunk = 0

    with executor(max_workers=n_workers) as executor:
        with tqdm(total=total_size, desc=f"{set_name} (chunked-single-pool)") as pbar:
            while (
                next_chunk < len(chunks) and len(pending_futures) < max_inflight_tasks
            ):
                future = executor.submit(
                    _transform_index_chunk,
                    dataset_object=dataset_object,
                    indices=chunks[next_chunk],
                    transform_fn=transform_fn,
                )
                pending_futures.add(future)
                next_chunk += 1

            while pending_futures:
                done, _ = wait(pending_futures, return_when=FIRST_COMPLETED)

                for future in done:
                    pending_futures.remove(future)

                    chunk_features, chunk_labels, chunk_skipped, chunk_errors = (
                        future.result()
                    )
                    set_features.extend(chunk_features)
                    set_labels.extend(chunk_labels)
                    skipped += chunk_skipped
                    errors += chunk_errors

                    pbar.update(len(chunk_features) + chunk_skipped + chunk_errors)

                    if next_chunk < len(chunks):
                        new_future = executor.submit(
                            _transform_index_chunk,
                            dataset_object=dataset_object,
                            indices=chunks[next_chunk],
                            transform_fn=transform_fn,
                        )
                        pending_futures.add(new_future)
                        next_chunk += 1

    return set_features, set_labels, skipped, errors


def preprocess_dataset_test_only(
    transform_name: str,
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]],
    single_json_dataset_path: Path | str,
    output_directory: Path | str = DATA_DIR,
    n_workers: int = 8,
    chunk_size: int = 64,
    max_inflight_tasks: int | None = None,
    n_samples: int = 0,
    seed: int = 42,
) -> None:
    """
    Extract a test-only cache from any SC2EGSet JSON file.

    Pools all indices (train + val + test of the library split), optionally
    subsamples ``n_samples``, and stores the result only in ``test_*``; train and
    val are empty. Intended for evaluating on an independent dataset; do not use
    it on the training JSON, since it includes training replays.

    Parameters
    ----------
    transform_name : str
        Transform name used in the output file name.
    transform_fn : Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]]
        Dense transform returning ``(features, label)`` or None.
    single_json_dataset_path : Path | str
        Path to the single-JSON file.
    output_directory : Path | str
        Directory for the cache file, by default ``DATA_DIR``.
    n_workers : int
        Worker processes, by default 8.
    chunk_size : int
        Indices per task, by default 64.
    max_inflight_tasks : int | None
        Cap on in-flight tasks, by default None.
    n_samples : int
        Number of valid replays to collect; 0 uses all, by default 0.
    seed : int
        Seed for the sampling order, by default 42.
    """
    logging.info(
        "SC2EGSet Dataset Pre-processing (test-only): "
        f"transform={transform_name}, n_workers={n_workers}, n_samples={n_samples}"
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

    all_indices: list[int] = sorted(
        list(train_dataset.indices)
        + list(test_dataset.indices)
        + list(val_dataset.indices)
    )
    total = len(all_indices)
    logging.info(f"  Total available replays: {total}")

    # Shuffle once with the seeded generator so each batch draws new indices
    # without repeats and without rebuilding a "used" set.
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(total, generator=generator).tolist()
    shuffled: list[int] = [all_indices[i] for i in perm]

    # Iterative batched processing: keep pulling batches from the shuffled
    # list until we have n_samples valid results (or the pool is exhausted).
    # The initial batch size is 1.5× the target to account for skips; subsequent
    # batches are sized from the observed yield rate.
    target = n_samples if n_samples > 0 else total
    batch_size = min(int(target * 1.5) + 1, total)

    all_features: list[torch.Tensor] = []
    all_labels: list = []
    total_skipped = 0
    total_errors = 0
    offset = 0

    while len(all_features) < target and offset < total:
        batch = shuffled[offset : offset + batch_size]
        offset += len(batch)

        test_dataset.indices = sorted(batch)
        batch_features, batch_labels, batch_skipped, batch_errors = (
            process_set_chunked_single_pool(
                dataset_object=test_dataset,
                executor=ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor,
                transform_fn=transform_fn,
                n_workers=n_workers,
                chunk_size=chunk_size,
                max_inflight_tasks=max_inflight_tasks,
                set_name="test",
            )
        )
        all_features.extend(batch_features)
        all_labels.extend(batch_labels)
        total_skipped += batch_skipped
        total_errors += batch_errors

        logging.info(
            f"  Progress: {len(all_features)}/{target} valid "
            f"({total_skipped} skipped, {total_errors} errors, {offset}/{total} processed)"
        )

        if len(all_features) < target and offset < total:
            remaining_needed = target - len(all_features)
            yield_rate = len(all_features) / max(offset, 1)
            batch_size = min(
                int(remaining_needed / max(yield_rate, 0.01) * 1.2) + 1,
                total - offset,
            )

    # Truncate to exactly target if we overshot
    all_features = all_features[:target]
    all_labels = all_labels[:target]

    logging.info(
        f"  Done: {len(all_features)} valid, {total_skipped} skipped, {total_errors} errors."
    )

    test_features_tensor = torch.stack(all_features)
    test_labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    empty_features = torch.empty(0, *test_features_tensor.shape[1:])
    empty_labels = torch.empty(0, dtype=torch.long)

    file_spec = CachedDatasetFileSpec(
        train_features=empty_features,
        train_labels=empty_labels,
        test_features=test_features_tensor,
        test_labels=test_labels_tensor,
        val_features=empty_features,
        val_labels=empty_labels,
        transform=transform_name,
    )

    os.makedirs(os.path.dirname(output_directory), exist_ok=True)
    n_tag = f"_{len(all_features)}" if n_samples > 0 else ""
    path_to_save = output_directory / f"cached_dataset_{transform_name}_test{n_tag}.pt"
    torch.save(asdict(file_spec), path_to_save)

    file_size_mb = os.path.getsize(path_to_save) / (1024 * 1024)
    logging.info(f"\n{'=' * 60}")
    logging.info("Test-only pre-processing complete!")
    logging.info(f"  Transform:     {transform_name}")
    logging.info(f"  Valid test:    {len(test_features_tensor)}")
    logging.info(f"  Feature shape: {test_features_tensor.shape}")
    logging.info(f"  Cache file:    {path_to_save} ({file_size_mb:.1f} MB)")
    logging.info(f"{'=' * 60}")


def preprocess_dataset(
    transform_name: str,
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]],
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
    Build ``cached_dataset_<transform>[_<n>].pt`` with splits from ``splits.json``.

    Parameters
    ----------
    transform_name : str
        Transform name used in the output file name.
    transform_fn : Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]]
        Dense transform returning ``(features, label)`` or None.
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

    Raises
    ------
    ValueError
        If a split ends up empty.
    """
    logging.info(
        f"SC2EGSet dense pre-processing: transform={transform_name}, "
        f"n_workers={n_workers}, chunk_size={chunk_size}, n_samples={n_samples}"
    )

    result = extract_with_manifest(
        load_source_dataset(single_json_dataset_path),
        functools.partial(process_replay, transform_fn=transform_fn),
        transform_name=transform_name,
        dataset_file=Path(single_json_dataset_path).name,
        splits_path=Path(splits_path),
        create_splits=create_splits,
        n_samples=n_samples,
        seed=seed,
        n_workers=n_workers,
        chunk_size=chunk_size,
    )

    tensors: dict[str, torch.Tensor] = {}
    for split in SPLIT_NAMES:
        pairs = result.outputs[split]
        if not pairs:
            raise ValueError(f"Split '{split}' is empty; nothing to cache.")
        tensors[f"{split}_features"] = torch.stack(
            [
                features
                if isinstance(features, torch.Tensor)
                else torch.tensor(features, dtype=torch.float32)
                for features, _ in pairs
            ]
        )
        tensors[f"{split}_labels"] = torch.tensor(
            [int(label) for _, label in pairs], dtype=torch.long
        )

    manifest = result.manifest
    file_spec = CachedDatasetFileSpec(
        **tensors,
        transform=transform_name,
        train_ids=manifest.ids("train"),
        val_ids=manifest.ids("val"),
        test_ids=manifest.ids("test"),
        splits_sha256=manifest.sha256(),
    )

    output_directory = Path(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    suffix = f"_{n_samples}" if n_samples > 0 else ""
    path_to_save = output_directory / f"cached_dataset_{transform_name}{suffix}.pt"
    torch.save(asdict(file_spec), path_to_save)

    logging.info("=" * 60)
    logging.info("Dense pre-processing complete!")
    logging.info(f"  Transform:        {transform_name}")
    for split in SPLIT_NAMES:
        logging.info(f"  {split:<6} rows:      {len(tensors[f'{split}_labels'])}")
    logging.info(f"  Skipped / errors: {result.n_skipped} / {result.n_errors}")
    logging.info(f"  Feature shape:    {tuple(tensors['train_features'].shape)}")
    logging.info(f"  Splits sha256:    {manifest.sha256()}")
    logging.info(f"  Cache file:       {path_to_save}")
    logging.info(f"{'=' * 60}")
