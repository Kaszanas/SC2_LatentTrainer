"""Pre-process the SC2EGSet dataset and cache transformed tensors to disk.

This script iterates through all replays once, applies the transform,
catches broken replays, and saves valid (features, label) pairs to a memmap
TensorDict cache directory. Subsequent training runs can load from this cache
instantly.

Usage:
    # Default (rich transform):
    uv run python src/latent_trainer/features/preprocess_dataset.py

    # Legacy economy-average transform:
    uv run python src/latent_trainer/features/preprocess_dataset.py --transform economy

On-disk format (memmap TensorDict):

    <output_directory>/cached_dataset_<transform_name>/
      train/
        features/   TensorDict batch_size=[N_train, 2]
                    nested keys: early, mid, late, final, delta,
                                 meta, units_born, units_killed, upgrade_count
        labels      Tensor [N_train], long  (0=loss, 1=win)
      val/
        features/   TensorDict batch_size=[N_val, 2]
        labels      Tensor [N_val], long
      test/
        features/   TensorDict batch_size=[N_test, 2]
        labels      Tensor [N_test], long
"""

import logging
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path
from typing import Callable

import torch
from sc2_datasets.lightning.sc2_egset_datamodule import (
    SC2EGSetDataModuleSingleJSON,
)
from sc2_datasets.replay_data.sc2_replay_data import SC2ReplayData
from tensordict import TensorDict
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from latent_trainer.settings import DATA_DIR


def _transform_single_object(
    dataset_object: DataLoader,
    index: int,
    transform_fn: Callable,
) -> tuple[TensorDict, int] | None | Exception:

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


def process_batch(
    executor: ProcessPoolExecutor | ThreadPoolExecutor,
    n_workers: int,
    dataset_object: Dataset,
    transform_fn: Callable,
    start_index: int,
    end_index: int,
) -> tuple[list[TensorDict], list[int], int, int]:
    set_features: list[TensorDict] = []
    set_labels: list[int] = []
    skipped = 0
    errors = 0

    with executor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                _transform_single_object,
                dataset_object=dataset_object,
                index=i,
                transform_fn=transform_fn,
            )
            for i in range(start_index, end_index)
        ]

        for future in tqdm(
            as_completed(futures),
            desc=f"  Batch {start_index}:{end_index} (parallel)",
            total=len(futures),
        ):
            result = future.result()

            if isinstance(result, Exception):
                errors += 1
            elif result is None:
                skipped += 1
            else:
                features, label = result
                set_features.append(features)
                set_labels.append(label)

    return set_features, set_labels, skipped, errors


def process_set(
    dataset_object: DataLoader,
    executor: ProcessPoolExecutor | ThreadPoolExecutor,
    transform_fn: Callable[[SC2ReplayData], tuple[TensorDict, int] | None] = None,
    n_workers: int = 24,
) -> tuple[list[TensorDict], list[int], int, int]:
    """
    Process a dataset split (train/test/val) in parallel,
    apply the transform, and return lists of features and labels along with counts of skipped and errored replays.

    Parameters
    ----------
    dataset_object : DataLoader
        Dataloader for the dataset split to process (train/test/val)
    transform_fn : Callable[[SC2ReplayData], tuple[TensorDict, int] | None], optional
        Function to apply as the transform, by default None
    n_workers : int, optional
        Number of worker processes to use for parallel processing, by default 24

    Returns
    -------
    tuple[list[TensorDict], list[int], int, int]
        A tuple containing:
        - List of feature TensorDicts (each batch_size=[2])
        - List of labels (int)
        - Count of skipped replays (where transform returned None)
        - Count of errors encountered during processing
    """

    skipped = 0
    errors = 0

    set_features: list[TensorDict] = []
    set_labels: list[int] = []

    batch_size = 20_000
    total_size = len(dataset_object)

    for start_index in range(0, total_size, batch_size):
        end_index = min(start_index + batch_size, total_size)
        batch_features, batch_labels, batch_skipped, batch_errors = process_batch(
            executor=executor,
            n_workers=n_workers,
            dataset_object=dataset_object,
            transform_fn=transform_fn,
            start_index=start_index,
            end_index=end_index,
        )

        set_features.extend(batch_features)
        set_labels.extend(batch_labels)
        skipped += batch_skipped
        errors += batch_errors

    return set_features, set_labels, skipped, errors


def process_replay(
    replay: SC2ReplayData,
    transform_fn: Callable[[SC2ReplayData], tuple[TensorDict, int] | None],
) -> tuple[TensorDict, int] | None:
    """Apply transform and return (features, label) or None."""
    result = transform_fn(replay)

    if result is None:
        return None

    features, label = result

    if features is None or label is None:
        return None

    if label == -1:  # Undecided/Draw/Tie
        return None

    return features, label


def check_split(
    train_dataset: DataLoader,
    test_dataset: DataLoader,
    val_dataset: DataLoader,
) -> None:

    total = len(train_dataset) + len(test_dataset) + len(val_dataset)
    logging.info(
        f"  Total replays: {total} (train: {len(train_dataset)}, test: {len(test_dataset)}, val: {len(val_dataset)})"
    )
    if total == 0:
        logging.warning("Dataset is empty! No replays found.")
        return

    train_frac = len(train_dataset) / total
    test_frac = len(test_dataset) / total
    val_frac = len(val_dataset) / total

    # Check if within 1.5% of target
    if (
        abs(train_frac - 0.8) > 0.015
        or abs(test_frac - 0.1) > 0.015
        or abs(val_frac - 0.1) > 0.015
    ):
        logging.warning(
            f"Dataset split deviates from 80/10/10! "
            f"Got: Train={train_frac:.2%}, Test={test_frac:.2%}, Val={val_frac:.2%}"
        )
        raise ValueError(
            f"Dataset split deviates from 80/10/10! "
            f"Got: Train={train_frac:.2%}, Test={test_frac:.2%}, Val={val_frac:.2%}"
        )

    return total


def debug_preprocess_dataset(
    single_json_dataset_path: Path | str,
    transform_fn: Callable[[SC2ReplayData], tuple[TensorDict, int] | None],
) -> None:

    # Initialize datamodule (this downloads + extracts if needed)
    logging.info("[1/3] Loading SC2EGSet datamodule (downloading if needed)...")

    datamodule = SC2EGSetDataModuleSingleJSON(
        json_path=single_json_dataset_path,
        download=False,
    )

    # Initialize the datamodule to get train/test/val splits (but skip any transforms for now)
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Get train, test, and val datasets
    train_dataset = datamodule.train_dataset
    test_dataset = datamodule.test_dataset
    val_dataset = datamodule.val_dataset

    train_dataset.indices = sorted(train_dataset.indices)
    test_dataset.indices = sorted(test_dataset.indices)
    val_dataset.indices = sorted(val_dataset.indices)

    for i in range(len(train_dataset)):
        replay = train_dataset[i]

        result = process_replay(
            replay=replay,
            transform_fn=transform_fn,
        )

        if result is None:
            logging.info(f"Replay {i} skipped (transform returned None)")
        else:
            features, label = result
            logging.info(f"Replay {i} processed successfully. Label: {label}")


def _save_dataset(
    output_directory: Path,
    transform_name: str,
    train_features: TensorDict,   # batch_size=[N_train, 2] — stacked/catted by caller
    train_labels: list[int],
    val_features: TensorDict,     # batch_size=[N_val, 2]
    val_labels: list[int],
    test_features: TensorDict,    # batch_size=[N_test, 2]
    test_labels: list[int],
) -> Path:
    """Wrap pre-stacked feature TensorDicts with labels and save as a memmap TensorDict.

    Returns the path to the saved cache directory.
    """
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_labels_tensor   = torch.tensor(val_labels,   dtype=torch.long)
    test_labels_tensor  = torch.tensor(test_labels,  dtype=torch.long)

    n_features = sum(t.numel() for t in train_features[0, 0].values(True, True))

    train_td = TensorDict(
        {"features": train_features, "labels": train_labels_tensor},
        batch_size=[len(train_labels_tensor)],
    )
    val_td = TensorDict(
        {"features": val_features, "labels": val_labels_tensor},
        batch_size=[len(val_labels_tensor)],
    )
    test_td = TensorDict(
        {"features": test_features, "labels": test_labels_tensor},
        batch_size=[len(test_labels_tensor)],
    )

    dataset_td = TensorDict(
        {"train": train_td, "val": val_td, "test": test_td},
        batch_size=[],
    )

    save_dir = output_directory / f"cached_dataset_{transform_name}"
    dataset_td.memmap(str(save_dir))

    logging.info(f"  Feature batch:    {train_features.shape}  n_features={n_features}")
    logging.info(f"  Cache dir:        {save_dir}")

    return save_dir


def preprocess_dataset(
    transform_name: str,
    transform_fn: Callable[[SC2ReplayData], tuple[TensorDict, int] | None],
    single_json_dataset_path: Path | str,
    output_directory: Path | str = DATA_DIR,
    n_workers: int = 24,
) -> None:
    """
    Preprocess a single JSON dataset and cache the transformed tensors to disk.

    Parameters
    ----------
    transform_name : str
        The name of the transformation (e.g. ``"rich"`` or ``"averaged_economy"``).
        Used to label the output directory as ``cached_dataset_<transform_name>``.
    transform_fn : Callable[[SC2ReplayData], tuple[TensorDict, int] | None]
        Function that maps a raw SC2ReplayData replay to a (features, label) pair.
    single_json_dataset_path : Path | str
        Path to the single-JSON index file for the dataset.
    output_directory : Path | str, optional
        Directory where the cached memmap directory will be written,
        by default ``./data``.
    n_workers : int, optional
        Number of worker processes for parallel replay processing, by default 24.
    """

    logging.info(f"SC2EGSet Dataset Pre-processing ({transform_name} transform)")

    # Initialize datamodule (this downloads + extracts if needed)
    logging.info("[1/3] Loading SC2EGSet datamodule (downloading if needed)...")

    datamodule = SC2EGSetDataModuleSingleJSON(
        json_path=single_json_dataset_path,
        download=False,
    )

    # Initialize the datamodule to get train/test/val splits (but skip any transforms for now)
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Get train, test, and val datasets
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

    # Process all replays
    logging.info("[2/3] Processing replays and applying transform...")

    logging.info("Processing training set...")
    train_features, train_labels, skipped_train, errors_train = process_set(
        executor=ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor,
        dataset_object=train_dataset,
        transform_fn=transform_fn,
        n_workers=n_workers,
    )
    logging.info("Processing test set...")
    test_features, test_labels, skipped_test, errors_test = process_set(
        executor=ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor,
        dataset_object=test_dataset,
        transform_fn=transform_fn,
        n_workers=n_workers,
    )
    logging.info("Processing validation set...")
    val_features, val_labels, skipped_val, errors_val = process_set(
        executor=ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor,
        dataset_object=val_dataset,
        transform_fn=transform_fn,
        n_workers=n_workers,
    )

    # Stack and save
    logging.info("\n[3/3] Saving cached dataset...")

    save_dir = _save_dataset(
        output_directory=output_directory,
        transform_name=transform_name,
        train_features=torch.stack(train_features),   # batch_size=[N_train, 2]
        train_labels=train_labels,
        val_features=torch.stack(val_features),       # batch_size=[N_val, 2]
        val_labels=val_labels,
        test_features=torch.stack(test_features),     # batch_size=[N_test, 2]
        test_labels=test_labels,
    )

    logging.info(f"\n{'=' * 60}")
    logging.info("Pre-processing complete!")
    logging.info(f"  Transform:        {transform_name}")
    logging.info(f"  Total replays:    {total}")
    logging.info(f"  Valid train:      {len(train_features)}")
    logging.info(f"  Valid val:        {len(val_features)}")
    logging.info(f"  Skipped (None):   {skipped_train + skipped_val + skipped_test}")
    logging.info(f"  Errors:           {errors_train + errors_val + errors_test}")
    logging.info(f"  Cache dir:        {save_dir}")
    logging.info(f"{'=' * 60}")


def _transform_index_chunk(
    dataset_object: DataLoader,
    indices: list[int],
    transform_fn: Callable,
) -> tuple[TensorDict | None, list[int], int, int]:
    """Process a contiguous chunk of replay indices in one worker call.

    Returns a stacked TensorDict batch_size=[chunk_valid, 2] rather than a list,
    reducing IPC pickling overhead when results are returned to the main process.
    Returns None for the features if every replay in the chunk was skipped/errored.
    """
    set_features: list[TensorDict] = []
    set_labels: list[int] = []
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
        elif result is None:
            skipped += 1
        else:
            features, label = result
            set_features.append(features)
            set_labels.append(label)

    stacked = torch.stack(set_features) if set_features else None
    return stacked, set_labels, skipped, errors


def process_set_chunked_single_pool(
    executor: ProcessPoolExecutor | ThreadPoolExecutor,
    dataset_object: DataLoader,
    transform_fn: Callable[[SC2ReplayData], tuple[TensorDict, int] | None] = None,
    n_workers: int = 8,
    chunk_size: int = 256,
    max_inflight_tasks: int | None = None,
    set_name: str = "train",
) -> tuple[list[TensorDict], list[int], int, int]:
    """Fast path: chunked index tasks with one persistent process pool.

    Each chunk is processed by a single worker call that returns a stacked
    TensorDict batch_size=[chunk_valid, 2] instead of individual batch_size=[2]
    items, reducing IPC overhead. The caller must ``torch.cat`` the returned list.
    """
    if max_inflight_tasks is None:
        max_inflight_tasks = max(2, int(n_workers * 1.1))

    total_size = len(dataset_object)
    chunks: list[list[int]] = [
        list(range(start, min(start + chunk_size, total_size)))
        for start in range(0, total_size, chunk_size)
    ]

    set_features: list[TensorDict] = []   # each item: batch_size=[chunk_valid, 2]
    set_labels: list[int] = []
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
                    if chunk_features is not None:
                        set_features.append(chunk_features)
                    set_labels.extend(chunk_labels)
                    skipped += chunk_skipped
                    errors += chunk_errors

                    pbar.update(len(chunk_labels) + chunk_skipped + chunk_errors)

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


def preprocess_dataset_chunked(
    transform_name: str,
    transform_fn: Callable[[SC2ReplayData], tuple[TensorDict, int] | None],
    single_json_dataset_path: Path | str,
    output_directory: Path | str = DATA_DIR,
    n_workers: int = 8,
    chunk_size: int = 64,
    max_inflight_tasks: int | None = None,
) -> None:
    """Alternative preprocess entrypoint for performance profiling.

    This leaves ``preprocess_dataset`` unchanged and writes a separate cache directory
    with ``_chunked`` suffix for side-by-side comparison.
    """

    logging.info(
        "SC2EGSet Dataset Pre-processing (chunked profile path): "
        f"transform={transform_name}, n_workers={n_workers}, "
        f"chunk_size={chunk_size}, max_inflight_tasks={max_inflight_tasks}"
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

    train_features, train_labels, skipped_train, errors_train = (
        process_set_chunked_single_pool(
            executor=ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor,
            dataset_object=train_dataset,
            transform_fn=transform_fn,
            n_workers=n_workers,
            chunk_size=chunk_size,
            max_inflight_tasks=max_inflight_tasks,
            set_name="train",
        )
    )

    test_features, test_labels, skipped_test, errors_test = (
        process_set_chunked_single_pool(
            executor=ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor,
            dataset_object=test_dataset,
            transform_fn=transform_fn,
            n_workers=n_workers,
            chunk_size=chunk_size,
            max_inflight_tasks=max_inflight_tasks,
            set_name="test",
        )
    )

    val_features, val_labels, skipped_val, errors_val = process_set_chunked_single_pool(
        executor=ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor,
        dataset_object=val_dataset,
        transform_fn=transform_fn,
        n_workers=n_workers,
        chunk_size=chunk_size,
        max_inflight_tasks=max_inflight_tasks,
        set_name="val",
    )

    save_dir = _save_dataset(
        output_directory=output_directory,
        transform_name=transform_name,
        train_features=torch.cat(train_features),   # batch_size=[N_train, 2]
        train_labels=train_labels,
        val_features=torch.cat(val_features),       # batch_size=[N_val, 2]
        val_labels=val_labels,
        test_features=torch.cat(test_features),     # batch_size=[N_test, 2]
        test_labels=test_labels,
    )

    logging.info(f"\n{'=' * 60}")
    logging.info("Chunked profile pre-processing complete!")
    logging.info(f"  Transform:        {transform_name}")
    logging.info(f"  Total replays:    {total}")
    logging.info(f"  Valid train:      {len(train_labels)}")
    logging.info(f"  Valid val:        {len(val_labels)}")
    logging.info(f"  Skipped (None):   {skipped_train + skipped_val + skipped_test}")
    logging.info(f"  Errors:           {errors_train + errors_val + errors_test}")
    logging.info(f"  Cache dir:        {save_dir}")
    logging.info(f"{'=' * 60}")
