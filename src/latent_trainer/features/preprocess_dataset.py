"""Pre-process the SC2EGSet dataset and cache transformed tensors to disk.

This script iterates through all replays once, applies the transform,
catches broken replays, and saves valid (features, label) pairs to a .pt file.
Subsequent training runs can load from this cache instantly.

Usage:
    # Default (rich transform):
    uv run python src/latent_trainer/features/preprocess_dataset.py

    # Legacy economy-average transform:
    uv run python src/latent_trainer/features/preprocess_dataset.py --transform economy
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor
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

from latent_trainer.features.rich_transform import rich_transform
from latent_trainer.features.type import CachedDatasetFileSpec


def _transform_single_object(
    dataset_object: DataLoader,
    index: int,
    transform_fn: Callable,
) -> tuple[torch.Tensor, torch.Tensor] | None | Exception:

    try:
        sc2_replay_data = dataset_object[index]

        result = process_replay(
            replay=sc2_replay_data,
            transform_fn=transform_fn,
        )

        if result is None:
            return None
    except Exception as e:
        return e

    return result


def submit_task(
    executor: ProcessPoolExecutor,
    dataset_object: DataLoader,
    index: int,
    transform_fn: Callable,
) -> tuple[torch.Tensor | None, Exception | None]:

    future = executor.submit(
        _transform_single_object,
        dataset_object=dataset_object,
        index=index,
        transform_fn=transform_fn,
    )
    return future


def process_set(
    dataloader_object: DataLoader,
    set_name: str = "train",
) -> tuple[list[torch.Tensor], list[int]]:
    """
    Process a dataset split (train/test/val) in parallel,
    with bounded in-flight futures and continuous replenishment.
    """

    dataset_features: list[torch.Tensor] = []
    dataset_labels: list[int] = []

    for batch in tqdm(dataloader_object, desc=f"Processing {set_name} set"):
        features, labels = batch

        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)

        dataset_features.append(features)
        dataset_labels.append(labels)

    return dataset_features, dataset_labels


def process_replay(
    replay: SC2ReplayData,
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Apply transform and return (features, label) or None."""
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


def preprocess_dataset(
    transform_name: str,
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]],
    single_json_dataset_path: Path | str,
    output_directory: Path | str = Path("./data").resolve(),
    n_workers: int = 24,
) -> None:
    """
    Preprocess a single JSON dataset and cache the transformed tensors to disk.

    Parameters
    ----------
    transform_name : str
        The name of the transformation (e.g. ``"rich"`` or ``"averaged_economy"``).
        Used to label the output file as ``cached_dataset_<transform_name>.pt``.
    transform_fn : Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]]
        Function that maps a raw SC2ReplayData replay to a (features, label) pair.
    dataset_name : str, optional
        Name of the dataset, matching the JSON file stem, by default ``"sc2egset_merged"``.
    single_json_dataset_path : Path | str, optional
        Path to the single-JSON index file for the dataset,
        by default ``H:/sc2egset_merged/sc2egset_merged.json``.
    output_directory : Path | str, optional
        Directory where the cached ``.pt`` file will be written,
        by default ``./data``.
    n_workers : int, optional
        Number of worker processes for parallel replay processing, by default 24.
    """

    dataset_name = single_json_dataset_path.stem

    output_directory = (
        output_directory
        if isinstance(output_directory, Path)
        else Path(output_directory).resolve()
    )

    logging.info(f"SC2EGSet Dataset Pre-processing ({transform_name} transform)")

    # Initialize datamodule (this downloads + extracts if needed)
    logging.info("[1/3] Loading SC2EGSet datamodule (downloading if needed)...")

    datamodule = SC2EGSetDataModuleSingleJSON(
        json_path=single_json_dataset_path,
        download=False,
        batch_size=128,
        num_workers=n_workers,
        transform=transform_fn,
    )

    # Initialize the datamodule to get train/test/val splits (but skip any transforms for now)
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Get train, test, and val datasets
    train_dataset = datamodule.train_dataset
    test_dataset = datamodule.test_dataset
    val_dataset = datamodule.val_dataset

    total = check_split(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
    )

    # Process all replays
    logging.info("[2/3] Processing replays and applying transform...")

    logging.info("  Processing training set...")

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    train_features, train_labels = process_set(
        dataloader_object=train_dataloader,
        set_name="train",
    )
    logging.info("  Processing test set...")
    test_features, test_labels = process_set(
        dataloader_object=test_dataloader,
        n_workers=n_workers,
        set_name="test",
    )
    logging.info("  Processing validation set...")
    val_features, val_labels = process_set(
        dataloader_object=val_dataloader,
        set_name="val",
    )

    # Stack into tensors
    logging.info("\n[3/3] Saving cached dataset...")

    train_features_tensor = torch.stack(train_features)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_features_tensor = torch.stack(test_features)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    val_features_tensor = torch.stack(val_features)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    file_spec = CachedDatasetFileSpec(
        train_features=train_features_tensor,
        train_labels=train_labels_tensor,
        test_features=test_features_tensor,
        test_labels=test_labels_tensor,
        val_features=val_features_tensor,
        val_labels=val_labels_tensor,
        transform=transform_name,
    )

    # Save
    os.makedirs(os.path.dirname(output_directory), exist_ok=True)
    path_to_save = (
        output_directory / f"{dataset_name}_cached_dataset_{transform_name}.pt"
    )
    torch.save(
        asdict(file_spec),
        path_to_save,
    )

    file_size_mb = os.path.getsize(path_to_save) / (1024 * 1024)

    logging.info(f"\n{'=' * 60}")
    logging.info("Pre-processing complete!")
    logging.info(f"  Transform:        {transform_name}")
    logging.info(f"  Total replays:    {total}")
    logging.info(f"  Valid train:      {len(train_features)}")
    logging.info(f"  Valid val:        {len(val_features)}")
    logging.info(f"  Feature shape:    {train_features_tensor.shape}")
    logging.info(f"  Cache file:       {output_directory} ({file_size_mb:.1f} MB)")
    logging.info(f"{'=' * 60}")


class TransformEnumFunction(click.Choice):
    """Custom Click Choice type that returns the actual transform function instead of the string name."""

    _TRANSFORM_NAMES: dict[Callable, str] = {
        rich_transform: "rich",
        economy_average_vs_outcome: "averaged_economy",
    }

    def convert(self, value, param, ctx):
        match value:
            case "rich":
                return rich_transform
            case "averaged_economy":
                return economy_average_vs_outcome
            case _:
                raise click.BadParameter(f"Invalid transform choice: {value}")
