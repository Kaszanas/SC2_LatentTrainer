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
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _transform_single_object(
    dataset_object: DataLoader,
    index: int,
    transform_fn: Callable,
) -> tuple[torch.Tensor, torch.Tensor] | None | Exception:

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


def process_set(
    dataset_object: DataLoader,
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]] = None,
    n_workers: int = 24,
) -> tuple[list[torch.Tensor], list[torch.Tensor], int, int]:
    """
    Process a dataset split (train/test/val) in parallel,
    apply the transform, and return lists of features and labels along with counts of skipped and errored replays.

    Parameters
    ----------
    dataset_object : DataLoader
        Dataloader for the dataset split to process (train/test/val)
    transform_fn : Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]], optional
        Function to apply as the transform, by default None
    n_workers : int, optional
        Number of worker processes to use for parallel processing, by default 24

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor], int, int]
        A tuple containing:
        - List of feature tensors
        - List of label tensors
        - Count of skipped replays (where transform returned None)
        - Count of errors encountered during processing
    """

    skipped = 0
    errors = 0

    set_features = []
    set_labels = []

    # Process training set
    logging.info("  Processing training set...")
    # for i in tqdm(range(len(dataset_object)), desc="  Train"):
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                _transform_single_object,
                dataset_object=dataset_object,
                index=i,
                transform_fn=transform_fn,
            )
            for i in range(len(dataset_object))
        ]

        for future in tqdm(
            as_completed(futures),
            desc="  Train (parallel)",
            total=len(futures),
        ):
            result = future.result()

            if isinstance(result, Exception):
                errors += 1
                if errors <= 5:
                    logging.info(f"    Error: {type(result).__name__}: {result}")
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


def preprocess_dataset(
    transform_name: str,
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]],
    dataset_name: str = "sc2egset_merged",
    single_json_dataset_path: Path | str = Path(
        "H:/sc2egset_merged/sc2egset_merged.json"
    ).resolve(),
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

    output_directory = (
        output_directory
        if isinstance(output_directory, Path)
        else Path(output_directory).resolve()
    )

    logging.info(f"SC2EGSet Dataset Pre-processing ({transform_name} transform)")

    # Initialize datamodule (this downloads + extracts if needed)
    logging.info("[1/3] Loading SC2EGSet datamodule (downloading if needed)...")

    datamodule = SC2EGSetDataModuleSingleJSON(
        dataset_name=dataset_name,
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

    total = len(train_dataset) + len(val_dataset)
    logging.info(
        f"  Total replays: {total} (train: {len(train_dataset)}, test: {len(test_dataset)}, val: {len(val_dataset)})"
    )

    # Process all replays
    logging.info("[2/3] Processing replays and applying transform...")

    logging.info("  Processing training set...")
    train_features, train_labels, skipped_train, errors_train = process_set(
        dataset_object=train_dataset,
        transform_fn=transform_fn,
        n_workers=n_workers,
    )
    logging.info("  Processing test set...")
    test_features, test_labels, skipped_test, errors_test = process_set(
        dataset_object=test_dataset,
        transform_fn=transform_fn,
        n_workers=n_workers,
    )
    logging.info("  Processing validation set...")
    val_features, val_labels, skipped_val, errors_val = process_set(
        dataset_object=val_dataset,
        transform_fn=transform_fn,
        n_workers=n_workers,
    )

    # Stack into tensors
    logging.info("\n[3/3] Saving cached dataset...")

    train_features_tensor = torch.stack(train_features)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_features_tensor = torch.stack(test_features)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    val_features_tensor = torch.stack(val_features)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    # Save
    os.makedirs(os.path.dirname(output_directory), exist_ok=True)
    path_to_save = output_directory / f"cached_dataset_{transform_name}.pt"
    torch.save(
        {
            "train_features": train_features_tensor,
            "train_labels": train_labels_tensor,
            "test_features": test_features_tensor,
            "test_labels": test_labels_tensor,
            "val_features": val_features_tensor,
            "val_labels": val_labels_tensor,
            "transform": transform_name,
        },
        path_to_save,
    )

    file_size_mb = os.path.getsize(path_to_save) / (1024 * 1024)

    logging.info(f"\n{'=' * 60}")
    logging.info("Pre-processing complete!")
    logging.info(f"  Transform:        {transform_name}")
    logging.info(f"  Total replays:    {total}")
    logging.info(f"  Valid train:      {len(train_features)}")
    logging.info(f"  Valid val:        {len(val_features)}")
    logging.info(f"  Skipped (None):   {skipped_train + skipped_val + skipped_test}")
    logging.info(f"  Errors:           {errors_train + errors_val + errors_test}")
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


@click.command(
    help="Pre-process the SC2_Dataset in a JSON format and cache transformed tensors to disk."
)
@click.option(
    "--transform",
    type=TransformEnumFunction(["rich", "averaged_economy"]),
    default="rich",
    show_default=True,
    help="Transform to use: 'rich' (temporal+meta+units, 204 features) or 'averaged_economy' (averaged economy, 39 features)",
)
@click.option(
    "--output-directory",
    type=click.Path(
        file_okay=False,
        writable=True,
        path_type=Path,
        resolve_path=True,
    ),
    help="Output directory for cached dataset (default: data/)",
)
@click.option(
    "--n-workers",
    type=int,
    default=24,
    show_default=True,
    help="Number of parallel workers for processing replays.",
)
def main(transform: Callable, output_directory: Path | None, n_workers: int) -> None:
    """Pre-process Single JSON SC2_Dataset and cache the transformed tensors to drive."""
    transform_name = TransformEnumFunction._TRANSFORM_NAMES[transform]
    output_path = output_directory or Path("data")
    preprocess_dataset(
        output_directory=output_path,
        transform_fn=transform,
        transform_name=transform_name,
        n_workers=n_workers,
    )


if __name__ == "__main__":
    main()
