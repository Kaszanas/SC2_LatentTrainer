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

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from sc2_datasets.lightning.sc2_egset_datamodule import (
    SC2EGSetDataModuleSingleJSON,
)
from sc2_datasets.replay_data.sc2_replay_data import SC2ReplayData
from sc2_datasets.transforms.pytorch.economy_vs_outcome import (
    economy_average_vs_outcome,
)

from src.latent_trainer.features.rich_transform import rich_transform


def _transform_single_object(
    dataset_object: DataLoader,
    index: int,
    transform_fn: Callable,
    is_raw_transform: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None | Exception:

    try:
        replay = dataset_object[index]

        result = process_replay(
            replay=replay,
            transform_fn=transform_fn,
            is_raw=is_raw_transform,
        )

        if result is None:
            return None
    except Exception as e:
        return e

    return result


def process_set(
    dataset_object: DataLoader,
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]] = None,
    is_raw_transform: bool = False,
    n_workers: int = 24,
):

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
                is_raw_transform=is_raw_transform,
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
    is_raw: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Apply transform and return (features, label) or None."""
    if is_raw:
        # Rich transform takes the raw replay directly
        result = transform_fn(replay)
    else:
        # Legacy transform is already applied via __getitem__ or manually
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
    output_path: Path | str = Path("data/cached_dataset.pt").resolve(),
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]]
    | None = None,
    transform_name: str = "rich",
    n_workers: int = 24,
) -> None:
    """Pre-process the entire SC2EGSet dataset and save to cache."""

    # The rich transform handles its own feature extraction from raw replay
    # So we pass transform=None to the datamodule
    is_raw_transform = transform_name == "rich"

    print("=" * 60)
    print(f"SC2EGSet Dataset Pre-processing ({transform_name} transform)")
    print("=" * 60)

    # Initialize datamodule (this downloads + extracts if needed)
    print("\n[1/3] Loading SC2EGSet datamodule (downloading if needed)...")

    datamodule = SC2EGSetDataModuleSingleJSON(
        dataset_name="sc2egset_merged",
        json_path=Path("H:\sc2egset_merged\sc2egset_merged.json"),
        download=False,
    )

    # datamodule = SC2EGSetDataModule(
    #     unpack_dir="./data/unpack",
    #     download_dir="./data/download",
    #     download=True,
    #     replaypacks=SC2EGSET_DATASET_REPLAYPACKS,
    #     transform=None,  # We'll apply transform manually to catch errors
    #     batch_size=1,
    #     num_workers=0,
    # )

    # Access the underlying dataset
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Get train and val datasets
    train_dataset = datamodule.train_dataset
    test_dataset = datamodule.test_dataset

    val_dataset = datamodule.val_dataset

    total = len(train_dataset) + len(val_dataset)
    print(
        f"  Total replays: {total} (train: {len(train_dataset)}, test: {len(test_dataset)}, val: {len(val_dataset)})"
    )

    # Process all replays
    print("\n[2/3] Processing replays and applying transform...")

    logging.info("  Processing training set...")
    train_features, train_labels, skipped_train, errors_train = process_set(
        dataset_object=train_dataset,
        transform_fn=transform_fn,
        is_raw_transform=is_raw_transform,
        n_workers=n_workers,
    )
    logging.info("  Processing test set...")
    test_features, test_labels, skipped_test, errors_test = process_set(
        dataset_object=test_dataset,
        transform_fn=transform_fn,
        is_raw_transform=is_raw_transform,
        n_workers=n_workers,
    )
    logging.info("  Processing validation set...")
    val_features, val_labels, skipped_val, errors_val = process_set(
        dataset_object=val_dataset,
        transform_fn=transform_fn,
        is_raw_transform=is_raw_transform,
        n_workers=n_workers,
    )

    # Stack into tensors
    print("\n[3/3] Saving cached dataset...")

    train_features_tensor = torch.stack(train_features)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_features_tensor = torch.stack(test_features)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    val_features_tensor = torch.stack(val_features)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
        output_path,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print("Pre-processing complete!")
    print(f"  Transform:        {transform_name}")
    print(f"  Total replays:    {total}")
    print(f"  Valid train:      {len(train_features)}")
    print(f"  Valid val:        {len(val_features)}")
    print(f"  Skipped (None):   {skipped_train + skipped_val + skipped_test}")
    print(f"  Errors:           {errors_train + errors_val + errors_test}")
    print(f"  Feature shape:    {train_features_tensor.shape}")
    print(f"  Cache file:       {output_path} ({file_size_mb:.1f} MB)")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Pre-process SC2EGSet dataset")
    parser.add_argument(
        "--transform",
        choices=["rich", "economy"],
        default="rich",
        help="Transform to use: 'rich' (temporal+meta+units, 204 features) or 'economy' (averaged economy, 39 features)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for cached dataset (default: data/cached_dataset_<transform>.pt)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=24,
        help="Number of parallel workers for processing replays (default: 24)",
    )
    args = parser.parse_args()

    if args.transform == "rich":
        transform_fn = rich_transform
        default_output = "data/cached_dataset_rich.pt"
    else:
        transform_fn = economy_average_vs_outcome
        default_output = "data/cached_dataset.pt"

    output_path = args.output or default_output
    preprocess_dataset(
        output_path=output_path,
        transform_fn=transform_fn,
        transform_name=args.transform,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
