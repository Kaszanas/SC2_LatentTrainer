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

import os
import sys
import argparse
import torch
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from sc2_datasets.lightning.sc2_egset_datamodule import SC2EGSetDataModule
from sc2_datasets.available_replaypacks import SC2EGSET_DATASET_REPLAYPACKS
from sc2_datasets.transforms.pytorch.economy_vs_outcome import (
    economy_average_vs_outcome,
)
from src.latent_trainer.features.rich_transform import rich_transform


def preprocess_dataset(
    output_path: str = "data/cached_dataset.pt",
    transform_fn=None,
    transform_name: str = "rich",
):
    """Pre-process the entire SC2EGSet dataset and save to cache."""

    # The rich transform handles its own feature extraction from raw replay
    # So we pass transform=None to the datamodule
    is_raw_transform = transform_name == "rich"

    print("=" * 60)
    print(f"SC2EGSet Dataset Pre-processing ({transform_name} transform)")
    print("=" * 60)

    # Initialize datamodule (this downloads + extracts if needed)
    print("\n[1/3] Loading SC2EGSet datamodule (downloading if needed)...")
    datamodule = SC2EGSetDataModule(
        unpack_dir="./data/unpack",
        download_dir="./data/download",
        download=True,
        replaypacks=SC2EGSET_DATASET_REPLAYPACKS,
        transform=None,  # We'll apply transform manually to catch errors
        batch_size=1,
        num_workers=0,
    )

    # Access the underlying dataset
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Get train and val datasets
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset

    total = len(train_dataset) + len(val_dataset)
    print(
        f"  Total replays: {total} (train: {len(train_dataset)}, val: {len(val_dataset)})"
    )

    # Process all replays
    print("\n[2/3] Processing replays and applying transform...")

    train_features = []
    train_labels = []
    val_features = []
    val_labels = []

    skipped = 0
    errors = 0

    def process_replay(replay, transform_fn, is_raw):
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

    # Process training set
    print("  Processing training set...")
    for i in tqdm(range(len(train_dataset)), desc="  Train"):
        try:
            replay = train_dataset[i]
            result = process_replay(replay, transform_fn, is_raw_transform)

            if result is None:
                skipped += 1
                continue

            features, label = result
            train_features.append(
                features
                if isinstance(features, torch.Tensor)
                else torch.tensor(features, dtype=torch.float32)
            )
            train_labels.append(label)

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    Error on train[{i}]: {type(e).__name__}: {e}")

    # Process validation set
    print("  Processing validation set...")
    for i in tqdm(range(len(val_dataset)), desc="  Val"):
        try:
            replay = val_dataset[i]
            result = process_replay(replay, transform_fn, is_raw_transform)

            if result is None:
                skipped += 1
                continue

            features, label = result
            val_features.append(
                features
                if isinstance(features, torch.Tensor)
                else torch.tensor(features, dtype=torch.float32)
            )
            val_labels.append(label)

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    Error on val[{i}]: {type(e).__name__}: {e}")

    # Stack into tensors
    print("\n[3/3] Saving cached dataset...")

    train_features_tensor = torch.stack(train_features)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_features_tensor = torch.stack(val_features)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "train_features": train_features_tensor,
            "train_labels": train_labels_tensor,
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
    print(f"  Skipped (None):   {skipped}")
    print(f"  Errors:           {errors}")
    print(f"  Feature shape:    {train_features_tensor.shape}")
    print(f"  Cache file:       {output_path} ({file_size_mb:.1f} MB)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
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
    )
