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
    NormalizedData,
    NormalizedDataWithLabels,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Normalisation
# ------------------------------------------------------------------
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
    std = train_flat.std(dim=0, keepdim=True) + 1e-8

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


# ------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------
def load_and_normalize(
    cache_path: Path,
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
    logger.info(f"Loading data from {cache_path}")
    cached: dict[str, torch.Tensor] = torch.load(f=str(cache_path), weights_only=True)

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
) -> NormalizedData:
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

    normalized_data = load_and_normalize(cache_path)
    input_dim = normalized_data.train_X.shape[-1]

    train_loader = DataLoader(
        CachedSC2Dataset(
            features=normalized_data.train_X, labels=normalized_data.train_y
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        CachedSC2Dataset(features=normalized_data.val_X, labels=normalized_data.val_y),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, input_dim


# ------------------------------------------------------------------
# Latent extraction
# ------------------------------------------------------------------
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
