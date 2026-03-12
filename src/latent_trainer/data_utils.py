"""Shared data loading and normalisation utilities.

This module provides the **single canonical implementations** of dataset
loading, normalisation, and latent extraction.  All training scripts,
analysis tools, and the hyperparameter search system import from here
instead of maintaining their own copies.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Protocol

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Protocol for any VAE-like encoder (LitVAE, SimpleVAE, suGuidedVAE…)
# ------------------------------------------------------------------

class Encoder(Protocol):
    """Structural type for any model that exposes an ``encode`` method.

    The ``encode`` method must return a tuple ``(mu, logvar)`` where:

    - **mu** — latent mean, shape ``[batch, latent_dim]``
    - **logvar** — latent log-variance, shape ``[batch, latent_dim]``
    """

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class NormalizedData(NamedTuple):
    """Return type of :func:`load_and_normalize`."""

    train_X: torch.Tensor
    train_y: torch.Tensor
    val_X: torch.Tensor
    val_y: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


# ------------------------------------------------------------------
# Dataset wrapper (used by train_model.py / GuidedVAE path)
# ------------------------------------------------------------------

class CachedSC2Dataset(Dataset):
    """Wraps pre-processed feature and label tensors as a PyTorch Dataset."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# ------------------------------------------------------------------
# Normalisation
# ------------------------------------------------------------------

def normalize(
    train_X: torch.Tensor,
    val_X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-feature z-score normalisation.  Fit on *train*, apply to both.

    Parameters
    ----------
    train_X:
        Training features, shape ``[N, ...]`` where the last dim is features.
    val_X:
        Validation features with the same trailing feature dimension.

    Returns
    -------
    tuple
        ``(normed_train, normed_val, mean, std)`` where *mean* and *std*
        have shape ``[1, F]`` and can be re-used for test-time normalisation.
    """
    shape = train_X.shape
    train_flat = train_X.reshape(-1, shape[-1])
    val_flat = val_X.reshape(-1, shape[-1])

    mean = train_flat.mean(dim=0, keepdim=True)
    std = train_flat.std(dim=0, keepdim=True) + 1e-8

    train_flat = (train_flat - mean) / std
    val_flat = (val_flat - mean) / std

    return train_flat.reshape(shape), val_flat.reshape(val_X.shape), mean, std


# ------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------

def load_and_normalize(
    cache_path: str,
) -> NormalizedData:
    """Load a cached ``.pt`` dataset and normalise features.

    The cache is expected to contain at least::

        {"train_features": ..., "train_labels": ...,
         "val_features": ...,   "val_labels": ...}

    Parameters
    ----------
    cache_path:
        Path to the ``.pt`` file produced by ``preprocess_dataset.py``.

    Returns
    -------
    tuple
        ``(train_X, train_y, val_X, val_y, mean, std)``
    """
    logger.info("Loading data from %s …", cache_path)
    cached = torch.load(cache_path, weights_only=True)

    train_X = cached["train_features"].float()
    train_y = cached["train_labels"].float()
    val_X = cached["val_features"].float()
    val_y = cached["val_labels"].float()

    train_X, val_X, mean, std = normalize(train_X, val_X)

    logger.info(
        "  Feature shape: %s  |  Train: %d  Val: %d",
        list(train_X.shape),
        len(train_X),
        len(val_X),
    )
    return NormalizedData(train_X, train_y, val_X, val_y, mean, std)


def load_cached_dataloaders(
    cache_path: str,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, int]:
    """Load cached dataset, normalise, and return ready-to-use DataLoaders.

    This is a convenience wrapper around :func:`load_and_normalize` for
    training loops that need DataLoaders rather than raw tensors.

    Returns
    -------
    tuple
        ``(train_loader, val_loader, input_dim)``
    """
    train_X, train_y, val_X, val_y, _mean, _std = load_and_normalize(cache_path)
    input_dim = train_X.shape[-1]

    train_loader = DataLoader(
        CachedSC2Dataset(train_X, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        CachedSC2Dataset(val_X, val_y),
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
