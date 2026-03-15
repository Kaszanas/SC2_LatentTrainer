from dataclasses import dataclass
from typing import Protocol

import torch
from torch.utils.data import Dataset


@dataclass
class NormalizedData:
    """Return type of :func:`load_and_normalize`."""

    train_X: torch.Tensor
    val_X: torch.Tensor
    test_X: torch.Tensor

    # Normalisation parameters (for test-time normalisation)
    mean: torch.Tensor
    std: torch.Tensor


@dataclass
class NormalizedDataWithLabels:
    train_X: torch.Tensor
    train_y: torch.Tensor
    val_X: torch.Tensor
    val_y: torch.Tensor
    test_X: torch.Tensor
    test_y: torch.Tensor

    # Normalisation parameters (for test-time normalisation)
    mean: torch.Tensor
    std: torch.Tensor


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


@dataclass
class CachedDatasetFileSpec:
    train_features: torch.Tensor
    train_labels: torch.Tensor
    val_features: torch.Tensor
    val_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor
    transform: str
