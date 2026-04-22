from dataclasses import dataclass
from typing import Protocol

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset


@dataclass
class NormalizedDataloaders:
    train_loader: DataLoader
    val_loader: DataLoader

    input_dim: int      # total scalar leaf count per player — used by model constructors
    mean: TensorDict    # per-key scalar, batch_size=[]
    std: TensorDict     # per-key scalar, batch_size=[]

    test_loader: DataLoader | None = None


@dataclass
class NormalizedData:
    """Return type of :func:`normalize`."""

    train_X: TensorDict
    val_X: TensorDict
    test_X: TensorDict

    mean: TensorDict    # per-key scalar, batch_size=[] — fit on train
    std: TensorDict     # per-key scalar, batch_size=[] — fit on train


@dataclass
class NormalizedDataWithLabels:
    train_X: TensorDict
    train_y: torch.Tensor   # long, values 0/1 — cast to float at the loss call site
    val_X: TensorDict
    val_y: torch.Tensor     # long
    test_X: TensorDict
    test_y: torch.Tensor    # long

    mean: TensorDict
    std: TensorDict


# Protocol for any VAE-like encoder (LitVAE, SimpleVAE, suGuidedVAE…)
class Encoder(Protocol):
    """Structural type for any model that exposes an ``encode`` method.

    The ``encode`` method must return a tuple ``(mu, logvar)`` where:

    - **mu** — latent mean, shape ``[batch, latent_dim]``
    - **logvar** — latent log-variance, shape ``[batch, latent_dim]``
    """

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


# Dataset wrapper (used by train_model.py / GuidedVAE path)
class CachedSC2Dataset(Dataset):
    """Wraps pre-processed feature and label tensors as a PyTorch Dataset."""

    def __init__(self, features: TensorDict, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[TensorDict, torch.Tensor]:
        return self.features[idx], self.labels[idx]


@dataclass
class CachedDatasetFileSpec:
    """In-memory container for a loaded split dataset.

    On-disk format (memmap TensorDict saved by preprocess_dataset.py):

    .. code-block:: text

        <cache_dir>/
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

    train: TensorDict   # batch_size=[N_train], keys: features, labels
    val: TensorDict     # batch_size=[N_val]
    test: TensorDict    # batch_size=[N_test]
    transform: str
    n_features: int     # total scalar leaf count per player (202 for rich transform)
