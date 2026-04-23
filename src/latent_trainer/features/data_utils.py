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
from tensordict import TensorDict
from torch.utils.data import DataLoader

from latent_trainer.features.type import (
    CachedSC2Dataset,
    Encoder,
    NormalizedData,
    NormalizedDataloaders,
    NormalizedDataWithLabels,
)

logger = logging.getLogger(__name__)


def _collate_sc2(
    batch: list[tuple[TensorDict, torch.Tensor]],
) -> tuple[TensorDict, torch.Tensor]:
    """Stack a list of (TensorDict, label) pairs into a batched tuple.

    PyTorch's default collate cannot handle TensorDicts correctly — it treats
    them as plain dicts and fails on nested indexing.  This collate uses
    ``torch.stack`` which TensorDict natively supports.
    """
    features, labels = zip(*batch)
    return torch.stack(list(features)), torch.stack(list(labels))


def normalize(
    train_X: TensorDict,
    val_X: TensorDict,
    test_X: TensorDict,
) -> NormalizedData:
    """Per-key z-score normalisation.  Fit on *train*, apply to val and test.

    Parameters
    ----------
    train_X:
        Training features, TensorDict with ``batch_size=[N_train, 2]``.
    val_X:
        Validation features with the same nested key structure.
    test_X:
        Test features with the same nested key structure.

    Returns
    -------
    NormalizedData
        Normalised TensorDicts and per-key normalisation parameters.
    """
    train_f = train_X.apply(lambda t: t.float())
    mean = train_f.apply(lambda t: t.mean(), batch_size=[])
    std = train_f.apply(lambda t: t.std() + 1e-8, batch_size=[])

    train_norm = (train_f - mean) / std
    val_norm = (val_X.apply(lambda t: t.float()) - mean) / std
    test_norm = (test_X.apply(lambda t: t.float()) - mean) / std

    return NormalizedData(
        train_X=train_norm,
        val_X=val_norm,
        test_X=test_norm,
        mean=mean,
        std=std,
    )


# Dataset loading
def load_and_normalize(
    cached_dataset_filepath: Path,
) -> NormalizedDataWithLabels:
    """
    Load a cached memmap TensorDict dataset and normalise features.

    The cache directory is expected to follow the layout produced by
    ``preprocess_dataset.py``::

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

    Parameters
    ----------
    cached_dataset_filepath : Path
        Path to the cache directory produced by ``preprocess_dataset.py``.

    Returns
    -------
    NormalizedDataWithLabels
        Normalised feature TensorDicts and long label tensors.
    """
    logger.info(f"Loading data from {cached_dataset_filepath}")

    full_td = TensorDict.load_memmap(str(cached_dataset_filepath))

    train_features = full_td["train", "features"]
    train_labels = full_td["train", "labels"]
    val_features = full_td["val", "features"]
    val_labels = full_td["val", "labels"]
    test_features = full_td["test", "features"]
    test_labels = full_td["test", "labels"]

    normalized_data = normalize(
        train_X=train_features,
        val_X=val_features,
        test_X=test_features,
    )

    logger.info(
        f"Feature shape: {list(normalized_data.train_X.shape)}  |  "
        f"Train: {len(normalized_data.train_X)}  Val: {len(normalized_data.val_X)}"
    )

    return NormalizedDataWithLabels(
        train_X=normalized_data.train_X,
        train_y=train_labels,   # long — cast to float at the loss call site if needed
        val_X=normalized_data.val_X,
        val_y=val_labels,
        test_X=normalized_data.test_X,
        test_y=test_labels,
        mean=normalized_data.mean,
        std=normalized_data.std,
    )


def load_cached_dataloaders(
    cache_path: Path,
    batch_size: int,
) -> NormalizedDataloaders:
    """
    Load cached dataset, normalise, and return ready-to-use DataLoaders.

    This is a convenience wrapper around :func:`load_and_normalize` for
    training loops that need DataLoaders rather than raw tensors.

    Parameters
    ----------
    cache_path : Path
        Path to the cached dataset directory produced by ``preprocess_dataset.py``.
    batch_size : int
        Batch size for the DataLoaders.

    Returns
    -------
    NormalizedDataloaders
        Contains the DataLoaders, ``input_dim``, and normalisation parameters.
    """

    normalized_data = load_and_normalize(cached_dataset_filepath=cache_path)

    # Total scalar leaf count per player — used by model constructors
    sample_player = normalized_data.train_X[0, 0]  # TensorDict batch_size=[]
    input_dim = sum(t.numel() for t in sample_player.values(True, True))

    train_loader = DataLoader(
        CachedSC2Dataset(
            features=normalized_data.train_X,
            labels=normalized_data.train_y,
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_sc2,
    )
    val_loader = DataLoader(
        CachedSC2Dataset(
            features=normalized_data.val_X,
            labels=normalized_data.val_y,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_sc2,
    )

    return NormalizedDataloaders(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        mean=normalized_data.mean,
        std=normalized_data.std,
    )


# Latent extraction
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
        NOTE: expects flat tensors [N, 2, F]. Needs updating when encoders accept TensorDict.
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
