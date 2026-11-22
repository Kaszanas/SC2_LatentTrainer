"""
Precomputed win/loss latent pools for single-replay serving.

The path-charting strategies (centroid/k-NN/KDE/OT targets) need a
population of encoded winning and losing latents to compare a new sample
against. Computing that from the full bulk cache at serve time would mean
shipping the entire training dataset to end users. A ReferencePack
precomputes it once (offline, alongside a trained checkpoint) into a small
artifact -- just [N, latent_dim] tensors, not the raw feature data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from latent_trainer.paths.data import encode_player, load_model_and_data


@dataclass
class ReferencePack:
    """
    Pre-encoded test-split latents used as the reference population when serving.

    Parameters
    ----------
    latents_p0 : torch.Tensor
        Player 1 latents, ``[N, latent_dim]``.
    latents_p1 : torch.Tensor
        Player 2 latents, ``[N, latent_dim]``.
    labels_tensor : torch.Tensor
        Labels, ``[N]``; 1 = player 1 won.
    """

    latents_p0: torch.Tensor
    latents_p1: torch.Tensor
    labels_tensor: torch.Tensor


def build_reference_pack(model_path: Path, dataset_path: Path) -> ReferencePack:
    """
    Encode the cached dataset's test split once, for reuse at serving time.

    Mirrors the encoding step of ``paths.data.prepare_path_context`` but keeps
    only latents and labels.

    Parameters
    ----------
    model_path : Path
        Guided VAE checkpoint.
    dataset_path : Path
        Dense feature cache.

    Returns
    -------
    ReferencePack
        Latents and labels of the test split, on CPU.
    """
    guided_vae, X, y = load_model_and_data(
        model_path=model_path,
        cached_dataset_filepath=dataset_path,
    )
    latents_p0 = encode_player(vae=guided_vae.model, data=X[:, 0, :])
    latents_p1 = encode_player(vae=guided_vae.model, data=X[:, 1, :])
    return ReferencePack(
        latents_p0=latents_p0.detach().cpu(),
        latents_p1=latents_p1.detach().cpu(),
        labels_tensor=y.detach().cpu(),
    )


def save_reference_pack(pack: ReferencePack, path: Path) -> None:
    """
    Save a reference pack with ``torch.save``.

    Parameters
    ----------
    pack : ReferencePack
        Pack to save.
    path : Path
        Destination file.
    """
    torch.save(
        {
            "latents_p0": pack.latents_p0,
            "latents_p1": pack.latents_p1,
            "labels_tensor": pack.labels_tensor,
        },
        str(path),
    )


def load_reference_pack(path: Path) -> ReferencePack:
    """
    Load a reference pack saved by :func:`save_reference_pack`.

    Parameters
    ----------
    path : Path
        Pack file.

    Returns
    -------
    ReferencePack
        The loaded pack.
    """
    data: dict[str, torch.Tensor] = torch.load(str(path), weights_only=True)
    return ReferencePack(
        latents_p0=data["latents_p0"],
        latents_p1=data["latents_p1"],
        labels_tensor=data["labels_tensor"],
    )
