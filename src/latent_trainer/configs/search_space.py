"""Hyperparameter and architecture search-space definitions.

Each ``get_*_search_space`` function receives an :class:`optuna.Trial` and
returns a flat dictionary of suggested parameters.  Architecture choices
(number of layers and per-layer width) are individual Optuna parameters
rather than categorical presets, giving the optimiser full freedom.
"""

from __future__ import annotations

import optuna


def build_hidden_layers(
    trial: optuna.Trial,
    prefix: str,
    min_layers: int = 1,
    max_layers: int = 4,
    min_width: int = 32,
    max_width: int = 512,
    width_step: int = 32,
) -> list[int]:
    """Ask Optuna for a flexible architecture: *n* layers with individual widths.

    Parameters
    ----------
    trial:
        Active Optuna trial.
    prefix:
        Namespace prefix so VAE and classifier dims don't collide
        (e.g. ``"vae"`` → ``"vae_n_layers"``, ``"vae_width_0"``).
    min_layers / max_layers:
        Bounds on the number of hidden layers.
    min_width / max_width / width_step:
        Bounds and step for each layer's width.

    Returns
    -------
    list[int]
        Ordered hidden-layer widths, e.g. ``[256, 128, 64]``.
    """
    n_layers = trial.suggest_int(f"{prefix}_n_layers", min_layers, max_layers)
    dims: list[int] = []
    for i in range(n_layers):
        w = trial.suggest_int(
            f"{prefix}_width_{i}", min_width, max_width, step=width_step
        )
        dims.append(w)
    return dims


def get_two_stage_search_space(trial: optuna.Trial) -> dict:
    """Return a params dict for the two-stage (VAE → Classifier) pipeline.

    Searched parameters:
    - ``latent_dim``: 8-64
    - ``vae_lr`` / ``cls_lr``: log-uniform 1e-4 … 1e-2
    - ``batch_size``: categorical {64, 128, 256, 512}
    - ``dropout``: uniform 0.1-0.5
    - ``vae_hidden_dims``: 1-3 layers, 64-512 wide
    - ``cls_hidden_dims``: 1-3 layers, 32-256 wide
    """
    return {
        "latent_dim": trial.suggest_int("latent_dim", 8, 64),
        "vae_lr": trial.suggest_float("vae_lr", 1e-4, 1e-2, log=True),
        "cls_lr": trial.suggest_float("cls_lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "vae_hidden_dims": build_hidden_layers(
            trial,
            "vae",
            min_layers=1,
            max_layers=3,
            min_width=64,
            max_width=512,
            width_step=64,
        ),
        "cls_hidden_dims": build_hidden_layers(
            trial,
            "cls",
            min_layers=1,
            max_layers=3,
            min_width=32,
            max_width=256,
            width_step=32,
        ),
    }


def get_guided_vae_search_space(trial: optuna.Trial) -> dict:
    """Return a params dict for the supervised Guided-VAE pipeline.

    Searched parameters:
    - ``nz`` (latent dim): 8-64
    - ``batch_size``: categorical {64, 128, 256, 512}
    - ``cls`` (classification weight): log-uniform 1-50
    - ``lr`` / ``lr_c``: log-uniform 1e-5 ... 1e-3
    - ``weight_decay`` / ``weight_decay_c``: log-uniform 1e-6 ... 1e-3
    """
    return {
        "nz": trial.suggest_int("nz", 8, 64),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "cls": trial.suggest_float("cls", 1.0, 50.0, log=True),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "lr_c": trial.suggest_float("lr_c", 1e-5, 1e-3, log=True),
        "weight_decay_c": trial.suggest_float("weight_decay_c", 1e-6, 1e-3, log=True),
        "encoder_hidden_dims": build_hidden_layers(
            trial,
            "enc",
            min_layers=1,
            max_layers=4,
            min_width=32,
            max_width=512,
            width_step=32,
        ),
    }


def reconstruct_hidden_dims(params: dict, prefix: str) -> list[int]:
    """Reconstruct a hidden-dims list from flat Optuna trial params.

    :func:`build_hidden_layers` stores architecture as individual Optuna
    parameters (``{prefix}_n_layers``, ``{prefix}_width_0``, …).  After a
    sweep this helper reads those flat params back into the ``list[int]``
    expected by ``LitVAE`` and ``LatentClassifier``.

    Parameters
    ----------
    params:
        Flat ``trial.params`` dict from a completed Optuna trial.
    prefix:
        The same prefix used when the search space was built (e.g. ``"vae"``
        or ``"cls"``).

    Returns
    -------
    list[int]
        Ordered hidden-layer widths, e.g. ``[256, 128]``.
    """
    n_layers: int = params[f"{prefix}_n_layers"]
    return [params[f"{prefix}_width_{i}"] for i in range(n_layers)]
