"""Hyperparameter and architecture search-space definitions.

Each ``get_*_search_space`` function receives an :class:`optuna.Trial` and
returns a flat dictionary of suggested parameters.  Architecture choices
(number of layers and per-layer width) are individual Optuna parameters
rather than categorical presets, giving the optimiser full freedom.
"""

from __future__ import annotations

import optuna

from latent_trainer.configs.hyperparam_settings import (
    VAE_HIDDEN_DIM_CHOICES,
)


def build_hidden_layers(
    trial: optuna.Trial,
    prefix: str,
    width_choices: list[int],
    min_layers: int = 1,
    max_layers: int = 4,
) -> list[int]:
    """Ask Optuna for a decreasing hidden-layer sequence from *width_choices*.

    Uses two parameters per prefix (``{prefix}_n_layers``, ``{prefix}_start_width``)
    so MLflow/Optuna logs show real layer sizes instead of list indices.

    Ordering remains encoded in the parameterisation itself (no post-hoc sorting):
    the selected start width is converted to an index and then clamped to ensure
    there are enough widths available to build ``n_layers`` strictly decreasing dims.

    Parameters
    ----------
    trial:
        Active Optuna trial.
    prefix:
        Namespace prefix so VAE and classifier dims don't collide
        (e.g. ``"enc"`` → ``"enc_n_layers"``, ``"enc_start_width"``).
    width_choices:
        Allowed layer widths in **ascending** order (e.g. ``[32, 64, 128, 256, 512]``).
    min_layers / max_layers:
        Bounds on the number of hidden layers.

    Returns
    -------
    list[int]
        Strictly decreasing hidden-layer widths, e.g. ``[256, 128, 64]``.
    """
    n_layers = trial.suggest_int(f"{prefix}_n_layers", low=min_layers, high=max_layers)
    # start_width is sampled from a static categorical distribution.
    # We clamp the derived index to guarantee n_layers strictly decreasing widths.
    start_width = trial.suggest_categorical(f"{prefix}_start_width", width_choices)
    start_idx = width_choices.index(start_width)
    actual_start = max(n_layers - 1, start_idx)
    return [width_choices[actual_start - i] for i in range(n_layers)]


def get_guided_vae_search_space(trial: optuna.Trial) -> dict:
    """Return a params dict for the supervised Guided-VAE pipeline.

    Encoder dims are always decreasing.  Latent dim (``nz``) is derived as a
    fraction of the last (smallest) encoder layer so it never exceeds the
    bottleneck width.

    Searched parameters:
    - ``enc_n_layers``, ``enc_start_width``: encoder architecture
    - ``nz_fraction``: fraction of last encoder layer used as latent dim
    - ``batch_size``: categorical {64, 128}
    - ``cls`` (classification weight): log-uniform 1-50
    - ``lr`` / ``lr_c``: log-uniform 1e-5 ... 1e-3
    - ``weight_decay`` / ``weight_decay_c``: log-uniform 1e-6 ... 1e-3
    - ``supervised_dim``: categorical {2, 4, 8}
    """
    encoder_hidden_dims = build_hidden_layers(
        trial,
        "enc",
        width_choices=VAE_HIDDEN_DIM_CHOICES,
        min_layers=2,
        max_layers=4,
    )

    # Latent dimension can be at most the width of the last encoder layer.
    # But it can also be smaller as in divided by 2 or 4,
    # but not smaller than 8 to avoid excessive bottlenecks.
    latent_dim_fraction = trial.suggest_categorical(
        "nz_fraction", choices=[0.25, 0.5, 1.0]
    )
    latent_dim = max(8, int(encoder_hidden_dims[-1] * latent_dim_fraction))

    # supervised_dim is sampled directly.  All choices are < 8 (the minimum
    # latent_dim), so the constraint supervised_dim < latent_dim is guaranteed.
    supervised_dim = trial.suggest_categorical("supervised_dim", choices=[1, 2, 4])

    return {
        "_trial_number": trial.number,
        "latent_dim": latent_dim,
        "supervised_dim": supervised_dim,
        "encoder_hidden_dims": encoder_hidden_dims,
        "batch_size": trial.suggest_categorical("batch_size", choices=[64]),
        "classification_weight": trial.suggest_float(
            "classification_weight", low=1.0, high=250.0, log=True
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", low=1e-5, high=1e-3, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", low=1e-6, high=1e-3, log=True
        ),
        "learning_rate_cls": trial.suggest_float(
            "learning_rate_cls", low=1e-5, high=1e-3, log=True
        ),
        "weight_decay_cls": trial.suggest_float(
            "weight_decay_cls", low=1e-6, high=1e-3, log=True
        ),
    }


def reconstruct_hidden_dims(
    params: dict,
    prefix: str,
    width_choices: list[int],
) -> list[int]:
    """Reconstruct a decreasing hidden-dims list from flat Optuna trial params.

    Mirrors the logic in :func:`build_hidden_layers` exactly.

    Parameters
    ----------
    params:
        Flat ``trial.params`` dict from a completed Optuna trial.
    prefix:
        The same prefix used when the search space was built (e.g. ``"enc"``).
    width_choices:
        The same ``width_choices`` list passed to :func:`build_hidden_layers`
        when the search space was built (must be ascending).

    Returns
    -------
    list[int]
        Decreasing hidden-layer widths, e.g. ``[256, 128]``.
    """
    n_layers: int = params[f"{prefix}_n_layers"]

    # Backward-compatible: support both old *_start_idx and new *_start_width.
    if f"{prefix}_start_width" in params:
        start_idx = width_choices.index(params[f"{prefix}_start_width"])
    else:
        start_idx = params[f"{prefix}_start_idx"]

    actual_start = max(n_layers - 1, start_idx)
    return [width_choices[actual_start - i] for i in range(n_layers)]


def reconstruct_guided_vae_nz(params: dict, encoder_hidden_dims: list[int]) -> int:
    """Derive ``nz`` from ``nz_fraction`` and the last encoder layer width."""
    return max(8, int(encoder_hidden_dims[-1] * params["nz_fraction"]))
