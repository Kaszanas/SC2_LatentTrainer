"""
Hyperparameter and architecture search-space definitions.

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
    """
    Ask Optuna for a strictly decreasing hidden-layer sequence.

    Samples ``{prefix}_n_layers`` and ``{prefix}_start_width`` so logs show real
    layer sizes. The start index is clamped so ``n_layers`` strictly decreasing
    widths are always available; no post-hoc sorting is needed.

    Parameters
    ----------
    trial : optuna.Trial
        Active Optuna trial.
    prefix : str
        Parameter name prefix, e.g. ``"enc"``.
    width_choices : list[int]
        Allowed widths in ascending order.
    min_layers : int
        Minimum number of hidden layers, by default 1.
    max_layers : int
        Maximum number of hidden layers, by default 4.

    Returns
    -------
    list[int]
        Strictly decreasing widths, e.g. ``[256, 128, 64]``.
    """
    n_layers = trial.suggest_int(f"{prefix}_n_layers", low=min_layers, high=max_layers)
    # start_width is sampled from a static categorical distribution.
    # We clamp the derived index to guarantee n_layers strictly decreasing widths.
    start_width = trial.suggest_categorical(f"{prefix}_start_width", width_choices)
    start_idx = width_choices.index(start_width)
    actual_start = max(n_layers - 1, start_idx)
    return [width_choices[actual_start - i] for i in range(n_layers)]


def get_guided_vae_search_space(trial: optuna.Trial) -> dict:
    """
    Sample one guided VAE configuration.

    Searched: ``enc_n_layers``/``enc_start_width`` (decreasing encoder),
    ``nz_fraction`` (latent dim as a fraction of the last encoder width, at
    least 8), ``batch_size`` {64, 128}, ``cls`` (classification weight,
    log 1-50), ``lr``/``lr_c`` (log 1e-5-1e-3), ``weight_decay``/
    ``weight_decay_c`` (log 1e-6-1e-3) and ``supervised_dim`` {2, 4, 8}.

    Parameters
    ----------
    trial : optuna.Trial
        Active Optuna trial.

    Returns
    -------
    dict
        Keyword arguments for :class:`LitGuidedVAE` plus ``batch_size`` and the
        trial number.
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
    """
    Reconstruct a decreasing hidden-dims list from flat Optuna trial params.

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
    """
    Derive the latent dim from ``nz_fraction`` and the last encoder width.

    Parameters
    ----------
    params : dict
        Flat Optuna parameters containing ``nz_fraction``.
    encoder_hidden_dims : list[int]
        Reconstructed encoder widths.

    Returns
    -------
    int
        ``max(8, int(encoder_hidden_dims[-1] * nz_fraction))``.
    """
    return max(8, int(encoder_hidden_dims[-1] * params["nz_fraction"]))


# Event-stream guided VAE:
EVENT_MODEL_KEYS: tuple[str, ...] = (
    "d_model",
    "n_heads",
    "n_layers",
    "ffn_mult",
    "dropout",
    "latent_dim",
    "supervised_dim",
    "decoder_width",
    "learning_rate",
    "weight_decay",
    "learning_rate_cls",
    "weight_decay_cls",
    "classification_weight",
    "kl_beta",
)
EVENT_DATA_KEYS: tuple[str, ...] = ("batch_size", "min_crop_s", "max_events_per_slice")


def get_event_guided_vae_search_space(trial: optuna.Trial) -> dict:
    """
    Sample one event-stream guided VAE configuration.

    ``supervised_dim`` is sampled as a fraction of ``latent_dim`` so it always
    leaves free dims; every width in ``d_model`` is divisible by every head
    count, so all combinations are valid.

    Parameters
    ----------
    trial : optuna.Trial
        Active Optuna trial (a ``FixedTrial`` rebuilds a stored config).

    Returns
    -------
    dict
        The keys of :data:`EVENT_MODEL_KEYS` and :data:`EVENT_DATA_KEYS`.
    """
    latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32, 64])
    supervised_fraction = trial.suggest_float("supervised_fraction", 0.125, 0.5)
    supervised_dim = min(
        latent_dim - 1, max(1, round(latent_dim * supervised_fraction))
    )
    return {
        "d_model": trial.suggest_categorical("d_model", [64, 128, 192, 256]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        "n_layers": trial.suggest_categorical("n_layers", [2, 3, 4, 6]),
        "ffn_mult": trial.suggest_categorical("ffn_mult", [2, 4]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "latent_dim": latent_dim,
        "supervised_dim": supervised_dim,
        "decoder_width": trial.suggest_categorical("decoder_width", [128, 256]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "learning_rate_cls": trial.suggest_float(
            "learning_rate_cls", 1e-5, 1e-3, log=True
        ),
        "weight_decay_cls": trial.suggest_float(
            "weight_decay_cls", 1e-6, 1e-3, log=True
        ),
        "classification_weight": trial.suggest_float(
            "classification_weight", 1.0, 50.0, log=True
        ),
        "kl_beta": trial.suggest_float("kl_beta", 1e-3, 1.0, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "min_crop_s": trial.suggest_categorical("min_crop_s", [120.0, 180.0, 300.0]),
        # Not searched: 128 drops 0.018% of events (busiest slice: 263), while
        # keeping all of them made large configs 3.4x slower per epoch.
        "max_events_per_slice": 128,
    }


def event_params_from_flat(flat: dict) -> dict:
    """
    Rebuild a full event-model config from stored raw Optuna parameters.

    Parameters
    ----------
    flat : dict
        Raw trial parameters (``trial.params``), e.g. read from Optuna storage.

    Returns
    -------
    dict
        The config :func:`get_event_guided_vae_search_space` produced for them.
    """
    return get_event_guided_vae_search_space(optuna.trial.FixedTrial(flat))
