# ──────────────────────────────────────────────────────────────
# Strategy: linear
# ──────────────────────────────────────────────────────────────
import numpy as np


def path_linear(
    z_start: np.ndarray,
    z_target: np.ndarray,
    *,
    n_waypoints: int = 20,
) -> np.ndarray:
    """Straight-line interpolation from *z_start* to *z_target*.

    Parameters
    ----------
    z_start:
        Starting latent code, shape ``(latent_dim,)``.
    z_target:
        Target latent code, shape ``(latent_dim,)``.
    n_waypoints:
        Number of evenly-spaced points along the path.

    Returns
    -------
    np.ndarray
        Path of shape ``(n_waypoints, latent_dim)``.
    """
    alphas = np.linspace(0.0, 1.0, n_waypoints)
    return np.array([(1.0 - a) * z_start + a * z_target for a in alphas])
