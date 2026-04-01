import numpy as np
import ot


# ──────────────────────────────────────────────────────────────
# Strategy: optimal transport
# ──────────────────────────────────────────────────────────────
def path_optimal_transport(
    z_start: np.ndarray,
    Z_win: np.ndarray,
    *,
    reg: float = 0.0,
    n_waypoints: int = 10,
) -> np.ndarray:
    """OT-barycentric target with linear interpolation.

    Computes the Wasserstein-barycentric target in the winning cloud
    using the POT library, then linearly interpolates toward it.

    Parameters
    ----------
    z_start:
        Starting latent code, shape ``(latent_dim,)``.
    Z_win:
        Winning-class latent codes, shape ``(N_win, latent_dim)``.
    reg:
        Entropic regularisation parameter.  0 = exact EMD.
    n_waypoints:
        Number of evenly-spaced points along the path.

    Returns
    -------
    np.ndarray
        Path of shape ``(n_waypoints, latent_dim)``.
    """

    a = np.array([1.0])
    M = ot.dist(z_start.reshape(1, -1), Z_win, metric="sqeuclidean")
    T = (
        ot.sinkhorn(a, ot.unif(len(Z_win)), M, reg=reg)
        if reg > 0
        else ot.emd(a, ot.unif(len(Z_win)), M)
    )
    weights = T[0]
    z_target = (weights[:, None] * Z_win).sum(0) / weights.sum()
    alphas = np.linspace(0, 1, n_waypoints)
    return np.array([(1 - a) * z_start + a * z_target for a in alphas])
