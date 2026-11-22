"""Iterative optimal-transport barycentric paths in latent space."""

import numpy as np
import ot


# Iterative ot flow:
def path_optimal_transport(
    z_start: np.ndarray,
    Z_win: np.ndarray,
    *,
    reg: float = 0.05,
    n_waypoints: int = 20,
    step_size: float = 0.1,
    opponent_z: np.ndarray | None = None,
    Z_loss: np.ndarray | None = None,
    k_opponents: int = 50,
) -> np.ndarray:
    """
    Move from ``z_start`` toward the winning distribution by repeated OT steps.

    Each step solves OT from the current point to the winners (log-domain
    Sinkhorn, or exact EMD for ``reg=0``) and moves ``step_size`` of the way to
    the barycentric target. With ``opponent_z`` and ``Z_loss``, the targets are
    restricted to the ``k_opponents`` winners whose beaten opponent is most
    similar to the current opponent, keeping the target weights uniform.

    Parameters
    ----------
    z_start : np.ndarray
        Starting latent, ``[D]``.
    Z_win : np.ndarray
        Winning latents, ``[N, D]``.
    reg : float
        Entropic regularisation; 0 uses exact EMD, by default 0.05.
    n_waypoints : int
        Points on the returned path, including the start, by default 20.
    step_size : float
        Fraction of the distance to the target moved per step, by default 0.1.
    opponent_z : np.ndarray | None
        Current opponent's latent, by default None.
    Z_loss : np.ndarray | None
        Latents of the opponents beaten by each row of ``Z_win``, by default None.
    k_opponents : int
        Winners kept when conditioning on the opponent, by default 50.

    Returns
    -------
    np.ndarray
        Path ``[n_waypoints, D]``.
    """
    path = [z_start.copy()]
    z_current = z_start.copy()

    if opponent_z is not None and Z_loss is not None:
        # Restrict the target set to the k_opponents winners whose beaten opponent
        # is most similar to the current opponent, then use a clean uniform b.
        # This avoids log(0) in sinkhorn_log that arises from zeros in a padded b.
        dists_opp = np.linalg.norm(Z_loss - opponent_z, axis=1)
        k = min(k_opponents, len(Z_win))
        top_k_idx = np.argpartition(dists_opp, k)[:k]
        Z_win = Z_win[top_k_idx]
    b = ot.unif(len(Z_win))
    a = np.array([1.0])

    for _ in range(n_waypoints - 1):
        # 1. Compute distance from current position to all target points
        M = ot.dist(z_current.reshape(1, -1), Z_win, metric="sqeuclidean")
        # Normalise so scale of latent space does not cause exp(-M/reg) underflow
        M = M / (M.max() + 1e-9)

        # 2. Compute the OT plan — log-domain Sinkhorn is numerically stable at any reg
        if reg > 0:
            T = ot.bregman.sinkhorn_log(a, b, M, reg=reg, numItermax=2000)
        else:
            T = ot.emd(a, b, M)

        # 3. Compute the local barycentric target
        weights = T[0] / (T[0].sum() + 1e-9)
        z_target = weights @ Z_win

        # 4. Take a small step toward the target (Euler step)
        # This is the "Flow": z_{t+1} = z_t + step_size * (z_target - z_t)
        z_current = z_current + step_size * (z_target - z_current)

        path.append(z_current.copy())

    return np.array(path)
