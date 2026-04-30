import numpy as np
import ot

# def path_optimal_transport(
#     z_start: np.ndarray,
#     Z_win: np.ndarray,
#     *,
#     reg: float = 0.0,
#     n_waypoints: int = 10,
# ) -> np.ndarray:
#     """OT-barycentric target with linear interpolation.

#     Computes the Wasserstein-barycentric target in the winning cloud
#     using the POT library, then linearly interpolates toward it.

#     Parameters
#     ----------
#     z_start:
#         Starting latent code, shape ``(latent_dim,)``.
#     Z_win:
#         Winning-class latent codes, shape ``(N_win, latent_dim)``.
#     reg:
#         Entropic regularisation parameter.  0 = exact EMD.
#     n_waypoints:
#         Number of evenly-spaced points along the path.

#     Returns
#     -------
#     np.ndarray
#         Path of shape ``(n_waypoints, latent_dim)``.
#     """

#     a = np.array([1.0])
#     M = ot.dist(z_start.reshape(1, -1), Z_win, metric="sqeuclidean")
#     T = (
#         ot.sinkhorn(a, ot.unif(len(Z_win)), M, reg=reg)
#         if reg > 0
#         else ot.emd(a, ot.unif(len(Z_win)), M)
#     )
#     weights = T[0]
#     z_target = (weights[:, None] * Z_win).sum(0) / weights.sum()
#     alphas = np.linspace(0, 1, n_waypoints)

#     return np.array([(1 - a) * z_start + a * z_target for a in alphas])


# Flow:
# def path_optimal_transport(
#     z_start: np.ndarray,
#     Z_win: np.ndarray,
#     *,
#     reg: float = 0.05,  # Sinkhorn is generally more stable for flows
#     n_waypoints: int = 10,
# ) -> np.ndarray:
#     """
#     Computes a path following the Optimal Transport displacement interpolant.
#     """
#     # 1. Compute the transport plan (T)
#     # We treat z_start as a Dirac mass and Z_win as a uniform empirical distribution
#     a = np.array([1.0])
#     b = ot.unif(len(Z_win))
#     M = ot.dist(z_start.reshape(1, -1), Z_win, metric="sqeuclidean")

#     if reg > 0:
#         T = ot.sinkhorn(a, b, M, reg=reg)
#     else:
#         T = ot.emd(a, b, M)

#     # 2. Identify the Barycentric Mapping
#     # In OT, the optimal map T(x) for a point x is:
#     # T(z_start) = (1 / a) * sum_j (T_{start, j} * Z_win_j)
#     # This represents where the mass at z_start 'wants' to go.
#     # z_target = (T[0, :] @ Z_win) / np.sum(T[0, :])

#     # To move toward the "center" of the winning class:
#     # The weights from the OT plan T define how z_start 'relates' to Z_win.
#     weights = T[0] / T[0].sum()
#     z_target = weights @ Z_win

#     # 3. Create the Path
#     # If you want a "flow", you can use the displacement: velocity = (z_target - z_start)
#     alphas = np.linspace(0, 1, n_waypoints)
#     path = np.array([(1 - a) * z_start + a * z_target for a in alphas])

#     return path


# Iterative ot flow:
def path_optimal_transport(
    z_start: np.ndarray,
    Z_win: np.ndarray,
    *,
    reg: float = 0.05,
    n_waypoints: int = 20,
    step_size: float = 0.1,  # How far to move toward the target at each step
) -> np.ndarray:
    """
    Generates a path by iteratively recalculating the OT target.
    This mimics a vector field flow rather than a linear interpolation.
    """
    path = [z_start.copy()]
    z_current = z_start.copy()

    # We treat Z_win as a fixed target distribution (uniform)
    b = ot.unif(len(Z_win))
    a = np.array([1.0])

    for _ in range(n_waypoints - 1):
        # 1. Compute distance from current position to all target points
        M = ot.dist(z_current.reshape(1, -1), Z_win, metric="sqeuclidean")
        # Normalise so scale of latent space does not cause exp(-M/reg) underflow
        M = M / (M.max() + 1e-9)

        # 2. Compute the OT plan — log-domain Sinkhorn is numerically stable at any reg
        if reg > 0:
            T = ot.sinkhorn_log(a, b, M, reg=reg)
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
