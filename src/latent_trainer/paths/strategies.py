"""Latent-space path generation strategies.

Four strategies for finding counterfactual improvement paths through a
model's latent space.  All strategies return numpy arrays of shape
``(n_waypoints, latent_dim)``.

The strategies are model-agnostic: they accept **callable** score/logit
functions rather than concrete model classes.  This lets the same code
work with both the two-stage pipeline (``LitVAE`` + ``LitClassifier``,
opponent-aware) and the single-model pipeline
(``VAEClassifierLightning``).

Example — two-stage (opponent-aware)::

    from functools import partial

    def _score(z: torch.Tensor, *, classifier, opponent_z, player_idx) -> torch.Tensor:
        opp = opponent_z.expand(z.shape[0], -1)
        if player_idx == 0:
            inp = torch.cat([z, opp], dim=1)
        else:
            inp = torch.cat([opp, z], dim=1)
        return classifier(inp)

    score_fn = partial(_score, classifier=clf, opponent_z=opp_z, player_idx=0)
    path = path_gradient_ascent(z_start, score_fn=score_fn, Z_all=Z_all)

Example — single model (VAEClassifierLightning)::

    path = path_gradient_ascent(
        z_start,
        score_fn=model.p_win,
        logit_fn=model.logit_win,
        Z_all=Z_all,
    )
"""

from __future__ import annotations

import enum
from typing import Callable

import numpy as np
import scipy.sparse.csgraph as csgraph
import torch
from sklearn.neighbors import KernelDensity, kneighbors_graph

# Lazy import — POT is only needed for OT strategy
_ot = None


def _get_ot():
    global _ot
    if _ot is None:
        import ot

        _ot = ot
    return _ot


class PathStrategy(enum.Enum):
    """Available path-finding strategies."""

    LINEAR = "linear"
    OPTIMAL_TRANSPORT = "optimal_transport"
    GRADIENT_ASCENT = "gradient_ascent"
    GEODESIC = "geodesic"


# ──────────────────────────────────────────────────────────────
# Strategy: linear
# ──────────────────────────────────────────────────────────────


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
    ot = _get_ot()
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


# ──────────────────────────────────────────────────────────────
# Strategy: gradient ascent (+ KDE density regularisation)
# ──────────────────────────────────────────────────────────────


def path_gradient_ascent(
    z_start: np.ndarray,
    *,
    score_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    logit_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    Z_all: np.ndarray,
    steps: int = 500,
    lr: float = 0.02,
    momentum: float = 0.9,
    density_weight: float = 0.3,
    kde_bandwidth: float = 0.5,
    n_waypoints: int = 10,
    convergence_threshold: float = 0.95,
    device: torch.device | None = None,
) -> np.ndarray:
    """Gradient ascent on P(win) regularised by KDE density.

    At each step the total gradient is::

        total_grad = grad_logit + density_weight * grad_kde

    The logit gradient comes from backpropagation through *logit_fn*.
    The KDE density gradient is computed via numerical central
    differences, keeping the path on the data manifold.

    You must provide at least one of *score_fn* or *logit_fn*:

    - *logit_fn* — pre-sigmoid logit; preferred because it gives
      stronger gradients far from the decision boundary.
    - *score_fn* — P(win) after sigmoid; used as fallback and for
      convergence checking.

    Parameters
    ----------
    z_start:
        Starting latent code, shape ``(latent_dim,)``.
    score_fn:
        ``z (N, D) → P(win) (N, 1)`` or ``(N,)``.
    logit_fn:
        ``z (N, D) → logit (N, 1)`` or ``(N,)``.  If not provided,
        *score_fn* is used for backprop instead.
    Z_all:
        All training-set latent codes, shape ``(N, latent_dim)``.
    steps:
        Maximum gradient ascent iterations.
    lr:
        Step size.
    momentum:
        Momentum coefficient for velocity-based updates.
    density_weight:
        Weight of the KDE density gradient relative to the logit gradient.
    kde_bandwidth:
        Bandwidth for the Gaussian KDE.
    n_waypoints:
        Number of evenly-spaced waypoints to resample from the trajectory.
    convergence_threshold:
        Stop early when P(win) exceeds this value.
    device:
        Torch device.  Defaults to CPU.

    Returns
    -------
    np.ndarray
        Path of shape ``(n_waypoints, latent_dim)``.
    """
    if score_fn is None and logit_fn is None:
        raise ValueError("Provide at least one of score_fn or logit_fn.")

    if device is None:
        device = torch.device("cpu")

    # Decide which function to backprop through
    backprop_fn = logit_fn if logit_fn is not None else score_fn

    kde = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth).fit(Z_all)
    z = torch.tensor(
        z_start.copy(), dtype=torch.float32, requires_grad=True, device=device
    )
    velocity = torch.zeros_like(z)
    trajectory = [z_start.copy()]

    for step in range(steps):
        if z.grad is not None:
            z.grad.zero_()

        out = backprop_fn(z.unsqueeze(0))
        out.backward()

        with torch.no_grad():
            grad_cls = z.grad.clone()

            # KDE density gradient via numerical central differences
            z_np = z.detach().cpu().numpy()
            grads_kde = np.zeros_like(z_np)
            eps = 1e-3
            for i in range(len(z_np)):
                zp, zm = z_np.copy(), z_np.copy()
                zp[i] += eps
                zm[i] -= eps
                grads_kde[i] = float(
                    kde.score_samples(zp.reshape(1, -1))[0]
                    - kde.score_samples(zm.reshape(1, -1))[0]
                ) / (2 * eps)

            total_grad = grad_cls + density_weight * torch.tensor(
                grads_kde, dtype=torch.float32, device=device
            )
            velocity = momentum * velocity + lr * total_grad
            z = z.detach() + velocity
            z.requires_grad_(True)
            trajectory.append(z.detach().cpu().numpy().copy())

            # Convergence check
            if score_fn is not None:
                current_p = score_fn(z.unsqueeze(0)).detach().squeeze().item()
            else:
                current_p = (
                    torch.sigmoid(logit_fn(z.unsqueeze(0))).detach().squeeze().item()
                )

            if step % 50 == 0:
                print(
                    f"    step {step:4d}  P(win)={current_p:.4f}"
                    f"  |grad|={grad_cls.norm().item():.5f}"
                )
            if current_p > convergence_threshold:
                print(f"  [GA]  Converged at step {step}  P(win)={current_p:.4f}")
                break

    trajectory = np.array(trajectory)
    indices = np.linspace(0, len(trajectory) - 1, n_waypoints, dtype=int)
    return trajectory[indices]


# ──────────────────────────────────────────────────────────────
# Strategy: geodesic (shortest path on kNN graph)
# ──────────────────────────────────────────────────────────────


def path_geodesic(
    z_start: np.ndarray,
    Z_win: np.ndarray,
    Z_all: np.ndarray,
    *,
    k: int = 12,
    n_waypoints: int = 10,
) -> np.ndarray:
    """Shortest path along a k-NN graph to the densest winning point.

    Builds a k-nearest-neighbours graph over all training latent codes
    (plus the start and target), then computes the shortest path using
    Dijkstra's algorithm.

    Parameters
    ----------
    z_start:
        Starting latent code, shape ``(latent_dim,)``.
    Z_win:
        Winning-class latent codes, shape ``(N_win, latent_dim)``.
    Z_all:
        All training latent codes, shape ``(N, latent_dim)``.
    k:
        Number of neighbours for the graph.
    n_waypoints:
        Number of evenly-spaced waypoints to resample from the path.

    Returns
    -------
    np.ndarray
        Path of shape ``(n_waypoints, latent_dim)``.
    """
    kde_win = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(Z_win)
    best_win = Z_win[np.argmax(kde_win.score_samples(Z_win))]
    Z_graph = np.vstack([Z_all, z_start.reshape(1, -1), best_win.reshape(1, -1)])
    src_idx, tgt_idx = len(Z_graph) - 2, len(Z_graph) - 1

    A = kneighbors_graph(Z_graph, n_neighbors=k, mode="distance", include_self=False)
    A = (A + A.T) / 2
    dist_matrix, predecessors = csgraph.shortest_path(
        A, method="D", directed=False, indices=src_idx, return_predecessors=True
    )

    path_indices, node = [], tgt_idx
    while node != src_idx and node >= 0:
        path_indices.append(node)
        node = predecessors[node]
    path_indices.append(src_idx)
    path_indices = path_indices[::-1]

    if len(path_indices) < 2:
        print("  [GEO] Warning: no path found — straight line fallback.")
        return path_linear(z_start, best_win, n_waypoints=n_waypoints)

    full_path = Z_graph[path_indices]
    indices = np.linspace(0, len(full_path) - 1, n_waypoints, dtype=int)
    return full_path[indices]
