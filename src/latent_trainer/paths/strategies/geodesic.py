# ──────────────────────────────────────────────────────────────
# Strategy: geodesic (shortest path on kNN graph)
# ──────────────────────────────────────────────────────────────
import numpy as np
from scipy.sparse import csgraph
from sklearn.neighbors import KernelDensity, kneighbors_graph

from latent_trainer.paths.strategies.linear import path_linear


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
        A,
        method="D",
        directed=False,
        indices=src_idx,
        return_predecessors=True,
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
