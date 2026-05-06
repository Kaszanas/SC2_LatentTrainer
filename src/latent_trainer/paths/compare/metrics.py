"""Pure metric functions for path-charting comparison.

All functions are model-agnostic and have no torch/sklearn dependencies,
making them independently unit-testable.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np


def first_crossover(p_win: np.ndarray, threshold: float = 0.5) -> int | None:
    """Return the first waypoint index where P(win) >= threshold, or None."""
    indices = np.where(p_win >= threshold)[0]
    return int(indices[0]) if len(indices) > 0 else None


def normalised_crossover_alpha(wp: int | None, n_waypoints: int) -> float | None:
    """Convert a waypoint index to a normalised alpha in [0, 1], or None."""
    if wp is None or n_waypoints <= 1:
        return None
    return float(wp) / float(n_waypoints - 1)


def auc_trapezoidal(p_win: np.ndarray) -> float:
    """Area under P(win) curve using the trapezoidal rule over alphas in [0, 1]."""
    if len(p_win) < 2:
        return float(p_win[0]) if len(p_win) == 1 else math.nan
    alphas = np.linspace(0.0, 1.0, len(p_win))
    return float(np.trapezoid(p_win, alphas))


def monotonicity_fraction(p_win: np.ndarray) -> float:
    """Fraction of consecutive pairs where P(win) is non-decreasing."""
    if len(p_win) < 2:
        return 1.0
    diffs = np.diff(p_win)
    return float((diffs >= 0).mean())


def path_length(path_z: np.ndarray) -> float:
    """Sum of Euclidean distances between consecutive waypoints."""
    if len(path_z) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path_z, axis=0), axis=1).sum())


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity between two feature sets."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def jaccard_matrix(top_k_dict: Mapping[str, Iterable[str]]) -> np.ndarray:
    """Compute an N × N Jaccard similarity matrix from a {name: features} mapping."""
    names = list(top_k_dict.keys())
    n = len(names)
    mat = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = jaccard(top_k_dict[a], top_k_dict[b])
    return mat


def sparsity_n_changed(
    delta: np.ndarray,
    norm_std: np.ndarray,
    threshold: float = 1.0,
) -> int:
    """Number of features that change by more than ``threshold`` standard deviations.

    Sparsity (Wachter et al., 2017) measures how many features a counterfactual
    path must modify to achieve the outcome flip.  A path that changes 12/196
    features is far more actionable as player feedback than one that changes
    150/196.  The 1σ threshold counts only features that changed meaningfully,
    ignoring near-zero perturbations.

    Parameters
    ----------
    delta:
        ``(n_features,)`` raw start→end delta in original (denormalized) scale.
    norm_std:
        ``(n_features,)`` or ``(1, n_features)`` per-feature standard deviation
        used during z-score normalization.
    threshold:
        Minimum number of standard deviations for a feature to count as changed.
    """
    std = np.asarray(norm_std).squeeze()
    return int(np.sum(np.abs(delta) / (std + 1e-8) > threshold))
