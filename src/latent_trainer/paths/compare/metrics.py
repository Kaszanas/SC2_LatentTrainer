"""
Pure metric functions for path-charting comparison.

All functions are model-agnostic and have no torch/sklearn dependencies,
making them independently unit-testable.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np


def first_crossover(p_win: np.ndarray, threshold: float = 0.5) -> int | None:
    """
    Return the first waypoint index where P(win) reaches a threshold.

    Parameters
    ----------
    p_win : np.ndarray
        P(win) per waypoint.
    threshold : float
        Threshold, by default 0.5.

    Returns
    -------
    int | None
        First index with ``p_win >= threshold``, or None.
    """
    indices = np.where(p_win >= threshold)[0]
    return int(indices[0]) if len(indices) > 0 else None


def normalised_crossover_alpha(wp: int | None, n_waypoints: int) -> float | None:
    """
    Convert a waypoint index to a path position in ``[0, 1]``.

    Parameters
    ----------
    wp : int | None
        Waypoint index, or None.
    n_waypoints : int
        Number of waypoints.

    Returns
    -------
    float | None
        ``wp / (n_waypoints - 1)``, or None if ``wp`` is None or the path has
        fewer than two waypoints.
    """
    if wp is None or n_waypoints <= 1:
        return None
    return float(wp) / float(n_waypoints - 1)


def auc_trapezoidal(p_win: np.ndarray) -> float:
    """
    Return the area under the P(win) curve over ``alpha`` in ``[0, 1]``.

    Parameters
    ----------
    p_win : np.ndarray
        P(win) per waypoint.

    Returns
    -------
    float
        Trapezoidal area; the single value for one waypoint, NaN for none.
    """
    if len(p_win) < 2:
        return float(p_win[0]) if len(p_win) == 1 else math.nan
    alphas = np.linspace(0.0, 1.0, len(p_win))
    return float(np.trapezoid(p_win, alphas))


def monotonicity_fraction(p_win: np.ndarray) -> float:
    """
    Return the fraction of consecutive steps where P(win) does not decrease.

    Parameters
    ----------
    p_win : np.ndarray
        P(win) per waypoint.

    Returns
    -------
    float
        Fraction in ``[0, 1]``; 1.0 for fewer than two waypoints.
    """
    if len(p_win) < 2:
        return 1.0
    diffs = np.diff(p_win)
    return float((diffs >= 0).mean())


def path_length(path_z: np.ndarray) -> float:
    """
    Return the sum of Euclidean distances between consecutive waypoints.

    Parameters
    ----------
    path_z : np.ndarray
        Waypoints, ``[n_waypoints, D]``.

    Returns
    -------
    float
        Path length; 0.0 for fewer than two waypoints.
    """
    if len(path_z) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path_z, axis=0), axis=1).sum())


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """
    Return the Jaccard similarity of two feature sets.

    Parameters
    ----------
    a : Iterable[str]
        First feature set.
    b : Iterable[str]
        Second feature set.

    Returns
    -------
    float
        ``|a & b| / |a | b|``; 1.0 if both are empty.
    """
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def jaccard_matrix(top_k_dict: Mapping[str, Iterable[str]]) -> np.ndarray:
    """
    Compute an N x N Jaccard similarity matrix from named feature sets.

    Parameters
    ----------
    top_k_dict : Mapping[str, Iterable[str]]
        Feature set per name.

    Returns
    -------
    np.ndarray
        Similarity matrix in the mapping's key order.
    """
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
    """
    Count features that change by more than ``threshold`` standard deviations.

    Sparsity (Wachter et al., 2017) measures how many features a counterfactual
    must change; fewer is more actionable.

    Parameters
    ----------
    delta : np.ndarray
        Feature change along the path, original scale.
    norm_std : np.ndarray
        Training feature standard deviations.
    threshold : float
        Threshold in standard deviations, by default 1.0.

    Returns
    -------
    int
        Number of features with ``|delta| / std > threshold``.
    """
    std = np.asarray(norm_std).squeeze()
    return int(np.sum(np.abs(delta) / (std + 1e-8) > threshold))
