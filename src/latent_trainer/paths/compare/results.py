"""Dataclasses for multi-method path-charting comparison results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class MethodSpec:
    """
    Specification of one path-charting method variant.

    Parameters
    ----------
    name : str
        Method identifier, e.g. ``"linear_centroid"``.
    display_name : str
        Name shown in plots and tables.
    strategy : str
        Underlying strategy (``linear``, ``gradient_ascent``,
        ``optimal_transport`` or ``neural_flow``).
    params : Mapping[str, Any]
        Strategy keyword arguments.
    requires_flow_checkpoint : bool
        Whether a trained flow checkpoint is needed, by default False.
    """

    name: str
    display_name: str
    strategy: str
    params: Mapping[str, Any]
    requires_flow_checkpoint: bool = False


DEFAULT_METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec(
        name="linear_centroid",
        display_name="Linear (centroid)",
        strategy="linear",
        params={"method": "centroid", "k_opponents": 50},
    ),
    MethodSpec(
        name="linear_nearest",
        display_name="Linear (k-NN)",
        strategy="linear",
        params={"method": "nearest", "k_neighbours": 5, "k_opponents": 50},
    ),
    MethodSpec(
        name="optimal_transport",
        display_name="Optimal Transport",
        strategy="optimal_transport",
        params={"reg": 0.01, "step_size": 0.1, "k_opponents": 50},
    ),
    MethodSpec(
        name="neural_flow",
        display_name="Neural Flow",
        strategy="neural_flow",
        params={"guidance_scale": 0.0},
        requires_flow_checkpoint=True,
    ),
    MethodSpec(
        name="gradient_ascent",
        display_name="Gradient Ascent",
        strategy="gradient_ascent",
        params={
            "steps": 1000,
            "lr": 0.005,
            "momentum": 0.5,
            "density_weight": 0.3,
            "kde_bandwidth": 0.5,
            "convergence_threshold": 0.95,
        },
    ),
)


@dataclass
class PathRunResult:
    """
    All metrics of a single (sample, method) run.

    Parameters
    ----------
    sample_idx : int
        Analysed game index.
    method_name : str
        Method identifier.
    player_idx : int
        Index of the losing player analysed (0 or 1).
    path_z : np.ndarray
        Path, ``[n_waypoints, latent_dim]``.
    p_win_curve : np.ndarray
        P(win) per waypoint.
    alphas : np.ndarray
        Normalised path positions in ``[0, 1]``.
    crossover_alpha : float | None
        Position where P(win) first reaches 0.5, or None.
    crossover_wp : int | None
        Waypoint index of the crossover, or None.
    success : bool
        Whether a crossover happened.
    p_win_start : float
        P(win) at the start.
    p_win_end : float
        P(win) at the end.
    p_win_gain : float
        ``p_win_end - p_win_start``.
    p_win_max : float
        Maximum P(win) along the path.
    auc_p_win : float
        Trapezoidal area under the P(win) curve.
    monotonicity : float
        Fraction of non-decreasing consecutive steps.
    dist_to_nearest_win_start : float
        Distance from the start to the nearest winning latent.
    dist_to_nearest_win_end : float
        Distance from the end to the nearest winning latent.
    dist_to_knn_win_start : float
        Mean distance from the start to the 5 nearest winners.
    dist_to_knn_win_end : float
        Mean distance from the end to the 5 nearest winners.
    kde_density_start : float
        Winner-KDE density at the start.
    kde_density_end : float
        Winner-KDE density at the end.
    path_length_l2 : float
        Path length in latent space.
    top_k_features_raw : tuple[str, ...]
        Top-k features of the raw feedback signal.
    top_k_features_mv : tuple[str, ...]
        Top-k features of the minimum-viable signal.
    top_k_features_weighted : tuple[str, ...]
        Top-k features of the P(win)-gain-weighted signal.
    wall_time_s : float
        Wall-clock seconds of the run.
    error : str | None
        Error message if the run failed, by default None.
    path_length_feature : float
        Path length in decoded feature space (original scale), by default NaN.
    n_features_changed : int
        Features changed by more than 1 training SD, by default 0.
    kde_density_path_mean : float
        Mean winner-KDE log-density along the path, by default NaN.
    recon_cycle_error_mean : float
        Mean ``||z - encode(decode(z))||`` along the path, by default NaN.
    recon_cycle_error_max : float
        Maximum ``||z - encode(decode(z))||`` along the path, by default NaN.
    """

    # Identity
    sample_idx: int
    method_name: str
    player_idx: int

    # Raw path data
    path_z: np.ndarray  # (n_waypoints, latent_dim)
    p_win_curve: np.ndarray  # (n_waypoints,)
    alphas: np.ndarray  # (n_waypoints,) normalised [0, 1]

    # Crossover metrics
    crossover_alpha: float | None  # normalised step where P(win) >= 0.5 first
    crossover_wp: int | None
    success: bool  # crossover_alpha is not None

    # P(win) magnitude metrics
    p_win_start: float
    p_win_end: float
    p_win_gain: float
    p_win_max: float
    auc_p_win: float  # trapezoidal area under P(win) curve
    monotonicity: float  # fraction of non-decreasing consecutive pairs

    # Geometry — nearest-win distance (topology-agnostic; centroid omitted)
    dist_to_nearest_win_start: float
    dist_to_nearest_win_end: float
    dist_to_knn_win_start: float  # mean k=5 nearest win distances at start
    dist_to_knn_win_end: float  # mean k=5 nearest win distances at end
    kde_density_start: float  # KDE(win_latents) evaluated at z_start
    kde_density_end: float  # KDE(win_latents) evaluated at z_end
    path_length_l2: float

    # Feature attribution (for Jaccard comparison across methods)
    top_k_features_raw: tuple[str, ...]
    top_k_features_mv: tuple[str, ...]
    top_k_features_weighted: tuple[str, ...]

    # Cost
    wall_time_s: float

    # Failure mode (None on success; error message on exception)
    error: str | None = None

    # New metrics:
    # Feature-space geometry (decoded to original scale before measuring)
    path_length_feature: float = math.nan   # sum-of-L2-steps in decoded feature space
    n_features_changed: int = 0             # sparsity: features with |Δ| > 1σ of training data

    # On-manifold quality (path-averaged, not just endpoint)
    kde_density_path_mean: float = math.nan  # mean log-density of win_kde along all waypoints

    # Reconstruction cycle consistency: decode → encode → compare
    recon_cycle_error_mean: float = math.nan  # mean ‖z_path − encode(decode(z_path))‖
    recon_cycle_error_max: float = math.nan   # max  ‖z_path − encode(decode(z_path))‖


@dataclass
class ComparisonReport:
    """
    Full result of comparing all methods across samples.

    Parameters
    ----------
    n_samples_requested : int
        Samples requested (0 = all).
    n_samples_evaluated : int
        Samples actually evaluated.
    methods : tuple[MethodSpec, ...]
        Compared methods.
    results : tuple[PathRunResult, ...]
        One result per (sample, method).
    seed : int
        Seed of the sample selection.
    model_path : Path
        Guided VAE checkpoint used.
    dataset_path : Path
        Dataset cache used.
    """

    n_samples_requested: int
    n_samples_evaluated: int
    methods: tuple[MethodSpec, ...]
    results: tuple[PathRunResult, ...]
    seed: int
    model_path: Path
    dataset_path: Path
