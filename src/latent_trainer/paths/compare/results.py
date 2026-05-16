"""Dataclasses for multi-method path-charting comparison results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class MethodSpec:
    """Specification for a single path-charting method variant."""

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
    """All metrics captured for a single (sample, method) run."""

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

    # ── New metrics ────────────────────────────────────────────────────────────
    # Feature-space geometry (decoded to original scale before measuring)
    path_length_feature: float = math.nan  # sum-of-L2-steps in decoded feature space
    n_features_changed: int = 0  # sparsity: features with |Δ| > 1σ of training data

    # On-manifold quality (path-averaged, not just endpoint)
    kde_density_path_mean: float = (
        math.nan
    )  # mean log-density of win_kde along all waypoints

    # Reconstruction cycle consistency: decode → encode → compare
    recon_cycle_error_mean: float = math.nan  # mean ‖z_path − encode(decode(z_path))‖
    recon_cycle_error_max: float = math.nan  # max  ‖z_path − encode(decode(z_path))‖


@dataclass
class ComparisonReport:
    """Full result of comparing all methods across N samples."""

    n_samples_requested: int
    n_samples_evaluated: int
    methods: tuple[MethodSpec, ...]
    results: tuple[PathRunResult, ...]
    seed: int
    model_path: Path
    dataset_path: Path
