"""Aggregate PathRunResults into per-method statistics."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from latent_trainer.paths.compare.metrics import jaccard
from latent_trainer.paths.compare.results import ComparisonReport, PathRunResult


@dataclass
class MethodStats:
    method_name: str
    display_name: str
    n_runs: int
    n_success: int
    success_rate: float

    # Crossover (over successful runs only)
    crossover_alpha_mean: float
    crossover_alpha_sd: float

    # P(win) metrics
    p_win_gain_mean: float
    p_win_gain_sd: float
    p_win_max_mean: float
    p_win_max_sd: float
    auc_mean: float
    auc_sd: float
    monotonicity_mean: float
    monotonicity_sd: float

    # P(win) absolute values at path start and end
    p_win_start_mean: float
    p_win_start_sd: float
    p_win_end_mean: float
    p_win_end_sd: float

    # Geometry (nearest-win; centroid omitted as unreliable for non-convex distributions)
    dist_nearest_win_end_mean: float
    dist_nearest_win_end_sd: float
    kde_density_shift_mean: float  # mean(density_end - density_start)
    kde_density_shift_sd: float

    # Conditional success — only samples where P(win_start) < 0.5
    cond_n_eligible: int
    cond_n_success: int
    cond_success_rate: float

    # SD-adjusted success — threshold = 0.5 + std(P(win_start)) across all runs
    p_win_threshold: float  # 0.5 + σ used as the bar
    sd_adj_n_eligible: int
    sd_adj_n_success: int
    sd_adj_success_rate: float

    # Normalised gain: ΔP(win) / (1 − P(win_start))
    p_win_gain_norm_mean: float
    p_win_gain_norm_sd: float

    # Latent norm along path (off-manifold diagnostic; computed from path_z)
    mean_z_norm_mean: float
    mean_z_norm_sd: float
    max_z_norm_mean: float
    max_z_norm_sd: float

    # Cost
    wall_time_mean_s: float
    wall_time_sd_s: float

    # ── New metrics ────────────────────────────────────────────────────────────
    # Feature-space path length (decoded to original scale)
    path_length_feature_mean: float
    path_length_feature_sd: float

    # Sparsity: mean number of features changed by > 1σ
    n_features_changed_mean: float
    n_features_changed_sd: float

    # On-manifold: path-averaged KDE log-density on winning distribution
    kde_density_path_mean_mean: float
    kde_density_path_mean_sd: float

    # Reconstruction cycle consistency
    recon_cycle_error_mean_mean: float
    recon_cycle_error_mean_sd: float


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], math.nan
    return statistics.mean(values), statistics.stdev(values)


def summarise(report: ComparisonReport) -> tuple[MethodStats, ...]:
    """Aggregate results into per-method statistics."""
    # Compute SD-adjusted threshold once from all runs (method-agnostic)
    all_starts = [
        r.p_win_start
        for r in report.results
        if r.error is None and not math.isnan(r.p_win_start)
    ]
    p_win_sd = statistics.stdev(all_starts) if len(all_starts) > 1 else 0.0
    p_win_threshold = 0.5 + p_win_sd

    method_display = {s.name: s.display_name for s in report.methods}
    groups: dict[str, list[PathRunResult]] = {s.name: [] for s in report.methods}
    for r in report.results:
        if r.method_name in groups:
            groups[r.method_name].append(r)

    stats_list = []
    for method_name, results in groups.items():
        good = [r for r in results if r.error is None]
        successful = [r for r in good if r.success]

        n_runs = len(results)
        n_success = len(successful)
        success_rate = n_success / n_runs if n_runs > 0 else math.nan

        crossover_alphas = [
            r.crossover_alpha for r in successful if r.crossover_alpha is not None
        ]
        ca_mean, ca_sd = _mean_sd(crossover_alphas)

        gains = [r.p_win_gain for r in good]
        gain_mean, gain_sd = _mean_sd(gains)

        maxes = [r.p_win_max for r in good]
        max_mean, max_sd = _mean_sd(maxes)

        aucs = [r.auc_p_win for r in good]
        auc_mean, auc_sd = _mean_sd(aucs)

        monos = [r.monotonicity for r in good]
        mono_mean, mono_sd = _mean_sd(monos)

        starts_pw = [r.p_win_start for r in good]
        start_pw_mean, start_pw_sd = _mean_sd(starts_pw)

        ends_pw = [r.p_win_end for r in good]
        end_pw_mean, end_pw_sd = _mean_sd(ends_pw)

        dists = [r.dist_to_nearest_win_end for r in good]
        dist_mean, dist_sd = _mean_sd(dists)

        shifts = [r.kde_density_end - r.kde_density_start for r in good]
        shift_mean, shift_sd = _mean_sd(shifts)

        cond_eligible = [
            r for r in good
            if not math.isnan(r.p_win_start) and r.p_win_start < 0.5
        ]
        cond_success_lst = [r for r in cond_eligible if r.success]
        cond_n_eligible = len(cond_eligible)
        cond_n_success = len(cond_success_lst)
        cond_success_rate = cond_n_success / cond_n_eligible if cond_n_eligible > 0 else math.nan

        sd_adj_success_lst = [r for r in cond_eligible if r.p_win_max >= p_win_threshold]
        sd_adj_n_eligible = cond_n_eligible
        sd_adj_n_success = len(sd_adj_success_lst)
        sd_adj_success_rate = sd_adj_n_success / sd_adj_n_eligible if sd_adj_n_eligible > 0 else math.nan

        norm_gains = [
            r.p_win_gain / (1.0 - r.p_win_start)
            for r in good
            if not math.isnan(r.p_win_start) and r.p_win_start < 1.0
        ]
        norm_gain_mean, norm_gain_sd = _mean_sd(norm_gains)

        mean_z_norms: list[float] = []
        max_z_norms: list[float] = []
        for r in good:
            if r.path_z is not None and len(r.path_z) > 0:
                norms = np.linalg.norm(r.path_z, axis=1)
                mean_z_norms.append(float(norms.mean()))
                max_z_norms.append(float(norms.max()))
        mean_z_mean, mean_z_sd = _mean_sd(mean_z_norms)
        max_z_mean, max_z_sd = _mean_sd(max_z_norms)

        times = [r.wall_time_s for r in good if not math.isnan(r.wall_time_s)]
        time_mean, time_sd = _mean_sd(times)

        # ── New metrics
        feat_lengths = [r.path_length_feature for r in good if not math.isnan(r.path_length_feature)]
        feat_len_mean, feat_len_sd = _mean_sd(feat_lengths)

        n_changed = [float(r.n_features_changed) for r in good]
        n_changed_mean, n_changed_sd = _mean_sd(n_changed)

        kde_path_vals = [r.kde_density_path_mean for r in good if not math.isnan(r.kde_density_path_mean)]
        kde_path_mean, kde_path_sd = _mean_sd(kde_path_vals)

        cycle_means = [r.recon_cycle_error_mean for r in good if not math.isnan(r.recon_cycle_error_mean)]
        cycle_mean, cycle_sd = _mean_sd(cycle_means)

        stats_list.append(
            MethodStats(
                method_name=method_name,
                display_name=method_display.get(method_name, method_name),
                n_runs=n_runs,
                n_success=n_success,
                success_rate=success_rate,
                crossover_alpha_mean=ca_mean,
                crossover_alpha_sd=ca_sd,
                p_win_start_mean=start_pw_mean,
                p_win_start_sd=start_pw_sd,
                p_win_end_mean=end_pw_mean,
                p_win_end_sd=end_pw_sd,
                p_win_gain_mean=gain_mean,
                p_win_gain_sd=gain_sd,
                p_win_max_mean=max_mean,
                p_win_max_sd=max_sd,
                auc_mean=auc_mean,
                auc_sd=auc_sd,
                monotonicity_mean=mono_mean,
                monotonicity_sd=mono_sd,
                dist_nearest_win_end_mean=dist_mean,
                dist_nearest_win_end_sd=dist_sd,
                kde_density_shift_mean=shift_mean,
                kde_density_shift_sd=shift_sd,
                cond_n_eligible=cond_n_eligible,
                cond_n_success=cond_n_success,
                cond_success_rate=cond_success_rate,
                p_win_threshold=p_win_threshold,
                sd_adj_n_eligible=sd_adj_n_eligible,
                sd_adj_n_success=sd_adj_n_success,
                sd_adj_success_rate=sd_adj_success_rate,
                p_win_gain_norm_mean=norm_gain_mean,
                p_win_gain_norm_sd=norm_gain_sd,
                mean_z_norm_mean=mean_z_mean,
                mean_z_norm_sd=mean_z_sd,
                max_z_norm_mean=max_z_mean,
                max_z_norm_sd=max_z_sd,
                wall_time_mean_s=time_mean,
                wall_time_sd_s=time_sd,
                path_length_feature_mean=feat_len_mean,
                path_length_feature_sd=feat_len_sd,
                n_features_changed_mean=n_changed_mean,
                n_features_changed_sd=n_changed_sd,
                kde_density_path_mean_mean=kde_path_mean,
                kde_density_path_mean_sd=kde_path_sd,
                recon_cycle_error_mean_mean=cycle_mean,
                recon_cycle_error_mean_sd=cycle_sd,
            )
        )

    return tuple(stats_list)


def jaccard_between_methods(
    report: ComparisonReport,
    signal: Literal["raw", "mv", "weighted"] = "weighted",
) -> pd.DataFrame:
    """Compute a methods x methods Jaccard similarity matrix for top-k features."""
    method_names = [s.name for s in report.methods]
    method_display = {s.name: s.display_name for s in report.methods}

    # Average Jaccard per method pair across all samples
    n = len(method_names)
    totals = np.zeros((n, n))
    counts = np.zeros((n, n), dtype=int)

    # Group results by sample
    samples: dict[int, dict[str, PathRunResult]] = {}
    for r in report.results:
        if r.error is not None:
            continue
        samples.setdefault(r.sample_idx, {})[r.method_name] = r

    attr = {
        "raw": "top_k_features_raw",
        "mv": "top_k_features_mv",
        "weighted": "top_k_features_weighted",
    }[signal]

    for sample_results in samples.values():
        for i, a in enumerate(method_names):
            for j, b in enumerate(method_names):
                if a in sample_results and b in sample_results:
                    fa = getattr(sample_results[a], attr)
                    fb = getattr(sample_results[b], attr)
                    totals[i, j] += jaccard(fa, fb)
                    counts[i, j] += 1

    with np.errstate(invalid="ignore"):
        mat = np.where(counts > 0, totals / counts, np.nan)

    display_names = [method_display.get(m, m) for m in method_names]
    return pd.DataFrame(mat, index=display_names, columns=display_names)
