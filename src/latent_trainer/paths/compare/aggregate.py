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
    # Geometry (nearest-win; centroid omitted as unreliable for non-convex distributions)
    dist_nearest_win_end_mean: float
    dist_nearest_win_end_sd: float
    kde_density_shift_mean: float  # mean(density_end - density_start)
    kde_density_shift_sd: float
    # Cost
    wall_time_mean_s: float
    wall_time_sd_s: float


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], math.nan
    return statistics.mean(values), statistics.stdev(values)


def summarise(report: ComparisonReport) -> tuple[MethodStats, ...]:
    """Aggregate results into per-method statistics."""
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

        dists = [r.dist_to_nearest_win_end for r in good]
        dist_mean, dist_sd = _mean_sd(dists)

        shifts = [r.kde_density_end - r.kde_density_start for r in good]
        shift_mean, shift_sd = _mean_sd(shifts)

        times = [r.wall_time_s for r in good if not math.isnan(r.wall_time_s)]
        time_mean, time_sd = _mean_sd(times)

        stats_list.append(
            MethodStats(
                method_name=method_name,
                display_name=method_display.get(method_name, method_name),
                n_runs=n_runs,
                n_success=n_success,
                success_rate=success_rate,
                crossover_alpha_mean=ca_mean,
                crossover_alpha_sd=ca_sd,
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
                wall_time_mean_s=time_mean,
                wall_time_sd_s=time_sd,
            )
        )

    return tuple(stats_list)


def jaccard_between_methods(
    report: ComparisonReport,
    signal: Literal["raw", "mv", "weighted"] = "raw",
) -> pd.DataFrame:
    """Compute a methods × methods Jaccard similarity matrix for top-k features."""
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
