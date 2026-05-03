"""Single-dataset comparison plots for multi-method path charting."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from latent_trainer.paths.compare.aggregate import (
    MethodStats,
    jaccard_between_methods,
)
from latent_trainer.paths.compare.results import ComparisonReport

# Consistent colour map keyed by method name
_PALETTE = [
    "#2196F3",  # blue        — linear_centroid
    "#03A9F4",  # light-blue  — linear_nearest
    "#F44336",  # red         — gradient_ascent
    "#FF9800",  # orange      — optimal_transport
    "#4CAF50",  # green       — geodesic
    "#9C27B0",  # purple      — neural_flow
]

_DPI = 150
_ALPHA_BAND = 0.25
_GRID_POINTS = 100


def _method_colours(report: ComparisonReport) -> dict[str, str]:
    return {s.name: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(report.methods)}


def _display_name(report: ComparisonReport) -> dict[str, str]:
    return {s.name: s.display_name for s in report.methods}


# P(win) curves — mean ± 1 SD band per method
def plot_pwin_curves_band(report: ComparisonReport, *, output_dir: Path) -> Path:
    colours = _method_colours(report)
    names = _display_name(report)
    grid = np.linspace(0.0, 1.0, _GRID_POINTS)

    # Group p_win_curves by method, interpolating to common grid
    method_curves: dict[str, list[np.ndarray]] = {s.name: [] for s in report.methods}
    for r in report.results:
        if r.error is not None or len(r.p_win_curve) < 2:
            continue
        interpolated = np.interp(grid, r.alphas, r.p_win_curve)
        method_curves[r.method_name].append(interpolated)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(
        0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.5, label="P(win)=0.5"
    )

    for method_name, curves in method_curves.items():
        if not curves:
            continue
        arr = np.stack(curves)  # (n_runs, 100)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0)
        c = colours[method_name]
        n = len(curves)
        ax.plot(
            grid, mean, color=c, linewidth=1.8, label=f"{names[method_name]} (n={n})"
        )
        ax.fill_between(grid, mean - sd, mean + sd, color=c, alpha=_ALPHA_BAND)

    ax.set_xlabel("Path progress α")
    ax.set_ylabel("P(win)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("P(win) along path — mean ± 1 SD across samples")
    fig.tight_layout()

    out = output_dir / "compare_pwin_curves.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


# Crossover α violin — only successful runs
def plot_crossover_violin(report: ComparisonReport, *, output_dir: Path) -> Path:
    colours = _method_colours(report)
    names = _display_name(report)

    data: dict[str, list[float]] = {s.name: [] for s in report.methods}
    success_counts: dict[str, int] = {s.name: 0 for s in report.methods}
    total_counts: dict[str, int] = {s.name: 0 for s in report.methods}

    for r in report.results:
        if r.error is not None:
            continue
        total_counts[r.method_name] += 1
        if r.success and r.crossover_alpha is not None:
            data[r.method_name].append(r.crossover_alpha)
            success_counts[r.method_name] += 1

    method_names = [s.name for s in report.methods if data[s.name]]
    if not method_names:
        return output_dir / "compare_crossover_violin.png"

    fig, ax = plt.subplots(figsize=(max(6, len(method_names) * 1.4), 5))
    positions = list(range(len(method_names)))

    for pos, method_name in zip(positions, method_names):
        vals = data[method_name]
        parts = ax.violinplot([vals], positions=[pos], showmedians=True, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(colours[method_name])
            pc.set_alpha(0.7)
        for part in ("cmedians", "cbars", "cmins", "cmaxes"):
            if part in parts:
                parts[part].set_color(colours[method_name])
        n_ok = success_counts[method_name]
        n_tot = total_counts[method_name]
        ax.text(
            pos,
            1.03,
            f"{n_ok}/{n_tot}\n({100 * n_ok / max(n_tot, 1):.0f}%)",
            ha="center",
            va="bottom",
            fontsize=7,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([names[m] for m in method_names], rotation=45, ha="right")
    ax.set_ylabel("Crossover α (normalised step where P(win) ≥ 0.5)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        "Crossover position distribution — successful runs only\n(n/total and success rate shown below)"
    )
    fig.tight_layout()

    out = output_dir / "compare_crossover_violin.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


# Success rate bar chart
def plot_success_rate_bar(stats: Sequence[MethodStats], *, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(max(6, len(stats) * 1.4), 4.5))
    x = np.arange(len(stats))
    colours = _PALETTE[: len(stats)]

    bars = ax.bar(x, [s.success_rate * 100 for s in stats], color=colours, alpha=0.85)
    for bar, s in zip(bars, stats):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{s.n_success}/{s.n_runs}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([s.display_name for s in stats], rotation=15, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 115)
    ax.axhline(50, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Fraction of samples where P(win) ≥ 0.5 was reached")
    fig.tight_layout()

    out = output_dir / "compare_success_rate.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


# P(win) gain violin
def plot_pwin_gain_violin(report: ComparisonReport, *, output_dir: Path) -> Path:
    colours = _method_colours(report)
    names = _display_name(report)

    data: dict[str, list[float]] = {s.name: [] for s in report.methods}
    for r in report.results:
        if r.error is None and not math.isnan(r.p_win_gain):
            data[r.method_name].append(r.p_win_gain)

    method_names = [s.name for s in report.methods if data[s.name]]
    if not method_names:
        return output_dir / "compare_pwin_gain.png"

    fig, ax = plt.subplots(figsize=(max(6, len(method_names) * 1.4), 5))
    positions = list(range(len(method_names)))

    for pos, method_name in zip(positions, method_names):
        vals = data[method_name]
        parts = ax.violinplot([vals], positions=[pos], showmedians=True, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(colours[method_name])
            pc.set_alpha(0.7)
        for part in ("cmedians", "cbars", "cmins", "cmaxes"):
            if part in parts:
                parts[part].set_color(colours[method_name])

    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([names[m] for m in method_names], rotation=15, ha="right")
    ax.set_ylabel("ΔP(win) = P(win) at end − P(win) at start")
    ax.set_title("P(win) gain distribution (all runs)")
    fig.tight_layout()

    out = output_dir / "compare_pwin_gain.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


# Nearest-win distance: start vs. end (paired)
def plot_nearest_win_distance(report: ComparisonReport, *, output_dir: Path) -> Path:
    colours = _method_colours(report)
    names = _display_name(report)
    method_names = [s.name for s in report.methods]

    starts: dict[str, list[float]] = {m: [] for m in method_names}
    ends: dict[str, list[float]] = {m: [] for m in method_names}
    for r in report.results:
        if r.error is None and not math.isnan(r.dist_to_nearest_win_start):
            starts[r.method_name].append(r.dist_to_nearest_win_start)
            ends[r.method_name].append(r.dist_to_nearest_win_end)

    active = [m for m in method_names if starts[m]]
    if not active:
        return output_dir / "compare_nearest_win_dist.png"

    fig, ax = plt.subplots(figsize=(max(6, len(active) * 1.4), 5))
    x = np.arange(len(active))
    w = 0.35

    for i, method_name in enumerate(active):
        c = colours[method_name]
        mean_s = np.mean(starts[method_name])
        mean_e = np.mean(ends[method_name])
        sd_s = np.std(starts[method_name])
        sd_e = np.std(ends[method_name])
        ax.bar(
            x[i] - w / 2,
            mean_s,
            w,
            yerr=sd_s,
            color=c,
            alpha=0.4,
            capsize=3,
            label="_nolegend_",
        )
        ax.bar(
            x[i] + w / 2,
            mean_e,
            w,
            yerr=sd_e,
            color=c,
            alpha=0.9,
            capsize=3,
            label="_nolegend_",
        )

    # Legend patches
    import matplotlib.patches as mpatches

    start_p = mpatches.Patch(color="grey", alpha=0.4, label="Start")
    end_p = mpatches.Patch(color="grey", alpha=0.9, label="End")
    ax.legend(handles=[start_p, end_p])

    ax.set_xticks(x)
    ax.set_xticklabels([names[m] for m in active], rotation=15, ha="right")
    ax.set_ylabel("Mean nearest-win distance (± 1 SD)")
    ax.set_title(
        "Distance to nearest winning latent: start vs. end\n"
        "(topology-agnostic; lower end = better)"
    )
    fig.tight_layout()

    out = output_dir / "compare_nearest_win_dist.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


# P(win) start vs. end paired bar
def plot_pwin_start_end(report: ComparisonReport, *, output_dir: Path) -> Path:
    colours = _method_colours(report)
    names = _display_name(report)
    method_names = [s.name for s in report.methods]

    starts: dict[str, list[float]] = {m: [] for m in method_names}
    ends: dict[str, list[float]] = {m: [] for m in method_names}
    for r in report.results:
        if r.error is None and not math.isnan(r.p_win_start):
            starts[r.method_name].append(r.p_win_start)
            ends[r.method_name].append(r.p_win_end)

    active = [m for m in method_names if starts[m]]
    if not active:
        return output_dir / "compare_pwin_start_end.png"

    fig, ax = plt.subplots(figsize=(max(6, len(active) * 1.4), 5))
    x = np.arange(len(active))
    w = 0.35

    for i, method_name in enumerate(active):
        c = colours[method_name]
        mean_s = np.mean(starts[method_name])
        mean_e = np.mean(ends[method_name])
        sd_s = np.std(starts[method_name])
        sd_e = np.std(ends[method_name])
        ax.bar(x[i] - w / 2, mean_s, w, yerr=sd_s, color=c, alpha=0.4, capsize=3)
        ax.bar(x[i] + w / 2, mean_e, w, yerr=sd_e, color=c, alpha=0.9, capsize=3)

    import matplotlib.patches as mpatches

    start_p = mpatches.Patch(color="grey", alpha=0.4, label="Start")
    end_p = mpatches.Patch(color="grey", alpha=0.9, label="End")
    ax.legend(handles=[start_p, end_p])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([names[m] for m in active], rotation=15, ha="right")
    ax.set_ylabel("Mean P(win) (± 1 SD)")
    ax.set_ylim(0, 1)
    ax.set_title(
        "P(win) at path start vs. end\n(higher end = better; dashed = 0.5 threshold)"
    )
    fig.tight_layout()

    out = output_dir / "compare_pwin_start_end.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


# KDE density shift violin
def plot_kde_density_shift(report: ComparisonReport, *, output_dir: Path) -> Path:
    colours = _method_colours(report)
    names = _display_name(report)

    data: dict[str, list[float]] = {s.name: [] for s in report.methods}
    for r in report.results:
        if r.error is None and not math.isnan(r.kde_density_start):
            data[r.method_name].append(r.kde_density_end - r.kde_density_start)

    method_names = [s.name for s in report.methods if data[s.name]]
    if not method_names:
        return output_dir / "compare_kde_shift.png"

    fig, ax = plt.subplots(figsize=(max(6, len(method_names) * 1.4), 5))
    positions = list(range(len(method_names)))

    for pos, method_name in zip(positions, method_names):
        vals = data[method_name]
        parts = ax.violinplot([vals], positions=[pos], showmedians=True, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(colours[method_name])
            pc.set_alpha(0.7)
        for part in ("cmedians", "cbars", "cmins", "cmaxes"):
            if part in parts:
                parts[part].set_color(colours[method_name])

    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([names[m] for m in method_names], rotation=15, ha="right")
    ax.set_ylabel("Log-density shift (end − start)")
    ax.set_title(
        "KDE density shift on winning distribution\n(positive = moved into denser winning region)"
    )
    fig.tight_layout()

    out = output_dir / "compare_kde_shift.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


# Feature Jaccard heatmap
def plot_feature_jaccard_heatmap(
    report: ComparisonReport,
    *,
    signal: str = "raw",
    output_dir: Path,
) -> Path:
    df = jaccard_between_methods(report, signal=signal)  # type: ignore[arg-type]
    if df.empty:
        return output_dir / f"compare_jaccard_{signal}.png"

    fig, ax = plt.subplots(figsize=(max(5, len(df) * 1.1), max(4, len(df) * 1.0)))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Jaccard similarity"},
    )
    ax.set_title(
        f"Feature agreement across methods — top-k features ({signal} signal)\n"
        f"(1.0 = identical feature sets)"
    )
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()

    out = output_dir / f"compare_jaccard_{signal}.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


# Convenience: render all plots:
def render_all(
    report: ComparisonReport,
    stats: Sequence[MethodStats],
    *,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    print("  Saving comparison plots...")
    for fn, label in [
        (lambda: plot_pwin_curves_band(report, output_dir=output_dir), "P(win) curves"),
        (
            lambda: plot_crossover_violin(report, output_dir=output_dir),
            "crossover violin",
        ),
        (lambda: plot_success_rate_bar(stats, output_dir=output_dir), "success rate"),
        (lambda: plot_pwin_gain_violin(report, output_dir=output_dir), "P(win) gain"),
        (
            lambda: plot_nearest_win_distance(report, output_dir=output_dir),
            "nearest-win dist",
        ),
        (
            lambda: plot_pwin_start_end(report, output_dir=output_dir),
            "P(win) start/end",
        ),
        (lambda: plot_kde_density_shift(report, output_dir=output_dir), "KDE shift"),
        (
            lambda: plot_feature_jaccard_heatmap(
                report, signal="raw", output_dir=output_dir
            ),
            "Jaccard heatmap",
        ),
    ]:
        try:
            p = fn()
            paths.append(p)
            print(f"    {label}: {p.name}")
        except Exception as exc:
            print(f"    WARNING: {label} failed — {exc}")
    return paths
