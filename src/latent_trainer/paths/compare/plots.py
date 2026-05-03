"""Single-dataset comparison plots for multi-method path charting."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from latent_trainer.paths.compare.aggregate import (
    MethodStats,
    jaccard_between_methods,
)
from latent_trainer.paths.compare.results import ComparisonReport

sns.set_theme(style="whitegrid", context="paper")

# Consistent colour palette keyed by method name (insertion order matches DEFAULT_METHOD_SPECS)
_PALETTE = [
    "#2196F3",  # blue        — linear_centroid
    "#03A9F4",  # light-blue  — linear_nearest
    "#F44336",  # red         — gradient_ascent
    "#FF9800",  # orange      — optimal_transport
    "#4CAF50",  # green       — geodesic
    "#9C27B0",  # purple      — neural_flow
]

_DPI = 150
_GRID_POINTS = 100


def _build_palette(report: ComparisonReport) -> dict[str, str]:
    """Return {display_name: hex_colour} for all methods in the report."""
    return {
        s.display_name: _PALETTE[i % len(_PALETTE)]
        for i, s in enumerate(report.methods)
    }


def _name_map(report: ComparisonReport) -> dict[str, str]:
    """Return {method_name: display_name}."""
    return {s.name: s.display_name for s in report.methods}


def plot_pwin_curves_band(report: ComparisonReport, *, output_dir: Path) -> Path:
    names = _name_map(report)
    palette = _build_palette(report)
    grid = np.linspace(0.0, 1.0, _GRID_POINTS)

    rows = []
    for r in report.results:
        if r.error is not None or len(r.p_win_curve) < 2:
            continue
        interp = np.interp(grid, r.alphas, r.p_win_curve)
        display = names.get(r.method_name, r.method_name)
        for alpha_val, p_val in zip(grid, interp):
            rows.append({"α": alpha_val, "P(win)": p_val, "Method": display})

    if not rows:
        return output_dir / "compare_pwin_curves.png"

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(
        0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.5, label="P(win)=0.5"
    )
    sns.lineplot(
        data=df,
        x="α",
        y="P(win)",
        hue="Method",
        errorbar="sd",
        palette=palette,
        linewidth=1.8,
        ax=ax,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("P(win) along path — mean ± 1 SD across samples")
    fig.tight_layout()

    out = output_dir / "compare_pwin_curves.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


def plot_crossover_violin(report: ComparisonReport, *, output_dir: Path) -> Path:
    names = _name_map(report)
    palette = _build_palette(report)

    success_counts: dict[str, int] = {s.name: 0 for s in report.methods}
    total_counts: dict[str, int] = {s.name: 0 for s in report.methods}
    rows = []
    for r in report.results:
        if r.error is not None:
            continue
        total_counts[r.method_name] += 1
        if r.success and r.crossover_alpha is not None:
            success_counts[r.method_name] += 1
            rows.append(
                {
                    "Crossover α": r.crossover_alpha,
                    "Method": names.get(r.method_name, r.method_name),
                }
            )

    if not rows:
        return output_dir / "compare_crossover_violin.png"

    df = pd.DataFrame(rows)
    method_order = [names[s.name] for s in report.methods if rows]

    fig, ax = plt.subplots(figsize=(max(6, df["Method"].nunique() * 1.4), 5))
    sns.violinplot(
        data=df,
        x="Method",
        y="Crossover α",
        hue="Method",
        order=method_order,
        palette=palette,
        inner="quart",
        legend=False,
        ax=ax,
    )

    for i, s in enumerate(report.methods):
        n_ok = success_counts[s.name]
        n_tot = total_counts[s.name]
        ax.text(
            i,
            1.03,
            f"{n_ok}/{n_tot}\n({100 * n_ok / max(n_tot, 1):.0f}%)",
            ha="center",
            va="bottom",
            fontsize=7,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(
        "Crossover position distribution — successful runs only\n(n/total and success rate shown above)"
    )
    fig.tight_layout()

    out = output_dir / "compare_crossover_violin.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


def plot_success_rate_bar(stats: Sequence[MethodStats], *, output_dir: Path) -> Path:
    if not stats:
        return output_dir / "compare_success_rate.png"

    palette = [_PALETTE[i % len(_PALETTE)] for i in range(len(stats))]
    df = pd.DataFrame(
        {
            "Method": [s.display_name for s in stats],
            "Success rate (%)": [s.success_rate * 100 for s in stats],
        }
    )

    fig, ax = plt.subplots(figsize=(max(6, len(stats) * 1.4), 4.5))
    bars = sns.barplot(
        data=df,
        x="Method",
        y="Success rate (%)",
        hue="Method",
        palette=palette,
        errorbar=None,
        legend=False,
        ax=ax,
    )
    for bar, s in zip(bars.patches, stats):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{s.n_success}/{s.n_runs}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.axhline(50, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0, 115)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.set_title("Fraction of samples where P(win) ≥ 0.5 was reached")
    fig.tight_layout()

    out = output_dir / "compare_success_rate.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


def plot_pwin_gain_violin(report: ComparisonReport, *, output_dir: Path) -> Path:
    names = _name_map(report)
    palette = _build_palette(report)

    rows = [
        {"ΔP(win)": r.p_win_gain, "Method": names.get(r.method_name, r.method_name)}
        for r in report.results
        if r.error is None and not math.isnan(r.p_win_gain)
    ]
    if not rows:
        return output_dir / "compare_pwin_gain.png"

    df = pd.DataFrame(rows)
    method_order = [
        names[s.name] for s in report.methods if names[s.name] in df["Method"].values
    ]

    fig, ax = plt.subplots(figsize=(max(6, df["Method"].nunique() * 1.4), 5))
    sns.violinplot(
        data=df,
        x="Method",
        y="ΔP(win)",
        hue="Method",
        order=method_order,
        palette=palette,
        inner="quart",
        legend=False,
        ax=ax,
    )
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylabel("ΔP(win) = P(win) at end − P(win) at start")
    ax.set_title("P(win) gain distribution (all runs)")
    fig.tight_layout()

    out = output_dir / "compare_pwin_gain.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


def plot_nearest_win_distance(report: ComparisonReport, *, output_dir: Path) -> Path:
    names = _name_map(report)

    rows = []
    for r in report.results:
        if r.error is None and not math.isnan(r.dist_to_nearest_win_start):
            display = names.get(r.method_name, r.method_name)
            rows.append(
                {
                    "Method": display,
                    "Phase": "Start",
                    "Distance": r.dist_to_nearest_win_start,
                }
            )
            rows.append(
                {
                    "Method": display,
                    "Phase": "End",
                    "Distance": r.dist_to_nearest_win_end,
                }
            )

    if not rows:
        return output_dir / "compare_nearest_win_dist.png"

    df = pd.DataFrame(rows)
    method_order = [
        names[s.name] for s in report.methods if names[s.name] in df["Method"].values
    ]

    fig, ax = plt.subplots(figsize=(max(6, len(method_order) * 1.4), 5))
    sns.barplot(
        data=df,
        x="Method",
        y="Distance",
        hue="Phase",
        order=method_order,
        palette=["#90CAF9", "#1565C0"],
        errorbar="sd",
        capsize=0.1,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylabel("Mean nearest-win distance (± 1 SD)")
    ax.set_title(
        "Distance to nearest winning latent: start vs. end\n(topology-agnostic; lower end = better)"
    )
    fig.tight_layout()

    out = output_dir / "compare_nearest_win_dist.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


def plot_pwin_start_end(report: ComparisonReport, *, output_dir: Path) -> Path:
    names = _name_map(report)

    rows = []
    for r in report.results:
        if r.error is None and not math.isnan(r.p_win_start):
            display = names.get(r.method_name, r.method_name)
            rows.append({"Method": display, "Phase": "Start", "P(win)": r.p_win_start})
            rows.append({"Method": display, "Phase": "End", "P(win)": r.p_win_end})

    if not rows:
        return output_dir / "compare_pwin_start_end.png"

    df = pd.DataFrame(rows)
    method_order = [
        names[s.name] for s in report.methods if names[s.name] in df["Method"].values
    ]

    fig, ax = plt.subplots(figsize=(max(6, len(method_order) * 1.4), 5))
    sns.barplot(
        data=df,
        x="Method",
        y="P(win)",
        hue="Phase",
        order=method_order,
        palette=["#90CAF9", "#1565C0"],
        errorbar="sd",
        capsize=0.1,
        ax=ax,
    )
    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.set_title(
        "P(win) at path start vs. end\n(higher end = better; dashed = 0.5 threshold)"
    )
    fig.tight_layout()

    out = output_dir / "compare_pwin_start_end.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


def plot_kde_density_shift(report: ComparisonReport, *, output_dir: Path) -> Path:
    names = _name_map(report)
    palette = _build_palette(report)

    rows = [
        {
            "Log-density shift": r.kde_density_end - r.kde_density_start,
            "Method": names.get(r.method_name, r.method_name),
        }
        for r in report.results
        if r.error is None and not math.isnan(r.kde_density_start)
    ]
    if not rows:
        return output_dir / "compare_kde_shift.png"

    df = pd.DataFrame(rows)
    method_order = [
        names[s.name] for s in report.methods if names[s.name] in df["Method"].values
    ]

    fig, ax = plt.subplots(figsize=(max(6, df["Method"].nunique() * 1.4), 5))
    sns.violinplot(
        data=df,
        x="Method",
        y="Log-density shift",
        hue="Method",
        order=method_order,
        palette=palette,
        inner="quart",
        legend=False,
        ax=ax,
    )
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.set_title(
        "KDE density shift on winning distribution\n(positive = moved into denser winning region)"
    )
    fig.tight_layout()

    out = output_dir / "compare_kde_shift.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


def plot_feature_jaccard_heatmap(
    report: ComparisonReport, *, signal: str = "raw", output_dir: Path
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
        f"Feature agreement across methods — top-k features ({signal} signal)\n(1.0 = identical feature sets)"
    )
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()

    out = output_dir / f"compare_jaccard_{signal}.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


def plot_cond_success_rate_bar(
    stats: Sequence[MethodStats], *, output_dir: Path
) -> Path:
    if not stats:
        return output_dir / "compare_cond_success_rate.png"

    threshold = stats[0].p_win_threshold

    rows = []
    for s in stats:
        rows.append(
            {
                "Method": s.display_name,
                "Tier": "Unconditional",
                "Rate (%)": s.success_rate * 100,
            }
        )
        cond = s.cond_success_rate * 100 if not math.isnan(s.cond_success_rate) else 0.0
        rows.append({"Method": s.display_name, "Tier": "Conditional", "Rate (%)": cond})
        sd = (
            s.sd_adj_success_rate * 100
            if not math.isnan(s.sd_adj_success_rate)
            else 0.0
        )
        rows.append({"Method": s.display_name, "Tier": "SD-adjusted", "Rate (%)": sd})

    df = pd.DataFrame(rows)
    tier_order = ["Unconditional", "Conditional", "SD-adjusted"]
    method_order = [s.display_name for s in stats]

    fig, ax = plt.subplots(figsize=(max(6, len(stats) * 1.8), 5))
    sns.barplot(
        data=df,
        x="Method",
        y="Rate (%)",
        hue="Tier",
        order=method_order,
        hue_order=tier_order,
        errorbar=None,
        ax=ax,
    )

    sd_adj_map = {
        s.display_name: (s.sd_adj_n_success, s.sd_adj_n_eligible) for s in stats
    }
    for bar, name in zip(ax.patches[2 * len(stats) :], [s.display_name for s in stats]):
        n_ok, n_el = sd_adj_map[name]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{n_ok}/{n_el}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.axhline(50, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0, 115)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.set_title(
        f"Success rate: unconditional / conditional / SD-adjusted\n(σ = {threshold - 0.5:.3f}, threshold = {threshold:.3f})"
    )
    fig.tight_layout()

    out = output_dir / "compare_cond_success_rate.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


def plot_z_norm_band(report: ComparisonReport, *, output_dir: Path) -> Path:
    names = _name_map(report)
    palette = _build_palette(report)
    grid = np.linspace(0.0, 1.0, _GRID_POINTS)

    rows = []
    ref_norms: list[float] = []

    for r in report.results:
        if r.error is not None or r.path_z is None or len(r.path_z) < 2:
            continue
        norms = np.linalg.norm(r.path_z, axis=1)
        interp = np.interp(grid, r.alphas, norms)
        display = names.get(r.method_name, r.method_name)
        ref_norms.append(float(norms[0]))
        for alpha_val, norm_val in zip(grid, interp):
            rows.append({"α": alpha_val, "‖z(α)‖": norm_val, "Method": display})

    if not rows:
        return output_dir / "compare_z_norm_band.png"

    df = pd.DataFrame(rows)
    ref_line = float(np.mean(ref_norms)) if ref_norms else None

    fig, ax = plt.subplots(figsize=(9, 5))
    if ref_line is not None:
        ax.axhline(
            ref_line,
            color="k",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            label=f"Mean ‖z_start‖ = {ref_line:.2f} (real data)",
        )
    sns.lineplot(
        data=df,
        x="α",
        y="‖z(α)‖",
        hue="Method",
        errorbar="sd",
        palette=palette,
        linewidth=1.8,
        ax=ax,
    )
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(
        "Latent norm along path — mean ± 1 SD\n(excursion above reference line = off-manifold)"
    )
    fig.tight_layout()

    out = output_dir / "compare_z_norm_band.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


def render_all(
    report: ComparisonReport, stats: Sequence[MethodStats], *, output_dir: Path
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
        (
            lambda: plot_cond_success_rate_bar(stats, output_dir=output_dir),
            "cond. success rate",
        ),
        (lambda: plot_z_norm_band(report, output_dir=output_dir), "latent norm band"),
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
