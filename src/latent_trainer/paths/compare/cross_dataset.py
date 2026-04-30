"""Cross-dataset comparison: compare results across multiple datasets with the same model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from latent_trainer.paths.compare.aggregate import summarise
from latent_trainer.paths.compare.latex import format_mean_sd
from latent_trainer.paths.compare.results import ComparisonReport

_DPI = 150
_ALPHA_BAND = 0.25
_GRID_POINTS = 100


@dataclass
class CrossDatasetComparison:
    reports: tuple[ComparisonReport, ...]
    labels: tuple[str, ...]
    model_path: Path


def build_cross_dataset(
    reports: Sequence[ComparisonReport],
    labels: Sequence[str],
) -> CrossDatasetComparison:
    """Build a CrossDatasetComparison, validating that all reports share the same model."""
    reports = tuple(reports)
    labels = tuple(labels)

    if len(reports) != len(labels):
        raise ValueError(f"Got {len(reports)} reports but {len(labels)} labels.")

    model_paths = {r.model_path for r in reports}
    if len(model_paths) > 1:
        print(
            f"  WARNING: reports use different model paths: {model_paths}. "
            "Cross-dataset comparison assumes the same model."
        )

    model_path = reports[0].model_path
    return CrossDatasetComparison(
        reports=reports,
        labels=labels,
        model_path=model_path,
    )


def _dataset_colours(cdc: CrossDatasetComparison) -> list[str]:
    base = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
    return [base[i % len(base)] for i in range(len(cdc.reports))]


def _method_names(cdc: CrossDatasetComparison) -> list[str]:
    seen = {}
    for r in cdc.reports:
        for s in r.methods:
            seen[s.name] = s.display_name
    return list(seen.keys()), seen


# 1. Grouped success rate bar chart
def plot_cross_success_rate(cdc: CrossDatasetComparison, *, output_dir: Path) -> Path:
    all_stats = [summarise(r) for r in cdc.reports]
    method_names_list = [s.name for s in cdc.reports[0].methods] if cdc.reports else []
    display = {s.name: s.display_name for s in cdc.reports[0].methods}
    ds_colours = _dataset_colours(cdc)

    x = np.arange(len(method_names_list))
    n_ds = len(cdc.reports)
    w = 0.8 / max(n_ds, 1)

    fig, ax = plt.subplots(figsize=(max(7, len(method_names_list) * 1.5), 5))

    for ds_idx, (stats, label, colour) in enumerate(
        zip(all_stats, cdc.labels, ds_colours)
    ):
        stat_map = {s.method_name: s for s in stats}
        offsets = x + (ds_idx - (n_ds - 1) / 2) * w
        heights = [
            stat_map[m].success_rate * 100 if m in stat_map else 0.0
            for m in method_names_list
        ]
        ax.bar(offsets, heights, w * 0.9, color=colour, alpha=0.8, label=label)

    ax.axhline(50, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [display.get(m, m) for m in method_names_list], rotation=15, ha="right"
    )
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 115)
    ax.legend(title="Dataset")
    ax.set_title("Success rate by method and dataset")
    fig.tight_layout()

    out = output_dir / "cross_success_rate.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


# 2. P(win) curves — one subplot per method, one band per dataset
def plot_cross_pwin_curves(cdc: CrossDatasetComparison, *, output_dir: Path) -> Path:
    method_names_list = [s.name for s in cdc.reports[0].methods] if cdc.reports else []
    display = {s.name: s.display_name for s in cdc.reports[0].methods}
    ds_colours = _dataset_colours(cdc)
    grid = np.linspace(0.0, 1.0, _GRID_POINTS)

    n_methods = len(method_names_list)
    ncols = min(3, n_methods)
    nrows = math.ceil(n_methods / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5), squeeze=False
    )

    for ax in axes.flat:
        ax.set_visible(False)

    for mi, method_name in enumerate(method_names_list):
        ax = axes[mi // ncols][mi % ncols]
        ax.set_visible(True)
        ax.axhline(0.5, color="k", linestyle="--", linewidth=0.7, alpha=0.5)

        for ds_idx, (report, label, colour) in enumerate(
            zip(cdc.reports, cdc.labels, ds_colours)
        ):
            curves = []
            for r in report.results:
                if (
                    r.method_name == method_name
                    and r.error is None
                    and len(r.p_win_curve) >= 2
                ):
                    curves.append(np.interp(grid, r.alphas, r.p_win_curve))
            if not curves:
                continue
            arr = np.stack(curves)
            mean = arr.mean(axis=0)
            sd = arr.std(axis=0)
            ax.plot(grid, mean, color=colour, linewidth=1.5, label=label)
            ax.fill_between(grid, mean - sd, mean + sd, color=colour, alpha=_ALPHA_BAND)

        ax.set_title(display.get(method_name, method_name), fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        if mi % ncols == 0:
            ax.set_ylabel("P(win)")
        if mi // ncols == nrows - 1:
            ax.set_xlabel("α")
        if mi == 0:
            ax.legend(fontsize=7)

    fig.suptitle("P(win) along path — mean ± 1 SD per dataset", y=1.01)
    fig.tight_layout()

    out = output_dir / "cross_pwin_curves.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# 3. P(win) gain — grouped violin
def plot_cross_gain_violin(cdc: CrossDatasetComparison, *, output_dir: Path) -> Path:
    method_names_list = [s.name for s in cdc.reports[0].methods] if cdc.reports else []
    display = {s.name: s.display_name for s in cdc.reports[0].methods}
    ds_colours = _dataset_colours(cdc)
    n_ds = len(cdc.reports)

    fig, ax = plt.subplots(figsize=(max(8, len(method_names_list) * n_ds * 0.8), 5))

    x_base = np.arange(len(method_names_list))
    width = 0.8 / max(n_ds, 1)

    for ds_idx, (report, label, colour) in enumerate(
        zip(cdc.reports, cdc.labels, ds_colours)
    ):
        for mi, method_name in enumerate(method_names_list):
            vals = [
                r.p_win_gain
                for r in report.results
                if r.method_name == method_name
                and r.error is None
                and not math.isnan(r.p_win_gain)
            ]
            if not vals:
                continue
            pos = x_base[mi] + (ds_idx - (n_ds - 1) / 2) * width
            parts = ax.violinplot(
                [vals], positions=[pos], showmedians=True, widths=width * 0.85
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(colour)
                pc.set_alpha(0.65)
            for part in ("cmedians", "cbars", "cmins", "cmaxes"):
                if part in parts:
                    parts[part].set_color(colour)

    # Legend
    patches = [
        mpatches.Patch(color=c, label=lbl) for c, lbl in zip(ds_colours, cdc.labels)
    ]
    ax.legend(handles=patches, title="Dataset", fontsize=8)

    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x_base)
    ax.set_xticklabels(
        [display.get(m, m) for m in method_names_list], rotation=15, ha="right"
    )
    ax.set_ylabel("ΔP(win)")
    ax.set_title("P(win) gain distribution by method and dataset")
    fig.tight_layout()

    out = output_dir / "cross_pwin_gain.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4. Crossover α — grouped violin
# ---------------------------------------------------------------------------


def plot_cross_crossover_violin(
    cdc: CrossDatasetComparison, *, output_dir: Path
) -> Path:
    method_names_list = [s.name for s in cdc.reports[0].methods] if cdc.reports else []
    display = {s.name: s.display_name for s in cdc.reports[0].methods}
    ds_colours = _dataset_colours(cdc)
    n_ds = len(cdc.reports)

    fig, ax = plt.subplots(figsize=(max(8, len(method_names_list) * n_ds * 0.8), 5))
    x_base = np.arange(len(method_names_list))
    width = 0.8 / max(n_ds, 1)

    for ds_idx, (report, label, colour) in enumerate(
        zip(cdc.reports, cdc.labels, ds_colours)
    ):
        for mi, method_name in enumerate(method_names_list):
            vals = [
                r.crossover_alpha
                for r in report.results
                if r.method_name == method_name
                and r.success
                and r.crossover_alpha is not None
            ]
            if not vals:
                continue
            pos = x_base[mi] + (ds_idx - (n_ds - 1) / 2) * width
            parts = ax.violinplot(
                [vals], positions=[pos], showmedians=True, widths=width * 0.85
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(colour)
                pc.set_alpha(0.65)
            for part in ("cmedians", "cbars", "cmins", "cmaxes"):
                if part in parts:
                    parts[part].set_color(colour)

    patches = [
        mpatches.Patch(color=c, label=lbl) for c, lbl in zip(ds_colours, cdc.labels)
    ]
    ax.legend(handles=patches, title="Dataset", fontsize=8)

    ax.set_xticks(x_base)
    ax.set_xticklabels(
        [display.get(m, m) for m in method_names_list], rotation=15, ha="right"
    )
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Crossover α")
    ax.set_title("Crossover position by method and dataset — successful runs only")
    fig.tight_layout()

    out = output_dir / "cross_crossover.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


# LaTeX cross-dataset table
def write_cross_dataset_table(
    cdc: CrossDatasetComparison,
    *,
    output_path: Path,
    label: str = "tab:cross-dataset",
    caption: str = "Cross-Dataset Comparison of Path-Charting Strategies",
) -> Path:
    """Write a cross-dataset comparison table with Δ columns."""
    all_stats = [summarise(r) for r in cdc.reports]
    method_names_list = [s.name for s in cdc.reports[0].methods] if cdc.reports else []
    display = {s.name: s.display_name for s in cdc.reports[0].methods}

    ds_labels = list(cdc.labels)
    n_ds = len(ds_labels)

    # Build header
    ds_cols = " & ".join(f"\\textbf{{{lab}}}" for lab in ds_labels)
    header = (
        r"\textbf{Method} & \textbf{Metric} & "
        + ds_cols
        + (r" & $\bm{\Delta}$" if n_ds == 2 else "")
        + r" \\"
    )

    rows = []
    metrics = [
        ("Success rate", lambda s: f"{s.success_rate:.3f}", None),
        (
            "Crossover α",
            lambda s: format_mean_sd(s.crossover_alpha_mean, s.crossover_alpha_sd),
            "crossover_alpha_mean",
        ),
        (
            "ΔP(win)",
            lambda s: format_mean_sd(s.p_win_gain_mean, s.p_win_gain_sd),
            "p_win_gain_mean",
        ),
        ("AUC", lambda s: format_mean_sd(s.auc_mean, s.auc_sd), "auc_mean"),
    ]

    for method_name in method_names_list:
        stat_per_ds = []
        for stats in all_stats:
            stat_map = {s.method_name: s for s in stats}
            stat_per_ds.append(stat_map.get(method_name))

        first_metric = True
        for metric_label, fmt_fn, delta_attr in metrics:
            vals_str = " & ".join(
                fmt_fn(s) if s is not None else "—" for s in stat_per_ds
            )
            delta_str = ""
            if n_ds == 2 and delta_attr and stat_per_ds[0] and stat_per_ds[1]:
                v0 = getattr(stat_per_ds[0], delta_attr, math.nan)
                v1 = getattr(stat_per_ds[1], delta_attr, math.nan)
                delta = (
                    v1 - v0 if (not math.isnan(v0) and not math.isnan(v1)) else math.nan
                )
                delta_str = (
                    f" & {'+' if delta >= 0 else ''}{delta:.3f}"
                    if not math.isnan(delta)
                    else " & —"
                )

            row_label = display.get(method_name, method_name) if first_metric else ""
            rows.append(f"{row_label} & {metric_label} & {vals_str}{delta_str} \\\\")
            first_metric = False
        rows.append(r"\addlinespace")

    rows_tex = "\n".join(rows)
    model_note = f"Model: \\texttt{{{str(cdc.model_path.name)}}}."
    note_text = f"Values are \\textit{{M}} (\\textit{{SD}}). {model_note}"
    if n_ds == 2:
        note_text += f" $\\Delta$ = {ds_labels[1]} $-$ {ds_labels[0]}."

    n_cols = 2 + n_ds + (1 if n_ds == 2 else 0)
    col_spec = "ll" + "c" * (n_cols - 2)

    tex = f"""%% Auto-generated by latent_trainer.paths.compare.cross_dataset
%% Requires: \\usepackage{{booktabs, bm}}
\\begin{{table}}[htbp]
\\centering
\\caption{{\\textit{{{caption}}}}}
\\label{{{label}}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
{header}
\\midrule
{rows_tex}
\\bottomrule
\\end{{tabular}}
\\\\[0.5em]
\\raggedright
\\small \\textit{{Note.}} {note_text}
\\end{{table}}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex, encoding="utf-8")
    return output_path


# Convenience: render all cross-dataset plots
def render_all_cross(cdc: CrossDatasetComparison, *, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    print("  Saving cross-dataset plots...")
    for fn, label in [
        (lambda: plot_cross_success_rate(cdc, output_dir=output_dir), "success rate"),
        (lambda: plot_cross_pwin_curves(cdc, output_dir=output_dir), "P(win) curves"),
        (lambda: plot_cross_gain_violin(cdc, output_dir=output_dir), "gain violin"),
        (
            lambda: plot_cross_crossover_violin(cdc, output_dir=output_dir),
            "crossover violin",
        ),
    ]:
        try:
            p = fn()
            paths.append(p)
            print(f"    {label}: {p.name}")
        except Exception as exc:
            print(f"    WARNING: {label} failed — {exc}")

    tex_path = output_dir / "cross_comparison_table.tex"
    try:
        write_cross_dataset_table(cdc, output_path=tex_path)
        paths.append(tex_path)
        print(f"    LaTeX table: {tex_path.name}")
    except Exception as exc:
        print(f"    WARNING: LaTeX table failed — {exc}")

    return paths
