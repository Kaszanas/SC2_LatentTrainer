"""Cross-dataset comparison: compare results across multiple datasets with the same model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from latent_trainer.paths.compare.aggregate import summarise
from latent_trainer.paths.compare.latex import format_mean_sd
from latent_trainer.paths.compare.results import ComparisonReport

sns.set_theme(style="whitegrid", context="paper")

_DPI = 300
_GRID_POINTS = 100

# Okabe-Ito colorblind-safe palette for dataset labels
_DS_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]


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

    return CrossDatasetComparison(
        reports=reports,
        labels=labels,
        model_path=reports[0].model_path,
    )


def _ds_palette(cdc: CrossDatasetComparison) -> dict[str, str]:
    return {lbl: _DS_PALETTE[i % len(_DS_PALETTE)] for i, lbl in enumerate(cdc.labels)}


def _method_display(cdc: CrossDatasetComparison) -> dict[str, str]:
    seen: dict[str, str] = {}
    for r in cdc.reports:
        for s in r.methods:
            seen[s.name] = s.display_name
    return seen


def plot_cross_success_rate(cdc: CrossDatasetComparison, *, output_dir: Path) -> Path:
    all_stats = [summarise(r) for r in cdc.reports]
    display = _method_display(cdc)
    method_names_list = [s.name for s in cdc.reports[0].methods] if cdc.reports else []

    rows = []
    for stats, label in zip(all_stats, cdc.labels):
        stat_map = {s.method_name: s for s in stats}
        for m in method_names_list:
            rate = stat_map[m].success_rate * 100 if m in stat_map else 0.0
            rows.append(
                {
                    "Method": display.get(m, m),
                    "Dataset": label,
                    "Success rate (%)": rate,
                }
            )

    if not rows:
        return output_dir / "cross_success_rate.pdf"

    df = pd.DataFrame(rows)
    palette = _ds_palette(cdc)
    method_order = [display.get(m, m) for m in method_names_list]

    fig, ax = plt.subplots(figsize=(max(7, len(method_names_list) * 1.5), 5))
    sns.barplot(
        data=df,
        x="Method",
        y="Success rate (%)",
        hue="Dataset",
        order=method_order,
        palette=palette,
        errorbar=None,
        ax=ax,
    )
    ax.axhline(50, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0, 115)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.set_title("Success rate by method and dataset")
    fig.tight_layout()

    out = output_dir / "cross_success_rate.pdf"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


def plot_cross_pwin_curves(cdc: CrossDatasetComparison, *, output_dir: Path) -> Path:
    display = _method_display(cdc)
    method_names_list = [s.name for s in cdc.reports[0].methods] if cdc.reports else []
    palette = _ds_palette(cdc)
    grid = np.linspace(0.0, 1.0, _GRID_POINTS)

    n_methods = len(method_names_list)
    ncols = min(3, n_methods)
    nrows = math.ceil(n_methods / ncols) if n_methods else 1
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5), squeeze=False
    )

    for ax in axes.flat:
        ax.set_visible(False)

    for mi, method_name in enumerate(method_names_list):
        ax = axes[mi // ncols][mi % ncols]
        ax.set_visible(True)
        ax.axhline(0.5, color="k", linestyle="--", linewidth=0.7, alpha=0.5)

        rows = []
        for report, label in zip(cdc.reports, cdc.labels):
            for r in report.results:
                if (
                    r.method_name == method_name
                    and r.error is None
                    and len(r.p_win_curve) >= 2
                ):
                    interp = np.interp(grid, r.alphas, r.p_win_curve)
                    for alpha_val, p_val in zip(grid, interp):
                        rows.append({"α": alpha_val, "P(win)": p_val, "Dataset": label})

        if rows:
            df = pd.DataFrame(rows)
            sns.lineplot(
                data=df,
                x="α",
                y="P(win)",
                hue="Dataset",
                errorbar="sd",
                palette=palette,
                linewidth=1.5,
                legend=(mi == 0),
                ax=ax,
            )
            if mi == 0:
                ax.legend(fontsize=7)

        ax.set_title(display.get(method_name, method_name), fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(win)" if mi % ncols == 0 else "")
        ax.set_xlabel("α" if mi // ncols == nrows - 1 else "")

    fig.suptitle("P(win) along path — mean ± 1 SD per dataset", y=1.01)
    fig.tight_layout()

    out = output_dir / "cross_pwin_curves.pdf"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cross_pwin_curves_per_method(
    cdc: CrossDatasetComparison, *, output_dir: Path
) -> list[Path]:
    """Per-method P(win) curves with ±1 SD band, one PDF per method, hued by dataset."""
    display = _method_display(cdc)
    method_names_list = [s.name for s in cdc.reports[0].methods] if cdc.reports else []
    palette = _ds_palette(cdc)
    grid = np.linspace(0.0, 1.0, _GRID_POINTS)

    saved: list[Path] = []
    for method_name in method_names_list:
        rows = []
        for report, label in zip(cdc.reports, cdc.labels):
            for r in report.results:
                if (
                    r.method_name == method_name
                    and r.error is None
                    and len(r.p_win_curve) >= 2
                ):
                    interp = np.interp(grid, r.alphas, r.p_win_curve)
                    for alpha_val, p_val in zip(grid, interp):
                        rows.append({"α": alpha_val, "P(win)": p_val, "Dataset": label})

        if not rows:
            continue

        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
        sns.lineplot(
            data=df,
            x="α",
            y="P(win)",
            hue="Dataset",
            errorbar="sd",
            palette=palette,
            linewidth=1.8,
            ax=ax,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_title(
            f"P(win) along path — {display.get(method_name, method_name)} (mean ± 1 SD per dataset)"
        )
        fig.tight_layout()

        out = output_dir / f"cross_pwin_curves_{method_name}.pdf"
        fig.savefig(out, dpi=_DPI)
        plt.close(fig)
        saved.append(out)

    return saved


def plot_cross_gain_violin(cdc: CrossDatasetComparison, *, output_dir: Path) -> Path:
    display = _method_display(cdc)
    method_names_list = [s.name for s in cdc.reports[0].methods] if cdc.reports else []
    palette = _ds_palette(cdc)

    rows = []
    for report, label in zip(cdc.reports, cdc.labels):
        for r in report.results:
            if r.error is None and not math.isnan(r.p_win_gain):
                rows.append(
                    {
                        "Method": display.get(r.method_name, r.method_name),
                        "Dataset": label,
                        "ΔP(win)": r.p_win_gain,
                    }
                )

    if not rows:
        return output_dir / "cross_pwin_gain.pdf"

    df = pd.DataFrame(rows)
    method_order = [display.get(m, m) for m in method_names_list]

    fig, ax = plt.subplots(
        figsize=(max(8, len(method_names_list) * len(cdc.reports) * 0.8), 5)
    )
    sns.violinplot(
        data=df,
        x="Method",
        y="ΔP(win)",
        hue="Dataset",
        order=method_order,
        palette=palette,
        inner="quart",
        ax=ax,
    )
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.set_title("P(win) gain distribution by method and dataset")
    fig.tight_layout()

    out = output_dir / "cross_pwin_gain.pdf"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


def plot_cross_crossover_violin(
    cdc: CrossDatasetComparison, *, output_dir: Path
) -> Path:
    display = _method_display(cdc)
    method_names_list = [s.name for s in cdc.reports[0].methods] if cdc.reports else []
    palette = _ds_palette(cdc)

    rows = []
    for report, label in zip(cdc.reports, cdc.labels):
        for r in report.results:
            if r.success and r.crossover_alpha is not None:
                rows.append(
                    {
                        "Method": display.get(r.method_name, r.method_name),
                        "Dataset": label,
                        "Crossover α": r.crossover_alpha,
                    }
                )

    if not rows:
        return output_dir / "cross_crossover.pdf"

    df = pd.DataFrame(rows)
    method_order = [display.get(m, m) for m in method_names_list]

    fig, ax = plt.subplots(
        figsize=(max(8, len(method_names_list) * len(cdc.reports) * 0.8), 5)
    )
    sns.violinplot(
        data=df,
        x="Method",
        y="Crossover α",
        hue="Dataset",
        order=method_order,
        palette=palette,
        inner="quart",
        cut=0,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.set_title("Crossover position by method and dataset — successful runs only")
    fig.tight_layout()

    out = output_dir / "cross_crossover.pdf"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    return out


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
    display = _method_display(cdc)

    ds_labels = list(cdc.labels)
    n_ds = len(ds_labels)

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
            "Crossover $\\alpha$",
            lambda s: format_mean_sd(s.crossover_alpha_mean, s.crossover_alpha_sd),
            "crossover_alpha_mean",
        ),
        (
            "$\\bm{\Delta}$P(win)",
            lambda s: format_mean_sd(s.p_win_gain_mean, s.p_win_gain_sd),
            "p_win_gain_mean",
        ),
        ("AUC", lambda s: format_mean_sd(s.auc_mean, s.auc_sd), "auc_mean"),
    ]

    for method_name in method_names_list:
        stat_per_ds = [
            {s.method_name: s for s in stats}.get(method_name) for stats in all_stats
        ]
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


def render_all_cross(cdc: CrossDatasetComparison, *, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    print("  Saving cross-dataset plots...")
    for fn, label in [
        (lambda: plot_cross_success_rate(cdc, output_dir=output_dir), "success rate"),
        (lambda: plot_cross_pwin_curves(cdc, output_dir=output_dir), "P(win) curves"),
        (
            lambda: plot_cross_pwin_curves_per_method(cdc, output_dir=output_dir),
            "P(win) curves (per method)",
        ),
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
