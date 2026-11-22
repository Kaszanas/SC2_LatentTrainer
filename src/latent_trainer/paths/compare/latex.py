"""
APA 7th edition LaTeX table generation for comparison results.

Uses raw f-string templates with booktabs for precise formatting control.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import pandas as pd

from latent_trainer.paths.compare.aggregate import MethodStats


def _fmt(value: float, decimals: int = 3) -> str:
    """
    Format a number to fixed decimals, or an em dash for NaN.

    Parameters
    ----------
    value : float
        Value to format.
    decimals : int
        Decimal places, by default 3.

    Returns
    -------
    str
        Formatted value.
    """
    if math.isnan(value):
        return "—"
    return f"{value:.{decimals}f}"


def format_mean_sd(mean: float, sd: float, decimals: int = 3) -> str:
    """
    Format a mean and SD for LaTeX, e.g. ``0.732 ± 0.041`` using the LaTeX ``pm`` sign.

    Parameters
    ----------
    mean : float
        Mean; NaN gives an em dash.
    sd : float
        Standard deviation; NaN prints the mean alone.
    decimals : int
        Decimal places, by default 3.

    Returns
    -------
    str
        Formatted string.
    """
    if math.isnan(mean):
        return "—"
    if math.isnan(sd):
        return _fmt(mean, decimals)
    return f"{mean:.{decimals}f} $\\pm$ {sd:.{decimals}f}"


def write_summary_table(
    stats: Sequence[MethodStats],
    *,
    output_path: Path,
    n_samples: int | None = None,
    label: str = "tab:path-comparison",
    caption: str = "Comparison of Latent-Space Path-Charting Strategies",
    note: str | None = None,
) -> Path:
    """
    Write the APA 7 method summary table to a ``.tex`` file.

    Columns: method, success rate, crossover alpha, delta P(win), AUC,
    monotonicity, nearest-win distance (end), feature path length, features
    changed, path KDE and time.

    Parameters
    ----------
    stats : Sequence[MethodStats]
        Per-method statistics.
    output_path : Path
        Destination ``.tex`` file.
    n_samples : int | None
        Sample count stated in the table note, by default None.
    label : str
        LaTeX label, by default ``"tab:path-comparison"``.
    caption : str
        Table caption.
    note : str | None
        Extra table note, by default None.

    Returns
    -------
    Path
        ``output_path``.
    """
    # n_note = f"$N = {n_samples}$ samples." if n_samples is not None else ""
    # default_note = (
    #     "Values are \\textit{M} (\\textit{SD}). "
    #     "Success rate = fraction of samples where $P(\\text{win}) \\geq 0.5$ was reached. "
    #     "Crossover $\\alpha$ is reported for successful runs only. "
    #     "Nearest-win distance is the Euclidean distance to the closest winning latent at path end; "
    #     "centroid distance is omitted as it is unreliable for non-convex winning distributions. "
    #     "Feature path length is the sum of L2 steps in decoded (original-scale) feature space. "
    #     "Features changed counts features with $|\\Delta| > 1\\sigma$ of training data (sparsity). "
    #     "Path KDE is the mean log-density of the winning-latent KDE averaged over all waypoints. "
    #     + n_note
    # )
    # note_text = note if note is not None else default_note

    header = (
        r"\textbf{Method} & \textbf{Success rate} & \textbf{Crossover $\alpha$} & "
        r"$\bm{\Delta}$\textbf{P(win)} & \textbf{AUC} & \textbf{Monotonicity} & "
        r"\textbf{Nearest-win dist.} & \textbf{Feat.\ path len.} & "
        r"\textbf{Feat.\ changed} & \textbf{Path KDE} & \textbf{Time (s)} \\"
    )

    rows = []
    for s in stats:
        row = (
            " & ".join(
                [
                    s.display_name,
                    f"{s.success_rate:.3f}",
                    format_mean_sd(s.crossover_alpha_mean, s.crossover_alpha_sd),
                    format_mean_sd(s.p_win_gain_mean, s.p_win_gain_sd),
                    format_mean_sd(s.auc_mean, s.auc_sd),
                    format_mean_sd(s.monotonicity_mean, s.monotonicity_sd),
                    format_mean_sd(
                        s.dist_nearest_win_end_mean,
                        s.dist_nearest_win_end_sd,
                        decimals=2,
                    ),
                    format_mean_sd(
                        s.path_length_feature_mean, s.path_length_feature_sd, decimals=1
                    ),
                    format_mean_sd(
                        s.n_features_changed_mean, s.n_features_changed_sd, decimals=1
                    ),
                    format_mean_sd(
                        s.kde_density_path_mean_mean,
                        s.kde_density_path_mean_sd,
                        decimals=2,
                    ),
                    format_mean_sd(s.wall_time_mean_s, s.wall_time_sd_s, decimals=1),
                ]
            )
            + r" \\"
        )
        rows.append(row)

    rows_tex = "\n".join(rows)

    tex = f"""%% Auto-generated by latent_trainer.paths.compare.latex
%% Requires: \\usepackage{{booktabs, bm}}
\\begin{{table}}[htbp]
\\centering
\\caption{{\\textit{{{caption}}}}}
\\label{{{label}}}
\\begin{{tabular}}{{lccccccrrrr}}
\\toprule
{header}
\\midrule
{rows_tex}
\\bottomrule
\\end{{tabular}}
\\\\[0.5em]
\\raggedright
\\end{{table}}
"""
    # \\small \\textit{{Note.}} {note_text}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex, encoding="utf-8")
    return output_path


def write_jaccard_table(
    df: pd.DataFrame,
    *,
    output_path: Path,
    signal: str = "raw",
    label: str = "tab:jaccard",
    caption: str = "Feature Agreement Across Path-Charting Methods (Jaccard Similarity)",
) -> Path:
    """
    Write a Jaccard similarity matrix as an APA 7 table.

    Parameters
    ----------
    df : pd.DataFrame
        Matrix from :func:`jaccard_between_methods`.
    output_path : Path
        Destination ``.tex`` file.
    signal : str
        Feedback signal the matrix was computed on (``raw``, ``mv`` or
        ``weighted``), by default "raw".
    label : str
        LaTeX label, by default ``"tab:jaccard"``.
    caption : str
        Table caption.

    Returns
    -------
    Path
        ``output_path``.
    """
    n = len(df)
    col_spec = "l" + "c" * n
    col_header = " & ".join([""] + [f"\\textbf{{{c}}}" for c in df.columns]) + r" \\"

    rows = []
    for row_label, row in df.iterrows():
        cells = [str(row_label)] + [("—" if math.isnan(v) else f"{v:.2f}") for v in row]
        rows.append(" & ".join(cells) + r" \\")

    rows_tex = "\n".join(rows)
    signal_label = {
        "raw": "full path",
        "mv": "minimum-viable",
        "weighted": "P(win)-weighted",
    }
    note_text = (
        f"Jaccard similarity of top-$k$ features across methods using the {signal_label.get(signal, signal)} signal. "
        "Values represent the mean Jaccard index across all evaluated samples. "
        "1.00 = identical feature sets; 0.00 = no overlap."
    )

    tex = f"""%% Auto-generated by latent_trainer.paths.compare.latex
%% Requires: \\usepackage{{booktabs}}
\\begin{{table}}[htbp]
\\centering
\\caption{{\\textit{{{caption}}}}}
\\label{{{label}}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
{col_header}
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
