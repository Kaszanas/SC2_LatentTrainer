"""
The one place where the look of every EDA figure and table is defined.

To retarget a journal, add or edit a :class:`PaperStyle` preset (column widths,
font size and family, LaTeX text rendering) and run the CLI with
``--style <preset> --replot``; nothing is recomputed. Colours are fixed per
entity (race, outcome) and were checked for colour-vision deficiency with the
dataviz palette validator; identity is never carried by colour alone (direct
labels, line style for outcome, and the saved table behind every figure).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


@dataclass(frozen=True)
class PaperStyle:
    """
    Page geometry and typography of one target venue.

    Parameters
    ----------
    name : str
        Preset name.
    single_width_in : float
        Width of a single-column figure in inches.
    double_width_in : float
        Width of a double-column (full text width) figure in inches.
    font_size : float
        Base font size in points; tick labels and legends use ``font_size - 1``.
    font_family : str
        ``"sans-serif"`` or ``"serif"``.
    usetex : bool
        Render text with LaTeX (requires a LaTeX installation).
    dpi : int
        Resolution of the PNG copies.
    """

    name: str
    single_width_in: float = 3.5
    double_width_in: float = 7.0
    font_size: float = 8.0
    font_family: str = "sans-serif"
    usetex: bool = False
    dpi: int = 300


# Presets: published column widths of common templates. Add a venue here.
PRESETS: dict[str, PaperStyle] = {
    "default": PaperStyle("default"),
    "ieee": PaperStyle("ieee", 3.5, 7.16, 8.0, "serif"),
    "elsevier": PaperStyle("elsevier", 3.54, 7.48, 8.0, "serif"),
    "springer": PaperStyle("springer", 3.31, 6.93, 8.0, "serif"),
    "acm": PaperStyle("acm", 3.33, 7.0, 8.0, "sans-serif"),
}

# Colours: one per entity, never by rank. Races take categorical slots 1-3 and
# outcome takes slots 7 and 4 of the reference palette, so they never collide.
RACE_COLORS: dict[str, str] = {"Terr": "#2a78d6", "Prot": "#eb6834", "Zerg": "#1baf7a"}
RACE_LABELS: dict[str, str] = {"Terr": "Terran", "Prot": "Protoss", "Zerg": "Zerg"}
OUTCOME_COLORS: dict[str, str] = {"winner": "#6250d6", "loser": "#eda100"}
OUTCOME_LINESTYLES: dict[str, str] = {"winner": "-", "loser": "--"}
# Single-series marks and sequential (ordinal) steps of the blue ramp.
NEUTRAL_COLOR = "#2a78d6"
SEQUENTIAL_BLUES: tuple[str, ...] = ("#86b6ef", "#2a78d6", "#104281")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
BAND_ALPHA = 0.18

_active: PaperStyle = PRESETS["default"]


def apply_style(style: PaperStyle | str = "default", **overrides) -> PaperStyle:
    """
    Set seaborn/matplotlib defaults for every subsequent figure.

    Parameters
    ----------
    style : PaperStyle | str
        A preset name from :data:`PRESETS` or a style object, by default
        ``"default"``.
    **overrides : Any
        Field overrides, e.g. ``font_size=9`` or ``usetex=True``.

    Returns
    -------
    PaperStyle
        The style now in effect.

    Raises
    ------
    KeyError
        If ``style`` names no preset.
    """
    global _active
    base = PRESETS[style] if isinstance(style, str) else style
    _active = replace(base, **overrides)

    sns.set_theme(context="paper", style="ticks")
    small = _active.font_size - 1
    plt.rcParams.update(
        {
            "font.family": _active.font_family,
            "font.size": _active.font_size,
            "axes.titlesize": _active.font_size,
            "axes.labelsize": _active.font_size,
            "xtick.labelsize": small,
            "ytick.labelsize": small,
            "legend.fontsize": small,
            "legend.title_fontsize": small,
            "legend.frameon": False,
            "text.usetex": _active.usetex,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK_SECONDARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "patch.linewidth": 0,
            # Embed TrueType fonts so the PDFs pass journal checks.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": _active.dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.dpi": 100,
        }
    )
    return _active


def active_style() -> PaperStyle:
    """
    Return the style set by the last :func:`apply_style` call.

    Returns
    -------
    PaperStyle
        The style in effect.
    """
    return _active


def figure_size(width: str = "single", aspect: float = 0.75) -> tuple[float, float]:
    """
    Return a figure size in inches for the active style.

    Parameters
    ----------
    width : str
        ``"single"`` or ``"double"`` column, by default ``"single"``.
    aspect : float
        Height as a fraction of the width, by default 0.75.

    Returns
    -------
    tuple[float, float]
        ``(width, height)`` in inches.

    Raises
    ------
    ValueError
        If ``width`` is neither ``"single"`` nor ``"double"``.
    """
    if width == "single":
        inches = _active.single_width_in
    elif width == "double":
        inches = _active.double_width_in
    else:
        raise ValueError(f"width must be 'single' or 'double', got {width!r}")
    return inches, inches * aspect


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> list[Path]:
    """
    Write a figure as vector PDF and a PNG copy, then close it.

    Parameters
    ----------
    fig : plt.Figure
        The figure.
    out_dir : Path
        Output directory.
    name : str
        File stem.

    Returns
    -------
    list[Path]
        The written files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def write_latex_table(
    table: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    precision: int = 3,
) -> Path:
    r"""
    Write a table as a ``booktabs`` LaTeX ``table`` environment.

    Column names and cells are LaTeX-escaped; the file is ready for
    ``\input`` in a document that loads ``booktabs``.

    Parameters
    ----------
    table : pd.DataFrame
        The table; its index is not written.
    path : Path
        Output ``.tex`` file.
    caption : str
        Table caption (plain text, escaped).
    label : str
        LaTeX label, e.g. ``tab:eda-composition``.
    precision : int
        Decimal places for floats, by default 3.

    Returns
    -------
    Path
        The written file.
    """
    styler = (
        table.style.hide(axis="index")
        .format(precision=precision, thousands=",", escape="latex", na_rep="--")
        .format_index(escape="latex", axis=1)
    )
    latex = styler.to_latex(
        hrules=True,
        caption=escape_latex(caption),
        label=label,
        position="t",
        position_float="centering",
        column_format="l" + "r" * (table.shape[1] - 1),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")
    return path


def escape_latex(text: str) -> str:
    """
    Escape the LaTeX special characters of plain text.

    Parameters
    ----------
    text : str
        Plain text.

    Returns
    -------
    str
        Text safe inside a LaTeX argument.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)
