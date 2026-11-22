"""
Generate the EDA figures, tables and report.

Usage:
    uv run --extra cuda python -m latent_trainer.analysis.eda --build-metadata
    uv run --extra cuda python -m latent_trainer.analysis.eda --style ieee --replot
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd

from latent_trainer.analysis.eda import style
from latent_trainer.analysis.eda.data import build_eda_data
from latent_trainer.analysis.eda.figures import OUTPUTS, Output
from latent_trainer.analysis.eda.metadata import build_metadata, load_metadata
from latent_trainer.features.data_utils import load_event_cache
from latent_trainer.features.splits import SPLIT_NAMES, SplitManifest
from latent_trainer.settings import DATA_DIR, LOGGING_FORMAT, OUTPUT_DIR

logger = logging.getLogger(__name__)


def _paths(out_dir: Path, name: str) -> dict[str, Path]:
    """
    Return where one output's files live.

    Parameters
    ----------
    out_dir : Path
        EDA output directory.
    name : str
        Output name.

    Returns
    -------
    dict[str, Path]
        ``table`` (Parquet), ``latex`` and ``figure_dir``.
    """
    return {
        "table": out_dir / "tables" / f"{name}.parquet",
        "latex": out_dir / "tables" / f"{name}.tex",
        "figure_dir": out_dir / "figures",
    }


def render(output: Output, table: pd.DataFrame, out_dir: Path) -> None:
    """
    Write an output's LaTeX table and figure from its table.

    Parameters
    ----------
    output : Output
        The output.
    table : pd.DataFrame
        Its computed table.
    out_dir : Path
        EDA output directory.
    """
    paths = _paths(out_dir, output.name)
    if output.latex:
        style.write_latex_table(
            table, paths["latex"], output.caption, f"tab:eda-{output.name}"
        )
    if output.plot is not None:
        style.save_figure(output.plot(table), paths["figure_dir"], output.name)


def write_report(out_dir: Path, done: dict[str, pd.DataFrame]) -> Path:
    """
    Write a Markdown index of every output with its caption and table.

    Outputs not produced in this run are listed from their saved tables, so a
    partial run (``--only``) keeps the report complete.

    Parameters
    ----------
    out_dir : Path
        EDA output directory.
    done : dict[str, pd.DataFrame]
        Tables of the outputs produced in this run.

    Returns
    -------
    Path
        The report file.
    """
    lines = [
        "# Exploratory data analysis",
        "",
        f"Style: `{style.active_style().name}`. Scope `train` = training split only.",
        "",
    ]
    for output in OUTPUTS:
        saved = _paths(out_dir, output.name)["table"]
        if output.name in done:
            table = done[output.name]
        elif saved.exists():
            table = pd.read_parquet(saved)
        else:
            continue
        lines += [f"## {output.name} (scope: {output.scope})", "", output.caption, ""]
        if output.plot is not None:
            lines += [f"![{output.name}](figures/{output.name}.png)", ""]
        if output.latex:
            lines += [f"LaTeX: `tables/{output.name}.tex`", ""]
        if len(table) <= 50:
            lines += [
                "```",
                table.to_string(index=False, float_format="%.4g"),
                "```",
                "",
            ]
        else:
            lines += [f"Table: `tables/{output.name}.parquet` ({len(table)} rows)", ""]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _check_cache_in_manifest(bundle, manifest: SplitManifest) -> None:
    """
    Check that every cached replay sits in the same split of the manifest.

    Parameters
    ----------
    bundle : EventDataBundle
        The loaded cache.
    manifest : SplitManifest
        The manifest.

    Raises
    ------
    ValueError
        If the cache has no replay ids or one is not in its manifest split.
    """
    for split in SPLIT_NAMES:
        ids = bundle.split_ids.get(split, [])
        if not ids:
            raise ValueError("the cache has no replay ids; rebuild it")
        missing = set(ids) - set(manifest.ids(split))
        if missing:
            raise ValueError(f"{len(missing)} cached {split} replays not in that split")


@click.command(help=__doc__.splitlines()[1])
@click.option(
    "--cache",
    type=click.Path(path_type=Path),
    default=DATA_DIR / "cached_dataset_events.pt",
    show_default=True,
)
@click.option(
    "--splits",
    type=click.Path(path_type=Path),
    default=DATA_DIR / "splits.json",
    show_default=True,
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(path_type=Path),
    default=OUTPUT_DIR / "eda",
    show_default=True,
)
@click.option(
    "--style",
    "style_name",
    type=click.Choice(sorted(style.PRESETS)),
    default="default",
    show_default=True,
)
@click.option("--font-size", type=float, default=None, help="Override the preset.")
@click.option("--font-family", type=click.Choice(["sans-serif", "serif"]), default=None)
@click.option("--usetex", is_flag=True, help="Render text with LaTeX.")
@click.option("--only", multiple=True, help="Output names to produce (repeatable).")
@click.option("--replot", is_flag=True, help="Redraw from saved tables only.")
@click.option(
    "--build-metadata",
    "rebuild_metadata",
    is_flag=True,
    help="(Re)build replay metadata first.",
)
@click.option(
    "--source",
    type=click.Path(path_type=Path),
    default=Path("H:/sc2egset_merged/sc2egset_merged.json"),
    show_default=True,
)
@click.option(
    "--skip-report",
    type=click.Path(path_type=Path),
    default=OUTPUT_DIR / "skipped_replays_events.json",
    show_default=True,
)
@click.option("--n_workers", type=int, default=24, show_default=True)
def main(
    cache: Path,
    splits: Path,
    out_dir: Path,
    style_name: str,
    font_size: float | None,
    font_family: str | None,
    usetex: bool,
    only: tuple[str, ...],
    replot: bool,
    rebuild_metadata: bool,
    source: Path,
    skip_report: Path,
    n_workers: int,
) -> None:
    """
    Produce the EDA outputs.

    Parameters
    ----------
    cache : Path
        Events cache.
    splits : Path
        Split manifest.
    out_dir : Path
        Output directory.
    style_name : str
        Style preset.
    font_size : float | None
        Font size override.
    font_family : str | None
        Font family override.
    usetex : bool
        Render text with LaTeX.
    only : tuple[str, ...]
        Restrict to these outputs.
    replot : bool
        Redraw from saved tables without recomputing.
    rebuild_metadata : bool
        Rebuild the replay metadata before the analysis.
    source : Path
        Single-JSON source dataset (for the metadata).
    skip_report : Path
        Skipped-replay report from ``scripts/report_skipped_replays.py``.
    n_workers : int
        Worker processes for the metadata pass.
    """
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
    # Font subsetting while embedding fonts in the PDFs logs every glyph.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    # Patch and year labels are categories on purpose.
    logging.getLogger("matplotlib.category").setLevel(logging.WARNING)
    overrides = {"usetex": usetex}
    if font_size is not None:
        overrides["font_size"] = font_size
    if font_family is not None:
        overrides["font_family"] = font_family
    style.apply_style(style_name, **overrides)

    selected = [o for o in OUTPUTS if not only or o.name in only]
    unknown = set(only) - {o.name for o in OUTPUTS}
    if unknown:
        raise click.BadParameter(f"unknown outputs: {sorted(unknown)}")

    done: dict[str, pd.DataFrame] = {}
    if replot:
        for output in selected:
            path = _paths(out_dir, output.name)["table"]
            if not path.exists():
                logger.warning(f"{output.name}: no saved table, skipped")
                continue
            table = pd.read_parquet(path)
            render(output, table, out_dir)
            done[output.name] = table
    else:
        manifest = SplitManifest.load(splits)
        if rebuild_metadata:
            build_metadata(source, splits, out_dir, n_workers=n_workers)
        bundle = load_event_cache(cache)
        _check_cache_in_manifest(bundle, manifest)
        try:
            metadata = load_metadata(out_dir, manifest)
        except FileNotFoundError as error:
            logger.warning(f"{error}; outputs needing metadata are skipped")
            metadata = None
        data = build_eda_data(bundle, metadata, skip_report)

        for output in selected:
            if output.needs_metadata and not data.has_metadata:
                logger.warning(f"{output.name}: needs metadata, skipped")
                continue
            if output.needs_skip_report and data.skip_report is None:
                logger.warning(f"{output.name}: needs {skip_report}, skipped")
                continue
            view = data.train_only() if output.scope == "train" else data
            logger.info(f"{output.name} ({output.scope})")
            table = output.compute(view)
            path = _paths(out_dir, output.name)["table"]
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(path, index=False)
            render(output, table, out_dir)
            done[output.name] = table

    report = write_report(out_dir, done)
    logger.info(f"{len(done)} outputs written; report: {report}")


if __name__ == "__main__":
    main()
