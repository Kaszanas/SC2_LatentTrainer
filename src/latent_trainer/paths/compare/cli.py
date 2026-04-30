"""CLI commands: compare, tune, compare-datasets."""

from __future__ import annotations

from pathlib import Path

import click

from latent_trainer.paths.compare.aggregate import jaccard_between_methods, summarise
from latent_trainer.paths.compare.cross_dataset import (
    build_cross_dataset,
    render_all_cross,
)
from latent_trainer.paths.compare.latex import write_jaccard_table, write_summary_table
from latent_trainer.paths.compare.orchestrator import (
    load_report,
    run_comparison,
    save_report,
)
from latent_trainer.paths.compare.plots import render_all
from latent_trainer.paths.compare.results import DEFAULT_METHOD_SPECS, MethodSpec
from latent_trainer.paths.compare.tune import (
    apply_config,
    load_config,
    save_config,
    tune_all_methods,
)
from latent_trainer.paths.options import global_options as _global_options
from latent_trainer.settings import DATA_DIR, OUTPUT_DIR


def _parse_methods(methods_str: str) -> list[MethodSpec]:
    if methods_str.strip().lower() == "all":
        return list(DEFAULT_METHOD_SPECS)
    requested = {m.strip() for m in methods_str.split(",")}
    result = [s for s in DEFAULT_METHOD_SPECS if s.name in requested]
    unknown = requested - {s.name for s in result}
    if unknown:
        raise click.UsageError(
            f"Unknown method(s): {', '.join(sorted(unknown))}. "
            f"Valid: {', '.join(s.name for s in DEFAULT_METHOD_SPECS)}"
        )
    return result


# compare
@click.command("compare")
@_global_options
@click.option(
    "--n_samples",
    type=int,
    default=0,
    show_default=True,
    help="Samples to evaluate. 0 = all available.",
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--methods",
    type=str,
    default="all",
    show_default=True,
    help="Comma-separated method names, or 'all'. "
    "Valid: linear_centroid, linear_nearest, gradient_ascent, "
    "optimal_transport, geodesic, neural_flow.",
)
@click.option(
    "--flow_checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    default=None,
    help="Required for neural_flow; skipped if absent.",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    default=None,
    help="JSON config from 'tune --output_config' to override per-method hyperparameters.",
)
@click.option(
    "--output_dir",
    type=click.Path(path_type=Path, resolve_path=True),
    default=None,
    help="Directory for plots and tables. Defaults to output/compare/.",
)
@click.option(
    "--save_raw",
    is_flag=True,
    default=False,
    help="Pickle the full ComparisonReport for later re-plotting.",
)
def cmd_compare(
    model_path: Path,
    dataset_filename: str,
    sample_idx: int | None,
    n_steps: int,
    top_k: int,
    n_samples: int,
    seed: int,
    methods: str,
    flow_checkpoint: Path | None,
    config: Path | None,
    output_dir: Path | None,
    save_raw: bool,
):
    """Aggregate path-charting metrics across N samples and all methods."""
    output_dir = output_dir or (OUTPUT_DIR / "compare")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = DATA_DIR / dataset_filename

    method_specs = _parse_methods(methods)

    if config is not None:
        cfg = load_config(config)
        method_specs = apply_config(method_specs, cfg)
        print(f"Loaded config from {config}")

    print(f"Methods: {[s.name for s in method_specs]}")
    print(f"Output: {output_dir}")

    report = run_comparison(
        model_path=model_path,
        dataset_path=dataset_path,
        method_specs=method_specs,
        n_samples=n_samples,
        n_steps=n_steps,
        top_k=top_k,
        seed=seed,
        flow_checkpoint=flow_checkpoint,
    )

    if save_raw:
        save_report(report, output_dir)

    stats = summarise(report)

    print("\nGenerating plots...")
    render_all(report, stats, output_dir=output_dir)

    print("Generating LaTeX tables...")
    write_summary_table(
        stats,
        output_path=output_dir / "table_summary.tex",
        n_samples=report.n_samples_evaluated,
    )
    jac_df = jaccard_between_methods(report, signal="raw")
    write_jaccard_table(jac_df, output_path=output_dir / "table_jaccard.tex")

    print("\nSummary:")
    _print_summary(stats)
    print(f"\nDone. Outputs in {output_dir}/")


# tune
@click.command("tune")
@_global_options
@click.option(
    "--methods",
    type=str,
    default="all",
    show_default=True,
    help="Methods to tune, comma-separated or 'all'. "
    "neural_flow skipped if no --flow_checkpoint.",
)
@click.option(
    "--n_trials",
    type=int,
    default=30,
    show_default=True,
    help="Optuna trials per method.",
)
@click.option(
    "--n_samples",
    type=int,
    default=0,
    show_default=True,
    help="Samples per trial. 0 = all.",
)
@click.option(
    "--metric",
    type=click.Choice(["auc", "success_rate", "p_win_gain"]),
    default="auc",
    show_default=True,
    help="Metric to maximise.",
)
@click.option(
    "--output_config",
    type=click.Path(path_type=Path, resolve_path=True),
    default=None,
    help="Write best params as JSON (use with compare --config).",
)
@click.option(
    "--flow_checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    default=None,
)
@click.option("--seed", type=int, default=42, show_default=True)
def cmd_tune(
    model_path: Path,
    dataset_filename: str,
    sample_idx: int | None,
    n_steps: int,
    top_k: int,
    methods: str,
    n_trials: int,
    n_samples: int,
    metric: str,
    output_config: Path | None,
    flow_checkpoint: Path | None,
    seed: int,
):
    """Find best hyperparameters for each method using Optuna."""
    dataset_path = DATA_DIR / dataset_filename
    method_specs = _parse_methods(methods)

    # Drop neural_flow if no checkpoint
    if flow_checkpoint is None:
        skipped = [s for s in method_specs if s.requires_flow_checkpoint]
        if skipped:
            print(f"Skipping {[s.name for s in skipped]} (no --flow_checkpoint).")
        method_specs = [s for s in method_specs if not s.requires_flow_checkpoint]

    print(
        f"Tuning {[s.name for s in method_specs]} — {n_trials} trials × {n_samples} samples each"
    )

    best = tune_all_methods(
        method_specs=method_specs,
        model_path=model_path,
        dataset_path=dataset_path,
        n_trials=n_trials,
        n_samples=n_samples,
        n_steps=n_steps,
        top_k=top_k,
        metric=metric,
        seed=seed,
        flow_checkpoint=flow_checkpoint,
    )

    output_config = output_config or (OUTPUT_DIR / "compare" / "best_configs.json")
    save_config(best, output_config)
    print(f"\nDone. Run: compare --config {output_config}")


# compare-datasets
@click.command("compare-datasets")
@click.option(
    "--reports",
    multiple=True,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    help="Paths to report.pkl files from 'compare --save_raw'.",
)
@click.option(
    "--labels",
    multiple=True,
    default=(),
    help="Dataset labels (same order as --reports). Defaults to 'Dataset 1', 'Dataset 2', ...",
)
@click.option(
    "--output_dir",
    type=click.Path(path_type=Path, resolve_path=True),
    default=None,
    help="Output directory. Defaults to output/cross_dataset/.",
)
def cmd_compare_datasets(
    reports: tuple[Path, ...],
    labels: tuple[str, ...],
    output_dir: Path | None,
):
    """Cross-dataset comparison from saved ComparisonReport pickles.

    Run 'compare --save_raw' once per dataset to generate the report.pkl files,
    then call this command to produce side-by-side plots and a LaTeX table.
    """
    output_dir = output_dir or (OUTPUT_DIR / "cross_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not labels:
        labels = tuple(f"Dataset {i + 1}" for i in range(len(reports)))
    if len(labels) != len(reports):
        raise click.UsageError("--labels must have the same count as --reports.")

    print("Loading reports...")
    loaded = []
    for p, lbl in zip(reports, labels):
        r = load_report(p)
        loaded.append(r)
        print(f"  {lbl}: {r.n_samples_evaluated} samples, {len(r.methods)} methods")

    cdc = build_cross_dataset(loaded, labels)
    render_all_cross(cdc, output_dir=output_dir)
    print(f"\nDone. Outputs in {output_dir}/")


def _print_summary(stats) -> None:
    header = f"  {'Method':<24} {'Success':>8}  {'Crossover α':>12}  {'ΔP(win)':>10}  {'AUC':>8}"
    print(header)
    print("  " + "─" * 66)
    for s in stats:
        ca = (
            f"{s.crossover_alpha_mean:.3f}"
            if not (s.crossover_alpha_mean != s.crossover_alpha_mean)
            else "—"
        )
        print(
            f"  {s.display_name:<24} {s.success_rate:>7.1%}  {ca:>12}  "
            f"{s.p_win_gain_mean:>+10.3f}  {s.auc_mean:>8.3f}"
        )
