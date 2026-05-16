from __future__ import annotations

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from mlflow import MlflowClient

from latent_trainer.settings import DEFAULT_MLFLOW_URI, LOGGING_FORMAT, PLOTS_DIR

logger = logging.getLogger(__name__)

_DPI = 300
_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
]

# Pairs of metrics to overlay on the same axes (train / val together)
_OVERLAY_PAIRS = [
    ("train_vae_loss_epoch", "val_vae_loss"),
    ("train_cls_loss_epoch", "val_cls_loss"),
    ("train_vae_acc_epoch", "val_acc"),
]


def _resolve_run(client, experiment_name: str, run_name: str):
    """Return the MLflow Run object matching experiment + run name."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise click.ClickException(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        max_results=5,
    )
    if not runs:
        raise click.ClickException(
            f"No run named '{run_name}' in experiment '{experiment_name}'."
        )
    if len(runs) > 1:
        logger.warning(f"Multiple runs match '{run_name}'; using the most recent one.")
    return sorted(runs, key=lambda r: r.info.start_time, reverse=True)[0]


def _get_history(client, run_id: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) arrays for a metric key."""
    history = client.get_metric_history(run_id, key)
    steps = np.array([m.step for m in history])
    values = np.array([m.value for m in history])
    return steps, values


def _plot_metric(
    steps: np.ndarray,
    values: np.ndarray,
    key: str,
    output_dir: Path,
    color: str = _PALETTE[0],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, values, color=color, lw=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel(key)
    ax.set_title(key, fontweight="bold")
    plt.tight_layout()
    safe_name = key.replace("/", "_").replace(" ", "_")
    path = output_dir / f"{safe_name}.pdf"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _plot_overlay(
    client,
    run_id: str,
    key_a: str,
    key_b: str,
    output_dir: Path,
) -> bool:
    """Plot two metrics on the same axes. Returns False if either key is missing."""
    try:
        steps_a, vals_a = _get_history(client, run_id, key_a)
        steps_b, vals_b = _get_history(client, run_id, key_b)
    except Exception:
        return False
    if len(steps_a) == 0 or len(steps_b) == 0:
        return False

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps_a, vals_a, color=_PALETTE[0], lw=1.5, label=key_a)
    ax.plot(steps_b, vals_b, color=_PALETTE[1], lw=1.5, linestyle="--", label=key_b)
    ax.set_xlabel("Epoch")
    combined = key_a.replace("train_", "").replace("_epoch", "")
    ax.set_ylabel(combined)
    ax.set_title(f"Train vs Val — {combined}", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    safe = combined.replace("/", "_").replace(" ", "_")
    path = output_dir / f"train_val_{safe}.pdf"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    logger.info(f"Saved overlay {path}")
    return True


def _print_summary(run, metrics_subset: list[str] | None) -> None:
    data = run.data.metrics
    keys = metrics_subset if metrics_subset else sorted(data.keys())
    print("\n" + "=" * 50)
    print(f"  RUN: {run.info.run_name}  ({run.info.run_id[:8]}...)")
    print("=" * 50)
    for k in keys:
        if k in data:
            print(f"  {k:<40s}  {data[k]:.6f}")
    print("=" * 50 + "\n")


@click.command()
@click.option("--experiment_name", required=True, help="MLflow experiment name.")
@click.option("--run_name", required=True, help="MLflow run name.")
@click.option(
    "--metrics",
    default=None,
    help="Comma-separated metric keys to export. Default: all metrics in the run.",
)
@click.option(
    "--output_dir",
    default=str(PLOTS_DIR / "model_training_plots"),
    show_default=True,
    type=click.Path(path_type=Path, resolve_path=True),
)
@click.option(
    "--mlflow_uri",
    default=DEFAULT_MLFLOW_URI,
    show_default=True,
)
@click.option(
    "--no_artifacts",
    is_flag=True,
    default=False,
    help="Skip downloading run artifacts (export curves only).",
)
def main(
    experiment_name: str,
    run_name: str,
    metrics: str | None,
    output_dir: Path,
    mlflow_uri: str,
    no_artifacts: bool,
) -> None:
    """Export training curves and artifacts from an MLflow run as PDFs."""
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = MlflowClient(tracking_uri=mlflow_uri)
    run = _resolve_run(client, experiment_name, run_name)
    run_id = run.info.run_id

    metrics_subset = [m.strip() for m in metrics.split(",")] if metrics else None
    _print_summary(run, metrics_subset)

    # Overlay pairs first (train+val on one plot)
    plotted_keys: set[str] = set()
    for key_a, key_b in _OVERLAY_PAIRS:
        if _plot_overlay(client, run_id, key_a, key_b, output_dir):
            plotted_keys.update([key_a, key_b])

    # Individual plots for remaining metrics
    keys_to_plot = metrics_subset or list(run.data.metrics.keys())
    for i, key in enumerate(keys_to_plot):
        if key in plotted_keys:
            continue
        steps, values = _get_history(client, run_id, key)
        if len(steps) == 0:
            logger.warning(f"No history for metric '{key}', skipping.")
            continue
        color = _PALETTE[i % len(_PALETTE)]
        _plot_metric(steps, values, key, output_dir, color=color)
        logger.info(f"Saved {key}.pdf")

    # Artifact download
    if not no_artifacts:
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        logger.info("Downloading run artifacts...")
        client.download_artifacts(run_id, path="", dst_path=str(artifacts_dir))
        logger.info(f"Artifacts saved to {artifacts_dir}/")

    print(f"Done. Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
