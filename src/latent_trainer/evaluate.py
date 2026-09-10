"""Test-set evaluation for a trained GuidedVAE checkpoint.

Runs a full forward pass on the held-out test split and produces:
  - Console summary of test metrics
  - confusion_matrix.pdf
  - roc_curve.pdf
  - calibration_curve.pdf
  - recon_error_dist.pdf  (train / val / test overlay)

Optionally logs test metrics back into the originating MLflow run.

Usage::

    uv run python -m latent_trainer.evaluate --model_path output/checkpoints/.../best.ckpt
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from latent_trainer.features.data_utils import load_and_normalize
from latent_trainer.features.type import CachedDatasetFileSpec, CachedSC2Dataset
from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE
from latent_trainer.settings import (
    DATA_DIR,
    DEFAULT_MLFLOW_URI,
    LOGGING_FORMAT,
    PLOTS_DIR,
)

logger = logging.getLogger(__name__)

_DPI = 300
_PALETTE = ["#0072B2", "#E69F00", "#009E73"]  # colorblind-safe


def _forward_pass(
    model: LitGuidedVAE,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run the full dataset through the model and collect raw outputs."""
    model.eval()
    model.to(device)

    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_recon_mse: list[np.ndarray] = []
    all_recon_mse_norm: list[np.ndarray] = []
    all_mu: list[np.ndarray] = []
    all_logvar: list[np.ndarray] = []

    loader = DataLoader(
        CachedSC2Dataset(features=X, labels=y),
        batch_size=batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            recon, mu, logvar, p_win = model.model(batch_X)

            # MSE in normalised (z-score) space — comparable to training VAE loss
            mse_norm = ((recon - batch_X) ** 2).mean(dim=(1, 2)).cpu().numpy()

            # MSE in original feature scale (for interpretability)
            if model.mean is not None and model.std is not None:
                recon_os = recon * model.std.to(device) + model.mean.to(device)
                x_os = batch_X * model.std.to(device) + model.mean.to(device)
            else:
                recon_os, x_os = recon, batch_X
            mse = ((recon_os - x_os) ** 2).mean(dim=(1, 2)).cpu().numpy()

            all_probs.append(p_win.squeeze(-1).cpu().numpy())
            all_targets.append(batch_y.squeeze(-1).cpu().numpy())
            all_recon_mse.append(mse)
            all_recon_mse_norm.append(mse_norm)
            all_mu.append(mu.flatten(start_dim=1).cpu().numpy())
            all_logvar.append(logvar.flatten(start_dim=1).cpu().numpy())

    return {
        "probs": np.concatenate(all_probs),
        "targets": np.concatenate(all_targets),
        "recon_mse": np.concatenate(all_recon_mse),
        "recon_mse_norm": np.concatenate(all_recon_mse_norm),
        "mu": np.concatenate(all_mu),
        "logvar": np.concatenate(all_logvar),
    }


def _compute_vae_metrics(
    recon_mse: np.ndarray,
    recon_mse_norm: np.ndarray,
    mu: np.ndarray,
    logvar: np.ndarray,
) -> dict[str, float]:
    """Test MSE (original and normalised scales) and KL divergence for the VAE."""
    test_mse = float(recon_mse.mean())
    test_mse_norm = float(recon_mse_norm.mean())
    # KL per sample: -0.5 * sum_j(1 + logvar_j - mu_j^2 - exp(logvar_j))
    kl_per_sample = -0.5 * (1.0 + logvar - mu ** 2 - np.exp(logvar)).sum(axis=1)
    test_kl = float(kl_per_sample.mean())
    return {"test_mse": test_mse, "test_mse_norm": test_mse_norm, "test_kl": test_kl}


def _compute_classifier_metrics(probs: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Accuracy, ROC-AUC, Brier score, and F1 for the win-probability classifier."""
    from sklearn.metrics import f1_score, roc_auc_score

    preds = (probs >= 0.5).astype(float)
    acc = float((preds == targets).mean()) * 100
    auc = float(roc_auc_score(targets, probs))
    brier = float(((probs - targets) ** 2).mean())
    f1 = float(f1_score(targets, preds))
    return {"test_acc": acc, "test_roc_auc": auc, "test_brier": brier, "test_f1": f1}


def _plot_confusion_matrix(
    probs: np.ndarray, targets: np.ndarray, output_dir: Path
) -> None:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(targets, (probs >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    labels = ["Loss", "Win"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.tight_layout()
    path = output_dir / "confusion_matrix.pdf"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_roc(probs: np.ndarray, targets: np.ndarray, output_dir: Path) -> None:
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=_PALETTE[0], lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = output_dir / "roc_curve.pdf"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    logger.info(f"Saved {path}")


def _plot_calibration(probs: np.ndarray, targets: np.ndarray, output_dir: Path) -> None:
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(targets, probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, marker="o", color=_PALETTE[0], lw=2, label="Model")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Perfect")
    ax.set_xlabel("Mean Predicted P(win)")
    ax.set_ylabel("Fraction of Wins")
    ax.set_title("Calibration Curve — Test Set", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "calibration_curve.pdf"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    logger.info(f"Saved {path}")


def _write_latex_table(
    columns: list[tuple[str, dict[str, float], dict[str, float]]],
    output_dir: Path,
) -> None:
    """Write evaluation table. columns = [(label, vae_metrics, cls_metrics), ...]."""
    n = len(columns)
    col_spec = "l" + "r" * n
    col_headers = " & ".join(f"\\textbf{{{label}}}" for label, _, _ in columns)
    n_cols_total = n + 1

    def _fmt_row(label: str, key: str, fmt: str) -> str:
        vals = " & ".join(
            f"{vm[key]:{fmt}}" if key in vm else f"{cm[key]:{fmt}}"
            for _, vm, cm in columns
        )
        return f"    {label} & {vals} \\\\"

    vae_rows = [
        _fmt_row(r"MSE (original scale)", "test_mse", ".1f"),
        _fmt_row(r"MSE (normalised scale)", "test_mse_norm", ".4f"),
        _fmt_row(r"KL Divergence", "test_kl", ".4f"),
    ]
    cls_rows = [
        _fmt_row(r"Accuracy (\%)", "test_acc", ".2f"),
        _fmt_row(r"ROC-AUC", "test_roc_auc", ".4f"),
        _fmt_row(r"F1 Score", "test_f1", ".4f"),
        _fmt_row(r"Brier Score", "test_brier", ".4f"),
    ]

    vae_rows_tex = "\n".join(vae_rows)
    cls_rows_tex = "\n".join(cls_rows)

    tex = f"""%% Auto-generated by latent_trainer.evaluate
%% Requires: \\usepackage{{booktabs}}
\\begin{{table}}[htbp]
\\centering
\\caption{{GuidedVAE Test-Set Evaluation}}
\\label{{tab:model-evaluation}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
\\textbf{{Metric}} & {col_headers} \\\\
\\midrule
\\multicolumn{{{n_cols_total}}}{{l}}{{\\textit{{VAE}}}} \\\\
{vae_rows_tex}
\\addlinespace
\\multicolumn{{{n_cols_total}}}{{l}}{{\\textit{{Classifier}}}} \\\\
{cls_rows_tex}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    path = output_dir / "evaluation_table.tex"
    path.write_text(tex, encoding="utf-8")
    logger.info(f"Saved {path}")


def _plot_recon_error(
    train_mse: np.ndarray,
    val_mse: np.ndarray,
    test_mse: np.ndarray,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for mse, label, color in zip(
        [train_mse, val_mse, test_mse],
        ["Train", "Val", "Test"],
        _PALETTE,
    ):
        ax.hist(mse, bins=60, alpha=0.55, color=color, label=label, density=True)
    ax.set_xlabel("Per-Sample Reconstruction MSE")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Error Distribution", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "recon_error_dist.pdf"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    logger.info(f"Saved {path}")


@click.command()
@click.option(
    "--model_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    help="Path to the trained GuidedVAE checkpoint (.ckpt).",
)
@click.option(
    "--dataset",
    default="cached_dataset_rich_sc2egset.pt",
    show_default=True,
    help="Cached dataset filename inside DATA_DIR.",
)
@click.option(
    "--batch_size",
    type=int,
    default=64,
    show_default=True,
)
@click.option(
    "--output_dir",
    default=str(PLOTS_DIR / "test"),
    show_default=True,
    type=click.Path(path_type=Path, resolve_path=True),
)
@click.option(
    "--mlflow_uri",
    default=DEFAULT_MLFLOW_URI,
    show_default=True,
    help="MLflow tracking URI. Set to empty string to skip MLflow logging.",
)
@click.option(
    "--run_id",
    default=None,
    help="MLflow run ID to log test metrics into. Skip to omit MLflow logging.",
)
@click.option(
    "--ood_dataset",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    help="Path to an OOD cached .pt dataset. Repeat for multiple OOD sets.",
)
@click.option(
    "--ood_label",
    multiple=True,
    default=(),
    help="Label for each OOD dataset (same order as --ood_dataset). "
         "Defaults to 'OOD 1', 'OOD 2', ...",
)
@click.option(
    "--max_input_dim",
    type=int,
    default=None,
    help=(
        "Truncate the loaded feature tensor to its first N columns before "
        "evaluating -- for checkpoints trained on a truncated bin-count cutoff "
        "(e.g. the k18 granular checkpoint, 703 of 781 dims). Per-feature "
        "normalization is slice-invariant, so this is equivalent to having "
        "cached only those columns. Applied to in-distribution and any OOD "
        "datasets alike."
    ),
)
def main(
    model_path: Path,
    dataset: str,
    batch_size: int,
    output_dir: Path,
    mlflow_uri: str,
    run_id: str | None,
    ood_dataset: tuple[Path, ...],
    ood_label: tuple[str, ...],
    max_input_dim: int | None,
) -> None:
    """Evaluate a GuidedVAE checkpoint on the held-out test set and optional OOD datasets."""
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model from {model_path}")
    model = LitGuidedVAE.load_from_checkpoint(model_path)
    model.eval()

    logger.info("Loading in-distribution dataset...")
    data = load_and_normalize(cached_dataset_filepath=DATA_DIR / dataset)
    if max_input_dim is not None:
        data.train_X = data.train_X[..., :max_input_dim]
        data.val_X = data.val_X[..., :max_input_dim]
        data.test_X = data.test_X[..., :max_input_dim]
        data.mean = data.mean[..., :max_input_dim]
        data.std = data.std[..., :max_input_dim]
        logger.info("Truncated features to first %d columns (max_input_dim).", max_input_dim)

    results: dict[str, dict] = {}
    for split, (X, y) in [
        ("train", (data.train_X, data.train_y)),
        ("val", (data.val_X, data.val_y)),
        ("test", (data.test_X, data.test_y)),
    ]:
        logger.info(f"Running forward pass on {split} split ({len(X)} samples)...")
        results[split] = _forward_pass(model, X, y, batch_size, device)

    # Resolve OOD labels
    labels = list(ood_label) + [f"OOD {i + 1}" for i in range(len(ood_dataset) - len(ood_label))]

    # Load and evaluate each OOD dataset using in-distribution normalisation stats
    ood_results: list[tuple[str, dict[str, float], dict[str, float]]] = []
    for path, label in zip(ood_dataset, labels):
        logger.info(f"Loading OOD dataset '{label}' from {path}...")
        cached: dict[str, torch.Tensor] = torch.load(str(path), weights_only=True)
        spec = CachedDatasetFileSpec(**cached)
        ood_features = spec.test_features.float()
        if max_input_dim is not None:
            ood_features = ood_features[..., :max_input_dim]
        X_ood = (ood_features - data.mean) / data.std
        y_ood = spec.test_labels.float()
        logger.info(f"Running forward pass on OOD '{label}' ({len(X_ood)} samples)...")
        res = _forward_pass(model, X_ood, y_ood, batch_size, device)
        v = _compute_vae_metrics(res["recon_mse"], res["recon_mse_norm"], res["mu"], res["logvar"])
        c = _compute_classifier_metrics(res["probs"], res["targets"])
        ood_results.append((label, v, c))

    test = results["test"]
    vae_metrics = _compute_vae_metrics(test["recon_mse"], test["recon_mse_norm"], test["mu"], test["logvar"])
    cls_metrics = _compute_classifier_metrics(test["probs"], test["targets"])
    all_metrics = {**vae_metrics, **cls_metrics}

    def _print_section(title: str, metrics: dict[str, float]) -> None:
        print(f"\n  {title}")
        print("=" * 54)
        for k, v in metrics.items():
            print(f"  {k:<22s} {v:.4f}")

    print("\n" + "=" * 54)
    print("  SC2EGSet (test)")
    _print_section("VAE", vae_metrics)
    _print_section("Classifier", cls_metrics)
    for label, v, c in ood_results:
        print("\n" + "=" * 54)
        print(f"  {label}")
        _print_section("VAE", v)
        _print_section("Classifier", c)
    print("=" * 54 + "\n")

    table_columns = [("SC2EGSet", vae_metrics, cls_metrics)] + ood_results
    _write_latex_table(table_columns, output_dir)
    _plot_confusion_matrix(test["probs"], test["targets"], output_dir)
    _plot_roc(test["probs"], test["targets"], output_dir)
    _plot_calibration(test["probs"], test["targets"], output_dir)
    _plot_recon_error(
        results["train"]["recon_mse"],
        results["val"]["recon_mse"],
        results["test"]["recon_mse"],
        output_dir,
    )

    if run_id and mlflow_uri:
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(all_metrics)
        logger.info(f"Logged test metrics to MLflow run {run_id}")
    elif run_id is None:
        logger.info("No --run_id supplied; skipping MLflow logging.")

    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
