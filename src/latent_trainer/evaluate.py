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
from latent_trainer.features.type import CachedSC2Dataset
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
    vae_metrics: dict[str, float],
    cls_metrics: dict[str, float],
    output_dir: Path,
) -> None:
    rows_vae = [
        (r"Test MSE (original scale)", f"{vae_metrics['test_mse']:.1f}"),
        (r"Test MSE (normalised scale)", f"{vae_metrics['test_mse_norm']:.4f}"),
        (r"Test KL Divergence", f"{vae_metrics['test_kl']:.4f}"),
    ]
    rows_cls = [
        (r"Test Accuracy (\%)", f"{cls_metrics['test_acc']:.2f}"),
        (r"Test ROC-AUC", f"{cls_metrics['test_roc_auc']:.4f}"),
        (r"Test F1 Score", f"{cls_metrics['test_f1']:.4f}"),
        (r"Test Brier Score", f"{cls_metrics['test_brier']:.4f}"),
    ]

    def _rows_tex(rows: list[tuple[str, str]]) -> str:
        return "\n".join(f"    {label} & {value} \\\\" for label, value in rows)

    tex = f"""%% Auto-generated by latent_trainer.evaluate
%% Requires: \\usepackage{{booktabs}}
\\begin{{table}}[htbp]
\\centering
\\caption{{GuidedVAE Test-Set Evaluation}}
\\label{{tab:model-evaluation}}
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
\\multicolumn{{2}}{{l}}{{\\textit{{VAE}}}} \\\\
{_rows_tex(rows_vae)}
\\addlinespace
\\multicolumn{{2}}{{l}}{{\\textit{{Classifier}}}} \\\\
{_rows_tex(rows_cls)}
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
def main(
    model_path: Path,
    dataset: str,
    batch_size: int,
    output_dir: Path,
    mlflow_uri: str,
    run_id: str | None,
) -> None:
    """Evaluate a GuidedVAE checkpoint on the held-out test set."""
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model from {model_path}")
    model = LitGuidedVAE.load_from_checkpoint(model_path)
    model.eval()

    logger.info("Loading dataset...")
    data = load_and_normalize(cached_dataset_filepath=DATA_DIR / dataset)

    splits = {
        "train": (data.train_X, data.train_y),
        "val": (data.val_X, data.val_y),
        "test": (data.test_X, data.test_y),
    }

    results: dict[str, dict] = {}
    for split, (X, y) in splits.items():
        logger.info(f"Running forward pass on {split} split ({len(X)} samples)...")
        results[split] = _forward_pass(model, X, y, batch_size, device)

    test = results["test"]
    vae_metrics = _compute_vae_metrics(test["recon_mse"], test["recon_mse_norm"], test["mu"], test["logvar"])
    cls_metrics = _compute_classifier_metrics(test["probs"], test["targets"])
    all_metrics = {**vae_metrics, **cls_metrics}

    print("\n" + "=" * 50)
    print("  TEST SET RESULTS — VAE")
    print("=" * 50)
    for k, v in vae_metrics.items():
        print(f"  {k:<20s} {v:.4f}")
    print("\n  TEST SET RESULTS — Classifier")
    print("=" * 50)
    for k, v in cls_metrics.items():
        print(f"  {k:<20s} {v:.4f}")
    print("=" * 50 + "\n")

    _write_latex_table(vae_metrics, cls_metrics, output_dir)
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
