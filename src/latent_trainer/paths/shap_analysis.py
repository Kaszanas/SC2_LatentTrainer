"""SHAP attribution analysis for the GuidedVAE.

Explains two model outputs:
  - **P(win)**: which player features drive the win-probability prediction.
  - **Reconstruction**: which input features contribute to reconstruction error,
    and how each input feature influences each reconstructed output feature.

Usage::

    uv run python -m latent_trainer.paths.shap_analysis \\
        --model_path output/checkpoints/.../best.ckpt \\
        --target both \\
        --n_background 200 \\
        --n_explain 500

The per-feature reconstruction heatmap is expensive (shap for ~196 outputs).
Skip it with ``--no_heatmap`` for a faster run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from latent_trainer.features.data_utils import load_and_normalize
from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE
from latent_trainer.paths.data import FEATURE_NAMES, build_granular_feature_names
from latent_trainer.settings import DATA_DIR, LOGGING_FORMAT, PLOTS_DIR

logger = logging.getLogger(__name__)

_GROUPS = ["early", "mid", "late", "final", "econDelta"]
_GROUP_COLORS = {
    "early": "#3498db",
    "mid": "#27ae60",
    "late": "#e67e22",
    "final": "#e74c3c",
    "econDelta": "#9b59b6",
    "meta": "#7f8c8d",
}


def _feature_group(name: str) -> str:
    for g in _GROUPS:
        if name.startswith(g):
            return g
    return "meta"


def _disable_inplace_ops(module: nn.Module) -> None:
    """Set inplace=False on all activation functions.

    shap.DeepExplainer installs backward hooks that are incompatible with
    inplace operations (ReLU(inplace=True) etc.).  This must be called before
    creating the explainer.
    """
    for m in module.modules():
        if hasattr(m, "inplace") and m.inplace:
            m.inplace = False


class PWinExplainer(nn.Module):
    """[batch, 2*input_dim] → [batch, 1]  P(win)."""

    def __init__(self, guided_vae: LitGuidedVAE) -> None:
        super().__init__()
        self.vae = guided_vae
        for p in self.vae.parameters():
            p.requires_grad_(False)
        _disable_inplace_ops(self.vae)

    def forward(self, X_flat: torch.Tensor) -> torch.Tensor:
        batch = X_flat.shape[0]
        X = X_flat.view(batch, 2, self.vae.model.input_dim)
        mu, _ = self.vae.model.encode(X)
        return self.vae.model.cls(mu)


class ReconExplainer(nn.Module):
    """[batch, 2*input_dim] → [batch, 1]  total MSE for player 0 (original scale)."""

    def __init__(self, guided_vae: LitGuidedVAE) -> None:
        super().__init__()
        self.vae = guided_vae
        for p in self.vae.parameters():
            p.requires_grad_(False)
        _disable_inplace_ops(self.vae)

    def forward(self, X_flat: torch.Tensor) -> torch.Tensor:
        batch = X_flat.shape[0]
        input_dim = self.vae.model.input_dim
        X = X_flat.view(batch, 2, input_dim)
        mu, _ = self.vae.model.encode(X)
        recon = self.vae.model.decode(mu)

        recon_p0 = recon[:, 0, :]
        target_p0 = X[:, 0, :]
        if self.vae.mean is not None and self.vae.std is not None:
            recon_p0 = recon_p0 * self.vae.std + self.vae.mean
            target_p0 = target_p0 * self.vae.std + self.vae.mean

        return ((recon_p0 - target_p0) ** 2).mean(dim=1, keepdim=True)


class ReconPerFeatureExplainer(nn.Module):
    """[batch, 2*input_dim] → [batch, input_dim]  per-feature squared error (player 0)."""

    def __init__(self, guided_vae: LitGuidedVAE) -> None:
        super().__init__()
        self.vae = guided_vae
        for p in self.vae.parameters():
            p.requires_grad_(False)
        _disable_inplace_ops(self.vae)

    def forward(self, X_flat: torch.Tensor) -> torch.Tensor:
        batch = X_flat.shape[0]
        input_dim = self.vae.model.input_dim
        X = X_flat.view(batch, 2, input_dim)
        mu, _ = self.vae.model.encode(X)
        recon = self.vae.model.decode(mu)

        recon_p0 = recon[:, 0, :]
        target_p0 = X[:, 0, :]
        if self.vae.mean is not None and self.vae.std is not None:
            recon_p0 = recon_p0 * self.vae.std + self.vae.mean
            target_p0 = target_p0 * self.vae.std + self.vae.mean

        return (recon_p0 - target_p0) ** 2


def _run_explainer(
    model: nn.Module,
    background: torch.Tensor,
    explain_set: torch.Tensor,
) -> np.ndarray:
    """Compute SHAP values using GradientExplainer.

    GradientExplainer uses plain autograd (integrated gradients over
    interpolated inputs) and handles any differentiable architecture,
    including models with LayerNorm — which DeepExplainer does not support
    correctly (it lacks a custom backward rule for LayerNorm, causing the
    additivity check to fail with large errors).
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "shap is required.  Install it with:  pip install shap  or  uv add shap"
        )

    model.eval()
    explainer = shap.GradientExplainer(model, background)
    shap_vals = explainer.shap_values(explain_set)
    return np.array(shap_vals)


def _bar_chart(
    importances: np.ndarray,
    feature_names: list[str],
    top_k: int,
    title: str,
    xlabel: str,
    save_path: Path,
    second_importances: np.ndarray | None = None,
    first_label: str = "Player 0",
    second_label: str = "Player 1",
) -> None:
    imp1d = np.asarray(importances).ravel()
    order = np.argsort(imp1d)[::-1][:top_k][::-1]
    def _strip_group(name: str) -> str:
        for g in _GROUPS:
            if name.startswith(g + "_"):
                return name[len(g) + 1:]
        return name

    raw_names = [str(feature_names[int(i)]) for i in order]
    names = [_strip_group(n) for n in raw_names]
    vals = imp1d[order]
    colors = [_GROUP_COLORS[_feature_group(n)] for n in raw_names]
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(10, max(4, top_k * 0.38)))
    if second_importances is not None:
        imp2d = np.asarray(second_importances).ravel()
        vals2 = imp2d[order]
        ax.barh(y - 0.2, vals, height=0.38, color=colors, alpha=0.9, label=first_label)
        ax.barh(
            y + 0.2, vals2, height=0.38, color=colors, alpha=0.45, label=second_label
        )
    else:
        ax.barh(y, vals, color=colors)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold")

    from matplotlib.patches import Patch

    group_patches = [Patch(color=c, label=g) for g, c in _GROUP_COLORS.items()]
    if second_importances is not None:
        from matplotlib.lines import Line2D

        player_handles = [
            Line2D([0], [0], color="grey", linewidth=6, alpha=0.9, label=first_label),
            Line2D([0], [0], color="grey", linewidth=6, alpha=0.45, label=second_label),
        ]
        ax.legend(
            handles=group_patches + player_handles,
            loc="lower right",
            fontsize=7,
            title="Group / Player",
        )
    else:
        ax.legend(
            handles=group_patches, loc="lower right", fontsize=7, title="Time window"
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved -> {save_path}")


def _beeswarm(
    shap_vals: np.ndarray,
    X_flat: np.ndarray,
    feature_names: list[str],
    input_dim: int,
    top_k: int,
    save_path: Path,
) -> None:
    shap_p0 = shap_vals[:, :input_dim]
    X_p0 = X_flat[:, :input_dim]
    order = np.argsort(np.abs(shap_p0).mean(0))[::-1][:top_k]

    n_cols = 4
    n_rows = (top_k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.6))
    axes = axes.flatten()

    feat_arr = np.array(feature_names)
    for rank, feat_idx in enumerate(order):
        ax = axes[rank]
        fname = feat_arr[feat_idx]
        color = _GROUP_COLORS[_feature_group(fname)]
        ax.scatter(
            X_p0[:, feat_idx], shap_p0[:, feat_idx], s=5, alpha=0.35, color=color
        )
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(fname, fontsize=7)
        ax.set_xlabel("Feature value", fontsize=6)
        ax.set_ylabel("SHAP", fontsize=6)
        ax.tick_params(labelsize=6)

    for i in range(len(order), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("SHAP vs. Feature Value  —  P(win)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved -> {save_path}")


def analyze_pwin(
    guided_vae: LitGuidedVAE,
    background: torch.Tensor,
    explain_set: torch.Tensor,
    feature_names: list[str],
    top_k: int,
    output_dir: Path,
) -> None:
    input_dim = guided_vae.model.input_dim
    bg_flat = background.reshape(len(background), -1)
    ex_flat = explain_set.reshape(len(explain_set), -1)

    wrapper = PWinExplainer(guided_vae).eval()
    logger.info("Running SHAP for P(win)...")
    shap_raw = _run_explainer(wrapper, bg_flat, ex_flat)
    # GradientExplainer may return a list (one array per output) or a plain array.
    # Normalise to 2-D [N, n_features] regardless of shap version.
    if isinstance(shap_raw, list):
        shap_vals = np.concatenate([np.asarray(s) for s in shap_raw], axis=-1)
    else:
        shap_vals = np.asarray(shap_raw)
    while shap_vals.ndim > 2:
        shap_vals = shap_vals.reshape(shap_vals.shape[0], -1)
        break

    logger.info(
        f"SHAP P(win) array shape: {shap_vals.shape}  (expected [N, {2 * input_dim}])"
    )

    mean_abs = np.abs(shap_vals).mean(0).ravel()  # always 1-D
    # Split into per-player halves if the full [N, 2*input_dim] is present;
    # fall back to player-0-only if the explainer collapsed the player axis.
    if len(mean_abs) >= 2 * input_dim:
        imp_p0 = mean_abs[:input_dim]
        imp_p1 = mean_abs[input_dim : 2 * input_dim]
    else:
        logger.warning(
            f"SHAP values have {len(mean_abs)} columns instead of {2 * input_dim}. "
            "Treating all as player-0 features; opponent bar omitted."
        )
        imp_p0 = mean_abs[:input_dim]
        imp_p1 = None

    _bar_chart(
        importances=imp_p0,
        feature_names=feature_names,
        top_k=top_k,
        title=f"P(win) SHAP — Top-{top_k} Features (both players)",
        xlabel="Mean |SHAP value|",
        save_path=output_dir / "shap_pwin_bar.pdf",
        second_importances=imp_p1,
        first_label="Player 0 (subject)",
        second_label="Player 1 (opponent)",
    )
    _beeswarm(
        shap_vals=shap_vals,
        X_flat=ex_flat.cpu().numpy(),
        feature_names=feature_names,
        input_dim=min(input_dim, shap_vals.shape[1]),
        top_k=min(top_k, 16),
        save_path=output_dir / "shap_pwin_beeswarm.pdf",
    )


@click.command()
@click.option(
    "--model_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    help="Path to the trained GuidedVAE checkpoint (.ckpt).",
)
@click.option(
    "--dataset_filename",
    default="cached_dataset_rich_sc2egset.pt",
    show_default=True,
    help="Filename of the cached dataset placed in DATA_DIR.",
)
@click.option(
    "--n_background",
    type=int,
    default=500,
    show_default=True,
    help="Background samples for DeepExplainer (more = more accurate, slower).",
)
@click.option(
    "--n_explain",
    type=int,
    default=1600,
    show_default=True,
    help="Number of test samples to explain.",
)
@click.option(
    "--top_k",
    type=int,
    default=20,
    show_default=True,
    help="Features shown in bar and beeswarm plots.",
)
@click.option(
    "--output_dir",
    default=str(PLOTS_DIR),
    show_default=True,
    type=click.Path(path_type=Path, resolve_path=True),
    help="Directory to save SHAP plots.",
)
@click.option(
    "--max_input_dim",
    type=int,
    default=None,
    help=(
        "Truncate the loaded feature tensor to its first N columns before "
        "explaining -- for a checkpoint trained on a truncated bin-count "
        "cutoff (e.g. the k18 granular checkpoint, 703 of 781 dims)."
    ),
)
def main(
    model_path: Path,
    dataset_filename: str,
    n_background: int,
    n_explain: int,
    top_k: int,
    output_dir: Path,
    max_input_dim: int | None,
) -> None:
    """SHAP attribution analysis for the GuidedVAE (P(win) and/or reconstruction)."""
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading GuidedVAE...")
    guided_vae = LitGuidedVAE.load_from_checkpoint(model_path)
    guided_vae.eval()

    logger.info("Loading dataset...")
    data = load_and_normalize(cached_dataset_filepath=DATA_DIR / dataset_filename)
    if max_input_dim is not None:
        data.test_X = data.test_X[..., :max_input_dim]
        logger.info("Truncated features to first %d columns (max_input_dim).", max_input_dim)

    rng = np.random.default_rng(42)
    n_test = len(data.test_X)
    idx = rng.permutation(n_test)
    background = data.test_X[idx[:n_background]]
    explain_set = data.test_X[idx[n_background : n_background + n_explain]]

    input_dim = guided_vae.model.input_dim
    feature_names: list[str] = FEATURE_NAMES
    if len(feature_names) != input_dim:
        if (input_dim - 1) % 39 == 0:
            feature_names = build_granular_feature_names(n_bins=(input_dim - 1) // 39)
            logger.info(
                f"FEATURE_NAMES length ({len(FEATURE_NAMES)}) != model input_dim "
                f"({input_dim}); using granular bin names instead."
            )
        else:
            logger.warning(
                f"FEATURE_NAMES length ({len(FEATURE_NAMES)}) != model input_dim ({input_dim}). "
                "Using generic names."
            )
            feature_names = [f"feat_{i}" for i in range(input_dim)]

    logger.info(
        f"input_dim={input_dim}  background={len(background)}  explain={len(explain_set)}"
    )

    analyze_pwin(
        guided_vae=guided_vae,
        background=background,
        explain_set=explain_set,
        feature_names=feature_names,
        top_k=top_k,
        output_dir=output_dir,
    )

    print(f"\nDone! SHAP plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
