# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------
from functools import partial

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from latent_trainer.paths.data import (
    FEATURE_NAMES,
    PathContext,
    compute_loss_latents,
    compute_win_latents,
    decode_features,
    opponent_aware_score,
)
from latent_trainer.paths.feedback import compute_feedback, print_feedback_report
from latent_trainer.paths.plot import (
    plot_distance,
    plot_feature_delta,
    plot_feature_evolution,
    plot_main_proj,
    plot_three_signal_feedback,
)
from latent_trainer.settings import OUTPUT_DIR, PLOTS_DIR

_TSNE_MAX = 3000


def _slice_proj(coords: np.ndarray, n_win: int, n_loss: int) -> tuple:
    win_c = coords[:n_win]
    loss_c = coords[n_win : n_win + n_loss]
    path_c = coords[n_win + n_loss :]
    return win_c, loss_c, path_c


def run_path_charting_pipeline(
    *,
    path_context: PathContext,
    n_steps: int,
    top_k: int,
    strategy: str,
    path_z_np: "np.ndarray",
) -> None:
    """Common post-path logic: P(win) curve, feedback, plots."""

    win_latents = compute_win_latents(
        labels_tensor=path_context.labels_tensor,
        latents_p0=path_context.latents_p0,
        latents_p1=path_context.latents_p1,
    )
    loss_latents = compute_loss_latents(
        labels_tensor=path_context.labels_tensor,
        latents_p0=path_context.latents_p0,
        latents_p1=path_context.latents_p1,
    )
    win_centroid = win_latents.mean(dim=0)

    opponent_z = (
        path_context.latents_p1[path_context.chosen]
        if path_context.player_idx == 0
        else path_context.latents_p0[path_context.chosen]
    )

    score_fn = partial(
        opponent_aware_score,
        guided_vae=path_context.guided_vae,
        opponent_z=opponent_z,
        player_idx=path_context.player_idx,
    )

    device = next(path_context.guided_vae.parameters()).device
    path_z_tensor = torch.tensor(path_z_np, dtype=torch.float32).to(device)
    alphas = np.linspace(0.0, 1.0, n_steps)

    with torch.no_grad():
        win_probs = score_fn(path_z_tensor).cpu().numpy()
    print(f"  P(win): {win_probs[0]:.3f} -> {win_probs[-1]:.3f}")

    print("Computing feedback...")
    feedback = compute_feedback(
        path_z=path_z_np,
        decode_fn=path_context.guided_vae.model.decode,
        score_fn=score_fn,
        norm_mean=path_context.guided_vae.mean,
        norm_std=path_context.guided_vae.std,
        feature_names=FEATURE_NAMES,
        top_k=top_k,
        method_name=strategy.upper(),
        device=device,
    )
    print_feedback_report(feedback=feedback, top_k=top_k)

    print("\nGenerating plots...")
    plot_three_signal_feedback(
        feedback=feedback,
        feature_names=FEATURE_NAMES,
        save_path=OUTPUT_DIR / f"feedback_{strategy}_three_signal.pdf",
        top_k=top_k,
    )
    path_features = decode_features(
        vae=path_context.guided_vae.model,
        z=path_z_tensor,
        norm_mean=path_context.guided_vae.mean,
        norm_std=path_context.guided_vae.std,
    )
    plot_feature_evolution(
        path_features=path_features,
        delta=feedback["_raw_delta"],
        feature_names=FEATURE_NAMES,
        n_top=min(5, top_k),
        alphas=alphas,
        save_path=PLOTS_DIR / f"feedback_{strategy}_feature_evolution.pdf",
    )
    plot_feature_delta(
        delta=feedback["_raw_delta"],
        feature_names=FEATURE_NAMES,
        top_k=top_k,
        save_path=PLOTS_DIR / f"feedback_{strategy}_feature_delta.pdf",
    )

    Z_win_np = win_latents.detach().cpu().numpy()
    Z_loss_np = loss_latents.detach().cpu().numpy()
    n_win, n_loss, n_path = len(Z_win_np), len(Z_loss_np), len(path_z_np)
    all_data = np.concatenate([Z_win_np, Z_loss_np, path_z_np])

    # --- PCA projection ---
    print("  Fitting PCA...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_data)
    win_c, loss_c, path_c = _slice_proj(coords, n_win, n_loss)
    plot_main_proj(
        win_c=win_c,
        loss_c=loss_c,
        path_c=path_c,
        alphas=alphas,
        win_probs=win_probs,
        save_path=PLOTS_DIR / f"feedback_{strategy}_latent_pca.pdf",
        proj_label="PC",
        subtitle=f" ({pca.explained_variance_ratio_[0]:.1%})",
    )

    # --- UMAP projection ---
    print("  Fitting UMAP...")
    umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = umap_reducer.fit_transform(all_data)
    win_c, loss_c, path_c = _slice_proj(coords, n_win, n_loss)
    plot_main_proj(
        win_c=win_c,
        loss_c=loss_c,
        path_c=path_c,
        alphas=alphas,
        win_probs=win_probs,
        save_path=PLOTS_DIR / f"feedback_{strategy}_latent_umap.pdf",
        proj_label="UMAP",
    )

    # --- t-SNE projection (subsample background; always keep path points) ---
    print("  Fitting t-SNE...")
    n_bg = n_win + n_loss
    rng = np.random.default_rng(42)
    max_bg = max(1, _TSNE_MAX - n_path)
    if n_bg > max_bg:
        bg_idx = np.sort(rng.choice(n_bg, size=max_bg, replace=False))
        sub_bg = all_data[bg_idx]
        # track original win/loss membership for colouring
        sub_is_win = bg_idx < n_win
    else:
        sub_bg = all_data[:n_bg]
        sub_is_win = np.arange(n_bg) < n_win
    sub_data = np.concatenate([sub_bg, path_z_np])
    perplexity = min(30, max(5, len(sub_data) // 10))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(sub_data)
    bg_coords = coords[: len(sub_bg)]
    sub_win_c = bg_coords[sub_is_win]
    sub_loss_c = bg_coords[~sub_is_win]
    sub_path_c = coords[len(sub_bg) :]
    plot_main_proj(
        win_c=sub_win_c,
        loss_c=sub_loss_c,
        path_c=sub_path_c,
        alphas=alphas,
        win_probs=win_probs,
        save_path=PLOTS_DIR / f"feedback_{strategy}_latent_tsne.pdf",
        proj_label="t-SNE",
        subtitle=f" (perp={perplexity})",
    )

    plot_distance(
        path_z=path_z_tensor,
        win_centroid=win_centroid,
        alphas=alphas,
        save_path=PLOTS_DIR / f"feedback_{strategy}_distance_curve.pdf",
    )
    print(f"\nDone! All plots saved to {PLOTS_DIR}/ (strategy={strategy})")
