# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------
from functools import partial

import numpy as np
import torch
from sklearn.decomposition import PCA

from latent_trainer.paths.data import (
    FEATURE_NAMES,
    decode_features,
    encode_player,
    load_model_and_data,
    opponent_aware_score,
)
from latent_trainer.paths.feedback import compute_feedback, print_feedback_report
from latent_trainer.paths.plot import (
    plot_distance,
    plot_feature_delta,
    plot_feature_evolution,
    plot_main,
    plot_three_signal_feedback,
)
from latent_trainer.settings import OUTPUT_DIR, PLOTS_DIR


def run_path_charting_pipeline(
    *,
    model: str,
    cache: str,
    chosen: int,
    player_idx: int,
    n_steps: int,
    top_k: int,
    strategy: str,
    path_z_np: "np.ndarray",
) -> None:
    """Common post-path logic: P(win) curve, feedback, plots."""

    vae, classifier, val_X, val_y, norm_mean, norm_std, _ = load_model_and_data(
        model, cache
    )
    labels = val_y.numpy()
    labels_tensor = torch.tensor(labels)

    latents_p0 = encode_player(vae, val_X[:, 0, :])
    latents_p1 = encode_player(vae, val_X[:, 1, :])

    # Win cloud: for each game, the winner's latent.
    # label=1 → p0 won; label=0 → p1 won.
    win_latents = torch.where((labels_tensor == 1).unsqueeze(1), latents_p0, latents_p1)
    loss_latents = torch.where(
        (labels_tensor == 0).unsqueeze(1), latents_p0, latents_p1
    )
    win_centroid = win_latents.mean(dim=0)

    opponent_idx = 1 - player_idx
    opponent_z = latents_p0[chosen] if opponent_idx == 0 else latents_p1[chosen]

    score_fn = partial(
        opponent_aware_score,
        classifier=classifier,
        opponent_z=opponent_z,
        player_idx=player_idx,
    )

    n_waypoints = n_steps
    path_z_tensor = torch.tensor(path_z_np, dtype=torch.float32)
    alphas = np.linspace(0.0, 1.0, n_waypoints)

    with torch.no_grad():
        win_probs = score_fn(path_z_tensor).numpy()
    print(f"  P(win): {win_probs[0]:.3f} -> {win_probs[-1]:.3f}")

    print("Computing three-signal feedback...")
    feedback = compute_feedback(
        path_z_np,
        decode_fn=vae.decode,
        score_fn=score_fn,
        norm_mean=norm_mean,
        norm_std=norm_std,
        feature_names=FEATURE_NAMES,
        top_k=top_k,
        method_name=strategy.upper(),
    )
    print_feedback_report(feedback=feedback, top_k=top_k)

    print("\nGenerating plots...")
    plot_three_signal_feedback(
        feedback=feedback,
        feature_names=FEATURE_NAMES,
        save_path=OUTPUT_DIR / f"feedback_{strategy}_three_signal.png",
        top_k=top_k,
    )
    path_features = decode_features(vae, path_z_tensor, norm_mean, norm_std)
    plot_feature_evolution(
        path_features=path_features,
        delta=feedback["_raw_delta"],
        feature_names=FEATURE_NAMES,
        n_top=min(5, top_k),
        alphas=alphas,
        save_path=PLOTS_DIR / f"feedback_{strategy}_feature_evolution.png",
    )
    plot_feature_delta(
        delta=feedback["_raw_delta"],
        feature_names=FEATURE_NAMES,
        top_k=top_k,
        save_path=PLOTS_DIR / f"feedback_{strategy}_feature_delta.png",
    )

    Z_win_np = win_latents.detach().cpu().numpy()
    all_data = np.concatenate(
        [Z_win_np, loss_latents.detach().cpu().numpy(), path_z_np]
    )
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_data)
    n_w, n_l = len(win_latents), len(loss_latents)
    win_c = coords[:n_w]
    loss_c = coords[n_w : n_w + n_l]
    path_c = coords[n_w + n_l :]

    plot_main(
        win_c=win_c,
        loss_c=loss_c,
        path_c=path_c,
        alphas=alphas,
        win_probs=win_probs,
        pca=pca,
        save_path=PLOTS_DIR / f"feedback_{strategy}_latent_path.png",
    )
    plot_distance(
        path_z=path_z_tensor,
        win_centroid=win_centroid,
        alphas=alphas,
        save_path=PLOTS_DIR / f"feedback_{strategy}_distance_curve.png",
    )
    print(f"\nDone! All plots saved to {PLOTS_DIR}/ (strategy={strategy})")
