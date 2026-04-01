# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from latent_trainer.paths.data import (
    FEATURE_NAMES,
    _decode_features,
    _encode_player,
    _load_model_and_data,
    _opponent_aware_score,
)
from latent_trainer.paths.feedback import compute_feedback, print_feedback_report
from latent_trainer.paths.plot import (
    _plot_distance,
    _plot_feature_delta,
    _plot_feature_evolution,
    _plot_main,
    _plot_three_signal_feedback,
)


def _run_pipeline(
    *,
    model: Path,
    cache: Path,
    chosen: int,
    player_idx: int,
    n_steps: int,
    top_k: int,
    strategy: str,
    path_z_np: "np.ndarray",
    output_dir: Path = Path("output"),
) -> None:
    """Common post-path logic: P(win) curve, feedback, plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    vae, classifier, val_X, val_y, norm_mean, norm_std, _ = _load_model_and_data(
        model, cache
    )
    labels = val_y.numpy()
    labels_tensor = torch.tensor(labels)

    latents_p0 = _encode_player(vae, val_X[:, 0, :])
    latents_p1 = _encode_player(vae, val_X[:, 1, :])

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
        _opponent_aware_score,
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
    print_feedback_report(feedback, top_k=top_k)

    print("\nGenerating plots...")
    _plot_three_signal_feedback(
        feedback,
        FEATURE_NAMES,
        output_dir / f"feedback_{strategy}_three_signal.png",
        top_k=top_k,
    )
    path_features = _decode_features(vae, path_z_tensor, norm_mean, norm_std)
    _plot_feature_evolution(
        path_features,
        feedback["_raw_delta"],
        FEATURE_NAMES,
        min(5, top_k),
        alphas,
        output_dir / f"feedback_{strategy}_feature_evolution.png",
    )
    _plot_feature_delta(
        feedback["_raw_delta"],
        FEATURE_NAMES,
        top_k,
        output_dir / f"feedback_{strategy}_feature_delta.png",
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

    _plot_main(
        win_c=win_c,
        loss_c=loss_c,
        path_c=path_c,
        alphas=alphas,
        win_probs=win_probs,
        pca=pca,
        save_path=output_dir / f"feedback_{strategy}_latent_path.png",
    )
    _plot_distance(
        path_z=path_z_tensor,
        win_centroid=win_centroid,
        alphas=alphas,
        save_path=output_dir / f"feedback_{strategy}_distance_curve.png",
    )
    print(f"\nDone! All plots saved to {output_dir} (strategy={strategy})")
