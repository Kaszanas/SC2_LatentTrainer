"""Feedback path finder: latent-space improvement guidance for SC2 players.

Loads the trained two-stage VAE, embeds player data, finds an improvement
path from a losing sample toward the winning class density, reconstructs
feature-level deltas, and produces visualisations.

Four strategies are available:

``linear``
    Linear interpolation toward a target (centroid or k-NN mean).
    Reports a single raw delta.  Fast and deterministic.

``gradient_ascent``
    Gradient ascent on P(win) regularised by a KDE density prior.
    The path stays on the data manifold while climbing toward the
    winning region.  Reports three complementary feedback signals:
    raw delta, minimum-viable delta, and P(win)-gain-weighted delta.

``optimal_transport``
    Finds a Wasserstein-barycentric target in the winning cloud
    and interpolates toward it.  Considers the full shape of the
    winning distribution instead of a single centroid.

``geodesic``
    Builds a k-nearest-neighbours graph on the training latent
    codes and computes the shortest path to the densest winning
    point.  Follows the data manifold by construction.

All strategies are **opponent-aware** — P(win) is computed using the
full ``(player_z, opponent_z)`` input to the classifier (for the
two-stage model).

Usage::

    # Linear interpolation toward win centroid:
    uv run python feedback_path.py linear --method centroid

    # Gradient ascent + KDE:
    uv run python feedback_path.py gradient-ascent --top-k 15

    # Optimal transport:
    uv run python feedback_path.py optimal-transport

    # Geodesic:
    uv run python feedback_path.py geodesic --geodesic-k 15

    # Custom GA hyperparameters:
    uv run python feedback_path.py gradient-ascent \\
        --ga-steps 800 --ga-lr 0.01 --density-weight 0.5

    # Global options come before the sub-command:
    uv run python feedback_path.py --player 2 --top-k 15 gradient-ascent
"""

from __future__ import annotations

import os
from functools import partial

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from train_two_stage import LatentClassifier, SimpleVAE, load_and_normalize

from latent_trainer.paths import (
    compute_feedback,
    path_geodesic,
    path_gradient_ascent,
    path_linear,
    path_optimal_transport,
    print_feedback_report,
)

# ---------------------------------------------------------------------------
# Feature names -- 203 per player
# ---------------------------------------------------------------------------

_ECON_FIELDS: list[str] = [
    "foodMade",
    "foodUsed",
    "mineralsCollectionRate",
    "mineralsCurrent",
    "mineralsFriendlyFireArmy",
    "mineralsFriendlyFireEconomy",
    "mineralsFriendlyFireTechnology",
    "mineralsKilledArmy",
    "mineralsKilledEconomy",
    "mineralsKilledTechnology",
    "mineralsLostArmy",
    "mineralsLostEconomy",
    "mineralsLostTechnology",
    "mineralsUsedActiveForces",
    "mineralsUsedCurrentArmy",
    "mineralsUsedCurrentEconomy",
    "mineralsUsedCurrentTechnology",
    "mineralsUsedInProgressArmy",
    "mineralsUsedInProgressEconomy",
    "mineralsUsedInProgressTechnology",
    "vespeneCollectionRate",
    "vespeneCurrent",
    "vespeneFriendlyFireArmy",
    "vespeneFriendlyFireEconomy",
    "vespeneFriendlyFireTechnology",
    "vespeneKilledArmy",
    "vespeneKilledEconomy",
    "vespeneKilledTechnology",
    "vespeneLostArmy",
    "vespeneLostEconomy",
    "vespeneLostTechnology",
    "vespeneUsedActiveForces",
    "vespeneUsedCurrentArmy",
    "vespeneUsedCurrentEconomy",
    "vespeneUsedCurrentTechnology",
    "vespeneUsedInProgressArmy",
    "vespeneUsedInProgressEconomy",
    "vespeneUsedInProgressTechnology",
    "workersActiveCount",
]


def _build_feature_names() -> list[str]:
    prefixes = ["early", "mid", "late", "final", "econDelta"]
    names: list[str] = []
    for prefix in prefixes:
        for field in _ECON_FIELDS:
            names.append(f"{prefix}_{field}")
    names.extend(["APM", "MMR", "SQ", "SupplyCapped%"])
    names.extend(["UnitsBorn", "UnitsKilled"])
    names.extend(["UpgradeCount", "GameDuration"])
    assert len(names) == 203
    return names


FEATURE_NAMES: list[str] = _build_feature_names()


# ---------------------------------------------------------------------------
# Model loading and encoding
# ---------------------------------------------------------------------------


def _load_model_and_data(model_path: str, cache_path: str) -> tuple:
    checkpoint = torch.load(model_path, weights_only=False)
    latent_dim = checkpoint["latent_dim"]
    input_dim = checkpoint["input_dim"]

    vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)
    vae.load_state_dict(checkpoint["vae_state"])
    vae.eval()

    classifier = LatentClassifier(latent_dim=latent_dim)
    classifier.load_state_dict(checkpoint["classifier_state"])
    classifier.eval()

    norm_mean = checkpoint["normalization"]["mean"]
    norm_std = checkpoint["normalization"]["std"]
    _, _, val_X, val_y, _, _ = load_and_normalize(cache_path)

    return vae, classifier, val_X, val_y, norm_mean, norm_std, latent_dim


@torch.no_grad()
def _encode_player(vae: SimpleVAE, data: torch.Tensor) -> torch.Tensor:
    mus = []
    for i in range(0, len(data), 256):
        mu, _ = vae.encode(data[i : i + 256])
        mus.append(mu)
    return torch.cat(mus, dim=0)


@torch.no_grad()
def _decode_features(vae, z, norm_mean, norm_std) -> np.ndarray:
    recon_norm = vae.decode(z)
    return (recon_norm * norm_std + norm_mean).cpu().numpy()


def _nearest_winning_target(sample_z, win_latents, k=5) -> torch.Tensor:
    dists = torch.cdist(sample_z.unsqueeze(0), win_latents.unsqueeze(0)).squeeze(0)
    _, indices = dists.topk(k, largest=False)
    return win_latents[indices.squeeze()].mean(dim=0)


# ---------------------------------------------------------------------------
# Opponent-aware score / logit functions
#
# These closures adapt the two-stage (player_z, opponent_z) classifier
# into the generic callable interface expected by path strategies.
# ---------------------------------------------------------------------------


def _opponent_aware_score(
    z: torch.Tensor,
    *,
    classifier: LatentClassifier,
    opponent_z: torch.Tensor,
    player_idx: int,
) -> torch.Tensor:
    """P(z wins) for each row in z against a fixed opponent."""
    n = z.shape[0]
    opp = opponent_z.unsqueeze(0).expand(n, -1)
    if player_idx == 0:
        combined = torch.cat([z, opp], dim=1)
        return classifier(combined).squeeze(-1)
    else:
        combined = torch.cat([opp, z], dim=1)
        return 1.0 - classifier(combined).squeeze(-1)


def _opponent_aware_logit(
    z: torch.Tensor,
    *,
    classifier: LatentClassifier,
    opponent_z: torch.Tensor,
    player_idx: int,
) -> torch.Tensor:
    """Pre-sigmoid logit for each row in z, with a fixed opponent.

    Strips the final Sigmoid layer from the classifier so we get
    raw logits — better gradients far from the decision boundary.
    """
    n = z.shape[0]
    opp = opponent_z.unsqueeze(0).expand(n, -1)
    if player_idx == 0:
        combined = torch.cat([z, opp], dim=1)
    else:
        combined = torch.cat([opp, z], dim=1)
    # Forward through all layers except the final Sigmoid
    h = combined
    for layer in list(classifier.net.children())[:-1]:
        h = layer(h)
    logit = h.squeeze(-1)
    # Negate for player 1 so the logit sign matches "z is winning".
    return logit if player_idx == 0 else -logit


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def _plot_main(win_c, loss_c, path_c, alphas, win_probs, pca, save_path):
    """Two-panel figure: latent space + P(win) curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Latent Space — Counterfactual Improvement Path",
        fontsize=15,
        fontweight="bold",
    )

    ax1.scatter(loss_c[:, 0], loss_c[:, 1], c="#e74c3c", alpha=0.2, s=5, label="Loss")
    ax1.scatter(win_c[:, 0], win_c[:, 1], c="#3498db", alpha=0.2, s=5, label="Win")

    ax1.plot(path_c[:, 0], path_c[:, 1], color="black", linewidth=2.5, zorder=5)
    ax1.scatter(
        path_c[1:-1, 0],
        path_c[1:-1, 1],
        c="gold",
        s=50,
        marker="D",
        edgecolors="black",
        linewidths=0.8,
        zorder=6,
        label="Improvement path",
    )
    ax1.scatter(
        path_c[0, 0],
        path_c[0, 1],
        c="red",
        s=140,
        marker="*",
        edgecolors="black",
        linewidths=1,
        zorder=7,
        label="New point (loss)",
    )
    ax1.scatter(
        path_c[-1, 0],
        path_c[-1, 1],
        c="blue",
        s=140,
        marker="*",
        edgecolors="black",
        linewidths=1,
        zorder=7,
        label="Target (win)",
    )

    for i in range(0, len(path_c) - 1, max(1, len(path_c) // 5)):
        ax1.annotate(
            "",
            xy=(path_c[i + 1, 0], path_c[i + 1, 1]),
            xytext=(path_c[i, 0], path_c[i, 1]),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )

    ax1.set_xlabel(f"z1 (PC1 {pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"z2 (PC2 {pca.explained_variance_ratio_[1]:.1%})")
    ax1.set_title("Latent space + improvement path", fontsize=11)
    ax1.legend(fontsize=8, markerscale=1.2, loc="best")
    ax1.grid(True, alpha=0.15)

    ax2.axhspan(0, 0.5, color="#e74c3c", alpha=0.08, label="Loss zone")
    ax2.axhspan(0.5, 1, color="#3498db", alpha=0.08, label="Win zone")
    ax2.axhline(
        0.5, color="grey", linewidth=1.2, linestyle="--", label="Decision boundary"
    )

    ax2.plot(alphas, win_probs, color="#27ae60", linewidth=2.5, zorder=5)
    ax2.scatter(
        alphas,
        win_probs,
        c="#27ae60",
        s=50,
        edgecolors="white",
        linewidths=0.8,
        zorder=6,
    )

    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("Path progress  (0=start, 1=end)")
    ax2.set_ylabel("P(win)")
    ax2.set_title("Win probability along path", fontsize=11)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path}")


def _plot_feature_delta(delta, feature_names, top_k, save_path):
    order = np.argsort(np.abs(delta))[::-1][:top_k]
    names = [feature_names[i] for i in order]
    values = delta[order]
    colours = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(3.5, top_k * 0.4)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature change (target - current)")
    ax.set_title(
        f"Top-{top_k} Feature Deltas for Improvement", fontsize=13, fontweight="bold"
    )
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path}")


def _plot_feature_evolution(
    path_features, delta, feature_names, n_top, alphas, save_path
):
    order = np.argsort(np.abs(delta))[::-1][:n_top]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx in order:
        ax.plot(alphas, path_features[:, idx], linewidth=2, label=feature_names[idx])
    ax.set_xlabel("Path progress  (0=start, 1=end)")
    ax.set_ylabel("Feature value (original scale)")
    ax.set_title(
        f"Top-{n_top} Feature Evolution Along Path", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path}")


def _plot_distance(path_z, win_centroid, alphas, save_path):
    dists = torch.norm(path_z - win_centroid.unsqueeze(0), dim=1).numpy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(alphas, dists, color="#8e44ad", linewidth=2.5, marker="o", markersize=5)
    ax.fill_between(alphas, dists, alpha=0.1, color="#8e44ad")
    ax.set_xlabel("Path progress  (0=start, 1=end)")
    ax.set_ylabel("Euclidean distance to winning centroid")
    ax.set_title("Distance to Win Region Along Path", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path}")


def _plot_three_signal_feedback(
    feedback: dict, feature_names, save_path: str, top_k: int = 10
):
    """Three-panel horizontal bar chart — one per signal."""
    signals = [
        ("_raw_delta", "raw_label", "Full path delta"),
        ("_mv_delta", "mv_label", "Minimum-viable delta"),
        ("_weighted_delta", "wgt_label", "P(win)-gain weighted delta"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, max(4.5, top_k * 0.4)))
    fig.suptitle(
        "Three-Signal Feedback Report",
        fontsize=15,
        fontweight="bold",
    )

    for ax, (delta_key, lbl_key, fallback_title) in zip(axes, signals):
        delta = feedback[delta_key]
        order = np.argsort(np.abs(delta))[::-1][:top_k]
        names = [feature_names[i] for i in order]
        values = delta[order]
        colours = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=colours, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Δ")
        ax.set_title(feedback.get(lbl_key, fallback_title), fontsize=10)
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.grid(True, axis="x", alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------


def _run_pipeline(
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
    os.makedirs("output", exist_ok=True)

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
        f"output/feedback_{strategy}_three_signal.png",
        top_k=top_k,
    )
    path_features = _decode_features(vae, path_z_tensor, norm_mean, norm_std)
    _plot_feature_evolution(
        path_features,
        feedback["_raw_delta"],
        FEATURE_NAMES,
        min(5, top_k),
        alphas,
        f"output/feedback_{strategy}_feature_evolution.png",
    )
    _plot_feature_delta(
        feedback["_raw_delta"],
        FEATURE_NAMES,
        top_k,
        f"output/feedback_{strategy}_feature_delta.png",
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
        win_c,
        loss_c,
        path_c,
        alphas,
        win_probs,
        pca,
        f"output/feedback_{strategy}_latent_path.png",
    )
    _plot_distance(
        path_z_tensor,
        win_centroid,
        alphas,
        f"output/feedback_{strategy}_distance_curve.png",
    )
    print(f"\nDone! All plots saved to output/ (strategy={strategy})")


# ---------------------------------------------------------------------------
# CLI group + sub-commands
# ---------------------------------------------------------------------------

_GLOBAL_OPTIONS = [
    click.option("--model", default="output/two_stage_model.pth", show_default=True),
    click.option("--cache", default="data/cached_dataset_rich.pt", show_default=True),
    click.option(
        "--sample-idx",
        type=int,
        default=None,
        help="Dataset index of the game to analyse (default: random).",
    ),
    click.option(
        "--n-steps",
        type=int,
        default=20,
        show_default=True,
        help="Number of waypoints along the path.",
    ),
    click.option(
        "--top-k",
        type=int,
        default=10,
        show_default=True,
        help="Number of top features to display.",
    ),
]


def _global_options(fn):
    """Decorator that attaches all global options to a sub-command."""
    for option in reversed(_GLOBAL_OPTIONS):
        fn = option(fn)
    return fn


@click.group()
def cli() -> None:
    """Latent-space improvement path finder for SC2 players.

    Choose a sub-command for the path-finding strategy:

    \b
      linear            Linear interpolation toward centroid / k-NN target.
      gradient-ascent   Gradient ascent with KDE density regularisation.
      optimal-transport Wasserstein-barycentric transport into win cloud.
      geodesic          Shortest path on a kNN latent-space graph.

    Global options (--model, --cache, --sample-idx, --n-steps, --top-k)
    must be placed BEFORE the sub-command name.
    The losing player is detected automatically from the game label.
    """


@cli.command("linear")
@_global_options
@click.option(
    "--method",
    type=click.Choice(["centroid", "nearest"]),
    default="centroid",
    show_default=True,
    help="Target selection: centroid of wins or nearest k-NN mean.",
)
@click.option(
    "--k-neighbours",
    type=int,
    default=5,
    show_default=True,
    help="k for nearest-neighbour target.",
)
def cmd_linear(model, cache, sample_idx, n_steps, top_k, method, k_neighbours):
    """Linear interpolation toward a winning target."""
    os.makedirs("output", exist_ok=True)

    print("Loading model and data...")
    vae, classifier, val_X, val_y, norm_mean, norm_std, _ = _load_model_and_data(
        model, cache
    )
    labels = val_y.numpy()
    labels_tensor = torch.tensor(labels)
    print(f"  Validation: {len(val_X)}")

    print("Encoding into latent space...")
    latents_p0 = _encode_player(vae, val_X[:, 0, :])
    latents_p1 = _encode_player(vae, val_X[:, 1, :])

    # Win cloud: label=1 → p0 won; label=0 → p1 won.
    win_latents = torch.where((labels_tensor == 1).unsqueeze(1), latents_p0, latents_p1)
    win_centroid = win_latents.mean(dim=0)

    n = len(labels)
    chosen = (
        sample_idx
        if (sample_idx is not None and sample_idx < n)
        else torch.randint(n, (1,)).item()
    )
    player_idx = int(labels[chosen])  # 0 if p0 lost, 1 if p1 lost
    sample_z = latents_p0[chosen] if player_idx == 0 else latents_p1[chosen]
    print(
        f"  Sample idx: {chosen} (label={int(labels[chosen])}, loser=player {player_idx})"
    )

    if method == "centroid":
        target_z = win_centroid.numpy()
        print("  Target: centroid")
    else:
        target_z = _nearest_winning_target(
            sample_z, win_latents, k=k_neighbours
        ).numpy()
        print(f"  Target: nearest (k={k_neighbours})")

    path_z_np = path_linear(
        sample_z.detach().cpu().numpy(), target_z, n_waypoints=n_steps
    )
    _run_pipeline(
        model=model,
        cache=cache,
        chosen=int(chosen),
        player_idx=player_idx,
        n_steps=n_steps,
        top_k=top_k,
        strategy="linear",
        path_z_np=path_z_np,
    )


@cli.command("gradient-ascent")
@_global_options
@click.option(
    "--ga-steps",
    type=int,
    default=500,
    show_default=True,
    help="Max gradient ascent steps.",
)
@click.option(
    "--ga-lr", type=float, default=0.02, show_default=True, help="Learning rate."
)
@click.option(
    "--ga-momentum", type=float, default=0.9, show_default=True, help="Momentum."
)
@click.option(
    "--density-weight",
    type=float,
    default=0.3,
    show_default=True,
    help="Weight of the KDE density gradient.",
)
@click.option(
    "--kde-bandwidth",
    type=float,
    default=0.5,
    show_default=True,
    help="Gaussian KDE bandwidth.",
)
@click.option(
    "--convergence-threshold",
    type=float,
    default=0.95,
    show_default=True,
    help="P(win) threshold for early stopping.",
)
def cmd_gradient_ascent(
    model,
    cache,
    sample_idx,
    n_steps,
    top_k,
    ga_steps,
    ga_lr,
    ga_momentum,
    density_weight,
    kde_bandwidth,
    convergence_threshold,
):
    """Gradient ascent on P(win) regularised by a KDE density prior."""
    os.makedirs("output", exist_ok=True)

    print("Loading model and data...")
    vae, classifier, val_X, val_y, norm_mean, norm_std, _ = _load_model_and_data(
        model, cache
    )
    labels = val_y.numpy()
    labels_tensor = torch.tensor(labels)
    print(f"  Validation: {len(val_X)}")

    print("Encoding into latent space...")
    latents_p0 = _encode_player(vae, val_X[:, 0, :])
    latents_p1 = _encode_player(vae, val_X[:, 1, :])

    n = len(labels)
    chosen = (
        sample_idx
        if (sample_idx is not None and sample_idx < n)
        else torch.randint(n, (1,)).item()
    )
    player_idx = int(labels[chosen])  # 0 if p0 lost, 1 if p1 lost
    sample_z = latents_p0[chosen] if player_idx == 0 else latents_p1[chosen]
    opponent_z = latents_p1[chosen] if player_idx == 0 else latents_p0[chosen]
    print(
        f"  Sample idx: {chosen} (label={int(labels[chosen])}, loser=player {player_idx})"
    )

    # All loser latents as reference distribution for KDE.
    all_loss_latents = torch.where(
        (labels_tensor == 0).unsqueeze(1), latents_p0, latents_p1
    )

    score_fn = partial(
        _opponent_aware_score,
        classifier=classifier,
        opponent_z=opponent_z,
        player_idx=player_idx,
    )
    logit_fn = partial(
        _opponent_aware_logit,
        classifier=classifier,
        opponent_z=opponent_z,
        player_idx=player_idx,
    )

    print("Running gradient ascent with KDE density regularisation...")
    path_z_np = path_gradient_ascent(
        sample_z.detach().cpu().numpy(),
        score_fn=score_fn,
        logit_fn=logit_fn,
        Z_all=all_loss_latents.detach().cpu().numpy(),
        steps=ga_steps,
        lr=ga_lr,
        momentum=ga_momentum,
        density_weight=density_weight,
        kde_bandwidth=kde_bandwidth,
        n_waypoints=n_steps,
        convergence_threshold=convergence_threshold,
    )
    _run_pipeline(
        model=model,
        cache=cache,
        chosen=int(chosen),
        player_idx=player_idx,
        n_steps=n_steps,
        top_k=top_k,
        strategy="gradient_ascent",
        path_z_np=path_z_np,
    )


@cli.command("optimal-transport")
@_global_options
@click.option(
    "--ot-reg",
    type=float,
    default=0.0,
    show_default=True,
    help="Entropic regularisation (0 = exact EMD).",
)
def cmd_optimal_transport(model, cache, sample_idx, n_steps, top_k, ot_reg):
    """Wasserstein-barycentric path into the winning distribution."""
    os.makedirs("output", exist_ok=True)

    print("Loading model and data...")
    vae, classifier, val_X, val_y, norm_mean, norm_std, _ = _load_model_and_data(
        model, cache
    )
    labels = val_y.numpy()
    labels_tensor = torch.tensor(labels)
    print(f"  Validation: {len(val_X)}")

    print("Encoding into latent space...")
    latents_p0 = _encode_player(vae, val_X[:, 0, :])
    latents_p1 = _encode_player(vae, val_X[:, 1, :])

    # Win cloud: label=1 → p0 won; label=0 → p1 won.
    win_latents = torch.where((labels_tensor == 1).unsqueeze(1), latents_p0, latents_p1)

    n = len(labels)
    chosen = (
        sample_idx
        if (sample_idx is not None and sample_idx < n)
        else torch.randint(n, (1,)).item()
    )
    player_idx = int(labels[chosen])  # 0 if p0 lost, 1 if p1 lost
    sample_z = latents_p0[chosen] if player_idx == 0 else latents_p1[chosen]
    print(
        f"  Sample idx: {chosen} (label={int(labels[chosen])}, loser=player {player_idx})"
    )

    print("Computing optimal transport path...")
    path_z_np = path_optimal_transport(
        sample_z.detach().cpu().numpy(),
        win_latents.detach().cpu().numpy(),
        reg=ot_reg,
        n_waypoints=n_steps,
    )
    _run_pipeline(
        model=model,
        cache=cache,
        chosen=int(chosen),
        player_idx=player_idx,
        n_steps=n_steps,
        top_k=top_k,
        strategy="optimal_transport",
        path_z_np=path_z_np,
    )


@cli.command("geodesic")
@_global_options
@click.option(
    "--geodesic-k",
    type=int,
    default=12,
    show_default=True,
    help="Number of neighbours for the kNN graph.",
)
def cmd_geodesic(model, cache, sample_idx, n_steps, top_k, geodesic_k):
    """Shortest path on a kNN latent-space graph."""
    os.makedirs("output", exist_ok=True)

    print("Loading model and data...")
    vae, classifier, val_X, val_y, norm_mean, norm_std, _ = _load_model_and_data(
        model, cache
    )
    labels = val_y.numpy()
    labels_tensor = torch.tensor(labels)
    print(f"  Validation: {len(val_X)}")

    print("Encoding into latent space...")
    latents_p0 = _encode_player(vae, val_X[:, 0, :])
    latents_p1 = _encode_player(vae, val_X[:, 1, :])

    # Win cloud: label=1 → p0 won; label=0 → p1 won.
    win_latents = torch.where((labels_tensor == 1).unsqueeze(1), latents_p0, latents_p1)
    # All latents for kNN graph.
    all_latents = torch.cat([latents_p0, latents_p1], dim=0)

    n = len(labels)
    chosen = (
        sample_idx
        if (sample_idx is not None and sample_idx < n)
        else torch.randint(n, (1,)).item()
    )
    player_idx = int(labels[chosen])  # 0 if p0 lost, 1 if p1 lost
    sample_z = latents_p0[chosen] if player_idx == 0 else latents_p1[chosen]
    print(
        f"  Sample idx: {chosen} (label={int(labels[chosen])}, loser=player {player_idx})"
    )

    print("Computing geodesic path on kNN graph...")
    path_z_np = path_geodesic(
        sample_z.detach().cpu().numpy(),
        win_latents.detach().cpu().numpy(),
        all_latents.detach().cpu().numpy(),
        k=geodesic_k,
        n_waypoints=n_steps,
    )
    _run_pipeline(
        model=model,
        cache=cache,
        chosen=int(chosen),
        player_idx=player_idx,
        n_steps=n_steps,
        top_k=top_k,
        strategy="geodesic",
        path_z_np=path_z_np,
    )


if __name__ == "__main__":
    cli()
