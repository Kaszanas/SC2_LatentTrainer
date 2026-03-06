"""Feedback path finder: latent-space improvement guidance for SC2 players.

Loads the trained two-stage VAE, embeds player data, finds an improvement
path from a losing sample toward the winning class density, reconstructs
feature-level deltas, and produces visualisations.

Usage:
    uv run python feedback_path.py --sample-idx 0 --method centroid
    uv run python feedback_path.py --sample-idx 42 --method nearest --top-k 15
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from train_two_stage import SimpleVAE, LatentClassifier, load_and_normalize

# ---------------------------------------------------------------------------
# Feature names -- 203 per player
# ---------------------------------------------------------------------------

_ECON_FIELDS: list[str] = [
    "foodMade", "foodUsed",
    "mineralsCollectionRate", "mineralsCurrent",
    "mineralsFriendlyFireArmy", "mineralsFriendlyFireEconomy",
    "mineralsFriendlyFireTechnology",
    "mineralsKilledArmy", "mineralsKilledEconomy", "mineralsKilledTechnology",
    "mineralsLostArmy", "mineralsLostEconomy", "mineralsLostTechnology",
    "mineralsUsedActiveForces", "mineralsUsedCurrentArmy",
    "mineralsUsedCurrentEconomy", "mineralsUsedCurrentTechnology",
    "mineralsUsedInProgressArmy", "mineralsUsedInProgressEconomy",
    "mineralsUsedInProgressTechnology",
    "vespeneCollectionRate", "vespeneCurrent",
    "vespeneFriendlyFireArmy", "vespeneFriendlyFireEconomy",
    "vespeneFriendlyFireTechnology",
    "vespeneKilledArmy", "vespeneKilledEconomy", "vespeneKilledTechnology",
    "vespeneLostArmy", "vespeneLostEconomy", "vespeneLostTechnology",
    "vespeneUsedActiveForces", "vespeneUsedCurrentArmy",
    "vespeneUsedCurrentEconomy", "vespeneUsedCurrentTechnology",
    "vespeneUsedInProgressArmy", "vespeneUsedInProgressEconomy",
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
# Core logic
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


def _find_path(start_z, target_z, n_steps) -> torch.Tensor:
    alphas = torch.linspace(0.0, 1.0, n_steps).unsqueeze(1)
    return start_z + alphas * (target_z - start_z)


def _nearest_winning_target(sample_z, win_latents, k=5) -> torch.Tensor:
    dists = torch.cdist(sample_z.unsqueeze(0), win_latents.unsqueeze(0)).squeeze(0)
    _, indices = dists.topk(k, largest=False)
    return win_latents[indices.squeeze()].mean(dim=0)


@torch.no_grad()
def _win_probability_along_path(classifier, path_z, opponent_z, player_idx):
    n = path_z.shape[0]
    opp = opponent_z.unsqueeze(0).expand(n, -1)
    if player_idx == 0:
        combined = torch.cat([path_z, opp], dim=1)
    else:
        combined = torch.cat([opp, path_z], dim=1)
    return classifier(combined).squeeze(-1).numpy()


# ---------------------------------------------------------------------------
# Visualisation (matplotlib)
# ---------------------------------------------------------------------------


def _plot_main(win_c, loss_c, path_c, alphas, win_probs, pca, save_path):
    """Two-panel figure: latent space + P(win) curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Latent Space -- Counterfactual Improvement Path",
        fontsize=15, fontweight="bold",
    )

    # --- Left: latent space ---
    ax1.scatter(loss_c[:, 0], loss_c[:, 1], c="#e74c3c", alpha=0.2, s=5, label="Loss")
    ax1.scatter(win_c[:, 0], win_c[:, 1], c="#3498db", alpha=0.2, s=5, label="Win")

    ax1.plot(path_c[:, 0], path_c[:, 1], color="black", linewidth=2.5, zorder=5)
    ax1.scatter(
        path_c[1:-1, 0], path_c[1:-1, 1],
        c="gold", s=50, marker="D", edgecolors="black", linewidths=0.8,
        zorder=6, label="Improvement path",
    )
    ax1.scatter(
        path_c[0, 0], path_c[0, 1],
        c="red", s=140, marker="*", edgecolors="black", linewidths=1,
        zorder=7, label="New point (loss)",
    )
    ax1.scatter(
        path_c[-1, 0], path_c[-1, 1],
        c="blue", s=140, marker="*", edgecolors="black", linewidths=1,
        zorder=7, label="Target (win)",
    )

    for i in range(0, len(path_c) - 1, max(1, len(path_c) // 5)):
        ax1.annotate(
            "", xy=(path_c[i + 1, 0], path_c[i + 1, 1]),
            xytext=(path_c[i, 0], path_c[i, 1]),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )

    ax1.set_xlabel(f"z1 (PC1 {pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"z2 (PC2 {pca.explained_variance_ratio_[1]:.1%})")
    ax1.set_title("Latent space + improvement path", fontsize=11)
    ax1.legend(fontsize=8, markerscale=1.2, loc="best")
    ax1.grid(True, alpha=0.15)

    # --- Right: P(win) curve ---
    # Zone shading
    ax2.axhspan(0, 0.5, color="#e74c3c", alpha=0.08, label="Loss zone")
    ax2.axhspan(0.5, 1, color="#3498db", alpha=0.08, label="Win zone")
    ax2.axhline(0.5, color="grey", linewidth=1.2, linestyle="--", label="Decision boundary")

    ax2.plot(
        alphas, win_probs,
        color="#27ae60", linewidth=2.5, zorder=5,
    )
    ax2.scatter(
        alphas, win_probs,
        c="#27ae60", s=50, edgecolors="white", linewidths=0.8, zorder=6,
    )

    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("Transport interpolation alpha  (0=loss, 1=win target)")
    ax2.set_ylabel("P(win)")
    ax2.set_title("Win probability along path", fontsize=11)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path}")


def _plot_feature_delta(delta, top_k, save_path):
    order = np.argsort(np.abs(delta))[::-1][:top_k]
    names = [FEATURE_NAMES[i] for i in order]
    values = delta[order]
    colours = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(3.5, top_k * 0.4)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature change (target - current)")
    ax.set_title(f"Top-{top_k} Feature Deltas for Improvement", fontsize=13, fontweight="bold")
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path}")


def _plot_feature_evolution(path_features, delta, n_top, alphas, save_path):
    order = np.argsort(np.abs(delta))[::-1][:n_top]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx in order:
        ax.plot(alphas, path_features[:, idx], linewidth=2, label=FEATURE_NAMES[idx])
    ax.set_xlabel("Transport interpolation alpha  (0=loss, 1=win target)")
    ax.set_ylabel("Feature value (original scale)")
    ax.set_title(f"Top-{n_top} Feature Evolution Along Path", fontsize=13, fontweight="bold")
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
    ax.set_xlabel("Transport interpolation alpha  (0=loss, 1=win target)")
    ax.set_ylabel("Euclidean distance to winning centroid")
    ax.set_title("Distance to Win Region Along Path", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find a latent-space improvement path for a losing SC2 player."
    )
    parser.add_argument("--model", default="output/two_stage_model.pth")
    parser.add_argument("--cache", default="data/cached_dataset_rich.pt")
    parser.add_argument("--sample-idx", type=int, default=None,
                        help="Index of a losing sample (default: random).")
    parser.add_argument("--method", choices=["centroid", "nearest"], default="centroid",
                        help="'centroid' or 'nearest' k-NN target.")
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k-neighbours", type=int, default=5)
    parser.add_argument("--player", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    # --- Load ---
    print("Loading model and data...")
    vae, classifier, val_X, val_y, norm_mean, norm_std, latent_dim = (
        _load_model_and_data(args.model, args.cache)
    )
    player_idx = args.player - 1
    opponent_idx = 1 - player_idx
    player_data = val_X[:, player_idx, :]
    opponent_data = val_X[:, opponent_idx, :]
    labels = val_y.numpy()
    print(f"  Validation: {len(val_X)},  Player {args.player}")

    # --- Encode ---
    print("Encoding into latent space...")
    all_latents = _encode_player(vae, player_data)
    opp_latents = _encode_player(vae, opponent_data)

    win_mask = labels == 1
    win_latents = all_latents[win_mask]
    loss_latents = all_latents[~win_mask]
    win_centroid = win_latents.mean(dim=0)
    print(f"  Win: {win_latents.shape[0]}, Loss: {loss_latents.shape[0]}")

    # --- Pick sample ---
    loss_indices = torch.where(torch.tensor(labels) == 0)[0]
    if args.sample_idx is not None and args.sample_idx < len(loss_indices):
        chosen = loss_indices[args.sample_idx]
    else:
        chosen = loss_indices[torch.randint(len(loss_indices), (1,)).item()]

    sample_z = all_latents[chosen]
    opponent_z = opp_latents[chosen]
    print(f"  Sample idx: {chosen.item()} (label={int(labels[chosen])})")

    # --- Target ---
    if args.method == "centroid":
        target_z = win_centroid
        print("  Method: centroid")
    else:
        target_z = _nearest_winning_target(sample_z, win_latents, k=args.k_neighbours)
        print(f"  Method: nearest (k={args.k_neighbours})")

    # --- Path ---
    path_z = _find_path(sample_z, target_z, n_steps=args.n_steps)
    alphas = np.linspace(0.0, 1.0, args.n_steps)

    # --- P(win) ---
    print("Computing win probability along path...")
    win_probs = _win_probability_along_path(classifier, path_z, opponent_z, player_idx)
    print(f"  P(win): {win_probs[0]:.3f} -> {win_probs[-1]:.3f}")

    # --- Decode ---
    print("Decoding path to feature space...")
    path_features = _decode_features(vae, path_z, norm_mean, norm_std)
    delta = path_features[-1] - path_features[0]

    # --- Report ---
    order = np.argsort(np.abs(delta))[::-1][: args.top_k]
    print(f"\n{'=' * 65}")
    print(f"  TOP-{args.top_k} FEATURES TO IMPROVE  (method={args.method})")
    print(f"{'=' * 65}")
    for rank, idx in enumerate(order, 1):
        d = "[+]" if delta[idx] > 0 else "[-]"
        print(
            f"  {rank:>2}. {FEATURE_NAMES[idx]:<45s}  "
            f"{d} {delta[idx]:>+10.2f}  "
            f"({path_features[0][idx]:.1f} -> {path_features[-1][idx]:.1f})"
        )
    print(f"{'=' * 65}\n")

    # --- PCA ---
    all_data = np.concatenate([win_latents.numpy(), loss_latents.numpy(), path_z.numpy()])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_data)
    n_w, n_l = len(win_latents), len(loss_latents)
    win_c, loss_c, path_c = coords[:n_w], coords[n_w:n_w + n_l], coords[n_w + n_l:]

    # --- Plots ---
    print("Generating plots...")
    _plot_main(win_c, loss_c, path_c, alphas, win_probs, pca, "output/feedback_latent_path.png")
    _plot_feature_delta(delta, args.top_k, "output/feedback_feature_delta.png")
    _plot_feature_evolution(path_features, delta, min(5, args.top_k), alphas, "output/feedback_feature_evolution.png")
    _plot_distance(path_z, win_centroid, alphas, "output/feedback_distance_curve.png")
    print("\nDone! All plots saved to output/")


if __name__ == "__main__":
    main()
