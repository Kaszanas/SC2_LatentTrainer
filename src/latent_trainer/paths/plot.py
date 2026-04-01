# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_main(win_c, loss_c, path_c, alphas, win_probs, pca, save_path):
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


def plot_feature_delta(delta, feature_names, top_k, save_path):
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


def plot_feature_evolution(
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


def plot_distance(path_z, win_centroid, alphas, save_path):
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


def plot_three_signal_feedback(
    feedback: dict,
    feature_names,
    save_path: str,
    top_k: int = 10,
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
