"""Visualize the VAE latent space using t-SNE and PCA.

Loads the trained two-stage model, extracts latent representations,
and creates 2D visualizations colored by match outcome.

Usage:
    uv run python visualize_latent.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from train_two_stage import SimpleVAE, load_and_normalize


def main():
    # Load model
    checkpoint = torch.load("output/two_stage_model.pth", weights_only=False)
    latent_dim = checkpoint["latent_dim"]
    input_dim = checkpoint["input_dim"]

    vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)
    vae.load_state_dict(checkpoint["vae_state"])
    vae.eval()

    # Load and normalize data
    train_X, train_y, val_X, val_y, _, _ = load_and_normalize(
        "data/cached_dataset_rich.pt"
    )

    # Use validation set for cleaner visualization
    print(f"Extracting latents for {len(val_X)} validation samples...")

    all_latents_p1 = []
    all_latents_p2 = []

    with torch.no_grad():
        for player_idx in range(2):
            player_data = val_X[:, player_idx, :]
            for i in range(0, len(player_data), 256):
                batch = player_data[i : i + 256]
                mu, _ = vae.encode(batch)
                if player_idx == 0:
                    all_latents_p1.append(mu)
                else:
                    all_latents_p2.append(mu)

    latents_p1 = torch.cat(all_latents_p1).numpy()  # [N, 32]
    latents_p2 = torch.cat(all_latents_p2).numpy()
    labels = val_y.numpy()

    # Concatenate both players for classification view
    latents_concat = np.concatenate([latents_p1, latents_p2], axis=1)  # [N, 64]

    # Player difference (interesting for symmetry)
    latents_diff = latents_p1 - latents_p2  # [N, 32]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle("VAE Latent Space Visualization", fontsize=16, fontweight="bold")

    views = [
        ("Player 1 Latent", latents_p1),
        ("Player 2 Latent", latents_p2),
        ("Player Difference (P1-P2)", latents_diff),
    ]

    # Row 1: PCA
    for idx, (title, data) in enumerate(views):
        ax = axes[0, idx]
        pca = PCA(n_components=2)
        coords = pca.fit_transform(data)

        win_mask = labels == 1
        lose_mask = labels == 0

        ax.scatter(
            coords[lose_mask, 0],
            coords[lose_mask, 1],
            c="#e74c3c",
            alpha=0.3,
            s=8,
            label=f"Loss ({lose_mask.sum()})",
        )
        ax.scatter(
            coords[win_mask, 0],
            coords[win_mask, 1],
            c="#2ecc71",
            alpha=0.3,
            s=8,
            label=f"Win ({win_mask.sum()})",
        )

        ax.set_title(f"PCA — {title}", fontsize=11)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(fontsize=9, markerscale=3)
        ax.grid(True, alpha=0.2)

    # Row 2: t-SNE
    print("Running t-SNE (this takes ~30s)...")
    for idx, (title, data) in enumerate(views):
        ax = axes[1, idx]
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            random_state=42,
            max_iter=1000,
            learning_rate="auto",
            init="pca",
        )
        coords = tsne.fit_transform(data)

        win_mask = labels == 1
        lose_mask = labels == 0

        ax.scatter(
            coords[lose_mask, 0],
            coords[lose_mask, 1],
            c="#e74c3c",
            alpha=0.3,
            s=8,
            label=f"Loss ({lose_mask.sum()})",
        )
        ax.scatter(
            coords[win_mask, 0],
            coords[win_mask, 1],
            c="#2ecc71",
            alpha=0.3,
            s=8,
            label=f"Win ({win_mask.sum()})",
        )

        ax.set_title(f"t-SNE — {title}", fontsize=11)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=9, markerscale=3)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("output/latent_space.png", dpi=150, bbox_inches="tight")
    print("Saved to output/latent_space.png")
    plt.close()

    # Also create a focused view: t-SNE on the concatenated latents (what the classifier sees)
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    print("Running t-SNE on concatenated latents...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    coords = tsne.fit_transform(latents_concat)

    win_mask = labels == 1
    lose_mask = labels == 0

    ax2.scatter(
        coords[lose_mask, 0],
        coords[lose_mask, 1],
        c="#e74c3c",
        alpha=0.3,
        s=12,
        label=f"Player 1 Loss ({lose_mask.sum()})",
    )
    ax2.scatter(
        coords[win_mask, 0],
        coords[win_mask, 1],
        c="#2ecc71",
        alpha=0.3,
        s=12,
        label=f"Player 1 Win ({win_mask.sum()})",
    )

    ax2.set_title(
        "t-SNE — Concatenated Latent Space (Classifier Input)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("t-SNE 1", fontsize=12)
    ax2.set_ylabel("t-SNE 2", fontsize=12)
    ax2.legend(fontsize=11, markerscale=3)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("output/latent_space_concat.png", dpi=150, bbox_inches="tight")
    print("Saved to output/latent_space_concat.png")
    plt.close()


if __name__ == "__main__":
    main()
