"""Visualize transformer attention patterns for SC2 match prediction.

Shows which feature groups the model attends to, how players interact
through cross-attention, and which features matter most for classification.

Usage:
    uv run python visualize_attention.py
    uv run python visualize_attention.py --n-samples 200
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from train_transformer import SC2Transformer
from latent_trainer.data_utils import load_and_normalize
from latent_trainer.features.feature_groups import FEATURE_GROUPS, NUM_GROUPS


# Token labels for visualization
TOKEN_LABELS = (
    ["[CLS]"] +
    [f"P1:{name}" for name, _, _ in FEATURE_GROUPS] +
    [f"P2:{name}" for name, _, _ in FEATURE_GROUPS]
)

SHORT_LABELS = (
    ["CLS"] +
    [f"P1:{n[:6]}" for n, _, _ in FEATURE_GROUPS] +
    [f"P2:{n[:6]}" for n, _, _ in FEATURE_GROUPS]
)


def extract_attention_weights(model, x):
    """Run forward pass and capture attention weights from all layers.
    
    Returns:
        attention_maps: list of [batch, n_heads, 17, 17] tensors, one per layer
        predictions: [batch, 1]
    """
    model.eval()
    attention_maps = []
    hooks = []

    # Register hooks on each attention layer
    for layer in model.transformer.layers:
        def hook_fn(module, input, output, attn_maps=attention_maps):
            # For TransformerEncoderLayer, we need to hook into the self_attn
            pass

        # Hook the self-attention module directly
        def make_attn_hook(layer_idx):
            def attn_hook(module, args, kwargs, output):
                # Re-run attention with need_weights=True
                pass
            return attn_hook

    # Alternative: manually compute attention
    with torch.no_grad():
        B = x.shape[0]
        device = x.device

        # Tokenize (replicate model.forward logic)
        p1_tokens = model.tokenizer(x[:, 0, :])
        p2_tokens = model.tokenizer(x[:, 1, :])

        player1_ids = torch.ones(B, NUM_GROUPS, dtype=torch.long, device=device)
        player2_ids = torch.full((B, NUM_GROUPS), 2, dtype=torch.long, device=device)
        p1_tokens = p1_tokens + model.player_embedding(player1_ids)
        p2_tokens = p2_tokens + model.player_embedding(player2_ids)

        positions = torch.arange(NUM_GROUPS, device=device).unsqueeze(0).expand(B, -1)
        p1_tokens = p1_tokens + model.position_embedding(positions)
        p2_tokens = p2_tokens + model.position_embedding(positions)

        cls_tokens = model.cls_token.expand(B, -1, -1)
        cls_player = torch.zeros(B, 1, dtype=torch.long, device=device)
        cls_tokens = cls_tokens + model.player_embedding(cls_player)

        sequence = torch.cat([cls_tokens, p1_tokens, p2_tokens], dim=1)

        # Run through each transformer layer manually to capture attention
        current = sequence
        for layer in model.transformer.layers:
            # Pre-norm
            normed = layer.norm1(current)

            # Self-attention with weights
            attn_output, attn_weights = layer.self_attn(
                normed, normed, normed,
                need_weights=True,
                average_attn_weights=False  # Get per-head weights
            )
            current = current + layer.dropout1(attn_output)

            # FFN
            normed2 = layer.norm2(current)
            current = current + layer.dropout2(layer.linear2(
                layer.dropout(layer.activation(layer.linear1(normed2)))
            ))

            attention_maps.append(attn_weights)  # [B, n_heads, 17, 17]

        # Final classification
        cls_output = current[:, 0, :]
        predictions = model.classifier(cls_output)

    return attention_maps, predictions


def plot_attention_heatmaps(attention_maps, save_path):
    """Plot average attention pattern per layer."""
    n_layers = len(attention_maps)
    n_heads = attention_maps[0].shape[1]

    # Average across samples and heads
    fig, axes = plt.subplots(1, n_layers, figsize=(7 * n_layers, 6))
    if n_layers == 1:
        axes = [axes]

    for layer_idx, attn in enumerate(attention_maps):
        avg_attn = attn.mean(dim=(0, 1)).numpy()  # [17, 17]
        ax = axes[layer_idx]
        im = ax.imshow(avg_attn, cmap='Blues', aspect='auto')
        ax.set_xticks(range(17))
        ax.set_yticks(range(17))
        ax.set_xticklabels(SHORT_LABELS, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(SHORT_LABELS, fontsize=7)
        ax.set_title(f"Layer {layer_idx + 1}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Key (attends to)")
        ax.set_ylabel("Query (attends from)")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Attention Patterns (averaged over samples and heads)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_cls_attention(attention_maps, save_path):
    """Plot what [CLS] token attends to — this is what drives the prediction."""
    n_layers = len(attention_maps)
    n_heads = attention_maps[0].shape[1]

    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 3 * n_layers))
    if n_layers == 1:
        axes = [axes]

    group_names = [n for n, _, _ in FEATURE_GROUPS]

    for layer_idx, attn in enumerate(attention_maps):
        # CLS is query at position 0
        cls_attn = attn[:, :, 0, :].mean(dim=0)  # [n_heads, 17]

        ax = axes[layer_idx]
        x = np.arange(17)
        width = 0.8 / n_heads

        for head_idx in range(n_heads):
            offset = (head_idx - n_heads / 2 + 0.5) * width
            bars = ax.bar(x + offset, cls_attn[head_idx].numpy(), width,
                          label=f'Head {head_idx + 1}', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_LABELS, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Attention Weight")
        ax.set_title(f"Layer {layer_idx + 1}: [CLS] Attention (what drives prediction)",
                      fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, ncol=n_heads)
        ax.grid(True, alpha=0.2, axis='y')

        # Add vertical line separating P1 and P2
        ax.axvline(x=8.5, color='red', linestyle='--', alpha=0.5, label='P1|P2 boundary')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_cross_player_attention(attention_maps, save_path):
    """Analyze how much players attend to each other vs themselves."""
    n_layers = len(attention_maps)

    # Define token regions
    cls_idx = [0]
    p1_range = list(range(1, 9))
    p2_range = list(range(9, 17))

    categories = ['Self (P1→P1)', 'Cross (P1→P2)', 'Self (P2→P2)', 'Cross (P2→P1)',
                   'CLS→P1', 'CLS→P2']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#e67e22', '#9b59b6', '#f39c12']

    fig, ax = plt.subplots(figsize=(10, 5))

    layer_data = {cat: [] for cat in categories}

    for layer_idx, attn in enumerate(attention_maps):
        avg = attn.mean(dim=(0, 1)).numpy()  # [17, 17]

        # P1 self-attention
        p1_self = avg[np.ix_(p1_range, p1_range)].mean()
        # P1 cross-attention to P2
        p1_cross = avg[np.ix_(p1_range, p2_range)].mean()
        # P2 self-attention
        p2_self = avg[np.ix_(p2_range, p2_range)].mean()
        # P2 cross-attention to P1
        p2_cross = avg[np.ix_(p2_range, p1_range)].mean()
        # CLS to P1
        cls_p1 = avg[0, p1_range].mean()
        # CLS to P2
        cls_p2 = avg[0, p2_range].mean()

        for cat, val in zip(categories, [p1_self, p1_cross, p2_self, p2_cross, cls_p1, cls_p2]):
            layer_data[cat].append(val)

    x = np.arange(n_layers)
    width = 0.12
    for i, (cat, vals) in enumerate(layer_data.items()):
        offset = (i - len(categories) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=cat, color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {i+1}" for i in range(n_layers)])
    ax.set_ylabel("Avg Attention Weight")
    ax.set_title("Self vs Cross-Player Attention by Layer", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_feature_importance(attention_maps, save_path):
    """Aggregate attention to show which feature groups matter most for prediction."""
    # Use the last layer's CLS attention as a proxy for feature importance
    last_attn = attention_maps[-1]  # [B, heads, 17, 17]
    cls_attn = last_attn[:, :, 0, 1:].mean(dim=(0, 1)).numpy()  # [16] — skip CLS self-attn

    group_names = [n.replace('_', '\n') for n, _, _ in FEATURE_GROUPS]

    # Separate P1 and P2
    p1_importance = cls_attn[:8]
    p2_importance = cls_attn[8:]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(group_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, p1_importance, width, label='Player 1',
                    color='#3498db', alpha=0.85)
    bars2 = ax.bar(x + width/2, p2_importance, width, label='Player 2',
                    color='#e74c3c', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=9)
    ax.set_ylabel("CLS Attention Weight (last layer)")
    ax.set_title("Feature Group Importance for Prediction", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="output/transformer_best.pth")
    parser.add_argument("--cache", default="data/cached_dataset_rich.pt")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of val samples to visualize")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    # Load model
    model = SC2Transformer(
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers
    )
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.eval()

    # Load data (use validation set)
    _, _, val_X, val_y = load_and_normalize(args.cache)
    val_X = val_X[:args.n_samples]
    val_y = val_y[:args.n_samples]

    print(f"Extracting attention from {len(val_X)} validation samples...")
    attention_maps, predictions = extract_attention_weights(model, val_X)

    pred_acc = ((predictions.squeeze() > 0.5).float() == val_y).float().mean() * 100
    print(f"  Prediction accuracy on subset: {pred_acc:.1f}%")

    print("\nGenerating visualizations...")
    plot_attention_heatmaps(attention_maps, "output/attn_heatmaps.png")
    plot_cls_attention(attention_maps, "output/attn_cls_focus.png")
    plot_cross_player_attention(attention_maps, "output/attn_cross_player.png")
    plot_feature_importance(attention_maps, "output/attn_feature_importance.png")

    print("\nDone! All plots saved to output/")


if __name__ == "__main__":
    main()
