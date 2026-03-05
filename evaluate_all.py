"""Evaluate all classifiers with comprehensive metrics.

Computes F1-score, MCC, balanced accuracy, precision, recall, specificity,
and confusion matrices for all trained models on the validation set.

Models evaluated:
    1. Diagnostic MLP (standalone classifier on raw features)
    2. Two-stage VAE classifier (frozen encoder + MLP on latent space)
    3. Transformer classifier (self-attention over feature groups)

Usage:
    uv run python evaluate_all.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import json

# --- Import models ---
from train_two_stage import SimpleVAE, LatentClassifier, load_and_normalize, extract_latents
from train_transformer import SC2Transformer


def load_data(cache_path="data/cached_dataset_rich.pt"):
    """Load and normalize data, return train/val sets."""
    train_X, train_y, val_X, val_y = load_and_normalize(cache_path)[:4]
    return train_X, train_y, val_X, val_y


def get_predictions(model, data_loader, device="cpu"):
    """Get raw probabilities and binary predictions from a model."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            prob = model(x_batch)
            all_probs.append(prob.cpu().squeeze())
            all_labels.append(y_batch.squeeze())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs > 0.5).astype(float)
    return probs, preds, labels


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall (Sensitivity)": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def print_metrics(name, metrics):
    """Pretty-print metrics for one model."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for key in ["Accuracy", "Balanced Accuracy", "F1-Score", "MCC",
                 "Precision", "Recall (Sensitivity)", "Specificity", "ROC-AUC"]:
        print(f"  {key:25s}: {metrics[key]:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"      Pred 0    Pred 1")
    print(f"  Actual 0:  {metrics['TN']:5d}    {metrics['FP']:5d}")
    print(f"  Actual 1:  {metrics['FN']:5d}    {metrics['TP']:5d}")


def build_and_train_mlp(train_X, train_y, val_X, val_y):
    """Train a fresh diagnostic MLP and return the model."""
    # Use "Combined" strategy (best from diagnostic_classifier.py)
    train_flat = train_X.reshape(train_X.size(0), -1)
    val_flat = val_X.reshape(val_X.size(0), -1)
    train_diff = train_X[:, 0, :] - train_X[:, 1, :]
    val_diff = val_X[:, 0, :] - val_X[:, 1, :]
    trX = torch.cat([train_flat, train_diff], dim=1)
    vaX = torch.cat([val_flat, val_diff], dim=1)

    input_dim = trX.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
        nn.Linear(64, 1), nn.Sigmoid(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()

    train_loader = DataLoader(TensorDataset(trX, train_y.unsqueeze(1)), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(vaX, val_y.unsqueeze(1)), batch_size=256, shuffle=False)

    best_val_acc = 0
    best_state = None
    no_improve = 0

    print("  Training diagnostic MLP (Combined [609])...")
    for epoch in range(200):
        model.train()
        for x_b, y_b in train_loader:
            pred = model(x_b)
            loss = criterion(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                pred = model(x_b)
                val_correct += ((pred > 0.5).float() == y_b).sum().item()
                val_total += y_b.numel()
                val_loss += criterion(pred, y_b).item()

        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 15:
            break

    model.load_state_dict(best_state)
    print(f"  Done (best val acc: {best_val_acc*100:.2f}%)")
    return model, val_loader


def load_two_stage(val_X, val_y, device="cpu"):
    """Load two-stage VAE + classifier and get predictions on validation set."""
    checkpoint = torch.load("output/two_stage_model.pth", weights_only=True)
    latent_dim = checkpoint['latent_dim']
    input_dim = checkpoint['input_dim']

    vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    vae.load_state_dict(checkpoint['vae_state'])

    classifier = LatentClassifier(latent_dim=latent_dim).to(device)
    classifier.load_state_dict(checkpoint['classifier_state'])

    # Extract latents
    val_z = extract_latents(vae, val_X, device)
    val_loader = DataLoader(
        TensorDataset(val_z, val_y.unsqueeze(1)), batch_size=256, shuffle=False
    )

    return classifier, val_loader


def load_transformer(val_X, val_y, d_model=64, n_heads=4, n_layers=3):
    """Load trained transformer and prepare dataloader."""
    model = SC2Transformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
    model.load_state_dict(torch.load("output/transformer_best.pth", weights_only=True))
    val_loader = DataLoader(
        TensorDataset(val_X, val_y.unsqueeze(1)), batch_size=256, shuffle=False
    )
    return model, val_loader


def plot_comparison(results, save_path):
    """Create comparison bar chart of key metrics across models."""
    metrics_to_plot = ["Accuracy", "Balanced Accuracy", "F1-Score", "MCC",
                       "Precision", "Recall (Sensitivity)", "Specificity", "ROC-AUC"]

    model_names = list(results.keys())
    n_metrics = len(metrics_to_plot)
    n_models = len(model_names)

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    for i, model_name in enumerate(model_names):
        values = [results[model_name][m] for m in metrics_to_plot]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name,
                      color=colors[i % len(colors)], alpha=0.85)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classification Metrics Comparison Across All Models",
                  fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.set_ylim(0, 1.12)
    ax.grid(True, alpha=0.2, axis='y')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Comparison chart saved: {save_path}")
    plt.close()


def plot_confusion_matrices(results, save_path):
    """Plot confusion matrices side by side."""
    model_names = list(results.keys())
    n = len(model_names)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        m = results[name]
        cm = np.array([[m['TN'], m['FP']], [m['FN'], m['TP']]])

        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_title(name, fontsize=11, fontweight='bold')

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]}',
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Loss', 'Win'])
        ax.set_yticklabels(['Loss', 'Win'])
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Confusion Matrices", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Confusion matrices saved: {save_path}")
    plt.close()


def main():
    os.makedirs("output", exist_ok=True)
    device = torch.device("cpu")

    print("Loading dataset...")
    train_X, train_y, val_X, val_y = load_data()
    print(f"  Train: {train_X.shape}, Val: {val_X.shape}")
    print(f"  Label balance — Train: {train_y.mean():.3f}, Val: {val_y.mean():.3f}")

    results = {}

    # --- 1. Diagnostic MLP ---
    print(f"\n{'#'*60}")
    print(f"  MODEL 1: Diagnostic MLP (Combined [609])")
    print(f"{'#'*60}")
    mlp_model, mlp_loader = build_and_train_mlp(train_X, train_y, val_X, val_y)
    probs, preds, labels = get_predictions(mlp_model, mlp_loader)
    results["Diagnostic MLP"] = compute_metrics(labels, preds, probs)
    print_metrics("Diagnostic MLP", results["Diagnostic MLP"])

    # --- 2. Two-Stage VAE ---
    print(f"\n{'#'*60}")
    print(f"  MODEL 2: Two-Stage VAE Classifier")
    print(f"{'#'*60}")
    if os.path.exists("output/two_stage_model.pth"):
        vae_model, vae_loader = load_two_stage(val_X, val_y, device)
        probs, preds, labels = get_predictions(vae_model, vae_loader)
        results["Two-Stage VAE"] = compute_metrics(labels, preds, probs)
        print_metrics("Two-Stage VAE", results["Two-Stage VAE"])
    else:
        print("  SKIPPED — output/two_stage_model.pth not found")
        print("  Run: uv run python train_two_stage.py")

    # --- 3. Transformer ---
    print(f"\n{'#'*60}")
    print(f"  MODEL 3: Transformer Classifier")
    print(f"{'#'*60}")
    if os.path.exists("output/transformer_best.pth"):
        tf_model, tf_loader = load_transformer(val_X, val_y)
        probs, preds, labels = get_predictions(tf_model, tf_loader)
        results["Transformer"] = compute_metrics(labels, preds, probs)
        print_metrics("Transformer", results["Transformer"])
    else:
        print("  SKIPPED — output/transformer_best.pth not found")
        print("  Run: uv run python train_transformer.py")

    # --- Summary table ---
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY: All Models Comparison")
    print(f"{'='*80}")

    headers = ["Model", "Acc", "Bal.Acc", "F1", "MCC", "Prec", "Recall", "Spec", "AUC"]
    row_fmt = "{:20s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}"
    print(row_fmt.format(*headers))
    print("-" * 80)

    for name, m in results.items():
        print(row_fmt.format(
            name,
            f"{m['Accuracy']:.4f}",
            f"{m['Balanced Accuracy']:.4f}",
            f"{m['F1-Score']:.4f}",
            f"{m['MCC']:.4f}",
            f"{m['Precision']:.4f}",
            f"{m['Recall (Sensitivity)']:.4f}",
            f"{m['Specificity']:.4f}",
            f"{m['ROC-AUC']:.4f}",
        ))

    # --- Plots ---
    plot_comparison(results, "output/metrics_comparison.png")
    plot_confusion_matrices(results, "output/confusion_matrices.png")

    # --- Save JSON ---
    json_path = "output/evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {json_path}")


if __name__ == "__main__":
    main()
