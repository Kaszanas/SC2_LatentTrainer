"""Overfitting & feature leakage analysis.

Two analyses:
1. K-Fold Cross-Validation — checks if results are stable across splits
2. Feature Ablation — removes feature groups one by one to identify
   which groups the model actually needs. If removing "final_state"
   drops accuracy to ~50%, the model was just reading the outcome.

Usage:
    uv run python analyze_overfitting.py
    uv run python analyze_overfitting.py --folds 5 --epochs 60
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from latent_trainer.features.feature_groups import FEATURE_GROUPS

from latent_trainer.data_utils import normalize


def load_all_data(cache_path="data/cached_dataset_rich.pt"):
    """Load and combine train+val for k-fold, return raw (unnormalized)."""
    cached = torch.load(cache_path, weights_only=True)
    # Combine train and val for proper k-fold
    all_X = torch.cat([cached['train_features'].float(),
                       cached['val_features'].float()], dim=0)
    all_y = torch.cat([cached['train_labels'].float(),
                       cached['val_labels'].float()], dim=0)
    return all_X, all_y





def build_mlp(input_dim):
    """Build diagnostic MLP."""
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
        nn.Linear(64, 1), nn.Sigmoid(),
    )


def train_and_eval(model, train_X, train_y, val_X, val_y, epochs=60, batch_size=256):
    """Train model and return val metrics."""
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()

    train_loader = DataLoader(TensorDataset(train_X, train_y.unsqueeze(1)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_y.unsqueeze(1)),
                            batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for x_b, y_b in train_loader:
            pred = model(x_b)
            loss = criterion(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                val_loss += criterion(model(x_b), y_b).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 12:
            break

    model.load_state_dict(best_state)
    model.eval()

    # Compute metrics
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            all_probs.append(model(x_b).squeeze())
            all_labels.append(y_b.squeeze())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs > 0.5).astype(float)

    return {
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "mcc": float(matthews_corrcoef(labels, preds)),
    }


def prepare_features(X, strategy="combined"):
    """Prepare features from [N, 2, F] to classification input."""
    flat = X.reshape(X.size(0), -1)
    diff = X[:, 0, :] - X[:, 1, :]
    if strategy == "combined":
        return torch.cat([flat, diff], dim=1)
    elif strategy == "flat":
        return flat
    else:
        return diff


# ═══════════════════════════════════════════════════════════
# ANALYSIS 1: K-FOLD CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════
def run_kfold(all_X, all_y, n_folds=5, epochs=60):
    """K-fold cross-validation to check stability."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 1: {n_folds}-Fold Cross-Validation")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_X, all_y)):
        train_X, val_X = all_X[train_idx], all_X[val_idx]
        train_y, val_y = all_y[train_idx], all_y[val_idx]

        # Normalize per fold
        train_X, val_X = normalize(train_X, val_X)[:2]

        # Prepare features
        train_feat = prepare_features(train_X)
        val_feat = prepare_features(val_X)

        model = build_mlp(train_feat.shape[1])
        metrics = train_and_eval(model, train_feat, train_y, val_feat, val_y, epochs=epochs)
        fold_results.append(metrics)

        print(f"  Fold {fold+1}: Acc={metrics['accuracy']:.4f}, "
              f"Bal.Acc={metrics['balanced_accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}, MCC={metrics['mcc']:.4f}")

    # Summary
    print(f"\n  {'-'*50}")
    for metric in ['accuracy', 'balanced_accuracy', 'f1', 'mcc']:
        values = [r[metric] for r in fold_results]
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {metric:20s}: {mean:.4f} ± {std:.4f}  "
              f"(range: {min(values):.4f} - {max(values):.4f})")

    return fold_results


# ═══════════════════════════════════════════════════════════
# ANALYSIS 2: FEATURE GROUP ABLATION
# ═══════════════════════════════════════════════════════════
def run_ablation(train_X_raw, train_y, val_X_raw, val_y, epochs=60):
    """Remove each feature group and measure impact."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 2: Feature Group Ablation")
    print(f"  (remove one group at a time → measure accuracy drop)")
    print(f"{'='*70}")

    # Baseline with all features
    train_X_norm, val_X_norm = normalize(train_X_raw.clone(), val_X_raw.clone())[:2]
    train_feat = prepare_features(train_X_norm)
    val_feat = prepare_features(val_X_norm)
    model = build_mlp(train_feat.shape[1])
    baseline = train_and_eval(model, train_feat, train_y, val_feat, val_y, epochs=epochs)
    print(f"\n  Baseline (all features): Acc={baseline['accuracy']:.4f}, "
          f"F1={baseline['f1']:.4f}, MCC={baseline['mcc']:.4f}")

    ablation_results = {}

    for group_name, start, end in FEATURE_GROUPS:
        # Zero out this feature group for both players
        train_X_abl = train_X_raw.clone()
        val_X_abl = val_X_raw.clone()
        train_X_abl[:, :, start:end] = 0
        val_X_abl[:, :, start:end] = 0

        train_X_norm, val_X_norm = normalize(train_X_abl, val_X_abl)[:2]
        train_feat = prepare_features(train_X_norm)
        val_feat = prepare_features(val_X_norm)
        model = build_mlp(train_feat.shape[1])
        metrics = train_and_eval(model, train_feat, train_y, val_feat, val_y, epochs=epochs)

        drop = baseline['accuracy'] - metrics['accuracy']
        ablation_results[group_name] = metrics
        ablation_results[group_name]['accuracy_drop'] = drop

        print(f"  Remove {group_name:18s} ({end-start:3d} feats): "
              f"Acc={metrics['accuracy']:.4f} (Δ={drop:+.4f}), "
              f"F1={metrics['f1']:.4f}, MCC={metrics['mcc']:.4f}")

    return baseline, ablation_results


# ═══════════════════════════════════════════════════════════
# ANALYSIS 3: PROGRESSIVE FEATURE REMOVAL (cumulative)
# ═══════════════════════════════════════════════════════════
def run_progressive(train_X_raw, train_y, val_X_raw, val_y, epochs=60):
    """Remove feature groups from most to least 'leaky' (post-game first)."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 3: Progressive Feature Removal")
    print(f"  (cumulative removal, most post-game → most pre-game)")
    print(f"{'='*70}")

    # Order from most post-game (leaky) to most pre-game (fair)
    removal_order = [
        ("final_state",    117, 156, "End-game economy snapshot"),
        ("late_economy",    78, 117, "Late-game economy (frames 15000+)"),
        ("economy_delta",  156, 195, "Economy rate of change"),
        ("unit_activity",  199, 201, "Units born/killed totals"),
        ("game_info",      201, 203, "Upgrades + duration"),
        ("mid_economy",     39,  78, "Mid-game economy (frames 5000-15000)"),
        ("meta_stats",     195, 199, "APM, MMR, SQ, supplyCapped%"),
        ("early_economy",    0,  39, "Early-game economy (frames 0-5000)"),
    ]

    removed_so_far = []
    train_X_prog = train_X_raw.clone()
    val_X_prog = val_X_raw.clone()

    for group_name, start, end, desc in removal_order:
        # Zero out this group
        train_X_prog[:, :, start:end] = 0
        val_X_prog[:, :, start:end] = 0
        removed_so_far.append(group_name)

        train_norm, val_norm = normalize(train_X_prog.clone(), val_X_prog.clone())[:2]
        train_feat = prepare_features(train_norm)
        val_feat = prepare_features(val_norm)
        model = build_mlp(train_feat.shape[1])
        metrics = train_and_eval(model, train_feat, train_y, val_feat, val_y, epochs=epochs)

        remaining = len(FEATURE_GROUPS) - len(removed_so_far)
        print(f"  Removed: {group_name:18s} → {remaining} groups left: "
              f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, MCC={metrics['mcc']:.4f}")


# ═══════════════════════════════════════════════════════════
# ANALYSIS 4: EARLY-GAME ONLY (can the model predict with just early data?)
# ═══════════════════════════════════════════════════════════
def run_early_only(train_X_raw, train_y, val_X_raw, val_y, epochs=60):
    """Use ONLY early + meta features (no post-game info)."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 4: Early-Game + Meta Only")
    print(f"  (only features available BEFORE game outcome is known)")
    print(f"{'='*70}")

    configs = {
        "Meta stats only (APM, MMR, SQ)": [("meta_stats", 195, 199)],
        "Early economy only": [("early_economy", 0, 39)],
        "Early + meta": [("early_economy", 0, 39), ("meta_stats", 195, 199)],
        "Early + mid economy": [("early_economy", 0, 39), ("mid_economy", 39, 78)],
        "Early + mid + meta": [("early_economy", 0, 39), ("mid_economy", 39, 78),
                                ("meta_stats", 195, 199)],
    }

    for name, groups in configs.items():
        # Select only these feature columns
        selected = []
        for _, start, end in groups:
            selected.extend(range(start, end))
        selected = sorted(selected)

        train_X_sel = train_X_raw[:, :, selected]
        val_X_sel = val_X_raw[:, :, selected]

        train_norm, val_norm = normalize(train_X_sel, val_X_sel)[:2]
        train_feat = prepare_features(train_norm)
        val_feat = prepare_features(val_norm)
        model = build_mlp(train_feat.shape[1])
        metrics = train_and_eval(model, train_feat, train_y, val_feat, val_y, epochs=epochs)

        print(f"  {name:35s} ({len(selected):3d} feats): "
              f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
              f"MCC={metrics['mcc']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/cached_dataset_rich.pt")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    # Load all data
    all_X, all_y = load_all_data(args.cache)
    print(f"Total samples: {len(all_X)}")
    print(f"Feature shape: {all_X.shape}")
    print(f"Label balance: {all_y.mean():.3f}")

    # Use original train/val split for ablation analyses
    cached = torch.load(args.cache, weights_only=True)
    train_X = cached['train_features'].float()
    train_y = cached['train_labels'].float()
    val_X = cached['val_features'].float()
    val_y = cached['val_labels'].float()

    # Run all analyses
    run_kfold(all_X, all_y, n_folds=args.folds, epochs=args.epochs)
    run_ablation(train_X, train_y, val_X, val_y, epochs=args.epochs)
    run_progressive(train_X, train_y, val_X, val_y, epochs=args.epochs)
    run_early_only(train_X, train_y, val_X, val_y, epochs=args.epochs)

    print(f"\n{'='*70}")
    print("  INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print("""
  If k-fold scores are tight (low std), the model is NOT overfitting
  the specific train/val split.

  If removing 'final_state' crashes accuracy, the model was largely
  reading end-game outcomes — high accuracy but questionable predictive value.

  If 'early_only' or 'meta_only' still show decent accuracy (>60-70%),
  the model has genuine predictive power from pre-game/early-game features.

  For a real-time prediction system, only early_economy + meta_stats
  would be available. The other features require the game to be over.
""")


if __name__ == "__main__":
    main()
