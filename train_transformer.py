"""Transformer-based SC2 match outcome classifier.

Architecture:
    1. Feature Tokenizer — splits 203 features per player into semantic groups,
       projects each group to a shared embedding dimension (d_model).
    2. Player + Positional Embeddings — each token knows which player it belongs
       to and what feature group it represents.
    3. Transformer Encoder — multi-head self-attention across all tokens from
       both players. The model learns which features matter and how they
       interact across players.
    4. [CLS] Token — a learnable classification token aggregates information
       for final prediction via an MLP head.

Feature groups (per player, 8 tokens):
    Token 0: Early economy  (39 features)
    Token 1: Mid economy    (39 features)
    Token 2: Late economy   (39 features)
    Token 3: Final state    (39 features)
    Token 4: Economy delta  (39 features)
    Token 5: Meta stats     (4 features: APM, MMR, SQ, supplyCapped%)
    Token 6: Unit activity  (2 features: born, killed)
    Token 7: Game info      (2 features: upgrades, duration)

Total: 8 tokens × 2 players + 1 [CLS] = 17 tokens

Usage:
    uv run python train_transformer.py
    uv run python train_transformer.py --d-model 128 --n-heads 4 --n-layers 4 --epochs 200
"""

import logging
import os
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from latent_trainer.data_utils import load_and_normalize


# Feature group definitions — imported from the features package
from latent_trainer.features.feature_groups import FEATURE_GROUPS, NUM_GROUPS

logger = logging.getLogger(__name__)


class FeatureTokenizer(nn.Module):
    """Splits raw features into groups and projects each to d_model."""

    def __init__(self, d_model=64):
        super().__init__()
        self.projections = nn.ModuleList()
        for name, start, end in FEATURE_GROUPS:
            dim = end - start
            self.projections.append(nn.Sequential(
                nn.Linear(dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            ))

    def forward(self, x):
        """
        Args:
            x: [batch, features] — 203 features for ONE player
        Returns:
            tokens: [batch, num_groups, d_model]
        """
        tokens = []
        for i, (name, start, end) in enumerate(FEATURE_GROUPS):
            group = x[:, start:end]
            tokens.append(self.projections[i](group))
        return torch.stack(tokens, dim=1)  # [B, 8, d_model]


class SC2Transformer(nn.Module):
    """Transformer classifier for SC2 match outcome prediction."""

    def __init__(self, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Feature tokenizer (shared for both players)
        self.tokenizer = FeatureTokenizer(d_model=d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Player embedding: 0 = CLS, 1 = Player 1, 2 = Player 2
        self.player_embedding = nn.Embedding(3, d_model)

        # Positional embedding for feature groups (0-7)
        self.position_embedding = nn.Embedding(NUM_GROUPS, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Args:
            x: [batch, 2, 203] — two players' features
        Returns:
            pred: [batch, 1] — win probability for player 1
        """
        B = x.shape[0]

        # Tokenize each player's features
        p1_tokens = self.tokenizer(x[:, 0, :])  # [B, 8, d_model]
        p2_tokens = self.tokenizer(x[:, 1, :])  # [B, 8, d_model]

        # Add player embeddings
        player1_ids = torch.ones(B, NUM_GROUPS, dtype=torch.long, device=x.device)
        player2_ids = torch.full((B, NUM_GROUPS), 2, dtype=torch.long, device=x.device)
        p1_tokens = p1_tokens + self.player_embedding(player1_ids)
        p2_tokens = p2_tokens + self.player_embedding(player2_ids)

        # Add positional embeddings (same positions for both players)
        positions = torch.arange(NUM_GROUPS, device=x.device).unsqueeze(0).expand(B, -1)
        p1_tokens = p1_tokens + self.position_embedding(positions)
        p2_tokens = p2_tokens + self.position_embedding(positions)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        cls_player = torch.zeros(B, 1, dtype=torch.long, device=x.device)
        cls_tokens = cls_tokens + self.player_embedding(cls_player)

        # Concatenate: [CLS] + P1 tokens + P2 tokens = [B, 17, d_model]
        sequence = torch.cat([cls_tokens, p1_tokens, p2_tokens], dim=1)

        # Self-attention across all tokens
        encoded = self.transformer(sequence)  # [B, 17, d_model]

        # Classification from [CLS] token
        cls_output = encoded[:, 0, :]  # [B, d_model]
        return self.classifier(cls_output)  # [B, 1]


# load_and_normalize is imported from latent_trainer.data_utils


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading dataset...")
    train_X, train_y, val_X, val_y, _mean, _std = load_and_normalize(args.cache)
    logger.info(f"  Train: {train_X.shape}, Val: {val_X.shape}")
    logger.info(f"  Label balance — Train: {train_y.mean():.3f}, Val: {val_y.mean():.3f}")

    train_loader = DataLoader(
        TensorDataset(train_X, train_y.unsqueeze(1)),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_X, val_y.unsqueeze(1)),
        batch_size=args.batch_size, shuffle=False
    )

    # Create model
    model = SC2Transformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    logger.info(f"\n{'='*60}")
    logger.info(f"SC2 Transformer Classifier")
    logger.info(f"  d_model={args.d_model}, heads={args.n_heads}, layers={args.n_layers}")
    logger.info(f"  Tokens: 8 groups × 2 players + 1 [CLS] = 17")
    logger.info(f"  Parameters: {count_parameters(model):,}")
    logger.info(f"{'='*60}\n")

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    best_val_acc = 0
    patience = args.patience
    no_improve = 0

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_correct = 0
        train_total = 0
        train_loss_sum = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_correct += ((pred > 0.5).float() == y_batch).sum().item()
            train_total += y_batch.numel()
            train_loss_sum += loss.item() * y_batch.shape[0]

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                val_correct += ((pred > 0.5).float() == y_batch).sum().item()
                val_total += y_batch.numel()
                val_loss_sum += loss.item() * y_batch.shape[0]

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        train_loss = train_loss_sum / train_total
        val_loss = val_loss_sum / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "output/transformer_best.pth")
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or no_improve == 0:
            logger.info(f"  Epoch {epoch:3d}: train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%, "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.1e} "
                  f"{'*BEST*' if no_improve == 0 else ''}")

        if no_improve >= patience:
            logger.info(f"  Early stopped at epoch {epoch}")
            break

    logger.info(f"\n{'='*60}")
    logger.info(f"TRANSFORMER TRAINING COMPLETE")
    logger.info(f"  Architecture: d_model={args.d_model}, heads={args.n_heads}, layers={args.n_layers}")
    logger.info(f"  Parameters:   {count_parameters(model):,}")
    logger.info(f"  Best val acc: {best_val_acc:.2f}%")
    logger.info(f"  Saved to:     output/transformer_best.pth")
    logger.info(f"{'='*60}")

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Transformer SC2 match classifier")
    parser.add_argument("--cache", default="data/cached_dataset_rich.pt")
    parser.add_argument("--d-model", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    train(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
