"""Two-stage training: VAE reconstruction first, then classifier on frozen latents.

Stage 1: Train VAE for reconstruction only (no classification head).
Stage 2: Freeze the VAE encoder, extract latent representations,
          train a proper MLP classifier on the full latent space.

This avoids the guided VAE's architectural bottleneck (cls head using
only 1 latent dim) and the adversarial step fighting the classifier.

Usage:
    uv run python train_two_stage.py
    uv run python train_two_stage.py --vae-epochs 200 --cls-epochs 100 --latent-dim 32
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from latent_trainer.data_utils import extract_latents, load_and_normalize

logger = logging.getLogger(__name__)


# --- VAE Model (reconstruction only) ---
class SimpleVAE(nn.Module):
    def __init__(self, input_dim: int = 203, latent_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss_fn(
    recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss = MSE reconstruction + KL divergence."""
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss


# --- Latent Classifier ---
class LatentClassifier(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        # Takes latent vectors from BOTH players: [batch, 2*latent_dim]
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def train_vae_stage(
    train_X: torch.Tensor,
    val_X: torch.Tensor,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> SimpleVAE:
    """Stage 1: Train VAE for reconstruction."""
    input_dim = train_X.shape[-1]  # 203

    # Flatten [N, 2, 203] -> [N*2, 203] for VAE training
    train_flat = train_X.reshape(-1, input_dim)
    val_flat = val_X.reshape(-1, input_dim)

    train_loader = DataLoader(
        TensorDataset(train_flat), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_flat), batch_size=batch_size, shuffle=False
    )

    vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float("inf")
    patience = 20
    no_improve = 0

    logger.info(f"\n{'=' * 60}")
    logger.info("STAGE 1: VAE Reconstruction Training")
    logger.info(f"  Input dim: {input_dim}, Latent dim: {latent_dim}")
    logger.info(f"  Train: {len(train_flat)}, Val: {len(val_flat)}")
    logger.info(f"{'=' * 60}")

    for epoch in range(epochs):
        vae.train()
        train_loss = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            recon, mu, logvar = vae(batch)
            loss, _, _ = vae_loss_fn(recon, batch, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = vae(batch)
                loss, _, _ = vae_loss_fn(recon, batch, mu, logvar)
                val_loss += loss.item()

        train_loss /= len(train_flat)
        val_loss /= len(val_flat)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(vae.state_dict(), "output/vae_best.pth")
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or no_improve == 0:
            logger.info(
                f"  Epoch {epoch:3d}: train_loss={train_loss:.2f}, val_loss={val_loss:.2f}, "
                f"lr={optimizer.param_groups[0]['lr']:.1e} {'*BEST*' if no_improve == 0 else ''}"
            )

        if no_improve >= patience:
            logger.info(f"  Early stopped at epoch {epoch}")
            break

    # Load best VAE
    vae.load_state_dict(torch.load("output/vae_best.pth", weights_only=True))
    logger.info(f"\n  Best VAE val_loss: {best_val_loss:.2f}")
    return vae


def train_classifier_stage(
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    val_z: torch.Tensor,
    val_y: torch.Tensor,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[LatentClassifier, float]:
    """Stage 2: Train classifier on frozen latent representations."""

    train_loader = DataLoader(
        TensorDataset(train_z, train_y.unsqueeze(1)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_z, val_y.unsqueeze(1)), batch_size=batch_size, shuffle=False
    )

    classifier = LatentClassifier(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()

    best_val_acc = 0
    patience = 15
    no_improve = 0

    logger.info(f"\n{'=' * 60}")
    logger.info("STAGE 2: Classifier on Frozen Latent Space")
    logger.info(f"  Input: [N, {latent_dim * 2}] (both players concatenated)")
    logger.info(f"  Train: {len(train_z)}, Val: {len(val_z)}")
    logger.info(f"{'=' * 60}")

    for epoch in range(epochs):
        classifier.train()
        train_correct = 0
        train_total = 0

        for z_batch, y_batch in train_loader:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device)
            pred = classifier(z_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_correct += ((pred > 0.5).float() == y_batch).sum().item()
            train_total += y_batch.numel()

        classifier.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for z_batch, y_batch in val_loader:
                z_batch, y_batch = z_batch.to(device), y_batch.to(device)
                pred = classifier(z_batch)
                loss = criterion(pred, y_batch)
                val_correct += ((pred > 0.5).float() == y_batch).sum().item()
                val_total += y_batch.numel()
                val_loss += loss.item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), "output/classifier_best.pth")
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or no_improve == 0:
            logger.info(
                f"  Epoch {epoch:3d}: train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%, "
                f"lr={optimizer.param_groups[0]['lr']:.1e} {'*BEST*' if no_improve == 0 else ''}"
            )

        if no_improve >= patience:
            logger.info(f"  Early stopped at epoch {epoch}")
            break

    logger.info(f"\n  >>> BEST classifier val accuracy: {best_val_acc:.2f}%")
    return classifier, best_val_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage VAE training")
    parser.add_argument(
        "--cache", default="data/cached_dataset_rich.pt", help="Path to cached dataset"
    )
    parser.add_argument(
        "--latent-dim", type=int, default=32, help="VAE latent dimension"
    )
    parser.add_argument(
        "--vae-epochs", type=int, default=200, help="Max epochs for VAE stage"
    )
    parser.add_argument(
        "--cls-epochs", type=int, default=100, help="Max epochs for classifier stage"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--vae-lr", type=float, default=1e-3, help="VAE learning rate")
    parser.add_argument(
        "--cls-lr", type=float, default=1e-3, help="Classifier learning rate"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs("output", exist_ok=True)

    # Load data
    logger.info("Loading dataset...")
    train_X, train_y, val_X, val_y, mean, std = load_and_normalize(args.cache)
    logger.info(f"  Train: {train_X.shape}, Val: {val_X.shape}")
    logger.info(
        f"  Label balance — Train: {train_y.mean():.3f}, Val: {val_y.mean():.3f}"
    )

    # Stage 1: Train VAE
    vae = train_vae_stage(
        train_X,
        val_X,
        latent_dim=args.latent_dim,
        epochs=args.vae_epochs,
        batch_size=args.batch_size,
        lr=args.vae_lr,
        device=device,
    )

    # Extract latent representations
    logger.info("\nExtracting latent representations...")
    train_z = extract_latents(vae, train_X, device)
    val_z = extract_latents(vae, val_X, device)
    logger.info(f"  Train latents: {train_z.shape}, Val latents: {val_z.shape}")

    # Stage 2: Train classifier on latent space
    classifier, best_acc = train_classifier_stage(
        train_z,
        train_y,
        val_z,
        val_y,
        latent_dim=args.latent_dim,
        epochs=args.cls_epochs,
        batch_size=args.batch_size,
        lr=args.cls_lr,
        device=device,
    )

    # Save combined model info
    torch.save(
        {
            "vae_state": vae.state_dict(),
            "classifier_state": classifier.state_dict(),
            "latent_dim": args.latent_dim,
            "input_dim": train_X.shape[-1],
            "normalization": {"mean": mean, "std": std},
            "best_val_acc": best_acc,
        },
        "output/two_stage_model.pth",
    )

    logger.info(f"\n{'=' * 60}")
    logger.info("TWO-STAGE TRAINING COMPLETE")
    logger.info(f"  VAE latent dim:      {args.latent_dim}")
    logger.info(f"  Best val accuracy:   {best_acc:.2f}%")
    logger.info("  Model saved to:      output/two_stage_model.pth")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    from latent_trainer.config import LOGGING_FORMAT

    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
    main()
