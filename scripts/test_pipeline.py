"""
Latent Space Optimal Transport Pipeline
========================================
Steps:
  1. Train a VAE (reconstruction) + classifier head (win/loss) on synthetic data
  2. Embed a fresh unseen point into latent space
  3. Use Optimal Transport to find the minimum-cost path from the new point
     to the opposite-class (win) distribution
  4. Decode the transport target back to input space (counterfactual)

Install deps:
    pip install torch scikit-learn POT matplotlib numpy
"""

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────
# CONFIGURATION & CONSTANTS
# ─────────────────────────────────────────────

INPUT_DIM = 10
LATENT_DIM = 2  # 2D so we can visualise it
N_SAMPLES = 800
EPOCHS = 60
BATCH_SIZE = 64
SEED = 42


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────


def make_data(n=N_SAMPLES):
    # Losses: cluster around negative means
    X_loss = np.random.randn(n // 2, INPUT_DIM) + np.array(
        [-2, -1, 0, 1, -1, 2, -2, 0, 1, -1]
    )
    # Wins:  cluster around positive means
    X_win = np.random.randn(n // 2, INPUT_DIM) + np.array(
        [2, 1, 0, -1, 1, -2, 2, 0, -1, 1]
    )
    X = np.vstack([X_loss, X_win]).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.float32)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def prepare_data():
    set_seed()
    X_all, y_all = make_data()
    scaler = StandardScaler().fit(X_all)
    X_scaled = scaler.transform(X_all).astype(np.float32)

    # Train / test split (hold out last 100 as "unseen")
    X_train, y_train = X_scaled[:-100], y_all[:-100]
    X_test, y_test = X_scaled[-100:], y_all[-100:]
    return X_train, y_train, X_test, y_test, scaler


def create_dataloader(X, y, batch_size=BATCH_SIZE):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# ─────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────


class VAEClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        # Encoder
        self.enc_shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(32, latent_dim)
        self.enc_logvar = nn.Linear(32, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

        # Classifier on latent z
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_shared(x)
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decoder(z)
        p_win = self.classifier(z)
        return x_hat, p_win, mu, logvar

    @torch.no_grad()
    def embed(self, x):
        """Return deterministic latent (μ) for inference."""
        mu, _ = self.encode(x)
        return mu

    @torch.no_grad()
    def decode(self, z):
        return self.decoder(z)


def vae_loss(x, x_hat, mu, logvar, p_win, y, beta=0.5, cls_weight=2.0):
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    cls = nn.functional.binary_cross_entropy(p_win.squeeze(), y)
    return recon + beta * kl + cls_weight * cls, recon, kl, cls


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────


def train_model(model, train_loader, epochs=EPOCHS):
    print("=" * 55)
    print("STEP 1 — Training VAE + Classifier")
    print("=" * 55)

    optimiser = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimiser.zero_grad()
            x_hat, p_win, mu, logvar = model(xb)
            loss, recon, kl, cls = vae_loss(xb, x_hat, mu, logvar, p_win, yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            avg = total_loss / len(train_loader)
            print(f"  Epoch {epoch:3d}/{epochs} | Loss {avg:.4f}")

    model.eval()
    print("Training complete.\n")
    return model


# ─────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────


def embed_and_select_subject(model, X_train, y_train, X_test, y_test):
    print("=" * 55)
    print("STEP 2 — Embedding training data & new point")
    print("=" * 55)

    with torch.no_grad():
        Z_train = model.embed(torch.tensor(X_train)).numpy()

    Z_win = Z_train[y_train == 1]
    Z_loss = Z_train[y_train == 0]

    # Pick one fresh unseen LOSS point as our subject
    loss_indices = np.where(y_test == 0)[0]
    if len(loss_indices) == 0:
        # Fallback if random seed produced no loss examples in test set (unlikely)
        print("Warning: No loss examples in test set. Using first test point.")
        x_new_raw = X_test[0]
    else:
        x_new_raw = X_test[loss_indices[0]]

    x_new_tensor = torch.tensor(x_new_raw).unsqueeze(0)  # shape (1, 10)

    with torch.no_grad():
        z_new = model.embed(x_new_tensor).numpy().squeeze()  # shape (2,)
        p_new = model.classifier(model.embed(x_new_tensor)).item()

    print(f"  New point latent coords : {z_new.round(3)}")
    print(f"  Model P(win) for new pt : {p_new:.3f}  (it's a predicted loss)\n")

    return Z_win, Z_loss, z_new, x_new_raw


def run_optimal_transport(z_new, Z_win):
    print("=" * 55)
    print("STEP 3 — Optimal Transport: loss → win distribution")
    print("=" * 55)

    # Source: single point
    source = z_new.reshape(1, -1)  # (1, latent_dim)
    target = Z_win  # (n_win, latent_dim)

    a = np.array([1.0])  # all mass on source
    b = ot.unif(len(target))  # uniform over win pts

    # Cost matrix: squared Euclidean distance
    M = ot.dist(source, target, metric="sqeuclidean")

    # Earth Mover's Distance transport plan
    T = ot.emd(a, b, M)

    # The transport target = weighted average of win points under plan
    weights = T[0]
    z_target = (weights[:, None] * target).sum(0) / weights.sum()

    # Wasserstein distance (transport cost)
    W_dist = (M * T).sum()

    print(f"  z_new    (loss) : {z_new.round(3)}")
    print(f"  z_target (win)  : {z_target.round(3)}")
    print(f"  Wasserstein distance (effort to become a winner): {W_dist:.4f}\n")

    return z_target


def decode_trajectory(model, scaler, z_new, z_target, x_new_raw, n_steps=8):
    print("=" * 55)
    print("STEP 4 — Decode counterfactual trajectory")
    print("=" * 55)

    alphas = np.linspace(0, 1, n_steps)

    waypoints_z = np.array([(1 - a) * z_new + a * z_target for a in alphas])
    waypoints_z_torch = torch.tensor(waypoints_z, dtype=torch.float32)

    with torch.no_grad():
        waypoints_x = model.decode(waypoints_z_torch).numpy()
        # waypoints_x_original = scaler.inverse_transform(waypoints_x)
        p_win_path = model.classifier(waypoints_z_torch).numpy().squeeze()

    print("  Waypoint  α     P(win)")
    print("  " + "-" * 28)
    for i, (alpha, p) in enumerate(zip(alphas, p_win_path)):
        marker = "  ← start" if i == 0 else ("  ← target" if i == n_steps - 1 else "")
        print(f"    {i + 1}      {alpha:.2f}    {p:.3f}{marker}")

    x_counterfactual = waypoints_x[-1]

    # Calculate original and counterfactual in input space
    x_orig = scaler.inverse_transform(x_new_raw.reshape(1, -1))[0]
    x_cf = scaler.inverse_transform(x_counterfactual.reshape(1, -1))[0]

    print(f"\n  Original  (loss) input (first 5 features): {x_orig[:5].round(2)}")
    print(f"  Counterfactual (win) input (first 5 feat) : {x_cf[:5].round(2)}")
    print(
        f"\n  Δ (change needed per feature, first 5)   : {(x_cf - x_orig)[:5].round(2)}"
    )

    return waypoints_z, p_win_path, alphas


def plot_results(
    Z_loss,
    Z_win,
    z_new,
    z_target,
    waypoints_z,
    probs,
    alphas,
    output_file="latent_ot_pipeline.png",
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Latent Space — Optimal Transport Counterfactual",
        fontsize=14,
        fontweight="bold",
    )

    # ── Left: Latent space overview
    ax = axes[0]
    ax.scatter(
        Z_loss[:, 0], Z_loss[:, 1], c="tomato", alpha=0.3, s=15, label="Loss (dirt)"
    )
    ax.scatter(
        Z_win[:, 0], Z_win[:, 1], c="steelblue", alpha=0.3, s=15, label="Win (holes)"
    )
    ax.scatter(*z_new, c="red", s=120, zorder=5, marker="*", label="New point (loss)")
    ax.scatter(
        *z_target, c="blue", s=120, zorder=5, marker="*", label="OT target (win)"
    )
    ax.annotate(
        "",
        xy=z_target,
        xytext=z_new,
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
    )
    for wz in waypoints_z:
        ax.scatter(*wz, c="gold", s=30, zorder=4)
    ax.set_title("Latent space + OT transport path")
    ax.legend(fontsize=8)
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")

    # ── Right: P(win) along the path
    ax2 = axes[1]
    ax2.plot(alphas, probs, "o-", color="mediumseagreen", lw=2, markersize=8)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Decision boundary")
    ax2.fill_between(
        alphas,
        probs,
        0.5,
        where=(probs > 0.5),
        alpha=0.15,
        color="steelblue",
        label="Win zone",
    )
    ax2.fill_between(
        alphas,
        probs,
        0.5,
        where=(probs < 0.5),
        alpha=0.15,
        color="tomato",
        label="Loss zone",
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Transport interpolation α  (0=loss, 1=win target)")
    ax2.set_ylabel("P(win)")
    ax2.set_title("Win probability along OT path")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {output_file}")


def main():
    # 1. Data Preparation
    X_train, y_train, X_test, y_test, scaler = prepare_data()
    train_loader = create_dataloader(X_train, y_train)

    # 2. Model Training
    model = VAEClassifier()
    model = train_model(model, train_loader)

    # 3. Embed & Select Subject
    Z_win, Z_loss, z_new, x_new_raw = embed_and_select_subject(
        model, X_train, y_train, X_test, y_test
    )

    # 4. Optimal Transport
    z_target = run_optimal_transport(z_new, Z_win)

    # 5. Counterfactual
    waypoints_z, probs, alphas = decode_trajectory(
        model, scaler, z_new, z_target, x_new_raw
    )

    # 6. Visualization
    plot_results(Z_loss, Z_win, z_new, z_target, waypoints_z, probs, alphas)

    print("\nDone. Full pipeline complete.")


if __name__ == "__main__":
    main()
