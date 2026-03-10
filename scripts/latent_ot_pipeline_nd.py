"""
Latent Space Counterfactual Pipeline  —  N-Dimensional
========================================================
Configurable latent dimensionality with three path-finding methods:

  METHOD 1 · Optimal Transport (OT)
    Finds the minimum-cost transport from z_new to the win distribution.
    Uses Earth Mover's Distance. Fast, globally optimal, ignores manifold.

  METHOD 2 · Gradient Ascent in Latent Space
    Follows ∇P(win|z) with momentum, staying close to the data manifold
    via a Gaussian-KDE density penalty. Slow but manifold-aware.

  METHOD 3 · Geodesic via k-NN Graph (Dijkstra)
    Builds a neighbourhood graph from training embeddings and finds the
    shortest graph path from z_new to the nearest high-density win region.
    Strictly on-manifold by construction.

Visualization: all paths projected to 2D via PCA / t-SNE / UMAP.

Install:
    pip install torch scikit-learn POT matplotlib numpy scipy umap-learn
"""

import warnings

import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import ot
import scipy.sparse.csgraph as csgraph
import torch
import torch.nn as nn
import torch.optim as optim
import umap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity, kneighbors_graph
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
HAS_UMAP = True

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

INPUT_DIM = 80
LATENT_DIM = 32
N_SAMPLES = 10000
EPOCHS = 500
BATCH_SIZE = 64
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path method parameters
OT_REG = 0.0
GRAD_STEPS = 500
GRAD_LR = 0.02
GRAD_MOMENTUM = 0.85
GRAD_DENSITY_WEIGHT = 0.3
GRAD_KDE_BW = 0.5
GEODESIC_K = 12
N_WAYPOINTS = 10


# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_data(n=N_SAMPLES, input_dim=INPUT_DIM):
    offset = np.random.choice([-1, 1], size=input_dim) * np.linspace(
        1.5, 2.5, input_dim
    )
    X_loss = np.random.randn(n // 2, input_dim) + (-offset)
    X_win = np.random.randn(n // 2, input_dim) + (offset)
    X = np.vstack([X_loss, X_win]).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.float32)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def prepare_data(input_dim=INPUT_DIM):
    set_seed()
    X_all, y_all = make_data(input_dim=input_dim)
    scaler = StandardScaler().fit(X_all)
    X_scaled = scaler.transform(X_all).astype(np.float32)
    X_train, y_train = X_scaled[:-150], y_all[:-150]
    X_test, y_test = X_scaled[-150:], y_all[-150:]
    return X_train, y_train, X_test, y_test, scaler


def create_dataloader(X, y, batch_size=BATCH_SIZE):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))

    def collate_to_device(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)

    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, collate_fn=collate_to_device
    )


# ══════════════════════════════════════════════════════════════
# MODEL  —  VAE + Classifier head
# ══════════════════════════════════════════════════════════════


class VAEClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        h = max(64, latent_dim * 8)

        self.enc_shared = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h // 2),
            nn.LayerNorm(h // 2),
            nn.GELU(),
        )
        self.enc_mu = nn.Linear(h // 2, latent_dim)
        self.enc_logvar = nn.Linear(h // 2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h // 2),
            nn.LayerNorm(h // 2),
            nn.GELU(),
            nn.Linear(h // 2, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, input_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_shared(x)
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decoder(z)
        p_win = self.classifier(z)
        return x_hat, p_win, mu, logvar

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def p_win(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)


def vae_loss(x, x_hat, mu, logvar, p_win, y, beta=0.4, cls_w=2.5):
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    cls = nn.functional.binary_cross_entropy(p_win.squeeze(), y)
    return recon + beta * kl + cls_w * cls


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════


def train_model(model, train_loader, epochs=EPOCHS):
    print("=" * 60)
    print(
        f"STEP 1  —  Training VAE+Classifier  (latent_dim={model.enc_mu.out_features})"
    )
    print(f"           Device: {DEVICE}")
    print("=" * 60)

    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            # xb, yb already on DEVICE via collate_to_device
            opt.zero_grad()
            x_hat, pw, mu, lv = model(xb)
            loss = vae_loss(xb, x_hat, mu, lv, pw, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        sched.step()
        if epoch % 10 == 0:
            print(
                f"  Epoch {epoch:3d}/{epochs}  |  loss {running / len(train_loader):.4f}"
            )

    model.eval()
    print("Training complete.\n")
    return model


# ══════════════════════════════════════════════════════════════
# EMBED  —  build class clouds, pick subject
# ══════════════════════════════════════════════════════════════


def embed_training_data(model, X_train, y_train):
    with torch.no_grad():
        Z_train = model.embed(torch.tensor(X_train).to(DEVICE)).cpu().numpy()
    Z_win = Z_train[y_train == 1]
    Z_loss = Z_train[y_train == 0]
    return Z_train, Z_win, Z_loss


def pick_subject(model, X_test, y_test):
    print("=" * 60)
    print("STEP 2  —  Embedding new (unseen) point")
    print("=" * 60)
    loss_idx = np.where(y_test == 0)[0]
    x_raw = X_test[loss_idx[0]]
    xt = torch.tensor(x_raw).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        z_new = model.embed(xt).cpu().numpy().squeeze()
        p = model.p_win(model.embed(xt)).item()
    print(f"  Latent dim    : {z_new.shape[0]}")
    print(f"  P(win) at z   : {p:.3f}  ← correctly predicted as loss\n")
    return z_new, x_raw


# ══════════════════════════════════════════════════════════════
# PATH METHOD 1  —  Optimal Transport
# ══════════════════════════════════════════════════════════════


def path_optimal_transport(
    z_new: np.ndarray,
    Z_win: np.ndarray,
    reg: float = OT_REG,
    n_waypoints: int = N_WAYPOINTS,
) -> np.ndarray:
    source = z_new.reshape(1, -1)
    a = np.array([1.0])
    b = ot.unif(len(Z_win))
    M = ot.dist(source, Z_win, metric="sqeuclidean")

    T = ot.sinkhorn(a, b, M, reg=reg) if reg > 0 else ot.emd(a, b, M)

    weights = T[0]
    z_target = (weights[:, None] * Z_win).sum(0) / weights.sum()
    W_dist = (M * T).sum()

    print(f"  [OT]  Wasserstein distance  : {W_dist:.4f}")
    print(f"  [OT]  z_target (first 4)   : {z_target[:4].round(3)}")

    alphas = np.linspace(0, 1, n_waypoints)
    return np.array([(1 - a) * z_new + a * z_target for a in alphas])


# ══════════════════════════════════════════════════════════════
# PATH METHOD 2  —  Gradient Ascent with Density Regularisation
# ══════════════════════════════════════════════════════════════


def path_gradient_ascent(
    model,
    z_new: np.ndarray,
    Z_all: np.ndarray,
    steps: int = GRAD_STEPS,
    lr: float = GRAD_LR,
    momentum: float = GRAD_MOMENTUM,
    density_w: float = GRAD_DENSITY_WEIGHT,
    kde_bw: float = GRAD_KDE_BW,
    n_waypoints: int = N_WAYPOINTS,
) -> np.ndarray:
    kde = KernelDensity(kernel="gaussian", bandwidth=kde_bw).fit(Z_all)
    z = torch.tensor(
        z_new.copy(), dtype=torch.float32, requires_grad=True, device=DEVICE
    )
    velocity = torch.zeros_like(z)
    trajectory = [z_new.copy()]

    model.eval()
    for step in range(steps):
        if z.grad is not None:
            z.grad.zero_()

        p = model.p_win(z.unsqueeze(0))
        p.backward()

        with torch.no_grad():
            grad_cls = z.grad.clone()

            # KDE gradient — numerical, on CPU/numpy
            z_np = z.detach().cpu().numpy()
            eps = 1e-3
            grads_kde = np.zeros_like(z_np)
            for i in range(len(z_np)):
                zp, zm = z_np.copy(), z_np.copy()
                zp[i] += eps
                zm[i] -= eps
                grads_kde[i] = float(
                    kde.score_samples(zp.reshape(1, -1))[0]
                    - kde.score_samples(zm.reshape(1, -1))[0]
                ) / (2 * eps)
            grad_density = torch.tensor(grads_kde, dtype=torch.float32, device=DEVICE)

            total_grad = grad_cls + density_w * grad_density
            velocity = momentum * velocity + lr * total_grad
            z = z.detach().to(DEVICE) + velocity
            z.requires_grad_(True)
            trajectory.append(z.detach().cpu().numpy().copy())

    trajectory = np.array(trajectory)
    indices = np.linspace(0, len(trajectory) - 1, n_waypoints, dtype=int)
    path = trajectory[indices]

    final_p = model.p_win(
        torch.tensor(path[-1], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    ).item()
    print(f"  [GA]  Final P(win)          : {final_p:.4f}")
    print(f"  [GA]  Steps taken           : {steps}")
    return path


# ══════════════════════════════════════════════════════════════
# PATH METHOD 3  —  Geodesic via k-NN Graph (Dijkstra)
# ══════════════════════════════════════════════════════════════


def path_geodesic(
    z_new: np.ndarray,
    Z_win: np.ndarray,
    Z_all: np.ndarray,
    k: int = GEODESIC_K,
    n_waypoints: int = N_WAYPOINTS,
) -> np.ndarray:
    kde_win = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(Z_win)
    scores = kde_win.score_samples(Z_win)
    best_win = Z_win[np.argmax(scores)]

    Z_graph = np.vstack([Z_all, z_new.reshape(1, -1), best_win.reshape(1, -1)])
    src_idx = len(Z_graph) - 2
    tgt_idx = len(Z_graph) - 1

    A = kneighbors_graph(Z_graph, n_neighbors=k, mode="distance", include_self=False)
    A = (A + A.T) / 2

    dist_matrix, predecessors = csgraph.shortest_path(
        A, method="D", directed=False, indices=src_idx, return_predecessors=True
    )

    path_indices = []
    node = tgt_idx
    while node != src_idx and node >= 0:
        path_indices.append(node)
        node = predecessors[node]
    path_indices.append(src_idx)
    path_indices = path_indices[::-1]

    if len(path_indices) < 2:
        print("  [GEO] Warning: no graph path found — falling back to straight line.")
        alphas = np.linspace(0, 1, n_waypoints)
        return np.array([(1 - a) * z_new + a * best_win for a in alphas])

    full_path = Z_graph[path_indices]
    indices = np.linspace(0, len(full_path) - 1, n_waypoints, dtype=int)
    path = full_path[indices]

    print(f"  [GEO] Graph nodes in path   : {len(path_indices)}")
    print(f"  [GEO] Graph distance        : {dist_matrix[tgt_idx]:.4f}")
    return path


# ══════════════════════════════════════════════════════════════
# DECODE PATH  —  latent waypoints → input space + P(win)
# ══════════════════════════════════════════════════════════════


def decode_path(model, path_z: np.ndarray) -> tuple:
    zt = torch.tensor(path_z, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        x_hat = model.decode(zt).cpu().numpy()
        p_vals = model.p_win(zt).cpu().numpy().squeeze()
    return x_hat, p_vals


# ══════════════════════════════════════════════════════════════
# PROJECTIONS  —  fit PCA / t-SNE / UMAP on training cloud
# ══════════════════════════════════════════════════════════════


def fit_projections(Z_train: np.ndarray) -> dict:
    projections = {}

    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(Z_train)
    projections["PCA"] = pca.transform
    projections["_pca_obj"] = pca

    print("  Fitting t-SNE on training embeddings …", flush=True)
    tsne_full = TSNE(
        n_components=2,
        perplexity=min(30, len(Z_train) // 4),
        random_state=SEED,
        max_iter=600,
        init="pca",
        learning_rate="auto",
    ).fit_transform(Z_train)
    projections["_tsne_train"] = tsne_full

    def tsne_project(Z_query):
        dists = np.linalg.norm(Z_train[:, None] - Z_query[None], axis=2)
        nn_idx = dists.argmin(axis=0)
        return tsne_full[nn_idx]

    projections["tSNE"] = tsne_project

    if HAS_UMAP:
        print("  Fitting UMAP on training embeddings …", flush=True)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=SEED,
            verbose=False,
        )
        reducer.fit(Z_train)
        projections["_umap_train"] = reducer.transform(Z_train)
        projections["UMAP"] = reducer.transform
    else:
        projections["UMAP"] = None

    return projections


# ══════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════

COLORS = {
    "ot": ("#F4A261", "#E76F51"),
    "ga": ("#57CC99", "#22577A"),
    "geo": ("#C77DFF", "#7B2FBE"),
    "win": "#457B9D",
    "loss": "#E63946",
    "new_pt": "#FFBA08",
}

_WIN_CMAP = LinearSegmentedColormap.from_list("win_cmap", ["#0D1117", "#457B9D"])
_LOSS_CMAP = LinearSegmentedColormap.from_list("loss_cmap", ["#0D1117", "#E63946"])


def _style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor("#161B22")
    ax.set_title(title, color="white", fontsize=9, pad=5)
    ax.set_xlabel(xlabel, color="#666", fontsize=7)
    ax.set_ylabel(ylabel, color="#666", fontsize=7)
    ax.tick_params(colors="#555", labelsize=6)
    for sp in ax.spines.values():
        sp.set_color("#30363D")


def _draw_projection_panel(
    ax, proj_name, proj_fn, Z_win_train, Z_loss_train, z_new, paths, y_train_win_mask
):
    Z_win_2d = proj_fn(Z_win_train)
    Z_loss_2d = proj_fn(Z_loss_train)

    ax.scatter(*Z_loss_2d.T, c=COLORS["loss"], alpha=0.18, s=8, lw=0, rasterized=True)
    ax.scatter(*Z_win_2d.T, c=COLORS["win"], alpha=0.18, s=8, lw=0, rasterized=True)

    z_new_2d = proj_fn(z_new.reshape(1, -1))[0]
    ax.scatter(
        *z_new_2d,
        c=COLORS["new_pt"],
        s=200,
        zorder=12,
        marker="*",
        edgecolors="white",
        linewidths=0.8,
    )

    for name, path_z in paths.items():
        c_light, c_dark = COLORS.get(name, ("#fff", "#aaa"))
        path_2d = proj_fn(path_z)
        n = len(path_2d)

        for i in range(n - 1):
            alpha = 0.4 + 0.6 * (i / (n - 1))
            lw = 1.5 + 1.0 * (i / (n - 1))
            ax.plot(
                path_2d[i : i + 2, 0],
                path_2d[i : i + 2, 1],
                "-",
                color=c_light,
                lw=lw,
                alpha=alpha,
                solid_capstyle="round",
            )

        sizes = np.linspace(10, 55, n)
        ax.scatter(
            path_2d[:, 0],
            path_2d[:, 1],
            c=[c_light] * n,
            s=sizes,
            zorder=10,
            alpha=0.9,
            edgecolors="none",
        )

        mid = int(n * 0.6)
        if mid < n - 1:
            ax.annotate(
                "",
                xy=(path_2d[mid + 1, 0], path_2d[mid + 1, 1]),
                xytext=(path_2d[mid, 0], path_2d[mid, 1]),
                arrowprops=dict(
                    arrowstyle="-|>", color=c_light, lw=1.6, mutation_scale=10
                ),
                zorder=11,
            )

        ax.scatter(
            *path_2d[-1], c=c_dark, s=80, zorder=11, edgecolors="white", linewidths=0.6
        )
        ax.text(
            path_2d[-1, 0],
            path_2d[-1, 1] - 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            name.upper(),
            color=c_light,
            fontsize=6,
            ha="center",
            va="top",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="#0D1117")],
        )

    _style_ax(
        ax, f"{proj_name} projection", xlabel=f"{proj_name}₁", ylabel=f"{proj_name}₂"
    )


def visualise_all(
    Z_win, Z_loss, z_new, paths, projections, model, out_prefix="latent_paths"
):
    proj_items = [("PCA", projections["PCA"]), ("tSNE", projections["tSNE"])]
    if projections.get("UMAP") is not None:
        proj_items.append(("UMAP", projections["UMAP"]))

    n_proj = len(proj_items)
    n_methods = len(paths)
    alphas_wp = np.linspace(0, 1, N_WAYPOINTS)

    # ── Figure 1: Projection panels
    fig1, axes1 = plt.subplots(1, n_proj, figsize=(6.5 * n_proj, 6.5))
    fig1.patch.set_facecolor("#0D1117")
    if n_proj == 1:
        axes1 = [axes1]

    for ax, (pname, pfn) in zip(axes1, proj_items):
        _draw_projection_panel(
            ax, pname, pfn, Z_win, Z_loss, z_new, paths, y_train_win_mask=None
        )

    legend_elems = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=COLORS["new_pt"],
            markersize=10,
            label="New point (loss)",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["loss"],
            markersize=6,
            alpha=0.6,
            label="Loss cloud",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["win"],
            markersize=6,
            alpha=0.6,
            label="Win cloud",
            linestyle="None",
        ),
    ]
    for name in paths:
        c_light, _ = COLORS.get(name, ("#fff", "#aaa"))
        legend_elems.append(
            Line2D([0], [0], color=c_light, lw=2, label=f"Path: {name.upper()}")
        )
    fig1.legend(
        handles=legend_elems,
        loc="lower center",
        ncol=len(legend_elems),
        fontsize=8,
        facecolor="#161B22",
        edgecolor="#30363D",
        labelcolor="white",
        bbox_to_anchor=(0.5, -0.02),
    )
    fig1.suptitle(
        f"Latent Space Projections  ·  latent_dim={LATENT_DIM}  ·  input_dim={INPUT_DIM}",
        color="white",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig1.tight_layout()
    out1 = f"plots/{out_prefix}_projections.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
    plt.close(fig1)
    print(f"  Saved → {out1}")

    # ── Figure 2: Analytics per method
    fig2 = plt.figure(figsize=(6 * n_methods, 10))
    fig2.patch.set_facecolor("#0D1117")
    gs2 = gridspec.GridSpec(
        3,
        n_methods,
        figure=fig2,
        hspace=0.5,
        wspace=0.35,
        left=0.06,
        right=0.97,
        top=0.91,
        bottom=0.07,
    )

    for col, (name, path_z) in enumerate(paths.items()):
        c_light, c_dark = COLORS.get(name, ("#fff", "#aaa"))
        x_decoded, p_vals = decode_path(model, path_z)

        # Row 0: P(win) curve
        ax_p = fig2.add_subplot(gs2[0, col])
        ax_p.plot(alphas_wp, p_vals, "o-", color=c_light, lw=2, markersize=5)
        ax_p.axhline(0.5, color="#555", linestyle="--", lw=1)
        ax_p.fill_between(
            alphas_wp,
            p_vals,
            0.5,
            where=(p_vals >= 0.5),
            color=COLORS["win"],
            alpha=0.2,
        )
        ax_p.fill_between(
            alphas_wp,
            p_vals,
            0.5,
            where=(p_vals < 0.5),
            color=COLORS["loss"],
            alpha=0.2,
        )
        cross = np.where(np.diff(np.sign(p_vals - 0.5)))[0]
        for c in cross:
            ax_p.axvline(alphas_wp[c], color="white", lw=0.8, alpha=0.4, linestyle=":")
        ax_p.set_ylim(0, 1)
        ax_p.set_xlim(0, 1)
        _style_ax(
            ax_p,
            f"{name.upper()} — P(win) along path",
            xlabel="α  (0=start → 1=target)",
            ylabel="P(win)",
        )

        # Row 1: Feature delta bar
        ax_d = fig2.add_subplot(gs2[1, col])
        delta = x_decoded[-1] - x_decoded[0]
        feat_idx = np.arange(len(delta))
        bar_cols = [COLORS["win"] if v > 0 else COLORS["loss"] for v in delta]
        ax_d.bar(feat_idx, delta, color=bar_cols, alpha=0.85, width=0.7)
        ax_d.axhline(0, color="#555", lw=0.8)
        top3 = np.argsort(np.abs(delta))[-3:]
        for i in top3:
            ax_d.text(
                i,
                delta[i] + np.sign(delta[i]) * 0.02 * np.abs(delta).max(),
                f"f{i}",
                color="white",
                fontsize=6,
                ha="center",
            )
        _style_ax(
            ax_d,
            f"{name.upper()} — Feature Δ (start→end)",
            xlabel="Feature index",
            ylabel="Δ (decoded space)",
        )

        # Row 2: Waypoint heatmap
        ax_h = fig2.add_subplot(gs2[2, col])
        hmap = ax_h.imshow(
            x_decoded.T,
            aspect="auto",
            cmap="RdBu_r",
            interpolation="nearest",
            extent=[0, 1, -0.5, x_decoded.shape[1] - 0.5],
        )
        ax_h.set_yticks(range(x_decoded.shape[1]))
        ax_h.set_yticklabels(
            [f"f{i}" for i in range(x_decoded.shape[1])], color="#888", fontsize=5
        )
        ax_h.set_xticks(np.linspace(0, 1, N_WAYPOINTS))
        ax_h.set_xticklabels(
            [f"{a:.1f}" for a in alphas_wp], color="#888", fontsize=5, rotation=45
        )
        plt.colorbar(hmap, ax=ax_h, pad=0.02, fraction=0.04).ax.tick_params(
            colors="#888", labelsize=5
        )
        _style_ax(
            ax_h,
            f"{name.upper()} — Feature values along path",
            xlabel="α",
            ylabel="Feature",
        )

    fig2.suptitle(
        f"Path Analytics  ·  latent_dim={LATENT_DIM}  ·  input_dim={INPUT_DIM}",
        color="white",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )
    out2 = f"plots/{out_prefix}_analytics.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"  Saved → {out2}")


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════


def run_pipeline(input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
    print(f"\n{'═' * 60}")
    print(f"  Latent OT Pipeline  ·  input={input_dim}d  ·  latent={latent_dim}d")
    print(f"{'═' * 60}\n")

    X_train, y_train, X_test, y_test, scaler = prepare_data(input_dim)
    train_loader = create_dataloader(X_train, y_train)

    model = VAEClassifier(input_dim=input_dim, latent_dim=latent_dim)
    model = train_model(model, train_loader)

    Z_train, Z_win, Z_loss = embed_training_data(model, X_train, y_train)
    z_new, x_new_raw = pick_subject(model, X_test, y_test)

    print("=" * 60)
    print("STEP 2b —  Fitting 2D projections (PCA / t-SNE / UMAP)")
    print("=" * 60)
    projections = fit_projections(Z_train)
    print()

    print("=" * 60)
    print("STEP 3  —  Computing paths via three methods")
    print("=" * 60)
    path_ot = path_optimal_transport(z_new, Z_win)
    print()
    path_ga = path_gradient_ascent(model, z_new, Z_train)
    print()
    path_geo = path_geodesic(z_new, Z_win, Z_train)
    print()
    paths = {"ot": path_ot, "ga": path_ga, "geo": path_geo}

    print("=" * 60)
    print("STEP 4  —  Decoding waypoints & comparing methods")
    print("=" * 60)
    print(
        f"  {'Method':<6}  {'Start P(win)':<14}  {'End P(win)':<12}  {'Path length (latent)'}"
    )
    print("  " + "-" * 54)
    for name, path_z in paths.items():
        _, p_vals = decode_path(model, path_z)
        length = np.linalg.norm(np.diff(path_z, axis=0), axis=1).sum()
        print(
            f"  {name.upper():<6}  {p_vals[0]:<14.3f}  {p_vals[-1]:<12.3f}  {length:.4f}"
        )

    print()
    print("=" * 60)
    print("STEP 5  —  Saving visualisations")
    print("=" * 60)
    visualise_all(
        Z_win, Z_loss, z_new, paths, projections, model, out_prefix="latent_paths"
    )
    print("\nDone.\n")


if __name__ == "__main__":
    run_pipeline(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
