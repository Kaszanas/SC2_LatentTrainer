"""Feedback path finder: latent-space improvement guidance for SC2 players.

Loads the trained two-stage VAE, embeds player data, finds an improvement
path from a losing sample toward the winning class density, reconstructs
feature-level deltas, and produces visualisations.

Two strategies are available:

``linear``
    Linear interpolation toward a target (centroid or k-NN mean).
    Reports a single raw delta.  Fast and deterministic.

``gradient_kde``
    Gradient ascent on P(win) regularised by a KDE density prior.
    The path stays on the data manifold while climbing toward the
    winning region.  Reports three complementary feedback signals:
    raw delta, minimum-viable delta, and P(win)-gain-weighted delta.

    The gradient has two components at each step::

        total_grad = grad_classifier + density_weight * grad_kde

    - ``grad_classifier``: direction that increases P(win) fastest
      (backprop through the opponent-aware binary classifier).
    - ``grad_kde``: direction that increases log-density under a
      Gaussian KDE fitted to all training latents (numerical gradient).

    The density term prevents the path from leaving the data manifold,
    ensuring that decoded features remain realistic.

Both strategies are **opponent-aware** — P(win) is computed using the
full ``(player_z, opponent_z)`` input to the classifier.

Usage::

    # Linear interpolation toward win centroid:
    uv run python feedback_path.py --strategy linear --method centroid

    # Gradient ascent + KDE with 15 top features:
    uv run python feedback_path.py --strategy gradient_kde --top-k 15

    # Custom GA hyperparameters:
    uv run python feedback_path.py --strategy gradient_kde \\
        --ga-steps 800 --ga-lr 0.01 --density-weight 0.5
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from train_two_stage import LatentClassifier, SimpleVAE, load_and_normalize

# REVIEW: This is not integrated into project structure.
# REVIEW: Not using click as in the other pieces of code.
# REVIEW:

# ---------------------------------------------------------------------------
# Feature names -- 203 per player
# ---------------------------------------------------------------------------

_ECON_FIELDS: list[str] = [
    "foodMade",
    "foodUsed",
    "mineralsCollectionRate",
    "mineralsCurrent",
    "mineralsFriendlyFireArmy",
    "mineralsFriendlyFireEconomy",
    "mineralsFriendlyFireTechnology",
    "mineralsKilledArmy",
    "mineralsKilledEconomy",
    "mineralsKilledTechnology",
    "mineralsLostArmy",
    "mineralsLostEconomy",
    "mineralsLostTechnology",
    "mineralsUsedActiveForces",
    "mineralsUsedCurrentArmy",
    "mineralsUsedCurrentEconomy",
    "mineralsUsedCurrentTechnology",
    "mineralsUsedInProgressArmy",
    "mineralsUsedInProgressEconomy",
    "mineralsUsedInProgressTechnology",
    "vespeneCollectionRate",
    "vespeneCurrent",
    "vespeneFriendlyFireArmy",
    "vespeneFriendlyFireEconomy",
    "vespeneFriendlyFireTechnology",
    "vespeneKilledArmy",
    "vespeneKilledEconomy",
    "vespeneKilledTechnology",
    "vespeneLostArmy",
    "vespeneLostEconomy",
    "vespeneLostTechnology",
    "vespeneUsedActiveForces",
    "vespeneUsedCurrentArmy",
    "vespeneUsedCurrentEconomy",
    "vespeneUsedCurrentTechnology",
    "vespeneUsedInProgressArmy",
    "vespeneUsedInProgressEconomy",
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
# Core shared logic
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


def _nearest_winning_target(sample_z, win_latents, k=5) -> torch.Tensor:
    dists = torch.cdist(sample_z.unsqueeze(0), win_latents.unsqueeze(0)).squeeze(0)
    _, indices = dists.topk(k, largest=False)
    return win_latents[indices.squeeze()].mean(dim=0)


def _opponent_aware_p_win(
    classifier,
    player_z,
    opponent_z,
    player_idx,
) -> np.ndarray:
    """Compute P(win) for each row in player_z (opponent_z is broadcast).

    Parameters
    ----------
    classifier:
        Trained ``LatentClassifier`` taking ``(player1_z, player2_z)``.
    player_z:
        ``(N, latent_dim)`` tensor of the *player* latent codes.
    opponent_z:
        ``(latent_dim,)`` tensor — the opponent's latent code (fixed).
    player_idx:
        0 for player-1, 1 for player-2.

    Returns
    -------
    np.ndarray of shape ``(N,)`` with P(win) values.
    """
    n = player_z.shape[0]
    opp = opponent_z.unsqueeze(0).expand(n, -1)
    if player_idx == 0:
        combined = torch.cat([player_z, opp], dim=1)
    else:
        combined = torch.cat([opp, player_z], dim=1)
    with torch.no_grad():
        return classifier(combined).squeeze(-1).numpy()


# ---------------------------------------------------------------------------
# Strategy: linear
#
# The simplest approach: pick a target in the winning region (centroid
# or k-NN average), then linearly interpolate from the losing sample.
# Fast, deterministic, but the straight line may cut through low-density
# regions where decoded features are unrealistic.
# ---------------------------------------------------------------------------


def _find_path_linear(start_z, target_z, n_steps) -> torch.Tensor:
    """Straight-line interpolation in latent space.

    Returns ``n_steps`` evenly-spaced points along the line from
    ``start_z`` to ``target_z``.
    """
    alphas = torch.linspace(0.0, 1.0, n_steps).unsqueeze(1)
    return start_z + alphas * (target_z - start_z)


# ---------------------------------------------------------------------------
# Strategy: gradient_kde
#
# Instead of picking a fixed target, this strategy uses gradient ascent
# to iteratively move the player's latent code toward higher P(win).
#
# At each step the total gradient is:
#   total_grad = grad_classifier  +  density_weight * grad_kde
#
# - grad_classifier comes from backprop through the classifier.
#   Because our classifier takes (player_z, opponent_z), only the
#   player dimensions carry gradient; the opponent is detached.
#
# - grad_kde comes from numerical differentiation of the log-density
#   under a Gaussian KDE fitted to all training latent codes.  This
#   regularises the path to stay in data-dense regions.
#
# Momentum-based updates (velocity) smooth out the trajectory.
# The path is downsampled to n_waypoints for uniform spacing.
# ---------------------------------------------------------------------------
def _gradient_ascent_path(
    vae,  # REVIEW: Why passing VAE if it is not used?
    classifier,
    sample_z: torch.Tensor,
    opponent_z: torch.Tensor,
    all_latents: np.ndarray,
    player_idx: int,
    *,
    steps: int = 500,
    lr: float = 0.02,
    momentum: float = 0.9,
    density_weight: float = 0.3,
    kde_bandwidth: float = 0.5,
    n_waypoints: int = 20,
    convergence_threshold: float = 0.95,
) -> np.ndarray:
    """Gradient ascent on P(win) regularised by KDE density.

    The gradient is backpropagated through the **opponent-aware**
    classifier: ``classifier(cat(player_z, opponent_z))``.  Only the
    player's latent is updated; the opponent is frozen.

    A KDE density term steers the path toward regions of high training
    data density, keeping it on the data manifold.

    Parameters
    ----------
    vae:
        Trained VAE (not used for gradient, but kept for interface symmetry).
    classifier:
        Trained ``LatentClassifier(latent_dim)`` — takes concatenated
        ``(player_z, opponent_z)`` input.
    sample_z:
        Starting latent code for the player  ``(latent_dim,)``.
    opponent_z:
        Opponent's latent code (fixed throughout) ``(latent_dim,)``.
    all_latents:
        ``(N, latent_dim)`` numpy array of all training latent codes
        (used to fit the KDE).
    player_idx:
        0 for player-1, 1 for player-2.
    steps:
        Maximum number of gradient ascent steps.
    lr:
        Learning rate for the update.
    momentum:
        Momentum coefficient for velocity-based updates.
    density_weight:
        Weighting of the KDE density gradient relative to the
        classifier gradient.
    kde_bandwidth:
        Bandwidth for the Gaussian KDE.
    n_waypoints:
        Number of evenly-spaced waypoints to resample from the
        full trajectory.
    convergence_threshold:
        Stop early when P(win) exceeds this value.

    Returns
    -------
    np.ndarray of shape ``(n_waypoints, latent_dim)``
    """
    # Fit KDE on all training latents
    kde = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth).fit(all_latents)

    # Initialise player latent as a gradient-tracked tensor
    z = sample_z.clone().detach().float().requires_grad_(True)
    opp = opponent_z.clone().detach().float()
    velocity = torch.zeros_like(z)
    trajectory = [sample_z.detach().cpu().numpy().copy()]

    classifier.eval()
    for step in range(steps):
        if z.grad is not None:
            z.grad.zero_()

        # Forward through opponent-aware classifier
        if player_idx == 0:
            combined = torch.cat([z.unsqueeze(0), opp.unsqueeze(0)], dim=1)
        else:
            combined = torch.cat([opp.unsqueeze(0), z.unsqueeze(0)], dim=1)

        logit = classifier(combined)  # (1, 1)
        logit.backward()

        with torch.no_grad():
            # ── Classifier gradient: direction that increases P(win)
            grad_cls = z.grad.clone()

            # ── KDE density gradient: numerical central difference
            #    For each latent dimension i, perturb z[i] by ±eps and
            #    measure the change in log-density.  This steers the path
            #    toward high-density regions of the training distribution.
            z_np = z.detach().cpu().numpy()
            grads_kde = np.zeros_like(z_np)
            eps = 1e-3
            for i in range(len(z_np)):
                zp, zm = z_np.copy(), z_np.copy()
                zp[i] += eps
                zm[i] -= eps
                grads_kde[i] = float(
                    kde.score_samples(zp.reshape(1, -1))[0]
                    - kde.score_samples(zm.reshape(1, -1))[0]
                ) / (2 * eps)

            total_grad = grad_cls + density_weight * torch.tensor(
                grads_kde,
                dtype=torch.float32,
            )
            velocity = momentum * velocity + lr * total_grad
            z = z.detach() + velocity
            z.requires_grad_(True)
            trajectory.append(z.detach().cpu().numpy().copy())

            # Check convergence
            if player_idx == 0:
                comb_check = torch.cat([z.unsqueeze(0), opp.unsqueeze(0)], dim=1)
            else:
                comb_check = torch.cat([opp.unsqueeze(0), z.unsqueeze(0)], dim=1)
            current_p = torch.sigmoid(classifier(comb_check)).item()

            if step % 50 == 0:
                print(
                    f"    step {step:4d}  P(win)={current_p:.4f}"
                    f"  |grad|={grad_cls.norm().item():.5f}"
                )
            if current_p > convergence_threshold:
                print(f"  [GA+KDE]  Converged at step {step}  P(win)={current_p:.4f}")
                break

    trajectory = np.array(trajectory)
    # Resample to n_waypoints
    indices = np.linspace(0, len(trajectory) - 1, n_waypoints, dtype=int)
    path = trajectory[indices]

    # Final P(win)
    final_z = torch.tensor(path[-1], dtype=torch.float32)
    if player_idx == 0:
        comb_final = torch.cat([final_z.unsqueeze(0), opp.unsqueeze(0)], dim=1)
    else:
        comb_final = torch.cat([opp.unsqueeze(0), final_z.unsqueeze(0)], dim=1)
    with torch.no_grad():
        final_p = torch.sigmoid(classifier(comb_final)).item()
    print(
        f"  [GA+KDE]  Final P(win): {final_p:.4f}"
        f"  |  steps taken: {len(trajectory) - 1}"
    )
    return path


# ---------------------------------------------------------------------------
# Three-signal feedback (used by gradient_kde strategy)
# ---------------------------------------------------------------------------


def _compute_three_signal_feedback(
    vae,
    classifier,
    path_z: np.ndarray,
    opponent_z: torch.Tensor,
    player_idx: int,
    norm_mean,
    norm_std,
    feature_names: list[str],
    top_k: int = 10,
) -> dict:
    """Compute three complementary feedback signals along a path.

    1. **Raw delta** (start → end) — overall direction of change.
    2. **Minimum-viable delta** — change only up to P(win)=0.5 crossover.
    3. **P(win)-gain-weighted delta** — features that moved *while* P(win) rose.

    All P(win) values are opponent-aware.

    Returns a dict with ranked tables and raw arrays for all three signals.
    """
    # Decode path to feature space
    path_tensor = torch.tensor(path_z, dtype=torch.float32)
    with torch.no_grad():
        x_decoded = vae.decode(path_tensor)
    x_orig = (x_decoded * norm_std + norm_mean).cpu().numpy()

    # Compute opponent-aware P(win) along path
    p_vals = _opponent_aware_p_win(classifier, path_tensor, opponent_z, player_idx)

    x_start = x_orig[0]
    x_end = x_orig[-1]

    # ── Signal 1: Raw start → end delta
    raw_delta = x_end - x_start

    # ── Signal 2: Minimum-viable delta (stop at P(win) ≥ 0.5 crossover)
    cross_idx = np.where(np.diff(np.sign(p_vals - 0.5)))[0]
    if len(cross_idx) > 0:
        threshold_wp = cross_idx[0] + 1
        mv_delta = x_orig[threshold_wp] - x_start
        mv_p = p_vals[threshold_wp]
        mv_label = f"waypoint {threshold_wp}/{len(p_vals) - 1}  (P(win)={mv_p:.3f})"
    else:
        mid = len(p_vals) // 2
        mv_delta = x_orig[mid] - x_start
        mv_p = p_vals[mid]
        mv_label = f"midpoint (P(win)={mv_p:.3f}, no crossing found)"

    # ── Signal 3: P(win)-gain-weighted delta
    p_gains = np.diff(p_vals).clip(min=0)
    x_steps = np.diff(x_orig, axis=0)
    if p_gains.sum() > 0:
        weighted_delta = (x_steps * p_gains[:, None]).sum(0) / p_gains.sum()
    else:
        weighted_delta = raw_delta.copy()

    # ── Build ranked tables
    def _rank_table(delta, label):
        ranked = np.argsort(np.abs(delta))[::-1]
        rows = []
        for i in ranked[:top_k]:
            rows.append(
                {
                    "priority": int(np.where(ranked == i)[0][0]) + 1,
                    "feature": feature_names[i],
                    "current": float(x_start[i]),
                    "target": float(x_start[i] + delta[i]),
                    "delta": float(delta[i]),
                    "direction": "▲" if delta[i] > 0 else "▼",
                }
            )
        return rows, label

    raw_rows, raw_lbl = _rank_table(raw_delta, "Full path  (start → end)")
    mv_rows, mv_lbl = _rank_table(mv_delta, f"Minimum viable  ({mv_label})")
    wgt_rows, wgt_lbl = _rank_table(weighted_delta, "P(win)-gain weighted")

    return {
        "raw": raw_rows,
        "raw_label": raw_lbl,
        "minimum_viable": mv_rows,
        "mv_label": mv_lbl,
        "gain_weighted": wgt_rows,
        "wgt_label": wgt_lbl,
        "_raw_delta": raw_delta,
        "_mv_delta": mv_delta,
        "_weighted_delta": weighted_delta,
        "_x_start": x_start,
        "_x_orig": x_orig,
        "_p_vals": p_vals,
        "_mv_crossover_wp": int(cross_idx[0]) if len(cross_idx) > 0 else None,
    }


def _print_feedback_report(feedback: dict, top_k: int = 10) -> None:
    """Pretty-print the three-signal feedback table to stdout."""
    header = f"  {'Feature':<22}  {'Current':>9}  {'Target':>9}  {'Δ':>9}"
    sep = "  " + "─" * 54

    print(f"\n{'═' * 60}")
    print(f"  [GA+KDE] FEEDBACK REPORT  —  top {top_k} features")
    print(f"{'═' * 60}")

    for key, lbl_key in [
        ("raw", "raw_label"),
        ("minimum_viable", "mv_label"),
        ("gain_weighted", "wgt_label"),
    ]:
        print(f"\n  ── {feedback[lbl_key]}")
        print(header)
        print(sep)
        for r in feedback[key]:
            print(
                f"  {r['feature']:<22}  {r['current']:>9.3f}"
                f"  {r['target']:>9.3f}  {r['direction']} {abs(r['delta']):>7.3f}"
            )


# ---------------------------------------------------------------------------
# Visualisation (shared)
# ---------------------------------------------------------------------------


# REVIEW: Type hints
def _plot_main(win_c, loss_c, path_c, alphas, win_probs, pca, save_path):
    """Two-panel figure: latent space + P(win) curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Latent Space — Counterfactual Improvement Path",
        fontsize=15,
        fontweight="bold",
    )

    # --- Left: latent space ---
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

    # --- Right: P(win) curve ---
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
    ax.set_title(
        f"Top-{top_k} Feature Deltas for Improvement", fontsize=13, fontweight="bold"
    )
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


def _plot_distance(path_z, win_centroid, alphas, save_path):
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


def _plot_three_signal_feedback(feedback: dict, save_path: str, top_k: int = 10):
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
        names = [FEATURE_NAMES[i] for i in order]
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find a latent-space improvement path for a losing SC2 player."
    )
    parser.add_argument("--model", default="output/two_stage_model.pth")
    parser.add_argument("--cache", default="data/cached_dataset_rich.pt")
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        help="Index of a losing sample (default: random).",
    )
    parser.add_argument(
        "--strategy",
        choices=["linear", "gradient_kde"],
        default="gradient_kde",
        help="Path-finding strategy: 'linear' (interpolation) "
        "or 'gradient_kde' (gradient ascent + KDE density).",
    )
    # linear-only options
    parser.add_argument(
        "--method",
        choices=["centroid", "nearest"],
        default="centroid",
        help="Target for linear strategy: 'centroid' or 'nearest' k-NN.",
    )
    parser.add_argument(
        "--k-neighbours",
        type=int,
        default=5,
        help="k for nearest-neighbour target (linear strategy).",
    )
    # gradient_kde options
    parser.add_argument(
        "--ga-steps", type=int, default=500, help="Max gradient ascent steps."
    )
    parser.add_argument(
        "--ga-lr", type=float, default=0.02, help="Gradient ascent learning rate."
    )
    parser.add_argument(
        "--ga-momentum", type=float, default=0.9, help="Gradient ascent momentum."
    )
    parser.add_argument(
        "--density-weight",
        type=float,
        default=0.3,
        help="Weight of the KDE density gradient.",
    )
    parser.add_argument(
        "--kde-bandwidth", type=float, default=0.5, help="Bandwidth for Gaussian KDE."
    )
    # shared
    parser.add_argument(
        "--n-steps", type=int, default=20, help="Number of waypoints along the path."
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of top features to display."
    )
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

    # ===================================================================
    # Strategy dispatch
    # ===================================================================
    strategy = args.strategy
    print(f"\nStrategy: {strategy}")

    if strategy == "linear":
        # ── Target ──
        if args.method == "centroid":
            target_z = win_centroid
            print("  Target: centroid")
        else:
            target_z = _nearest_winning_target(
                sample_z,
                win_latents,
                k=args.k_neighbours,
            )
            print(f"  Target: nearest (k={args.k_neighbours})")

        # ── Path ──
        path_z = _find_path_linear(sample_z, target_z, n_steps=args.n_steps)
        alphas = np.linspace(0.0, 1.0, args.n_steps)

        # ── P(win) ──
        print("Computing win probability along path...")
        win_probs = _opponent_aware_p_win(
            classifier,
            path_z,
            opponent_z,
            player_idx,
        )
        print(f"  P(win): {win_probs[0]:.3f} -> {win_probs[-1]:.3f}")

        # ── Decode ──
        print("Decoding path to feature space...")
        path_features = _decode_features(vae, path_z, norm_mean, norm_std)
        delta = path_features[-1] - path_features[0]

        # ── Report (single raw delta) ──
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

        # ── Plots ──
        print("Generating plots...")
        _plot_feature_delta(delta, args.top_k, "output/feedback_feature_delta.png")
        _plot_feature_evolution(
            path_features,
            delta,
            min(5, args.top_k),
            alphas,
            "output/feedback_feature_evolution.png",
        )

    elif strategy == "gradient_kde":
        # ── Path ──
        print("Running gradient ascent with KDE density regularisation...")
        path_z_np = _gradient_ascent_path(
            vae,
            classifier,
            sample_z,
            opponent_z,
            all_latents=all_latents.numpy(),
            player_idx=player_idx,
            steps=args.ga_steps,
            lr=args.ga_lr,
            momentum=args.ga_momentum,
            density_weight=args.density_weight,
            kde_bandwidth=args.kde_bandwidth,
            n_waypoints=args.n_steps,
        )
        path_z = torch.tensor(path_z_np, dtype=torch.float32)
        alphas = np.linspace(0.0, 1.0, args.n_steps)

        # ── P(win) ──
        win_probs = _opponent_aware_p_win(
            classifier,
            path_z,
            opponent_z,
            player_idx,
        )
        print(f"  P(win): {win_probs[0]:.3f} -> {win_probs[-1]:.3f}")

        # ── Three-signal feedback ──
        print("Computing three-signal feedback...")
        feedback = _compute_three_signal_feedback(
            vae,
            classifier,
            path_z_np,
            opponent_z,
            player_idx,
            norm_mean,
            norm_std,
            FEATURE_NAMES,
            top_k=args.top_k,
        )
        _print_feedback_report(feedback, top_k=args.top_k)

        # ── Plots ──
        print("\nGenerating plots...")
        _plot_three_signal_feedback(
            feedback,
            "output/feedback_three_signal.png",
            top_k=args.top_k,
        )
        # Also generate the feature evolution using the raw delta
        path_features = _decode_features(vae, path_z, norm_mean, norm_std)
        _plot_feature_evolution(
            path_features,
            feedback["_raw_delta"],
            min(5, args.top_k),
            alphas,
            "output/feedback_feature_evolution.png",
        )

    # --- Shared plots (both strategies) ---
    all_data = np.concatenate(
        [
            win_latents.numpy(),
            loss_latents.numpy(),
            path_z.numpy() if isinstance(path_z, torch.Tensor) else path_z,
        ]
    )
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_data)
    n_w, n_l = len(win_latents), len(loss_latents)
    win_c, loss_c, path_c = coords[:n_w], coords[n_w : n_w + n_l], coords[n_w + n_l :]

    _plot_main(
        win_c, loss_c, path_c, alphas, win_probs, pca, "output/feedback_latent_path.png"
    )

    if isinstance(path_z, torch.Tensor):
        _plot_distance(
            path_z, win_centroid, alphas, "output/feedback_distance_curve.png"
        )
    else:
        _plot_distance(
            torch.tensor(path_z, dtype=torch.float32),
            win_centroid,
            alphas,
            "output/feedback_distance_curve.png",
        )

    print("\nDone! All plots saved to output/")


if __name__ == "__main__":
    main()
