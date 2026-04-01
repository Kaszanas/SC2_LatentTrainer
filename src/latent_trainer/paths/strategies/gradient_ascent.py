# ──────────────────────────────────────────────────────────────
# Strategy: gradient ascent (+ KDE density regularisation)
# ──────────────────────────────────────────────────────────────
from typing import Callable

import numpy as np
import torch
from sklearn.neighbors import KernelDensity


def path_gradient_ascent(
    z_start: np.ndarray,
    *,
    score_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    logit_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    Z_all: np.ndarray,
    steps: int = 500,
    lr: float = 0.02,
    momentum: float = 0.9,
    density_weight: float = 0.3,
    kde_bandwidth: float = 0.5,
    n_waypoints: int = 10,
    convergence_threshold: float = 0.95,
    device: torch.device | None = None,
) -> np.ndarray:
    """Gradient ascent on P(win) regularised by KDE density.

    At each step the total gradient is::

        total_grad = grad_logit + density_weight * grad_kde

    The logit gradient comes from backpropagation through *logit_fn*.
    The KDE density gradient is computed via numerical central
    differences, keeping the path on the data manifold.

    You must provide at least one of *score_fn* or *logit_fn*:

    - *logit_fn* — pre-sigmoid logit; preferred because it gives
      stronger gradients far from the decision boundary.
    - *score_fn* — P(win) after sigmoid; used as fallback and for
      convergence checking.

    Parameters
    ----------
    z_start:
        Starting latent code, shape ``(latent_dim,)``.
    score_fn:
        ``z (N, D) → P(win) (N, 1)`` or ``(N,)``.
    logit_fn:
        ``z (N, D) → logit (N, 1)`` or ``(N,)``.  If not provided,
        *score_fn* is used for backprop instead.
    Z_all:
        All training-set latent codes, shape ``(N, latent_dim)``.
    steps:
        Maximum gradient ascent iterations.
    lr:
        Step size.
    momentum:
        Momentum coefficient for velocity-based updates.
    density_weight:
        Weight of the KDE density gradient relative to the logit gradient.
    kde_bandwidth:
        Bandwidth for the Gaussian KDE.
    n_waypoints:
        Number of evenly-spaced waypoints to resample from the trajectory.
    convergence_threshold:
        Stop early when P(win) exceeds this value.
    device:
        Torch device.  Defaults to CPU.

    Returns
    -------
    np.ndarray
        Path of shape ``(n_waypoints, latent_dim)``.
    """
    if score_fn is None and logit_fn is None:
        raise ValueError("Provide at least one of score_fn or logit_fn.")

    if device is None:
        device = torch.device("cpu")

    # Decide which function to backprop through
    backprop_fn = logit_fn if logit_fn is not None else score_fn

    kde = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth).fit(Z_all)
    z = torch.tensor(
        z_start.copy(), dtype=torch.float32, requires_grad=True, device=device
    )
    velocity = torch.zeros_like(z)
    trajectory = [z_start.copy()]

    for step in range(steps):
        if z.grad is not None:
            z.grad.zero_()

        out = backprop_fn(z.unsqueeze(0))
        out.backward()

        with torch.no_grad():
            grad_cls = z.grad.clone()

            # KDE density gradient via numerical central differences
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
                grads_kde, dtype=torch.float32, device=device
            )
            velocity = momentum * velocity + lr * total_grad
            z = z.detach() + velocity
            z.requires_grad_(True)
            trajectory.append(z.detach().cpu().numpy().copy())

            # Convergence check
            if score_fn is not None:
                current_p = score_fn(z.unsqueeze(0)).detach().squeeze().item()
            else:
                current_p = (
                    torch.sigmoid(logit_fn(z.unsqueeze(0))).detach().squeeze().item()
                )

            if step % 50 == 0:
                print(
                    f"    step {step:4d}  P(win)={current_p:.4f}"
                    f"  |grad|={grad_cls.norm().item():.5f}"
                )
            if current_p > convergence_threshold:
                print(f"  [GA]  Converged at step {step}  P(win)={current_p:.4f}")
                break

    trajectory = np.array(trajectory)
    indices = np.linspace(0, len(trajectory) - 1, n_waypoints, dtype=int)
    return trajectory[indices]
