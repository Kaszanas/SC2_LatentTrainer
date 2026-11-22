"""Gradient ascent on P(win) regularised by latent density."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def _kde_log_density(
    z: torch.Tensor,
    Z_ref: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    """
    Differentiable Gaussian KDE log-density (up to a constant), on device.

    Parameters
    ----------
    z : torch.Tensor
        Query point, ``[D]``.
    Z_ref : torch.Tensor
        Reference points, ``[N, D]``.
    bandwidth : float
        Gaussian kernel bandwidth.

    Returns
    -------
    torch.Tensor
        Scalar log-density.
    """
    diffs = z.unsqueeze(0) - Z_ref  # (N, D)
    return torch.logsumexp(-0.5 * (diffs / bandwidth).pow(2).sum(-1), dim=0)


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
    """
    Climb P(win) with momentum, regularised by a differentiable KDE density.

    Gradients of the classifier and of the KDE come from autograd; runs on GPU if
    available. Stops early once P(win) exceeds ``convergence_threshold``.

    Parameters
    ----------
    z_start : np.ndarray
        Starting latent, ``[D]``.
    score_fn : Callable[[torch.Tensor], torch.Tensor] | None
        P(win) of a latent; used for gradients if ``logit_fn`` is None and for
        the stopping check.
    logit_fn : Callable[[torch.Tensor], torch.Tensor] | None
        Win logit, preferred for gradients (no sigmoid saturation).
    Z_all : np.ndarray
        Reference latents for the density term, ``[N, D]``.
    steps : int
        Maximum iterations, by default 500.
    lr : float
        Step size, by default 0.02.
    momentum : float
        Momentum coefficient, by default 0.9.
    density_weight : float
        Weight of the KDE gradient, by default 0.3.
    kde_bandwidth : float
        KDE bandwidth, by default 0.5.
    n_waypoints : int
        Evenly spaced waypoints returned from the trajectory, by default 10.
    convergence_threshold : float
        P(win) at which to stop, by default 0.95.
    device : torch.device | None
        Device; None picks CUDA if available.

    Returns
    -------
    np.ndarray
        Waypoints ``[n_waypoints, D]`` from start to end.

    Raises
    ------
    ValueError
        If neither ``score_fn`` nor ``logit_fn`` is given.
    """
    if score_fn is None and logit_fn is None:
        raise ValueError("Provide at least one of score_fn or logit_fn.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backprop_fn = logit_fn if logit_fn is not None else score_fn

    Z_ref = torch.tensor(Z_all, dtype=torch.float32, device=device)
    z = torch.tensor(
        z_start.copy(), dtype=torch.float32, device=device, requires_grad=True
    )
    velocity = torch.zeros_like(z)
    trajectory = [z_start.copy()]

    for step in range(steps):
        # Classifier gradient (backprop through logit/score)
        if z.grad is not None:
            z.grad.zero_()
        logit_out = backprop_fn(z.unsqueeze(0))
        logit_out.backward()
        grad_cls = z.grad.detach().clone()

        # KDE density gradient (fresh autograd graph, outside no_grad)
        z_kde = z.detach().requires_grad_(True)
        log_dens = _kde_log_density(z_kde, Z_ref, kde_bandwidth)
        log_dens.backward()
        grad_kde = z_kde.grad.detach().clone()

        # Parameter update (no autograd needed)
        with torch.no_grad():
            total_grad = grad_cls + density_weight * grad_kde
            velocity = momentum * velocity + lr * total_grad
            z = (z.detach() + velocity).requires_grad_(True)
            trajectory.append(z.detach().cpu().numpy().copy())

        # Convergence check
        if step % 50 == 0 or step < 5:
            with torch.no_grad():
                if score_fn is not None:
                    current_p = score_fn(z.unsqueeze(0)).squeeze().item()
                else:
                    current_p = torch.sigmoid(logit_fn(z.unsqueeze(0))).squeeze().item()
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
