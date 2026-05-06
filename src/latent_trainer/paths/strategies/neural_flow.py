"""Neural flow path strategy using the trained OT-Flow Matching model."""

from __future__ import annotations

import numpy as np

from latent_trainer.paths.flow import LitOTFlowMatching


def path_neural_flow(
    z_start: np.ndarray,
    flow_model: LitOTFlowMatching,
    n_waypoints: int = 20,
) -> np.ndarray:
    """Generate a counterfactual path via Euler integration of the learned velocity field.

    Parameters
    ----------
    z_start:
        Starting (losing) latent vector, shape ``(latent_dim,)``.
    flow_model:
        Trained :class:`~latent_trainer.paths.flow.LitOTFlowMatching` instance.
    n_waypoints:
        Number of integration steps; output has shape ``(n_waypoints + 1, latent_dim)``.

    Returns
    -------
    np.ndarray
        Path array of shape ``(n_waypoints + 1, latent_dim)``.
    """
    # predict_path returns steps+1 points; pass steps-1 so output is n_waypoints points,
    # matching the (n_waypoints, latent_dim) contract of all other strategies.
    return flow_model.predict_path(z_start, steps=n_waypoints - 1)
