"""Latent-space path generation strategies.

Four strategies for finding counterfactual improvement paths through a
model's latent space.  All strategies return numpy arrays of shape
``(n_waypoints, latent_dim)``.

The strategies are model-agnostic: they accept **callable** score/logit
functions rather than concrete model classes.  This lets the same code
work with both the two-stage pipeline (``LitVAE`` + ``LitClassifier``,
opponent-aware) and the single-model pipeline
(``VAEClassifierLightning``).

Example — two-stage (opponent-aware)::

    from functools import partial

    def _score(z: torch.Tensor, *, classifier, opponent_z, player_idx) -> torch.Tensor:
        opp = opponent_z.expand(z.shape[0], -1)
        if player_idx == 0:
            inp = torch.cat([z, opp], dim=1)
        else:
            inp = torch.cat([opp, z], dim=1)
        return classifier(inp)

    score_fn = partial(_score, classifier=clf, opponent_z=opp_z, player_idx=0)
    path = path_gradient_ascent(z_start, score_fn=score_fn, Z_all=Z_all)

Example — single model (VAEClassifierLightning)::

    path = path_gradient_ascent(
        z_start,
        score_fn=model.p_win,
        logit_fn=model.logit_win,
        Z_all=Z_all,
    )
"""

from __future__ import annotations

import enum


class PathStrategy(enum.Enum):
    """Available path-finding strategies."""

    LINEAR = "linear"
    OPTIMAL_TRANSPORT = "optimal_transport"
    GRADIENT_ASCENT = "gradient_ascent"
    GEODESIC = "geodesic"
