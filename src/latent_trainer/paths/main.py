"""Feedback path finder: latent-space improvement guidance for SC2 players.

Loads the trained two-stage VAE, embeds player data, finds an improvement
path from a losing sample toward the winning class density, reconstructs
feature-level deltas, and produces visualisations.

Four strategies are available:

``linear``
    Linear interpolation toward a target (centroid or k-NN mean).
    Reports a single raw delta.  Fast and deterministic.

``gradient_ascent``
    Gradient ascent on P(win) regularised by a KDE density prior.
    The path stays on the data manifold while climbing toward the
    winning region.  Reports three complementary feedback signals:
    raw delta, minimum-viable delta, and P(win)-gain-weighted delta.

``optimal_transport``
    Finds a Wasserstein-barycentric target in the winning cloud
    and interpolates toward it.  Considers the full shape of the
    winning distribution instead of a single centroid.

All strategies are **opponent-aware** — P(win) is computed using the
full ``(player_z, opponent_z)`` input to the classifier (for the
two-stage model).

Usage::

    # Linear interpolation toward win centroid:
    uv run python feedback_path.py linear --method centroid

    # Gradient ascent + KDE:
    uv run python feedback_path.py gradient-ascent --top-k 15

    # Optimal transport:
    uv run python feedback_path.py optimal-transport

    # Custom GA hyperparameters:
    uv run python feedback_path.py gradient-ascent \\
        --ga-steps 800 --ga-lr 0.01 --density-weight 0.5

    # Global options come before the sub-command:
    uv run python feedback_path.py --player 2 --top-k 15 gradient-ascent
"""

from __future__ import annotations

from latent_trainer.paths.cli import cli

if __name__ == "__main__":
    cli()
