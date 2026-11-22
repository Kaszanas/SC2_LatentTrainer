"""
Path-charting CLI: latent-space improvement guidance for a losing player.

Loads a trained guided VAE, encodes a game, charts a path from the loser's
latent toward the winning region, decodes it to feature-level deltas and
plots the result. Strategies: ``linear`` (toward the winners' centroid or
k-NN mean), ``gradient_ascent`` (P(win) ascent with a KDE density prior),
``optimal_transport`` (iterative OT barycentric steps) and ``neural_flow``
(a trained OT flow-matching model). P(win) is opponent-aware throughout.

Usage: ``uv run --extra cuda python src/latent_trainer/paths/main.py gradient_ascent --top-k 15``
"""

from __future__ import annotations

from latent_trainer.paths.cli import cli

if __name__ == "__main__":
    cli()
