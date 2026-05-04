"""latent_trainer.paths — path-generation strategies and feedback analysis."""

from latent_trainer.paths.feedback import compute_feedback, print_feedback_report
from latent_trainer.paths.strategies import (
    PathStrategy,
    path_gradient_ascent,
    path_linear,
    path_neural_flow,
    path_optimal_transport,
)

__all__ = [
    "PathStrategy",
    "path_linear",
    "path_optimal_transport",
    "path_gradient_ascent",
    "path_neural_flow",
    "compute_feedback",
    "print_feedback_report",
]
