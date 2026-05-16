from __future__ import annotations

import enum


class PathStrategy(enum.Enum):
    """Available path-finding strategies."""

    LINEAR = "linear"
    OPTIMAL_TRANSPORT = "optimal_transport"
    GRADIENT_ASCENT = "gradient_ascent"
    NEURAL_FLOW = "neural_flow"
