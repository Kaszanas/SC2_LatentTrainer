"""Multi-method path-charting comparison package."""

from latent_trainer.paths.compare.aggregate import MethodStats, summarise
from latent_trainer.paths.compare.cross_dataset import (
    CrossDatasetComparison,
    build_cross_dataset,
)
from latent_trainer.paths.compare.orchestrator import run_comparison
from latent_trainer.paths.compare.results import (
    DEFAULT_METHOD_SPECS,
    ComparisonReport,
    MethodSpec,
    PathRunResult,
)

__all__ = [
    "ComparisonReport",
    "CrossDatasetComparison",
    "DEFAULT_METHOD_SPECS",
    "MethodSpec",
    "MethodStats",
    "PathRunResult",
    "build_cross_dataset",
    "run_comparison",
    "summarise",
]
