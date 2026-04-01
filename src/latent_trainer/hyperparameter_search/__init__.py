"""HPO (Hyper-Parameter Optimisation) package.

Re-exports all public entry-points so callers can import from either the
package root or the specific submodule::

    # Both of these work:
    from latent_trainer.hyperparameter_search import run_hyperparameter_search
    from latent_trainer.hyperparameter_search.two_stage import run_hyperparameter_search
"""

from latent_trainer.hyperparameter_search.guided_vae import (
    run_guided_vae_best,
    run_guided_vae_hyperparameter_search,
)
from latent_trainer.hyperparameter_search.two_stage import (
    run_two_stage_best,
    run_two_stage_hyperparameter_search,
    train_two_stage_pipeline,
)

__all__ = [
    "run_guided_vae_best",
    "run_guided_vae_hyperparameter_search",
    "run_two_stage_hyperparameter_search",
    "run_two_stage_best",
    "train_two_stage_pipeline",
]
