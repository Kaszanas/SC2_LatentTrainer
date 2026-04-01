"""Lightning modules for the latent-trainer project."""

from latent_trainer.models.lightning.lit_classifier import LatentClassifier
from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE
from latent_trainer.models.lightning.lit_vae import LitVAE

__all__ = ["LatentClassifier", "LitGuidedVAE", "LitVAE"]
