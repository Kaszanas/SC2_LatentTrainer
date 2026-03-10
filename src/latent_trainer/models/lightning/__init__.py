"""Lightning modules for the latent-trainer project."""

from latent_trainer.models.lightning.lit_classifier import LitClassifier
from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE
from latent_trainer.models.lightning.lit_vae import LitVAE

__all__ = ["LitClassifier", "LitGuidedVAE", "LitVAE"]
