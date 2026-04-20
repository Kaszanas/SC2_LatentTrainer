"""Loss functions for the guided VAE training loop.

``loss_supervised`` is used by :func:`latent_trainer.models.train_model.train_guided`
for the ``suGuidedVAE`` (tabular SC2 features).
"""

import torch
import torch.nn.functional as F


def loss_supervised(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VAE loss: MSE reconstruction + KL divergence.

    Parameters
    ----------
    recon_x : Tensor
        Reconstructed features from the decoder.
    x : Tensor
        Original input features.
    mu : Tensor
        Latent mean from the encoder.
    logvar : Tensor
        Latent log-variance from the encoder.

    Returns
    -------
    (total_loss, mse_loss)
        Total loss is MSE + KLD, with both computed as means over the batch.
    """
    MSE = F.mse_loss(recon_x, x, reduction="mean")
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD, MSE
