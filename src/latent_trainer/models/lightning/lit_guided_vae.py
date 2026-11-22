"""Lightning module training the guided VAE with adversarial disentanglement."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from latent_trainer.models.guided_vae import Classifier, suGuidedVAE
from latent_trainer.models.lightning.guided_base import GuidedAdversarialModule
from latent_trainer.models.losses import loss_supervised


class LitGuidedVAE(GuidedAdversarialModule):
    """
    Lightning module for supervised guided VAE training.

    Uses the three-step adversarial loop of :class:`GuidedAdversarialModule`:
    (1) the VAE on reconstruction + KL + weighted outcome BCE, (2) the adversary
    on the detached free dims, (3) the VAE to push the adversary towards 0.5.
    Both heads output probabilities.

    Parameters
    ----------
    input_dim : int
        Input features per player.
    encoder_hidden_dims : list[int]
        Encoder hidden widths; the decoder mirrors them.
    supervised_dim : int
        Latent dims per player used by the outcome classifier.
    latent_dim : int
        Size of the latent distribution per player.
    learning_rate : float
        Learning rate of the VAE and adversarial optimizers, by default 1e-4.
    weight_decay : float
        Weight decay of the VAE and adversarial optimizers, by default 1e-5.
    learning_rate_cls : float
        Learning rate of the adversary's optimizer, by default 1e-4.
    weight_decay_cls : float
        Weight decay of the adversary's optimizer, by default 1e-4.
    classification_weight : float
        Weight of the outcome and adversarial BCE terms, by default 50.0.
    mean : torch.Tensor | None
        Training feature mean, stored as a buffer for inference, by default None.
    std : torch.Tensor | None
        Training feature std, stored as a buffer for inference, by default None.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: list[int],
        supervised_dim: int,
        latent_dim: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        learning_rate_cls: float = 1e-4,
        weight_decay_cls: float = 1e-4,
        classification_weight: float = 50.0,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Dimensions:
        self.input_dim = input_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.supervised_dim = supervised_dim
        self.latent_dim = latent_dim

        # Normalization stats, needed for working with unnormalized data
        # after training (e.g. when encoding new samples with the trained model):
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        # Hyperparameters:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.learning_rate_cls = learning_rate_cls
        self.weight_decay_cls = weight_decay_cls
        self.classification_weight = classification_weight

        # Manual optimisation (3 optimizers), required for the
        # alternating adversarial training steps:
        self.automatic_optimization = False

        self.model = suGuidedVAE(
            latent_dim=latent_dim,
            input_dim=input_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            supervised_dim=supervised_dim,
        )

        # Adversarial classifier operating on the non-supervised latent dims for adversarial disentanglement.
        # Pushes the VAE to learn class-discriminative information in the supervised dims, leaving the rest
        # free of class information and hopefully more disentangled:
        self.adversarial_classifier = Classifier(
            latent_dim=latent_dim,
            supervised_dim=supervised_dim,
        )

    # Hooks of GuidedAdversarialModule:
    def _unpack(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split ``(features [B, 2, F], labels [B])`` and prepare the labels.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            Features and labels.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Features and ``[B, 1]`` float labels.
        """
        return batch[0], self._prepare_label(label=batch[1])

    def _vae_forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Reconstruct the features and predict the outcome.

        Parameters
        ----------
        inputs : torch.Tensor
            Features ``[B, 2, F]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict]
            MSE + KL loss, win probability ``[B, 1]``, no extra logs.
        """
        recon_batch, mu, logvar, probability = self.model(inputs)
        vae_loss = loss_supervised(recon_batch, inputs, mu, logvar)[0]
        return vae_loss, probability, {}

    def _outcome_loss(
        self, outcome: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Return the BCE of a probability against a target.

        Parameters
        ----------
        outcome : torch.Tensor
            Win probability ``[B, 1]``.
        target : torch.Tensor
            Target ``[B, 1]``.

        Returns
        -------
        torch.Tensor
            Mean binary cross-entropy.
        """
        return F.binary_cross_entropy(outcome, target, reduction="mean")

    def _predicts_player_1(self, outcome: torch.Tensor) -> torch.Tensor:
        """
        Threshold a probability at 0.5.

        Parameters
        ----------
        outcome : torch.Tensor
            Win probability ``[B, 1]``.

        Returns
        -------
        torch.Tensor
            0/1 predictions as float.
        """
        return (outcome > 0.5).float()

    def _sample_z(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode and sample per-player latents.

        Parameters
        ----------
        inputs : torch.Tensor
            Features ``[B, 2, F]``.

        Returns
        -------
        torch.Tensor
            Latents ``[B, 2, latent_dim]``.
        """
        mu, logvar = self.model.encode(inputs)
        return self.model.reparameterize(mu, logvar)

    def forward(self, x: torch.Tensor):  # noqa: D401
        """
        Run the underlying :class:`suGuidedVAE`.

        Parameters
        ----------
        x : torch.Tensor
            Features ``[B, 2, F]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ``(reconstruction, mu, logvar, win_probability)``.
        """
        return self.model(x)
