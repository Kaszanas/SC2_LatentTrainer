"""Lightning module training the event-stream guided VAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from latent_trainer.models.event_guided_vae import PairHead, suEventGuidedVAE
from latent_trainer.models.lightning.guided_base import GuidedAdversarialModule


class LitEventGuidedVAE(GuidedAdversarialModule):
    """
    Lightning module for :class:`suEventGuidedVAE` with the guided adversarial loop.

    The VAE loss is the masked MSE over every observed slice (in the model's
    scaled channel space) plus ``beta * KL``, with ``beta`` ramping linearly
    from 0 to ``kl_beta`` over ``kl_warmup_steps`` training batches. Both
    heads output logits (safe under bf16 autocast) and are antisymmetric in
    the players. Call :meth:`fit_input_transform` on the raw training stats
    before training; the fitted scaling is saved in the checkpoint.

    Parameters
    ----------
    channel_names : list[str]
        Raw stat channel names, in cache order.
    unit_vocab : list[str]
        Unit/upgrade vocabulary (saved for inference).
    event_type_names : list[str]
        Event type vocabulary.
    race_names : list[str]
        Race vocabulary.
    d_model : int
        Model width, by default 128.
    n_heads : int
        Attention heads, by default 4.
    n_layers : int
        Level-2 transformer layers, by default 4.
    ffn_mult : int
        Feed-forward width multiple, by default 4.
    dropout : float
        Dropout, by default 0.1.
    latent_dim : int
        Latent size per player, by default 32.
    supervised_dim : int
        Latent dims per player read by the outcome head, by default 8.
    decoder_width : int
        Decoder hidden width, by default 256.
    decoder_layers : int
        Decoder hidden layers, by default 3.
    learning_rate : float
        Learning rate of the VAE optimizers, by default 3e-4.
    weight_decay : float
        Weight decay of the VAE optimizers, by default 1e-4.
    learning_rate_cls : float
        Learning rate of the adversary, by default 1e-4.
    weight_decay_cls : float
        Weight decay of the adversary, by default 1e-4.
    classification_weight : float
        Weight of the outcome and adversarial losses, by default 10.0.
    kl_beta : float
        Final KL weight, by default 1.0.
    kl_warmup_steps : int
        Training batches over which the KL weight ramps up; 0 disables the
        warm-up, by default 0.
    data_config : dict | None
        Dataset settings to record with the model (``slice_ticks``,
        ``drop_tail_frac``, ``min_crop_s``, ``max_events_per_slice``), by
        default None.
    """

    def __init__(
        self,
        channel_names: list[str],
        unit_vocab: list[str],
        event_type_names: list[str],
        race_names: list[str],
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        latent_dim: int = 32,
        supervised_dim: int = 8,
        decoder_width: int = 256,
        decoder_layers: int = 3,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        learning_rate_cls: float = 1e-4,
        weight_decay_cls: float = 1e-4,
        classification_weight: float = 10.0,
        kl_beta: float = 1.0,
        kl_warmup_steps: int = 0,
        data_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.supervised_dim = supervised_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.learning_rate_cls = learning_rate_cls
        self.weight_decay_cls = weight_decay_cls
        self.classification_weight = classification_weight
        self.kl_beta = kl_beta
        self.kl_warmup_steps = kl_warmup_steps
        self.automatic_optimization = False

        self.model = suEventGuidedVAE(
            channel_names=channel_names,
            n_units=len(unit_vocab),
            n_event_types=len(event_type_names),
            n_races=len(race_names),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_mult=ffn_mult,
            dropout=dropout,
            latent_dim=latent_dim,
            supervised_dim=supervised_dim,
            decoder_width=decoder_width,
            decoder_layers=decoder_layers,
        )
        self.adversarial_classifier = PairHead(latent_dim - supervised_dim)
        # Training batches seen, for the KL warm-up; a buffer so it is checkpointed.
        self.register_buffer("kl_step", torch.tensor(0, dtype=torch.long))

    def fit_input_transform(self, raw_train_stats: torch.Tensor) -> LitEventGuidedVAE:
        """
        Fit the model's input scaling on the raw training stat values.

        Parameters
        ----------
        raw_train_stats : torch.Tensor
            ``bundle.train.stat_values`` ``[N, C]``.

        Returns
        -------
        LitEventGuidedVAE
            ``self``.
        """
        self.model.transform.fit(raw_train_stats)
        return self

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """
        Run the underlying :class:`suEventGuidedVAE`.

        Parameters
        ----------
        batch : dict
            Output of ``collate_event_sequences``.

        Returns
        -------
        dict[str, torch.Tensor]
            See :meth:`suEventGuidedVAE.forward`.
        """
        return self.model(batch)

    def _current_beta(self) -> float:
        """
        Return the KL weight for this step (full weight outside training).

        Returns
        -------
        float
            ``kl_beta`` scaled by the warm-up progress while training.
        """
        if not self.training or self.kl_warmup_steps <= 0:
            return self.kl_beta
        return self.kl_beta * min(1.0, int(self.kl_step) / self.kl_warmup_steps)

    # Hooks of GuidedAdversarialModule:
    def _unpack(self, batch: dict) -> tuple[dict, torch.Tensor]:
        """
        Return the batch itself and its ``[B, 1]`` float labels.

        Parameters
        ----------
        batch : dict
            Output of ``collate_event_sequences``.

        Returns
        -------
        tuple[dict, torch.Tensor]
            The batch and its labels.
        """
        return batch, self._prepare_label(batch["label"])

    def _vae_forward(self, inputs: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Reconstruct every observed slice and predict the outcome.

        Parameters
        ----------
        inputs : dict
            Output of ``collate_event_sequences``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict]
            ``recon + beta * kl``, the outcome logit ``[B, 1]``, and
            ``recon_loss``, ``kl`` and ``kl_beta`` for logging.
        """
        out = self.model(inputs)
        mask = out["slice_mask"].unsqueeze(-1).float()
        n_channels = out["recon"].shape[-1]
        squared = (out["recon"].float() - out["target"].float()) ** 2 * mask
        recon_loss = squared.sum() / (mask.sum() * n_channels).clamp(min=1.0)
        mu, logvar = out["mu"].float(), out["logvar"].float()
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        beta = self._current_beta()
        if self.training:
            self.kl_step += 1
        logs = {"recon_loss": recon_loss, "kl": kl, "kl_beta": beta}
        return recon_loss + beta * kl, out["logit"], logs

    def _outcome_loss(
        self, outcome: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Return the BCE of a logit against a target.

        Parameters
        ----------
        outcome : torch.Tensor
            Logits ``[B, 1]``.
        target : torch.Tensor
            Target probabilities ``[B, 1]``.

        Returns
        -------
        torch.Tensor
            Mean binary cross-entropy.
        """
        return F.binary_cross_entropy_with_logits(outcome.float(), target)

    def _predicts_player_1(self, outcome: torch.Tensor) -> torch.Tensor:
        """
        Threshold a logit at 0.

        Parameters
        ----------
        outcome : torch.Tensor
            Logits ``[B, 1]``.

        Returns
        -------
        torch.Tensor
            0/1 predictions as float.
        """
        return (outcome > 0).float()

    def _sample_z(self, inputs: dict) -> torch.Tensor:
        """
        Encode and sample per-player latents.

        Parameters
        ----------
        inputs : dict
            Output of ``collate_event_sequences``.

        Returns
        -------
        torch.Tensor
            Latents ``[B, 2, latent_dim]``.
        """
        mu, logvar = self.model.encode(inputs)
        return self.model.reparameterize(mu, logvar)
