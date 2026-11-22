"""
The guided-VAE training loop shared by every guided model.

Manual optimisation with three optimizers per step:

1. VAE: reconstruction + KL + ``classification_weight`` x outcome loss on the
   supervised latent dims.
2. Adversary: learns the outcome from the *detached* free latent dims.
3. VAE again: pushes the adversary's prediction towards 0.5, so the free dims
   carry no outcome information.

Subclasses only say how a batch is unpacked, how the VAE forward and its loss
are computed, how the outcome loss and accuracy read the head's output
(probabilities or logits), and how latents are sampled. The base class has no
parameters, so it never changes a checkpoint's keys.
"""

from __future__ import annotations

from typing import Any

import lightning as pl
import torch
from torch import nn, optim


class GuidedAdversarialModule(pl.LightningModule):
    """
    Three-step adversarial training loop for guided VAEs.

    Subclasses set ``model``, ``adversarial_classifier``, ``supervised_dim``,
    ``learning_rate``, ``weight_decay``, ``learning_rate_cls``,
    ``weight_decay_cls``, ``classification_weight`` and
    ``automatic_optimization = False``, and implement the hooks below.
    """

    model: nn.Module
    adversarial_classifier: nn.Module

    # Hooks:
    def _unpack(self, batch: Any) -> tuple[Any, torch.Tensor]:
        """
        Split a batch into model inputs and ``[B, 1]`` float labels.

        Parameters
        ----------
        batch : Any
            One batch from the dataloader.

        Returns
        -------
        tuple[Any, torch.Tensor]
            ``(inputs, labels)``.
        """
        raise NotImplementedError

    def _vae_forward(self, inputs: Any) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Run the VAE and return its loss and the outcome head's output.

        Parameters
        ----------
        inputs : Any
            Model inputs from :meth:`_unpack`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict]
            ``(vae_loss, outcome, extra_logs)``; ``extra_logs`` maps metric
            name suffixes to values.
        """
        raise NotImplementedError

    def _outcome_loss(
        self, outcome: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Return the binary cross-entropy of a head's output against a target.

        Parameters
        ----------
        outcome : torch.Tensor
            Head output (probability or logit, per subclass), ``[B, 1]``.
        target : torch.Tensor
            Target probability, ``[B, 1]``.

        Returns
        -------
        torch.Tensor
            Mean loss.
        """
        raise NotImplementedError

    def _predicts_player_1(self, outcome: torch.Tensor) -> torch.Tensor:
        """
        Return whether a head's output predicts that player 1 wins.

        Parameters
        ----------
        outcome : torch.Tensor
            Head output, ``[B, 1]``.

        Returns
        -------
        torch.Tensor
            Float tensor of 0/1 predictions, ``[B, 1]``.
        """
        raise NotImplementedError

    def _sample_z(self, inputs: Any) -> torch.Tensor:
        """
        Encode the inputs and sample per-player latents.

        Parameters
        ----------
        inputs : Any
            Model inputs from :meth:`_unpack`.

        Returns
        -------
        torch.Tensor
            Latents ``[B, 2, latent_dim]``.
        """
        raise NotImplementedError

    # Shared helpers:
    @staticmethod
    def _prepare_label(label: torch.Tensor) -> torch.Tensor:
        """
        Coerce labels to ``[B, 1]`` float32 and check they are 0 or 1.

        Parameters
        ----------
        label : torch.Tensor
            Labels ``[B]`` or ``[B, 1]``.

        Returns
        -------
        torch.Tensor
            Labels as ``[B, 1]`` float32.

        Raises
        ------
        ValueError
            If any label is outside ``[0, 1]``; invalid samples must be filtered
            upstream.
        """
        if label.dtype != torch.float32:
            label = label.float()
        if label.dim() == 1:
            label = label.unsqueeze(1)

        if (label < 0).any() or (label > 1).any():
            raise ValueError(
                "Batch contains labels outside [0, 1]. "
                "Filter invalid samples in the dataset/dataloader, not here."
            )
        return label

    def _slice_free_dims(self, z: torch.Tensor) -> torch.Tensor:
        """
        Slice the free (non-supervised) latent dims passed to the adversary.

        Parameters
        ----------
        z : torch.Tensor
            Latents ``[B, latent_dim]`` or ``[B, 2, latent_dim]``.

        Returns
        -------
        torch.Tensor
            ``z[..., supervised_dim:]``.
        """
        return (
            z[:, :, self.supervised_dim :]
            if z.dim() == 3
            else z[:, self.supervised_dim :]
        )

    def _accuracy(self, outcome: torch.Tensor, label: torch.Tensor) -> float:
        """
        Return the percentage of correct outcome predictions.

        Parameters
        ----------
        outcome : torch.Tensor
            Head output, ``[B, 1]``.
        label : torch.Tensor
            Labels, ``[B, 1]``.

        Returns
        -------
        float
            Accuracy in percent.
        """
        correct = self._predicts_player_1(outcome).eq(label).sum().item()
        return correct / label.numel() * 100

    # Training / validation:
    def training_step(self, batch, batch_idx: int):
        """
        Run the three optimisation steps on one batch.

        Parameters
        ----------
        batch : Any
            One batch from the dataloader.
        batch_idx : int
            Index of the batch in the epoch.

        Returns
        -------
        torch.Tensor
            Sum of the three losses (for logging; optimisation is manual).
        """
        inputs, label = self._unpack(batch)
        optimizer_vae, optimizer_classification, optimizer_adversarial = (
            self.optimizers()
        )

        # Step 1: VAE
        optimizer_vae.zero_grad()
        vae_loss, outcome, extra = self._vae_forward(inputs)
        cls_loss = self._outcome_loss(outcome, label)
        vae_total = vae_loss + cls_loss * self.classification_weight
        acc = self._accuracy(outcome, label)

        self.log("train_vae_loss", vae_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_cls_loss", cls_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_vae_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        for name, value in extra.items():
            self.log(f"train_{name}", value, on_step=True, on_epoch=True)

        self.manual_backward(vae_total)
        optimizer_vae.step()

        # Step 2: Adversarial classifier
        optimizer_classification.zero_grad()
        z = self._sample_z(inputs).detach()
        cls1 = self.adversarial_classifier(self._slice_free_dims(z))
        adversarial_classification_loss = (
            self._outcome_loss(cls1, label) * self.classification_weight
        )
        adversarial_classification_accuracy = self._accuracy(cls1, label)

        self.log(
            "train_adv_cls_loss",
            adversarial_classification_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_adv_cls_acc",
            adversarial_classification_accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        self.manual_backward(adversarial_classification_loss)
        optimizer_classification.step()

        # Step 3: Adversarial (VAE fools the classifier)
        optimizer_adversarial.zero_grad()
        z = self._sample_z(inputs)
        cls2 = self.adversarial_classifier(self._slice_free_dims(z))
        label_half = torch.empty_like(label).fill_(0.5)
        adv_loss = self._outcome_loss(cls2, label_half) * self.classification_weight

        self.log("train_adv_loss", adv_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.manual_backward(adv_loss)
        optimizer_adversarial.step()

        return vae_total + adversarial_classification_loss + adv_loss

    def on_train_epoch_end(self) -> None:
        """Step the three cosine LR schedulers."""
        sch1, sch2, sch3 = self.lr_schedulers()
        sch1.step()
        sch2.step()
        sch3.step()

    def validation_step(self, batch, batch_idx):
        """
        Compute and log ``val_loss``, ``val_vae_loss``, ``val_cls_loss`` and ``val_acc``.

        Parameters
        ----------
        batch : Any
            One batch from the dataloader.
        batch_idx : int
            Index of the batch in the epoch.

        Returns
        -------
        torch.Tensor
            VAE loss plus weighted classification loss.
        """
        inputs, label = self._unpack(batch)
        vae_loss, outcome, extra = self._vae_forward(inputs)
        cls_loss = self._outcome_loss(outcome, label)
        total_loss = vae_loss + cls_loss * self.classification_weight
        acc = self._accuracy(outcome, label)

        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val_vae_loss", vae_loss, prog_bar=True, sync_dist=True)
        self.log("val_cls_loss", cls_loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        for name, value in extra.items():
            self.log(f"val_{name}", value, sync_dist=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        """
        Evaluate a test batch exactly like a validation batch.

        Parameters
        ----------
        batch : Any
            One batch from the dataloader.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            VAE loss plus weighted classification loss.
        """
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """
        Create the VAE, adversary and adversarial optimizers with cosine schedulers.

        Returns
        -------
        tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]
            Three AdamW optimizers and their ``CosineAnnealingLR`` schedulers.

        Raises
        ------
        ValueError
            If the trainer has no ``max_epochs`` (needed for ``T_max``).
        """
        if not self.trainer.max_epochs:
            raise ValueError(
                "Trainer max_epochs must be set for LR scheduler configuration."
            )

        opt_vae = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        opt_cls = optim.AdamW(
            self.adversarial_classifier.parameters(),
            lr=self.learning_rate_cls,
            weight_decay=self.weight_decay_cls,
        )
        # Adversarial optimizer updates VAE params to fool the classifier
        opt_adv = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        epochs = self.trainer.max_epochs
        sched_vae = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=opt_vae, T_max=epochs
        )
        sched_cls = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=opt_cls, T_max=epochs
        )
        sched_adv = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=opt_adv, T_max=epochs
        )

        return [opt_vae, opt_cls, opt_adv], [sched_vae, sched_cls, sched_adv]
