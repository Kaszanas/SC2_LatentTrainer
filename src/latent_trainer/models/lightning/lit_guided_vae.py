"""Lightning module for supervised Guided VAE with adversarial classifier.

Wraps :class:`latent_trainer.models.guided_vae.suGuidedVAE` and
:class:`latent_trainer.models.guided_vae.Classifier` into a single
Lightning module with three-optimizer manual optimization:

1. **VAE optimizer** — reconstructs inputs and produces classification logits.
2. **Classifier optimizer** — trains a dedicated classifier on the latent
   space (excluding the first latent dimension used for classification).
3. **Adversarial optimizer** — (currently disabled) pushes the classifier
   toward chance level so the VAE doesn't leak class info into non-class dims.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
import torch.optim as optim

from latent_trainer.models.guided_vae import Classifier, suGuidedVAE
from latent_trainer.models.losses import loss_supervised


class LitGuidedVAE(L.LightningModule):
    """Lightning module for supervised Guided VAE training with adversarial classifier."""

    def __init__(
        self,
        n_vae_dis: int = 16,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_c: float = 1e-4,
        weight_decay_c: float = 1e-4,
        w_cls: float = 50_000.0,
        input_dim: int = 2,
    ) -> None:
        """Initialise the LitGuidedVAE module.

        Parameters
        ----------
        n_vae_dis:
            Size of the VAE latent distribution.
        lr:
            Learning rate for the VAE optimizer.
        weight_decay:
            Weight decay for the VAE optimizer.
        lr_c:
            Learning rate for the classifier optimizer.
        weight_decay_c:
            Weight decay for the classifier optimizer.
        w_cls:
            Weight for the classification loss.
        input_dim:
            Input feature dimension (depends on the transform used).
        """
        super().__init__()
        self.save_hyperparameters()

        # Manual optimisation (3 optimizers)
        self.automatic_optimization = False

        self.model = suGuidedVAE(n_vae_dis=n_vae_dis, input_dim=input_dim)
        self.classifier = Classifier(n_vae_dis=n_vae_dis)

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_c = lr_c
        self.weight_decay_c = weight_decay_c
        self.w_cls = w_cls

        self.total_valid_samples = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_labels(
        data: torch.Tensor,
        label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Normalise labels and filter out invalid (-1) entries.

        Returns ``(valid_data, valid_label)`` or ``None`` if the entire
        batch is invalid.
        """
        if label.dtype == torch.int8:
            label = label.float()
        if label.dim() == 1:
            label = label.unsqueeze(1)
        if data.dim() == 3 and label.dim() == 2:
            label = label.unsqueeze(1).expand(-1, data.shape[1], -1)
        if label.dtype != torch.float32:
            label = label.float()

        # Filter invalid labels
        if data.dim() == 3 and label.dim() == 3:
            valid = (label != -1).all(dim=1).all(dim=1)
        else:
            valid = (label != -1).squeeze()

        if valid.sum() == 0:
            return None

        return data[valid], torch.clamp(label[valid], 0, 1)

    @staticmethod
    def _slice_latent(z: torch.Tensor) -> torch.Tensor:
        """Exclude first latent dim (used for classification)."""
        return z[:, :, 1:] if z.dim() == 3 else z[:, 1:]

    # ------------------------------------------------------------------
    # Forward / training / validation
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):  # noqa: D401
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        result = self._prepare_labels(batch[0], batch[1])
        if result is None:
            self.log("train_skip_batch", 1)
            return None
        valid_data, valid_label = result

        opt_vae, opt_cls, opt_adv = self.optimizers()
        batch_valid = valid_data.shape[0]
        self.total_valid_samples += batch_valid

        # ---- Step 1: VAE ------------------------------------------------
        opt_vae.zero_grad()
        recon_batch, mu, logvar, re = self.model(valid_data)
        vae_loss = loss_supervised(recon_batch, valid_data, mu, logvar)[0]
        cls_loss = F.binary_cross_entropy(re, valid_label, reduction="sum")
        vae_total = vae_loss + cls_loss * self.w_cls

        pred = (re > 0.5).float()
        acc = pred.eq(valid_label).sum().item() / valid_label.numel() * 100

        self.log("train_vae_loss", vae_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_cls_loss", cls_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_vae_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_valid_samples", batch_valid, on_epoch=True)

        self.manual_backward(vae_total)
        opt_vae.step()

        # ---- Step 2: Classifier ------------------------------------------
        opt_cls.zero_grad()
        mu, logvar = self.model.encode(valid_data)
        z = self.model.reparameterize(mu, logvar).detach()
        cls1 = self.classifier(self._slice_latent(z))
        c_loss = F.binary_cross_entropy(cls1, valid_label, reduction="sum") * self.w_cls

        c_acc = (
            (cls1 > 0.5).float().eq(valid_label).sum().item()
            / valid_label.numel()
            * 100
        )

        self.log("train_c_loss", c_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_c_acc", c_acc, prog_bar=True, on_step=True, on_epoch=True)

        self.manual_backward(c_loss)
        opt_cls.step()

        # ---- Step 3: Adversarial (currently disabled) --------------------
        opt_adv.zero_grad()
        mu, logvar = self.model.encode(valid_data)
        z = self.model.reparameterize(mu, logvar)
        cls2 = self.classifier(self._slice_latent(z))
        label_half = torch.empty_like(valid_label).fill_(0.5)
        adv_loss = F.binary_cross_entropy(cls2, label_half, reduction="sum")
        adv_loss *= 0.0  # Disabled: adversarial step cancels classifier learning

        self.log("train_adv_loss", adv_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.manual_backward(adv_loss)
        opt_adv.step()

        return vae_total + c_loss + adv_loss

    def on_train_epoch_end(self) -> None:
        self.log("epoch_valid_samples", self.total_valid_samples)
        self.total_valid_samples = 0

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None

        result = self._prepare_labels(batch[0], batch[1])
        if result is None:
            return None
        valid_data, valid_label = result

        recon_batch, mu, logvar, re = self.model(valid_data)
        vae_loss = loss_supervised(recon_batch, valid_data, mu, logvar)[0]
        cls_loss = F.binary_cross_entropy(re, valid_label, reduction="sum")

        pred = (re > 0.5).float()
        acc = pred.eq(valid_label).sum().item() / valid_label.numel() * 100

        self.log("val_vae_loss", vae_loss, prog_bar=True, sync_dist=True)
        self.log("val_cls_loss", cls_loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)

        return vae_loss

    def test_step(self, batch, batch_idx):
        # Re-use validation logic for test evaluation
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        opt_vae = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        opt_cls = optim.Adam(
            self.classifier.parameters(),
            lr=self.lr_c,
            weight_decay=self.weight_decay_c,
        )
        # Adversarial optimizer (VAE params, trained adversarially)
        opt_adv = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return [opt_vae, opt_cls, opt_adv], []
