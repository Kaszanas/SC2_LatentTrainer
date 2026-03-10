"""Lightning VAE module for two-stage training.

Configurable encoder/decoder architecture via ``hidden_dims`` list, enabling
architecture search over the number of layers and their widths.
"""

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LitVAE(L.LightningModule):
    """Variational Auto-Encoder with configurable hidden-layer architecture.

    Parameters
    ----------
    input_dim:
        Dimensionality of the input features (per player).
    latent_dim:
        Size of the latent space.
    hidden_dims:
        List of hidden-layer widths for encoder (decoder mirrors in reverse).
        Defaults to ``[256, 128]`` when *None*.
    lr:
        Learning rate for Adam optimiser.
    """

    def __init__(
        self,
        input_dim: int = 203,
        latent_dim: int = 32,
        hidden_dims: list[int] | None = None,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr

        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Encoder
        encoder_layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU(True))
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder (mirrors encoder)
        decoder_layers: list[nn.Module] = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU(True))
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    # ------------------------------------------------------------------
    # Forward pass helpers
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def _common_step(
        self, batch: tuple[torch.Tensor], batch_idx: int, stage: str
    ) -> torch.Tensor:
        (x,) = batch
        recon_x, mu, logvar = self(x)

        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss

        batch_size = x.size(0)
        self.log(f"{stage}_loss", loss / batch_size, prog_bar=True)
        self.log(f"{stage}_recon_loss", recon_loss / batch_size, prog_bar=False)
        self.log(f"{stage}_kl_loss", kl_loss / batch_size, prog_bar=False)
        return loss

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> None:
        self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
