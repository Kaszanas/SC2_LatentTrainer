"""Lightning Classifier module for two-stage training.

Operates on concatenated latent vectors from both players and predicts
match outcome.  Architecture (hidden layers, dropout) is fully configurable
to support architecture search.
"""

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim


class LitClassifier(L.LightningModule):
    """Binary classifier on frozen latent representations.

    Parameters
    ----------
    latent_dim:
        Dimensionality of a **single** player's latent vector.  Input to
        the classifier is ``2 * latent_dim`` (both players concatenated).
    hidden_dims:
        List of hidden-layer widths.  Defaults to ``[128, 64]`` when *None*.
    lr:
        Learning rate for Adam optimiser.
    dropout:
        Dropout probability applied after each hidden block.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: list[int] | None = None,
        lr: float = 1e-3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.lr = lr
        self.dropout = dropout

        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Takes latent vectors from BOTH players: [batch, 2*latent_dim]
        layers: list[nn.Module] = []
        in_dim = latent_dim * 2
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.BCELoss()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def _common_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        stage: str,
    ) -> torch.Tensor:
        z_batch, y_batch = batch
        pred = self(z_batch)
        loss = self.loss_fn(pred, y_batch)

        preds = (pred > 0.5).float()
        acc = (preds == y_batch).float().mean() * 100

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        self._common_step(batch, batch_idx, "val")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
