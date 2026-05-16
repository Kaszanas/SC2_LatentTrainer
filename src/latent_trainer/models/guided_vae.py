import torch
from torch import nn


class suGuidedVAE(nn.Module):
    def __init__(
        self,
        encoder_hidden_dims: list[int],
        latent_dim: int,
        input_dim: int,
        supervised_dim: int,
    ):
        """


        Parameters
        ----------
        encoder_hidden_dims : list[int]
            Defines the hidden layer widths for the encoder.  The decoder mirrors these in
            reverse order.  For example, ``[64, 128, 256, 512]``.
        latent_dim : int
            Size of the VAE latent distribution.
        input_dim : int
            Number of input features.
        classifier_dim : int
            Number of latent dims (per player) reserved for supervised classification.
            The remaining ``latent_dim - classifier_dim`` dims are adversarially disentangled.

        Raises
        ------
        ValueError
            If ``classifier_dim`` is not in the range (0, ``latent_dim``).
        """

        super().__init__()

        if not (0 < supervised_dim < latent_dim):
            raise ValueError(
                f"k_cls_dims must be in (0, n_vae_dis). Got k_cls_dims={supervised_dim}, n_vae_dis={latent_dim}."
            )

        self.n_vae_dis = latent_dim
        self.n_free_dims = latent_dim - supervised_dim
        self.input_dim = input_dim

        self.supervised_dim = supervised_dim

        # Encoder: input_dim → hidden_dims → n_vae_dis*2 (mu and logvar)
        enc_layers: list[nn.Module] = []
        in_dim = input_dim
        for h in encoder_hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.ReLU(True)]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim * 2))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: n_vae_dis → reversed(hidden_dims) → input_dim
        dec_layers: list[nn.Module] = []
        in_dim = latent_dim
        for h in reversed(encoder_hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), nn.ReLU(True)]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Opponent-aware classifier using only the first supervised_dim from each player
        self.classifier = nn.Sequential(
            nn.Linear(supervised_dim * 2, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        if len(x.shape) == 3:
            batch_size, num_players, num_features = x.shape
            x = x.view(batch_size * num_players, num_features)

        x = self.encoder(x)
        mu = x[:, : self.n_vae_dis]
        logvar = x[:, self.n_vae_dis :]

        # Clamp logvar to prevent numerical instability in reparameterization
        # Typical range: -20 to 2 (std from ~exp(-10)=4.5e-5 to ~exp(1)=2.7)
        logvar = torch.clamp(logvar, min=-20, max=2)

        if len(original_shape) == 3:
            mu = mu.view(batch_size, num_players, self.n_vae_dis)
            logvar = logvar.view(batch_size, num_players, self.n_vae_dis)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        original_shape = z.shape
        if len(z.shape) == 3:
            batch_size, num_players, latent_dim = z.shape
            z = z.view(batch_size * num_players, latent_dim)

        output = self.decoder(z)

        if len(original_shape) == 3:
            output = output.view(batch_size, num_players, -1)

        return output

    def cls(self, z: torch.Tensor) -> torch.Tensor:
        # Use only the first supervised_dim from each player,
        # concatenated opponent-aware
        if z.dim() == 3:
            batch_size, num_players, _ = z.shape
            z_cls = z[:, :, : self.supervised_dim]  # [batch, 2, self.supervised_dim]
            z_cls = z_cls.reshape(batch_size, num_players * self.supervised_dim)
        else:
            z_cls = z[:, : self.supervised_dim]
        return self.classifier(z_cls)  # [batch, 1]

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        classification = self.cls(z)
        reconstruction = self.decode(z)

        return reconstruction, mu, logvar, classification


class Classifier(nn.Module):
    """Adversarial classifier operating on the non-supervised latent dims."""

    def __init__(self, latent_dim: int, supervised_dim: int) -> None:
        super().__init__()
        if not (0 < supervised_dim < latent_dim):
            raise ValueError(
                f"Classifier dimension must be in (0, latent_dim). Got classifier_dim={supervised_dim}, latent_dim={latent_dim}."
            )

        # Adversarial classifier operates on the non supervised dimensions coming out
        # of the VAE encoder, since we have two players,
        # we need to concatenate those dims for both players before classifying.
        # Therefore the input dimension is number of the number of supervised
        # dimensions subtracted from the total latent dim, multiplied by 2 for both players:
        in_dim = (latent_dim - supervised_dim) * 2
        self.cls_sq = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 2, n_vae_dis - k_cls_dims] — concat players before classifying
        if x.dim() == 3:
            batch_size, num_players, latent_dim = x.shape
            x = x.reshape(batch_size, num_players * latent_dim)
        return self.cls_sq(x)  # [batch, 1]
