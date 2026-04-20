"""SC2 Guided VAE model definitions.

Active models:
- ``suGuidedVAE``: Supervised Guided VAE for tabular SC2 features.
- ``Classifier``: Adversarial classifier used during guided VAE training.
"""

import torch
from torch import nn


# REVIEW: This architecture is hardcoded.
# REVIEW: We are still using in training, but I thought we wanted to be able
# REVIEW: To do architecture search as well?
# REVIEW: Other models are defined directly in Lightning while this one is
# REVIEW: left as a PyTorch module. Why is that?
# REVIEW: No type hints are applied here.
class suGuidedVAE(nn.Module):
    def __init__(
        self,
        n_vae_dis: int = 16,
        input_dim: int = 39,
        encoder_hidden_dims: list[int] | None = None,
    ):
        super().__init__()

        self.n_vae_dis = n_vae_dis
        self.input_dim = input_dim

        if encoder_hidden_dims is None:
            encoder_hidden_dims = [64, 128, 256]

        # Encoder: input_dim → hidden_dims → n_vae_dis*2 (mu and logvar)
        enc_layers: list[nn.Module] = []
        in_dim = input_dim
        for h in encoder_hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.ReLU(True)]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, n_vae_dis * 2))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: n_vae_dis → reversed(hidden_dims) → input_dim
        dec_layers: list[nn.Module] = []
        in_dim = n_vae_dis
        for h in reversed(encoder_hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), nn.ReLU(True)]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.classifier = nn.Sequential(
            nn.Linear(n_vae_dis * 2, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        # Handle 3D input [batch, num_players, features]
        original_shape = x.shape
        if len(x.shape) == 3:
            batch_size, num_players, num_features = x.shape
            # Reshape to [batch*num_players, num_features] for encoding
            x = x.view(batch_size * num_players, num_features)

        x = self.encoder(x)
        mu = x[:, : self.n_vae_dis]
        logvar = x[:, self.n_vae_dis :]

        # Clamp logvar to prevent numerical instability in reparameterization
        # Typical range: -20 to 2 (std from ~exp(-10)=4.5e-5 to ~exp(1)=2.7)
        logvar = torch.clamp(logvar, min=-20, max=2)

        # Reshape back to 3D if input was 3D
        if len(original_shape) == 3:
            mu = mu.view(batch_size, num_players, self.n_vae_dis)
            logvar = logvar.view(batch_size, num_players, self.n_vae_dis)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Handle 3D input [batch, num_players, latent_dim]
        original_shape = z.shape
        if len(z.shape) == 3:
            batch_size, num_players, latent_dim = z.shape
            # Reshape to [batch*num_players, latent_dim] for decoding
            z = z.view(batch_size * num_players, latent_dim)

        output = self.decoder(z)

        # Reshape back to 3D if input was 3D
        if len(original_shape) == 3:
            output = output.view(batch_size, num_players, -1)

        return output

    def cls(self, z):
        # Expects 3D input [batch, num_players=2, latent_dim]
        # Concatenates both players along feature dim for opponent-aware prediction
        if z.dim() == 3:
            batch_size, num_players, latent_dim = z.shape
            z = z.view(batch_size, num_players * latent_dim)
        return self.classifier(z)  # [batch, 1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, self.cls(z)


class Classifier(nn.Module):
    def __init__(self, n_vae_dis=16):
        super(Classifier, self).__init__()

        self.cls_sq = nn.Sequential(
            nn.Linear((n_vae_dis - 1) * 2, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Expects 3D [batch, num_players=2, latent_dim-1]
        # Concatenates both players for opponent-aware prediction
        if x.dim() == 3:
            batch_size, num_players, latent_dim = x.shape
            x = x.view(batch_size, num_players * latent_dim)
        return self.cls_sq(x)  # [batch, 1]
