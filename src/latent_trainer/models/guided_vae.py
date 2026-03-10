"""SC2 Guided VAE model definitions.

Active models:
- ``suGuidedVAE``: Supervised Guided VAE for tabular SC2 features.
- ``Classifier``: Adversarial classifier used during guided VAE training.
"""

import torch
from torch import nn


class suGuidedVAE(nn.Module):
    def __init__(self, n_vae_dis=16, input_dim=39):
        super().__init__()

        self.n_vae_dis = n_vae_dis
        self.input_dim = input_dim

        # Simple encoder for input data with shape [batch, input_dim]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, n_vae_dis * 2),  # mu and logvar
        )

        # Simple decoder to reconstruct input data
        self.decoder = nn.Sequential(
            nn.Linear(n_vae_dis, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, input_dim),  # Back to original input dimension
        )

        self.cls_sq = nn.Sequential(
            nn.Linear(1, 32),
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
        # Handle both 2D [batch, latent_dim] and 3D [batch, num_players, latent_dim]
        original_shape = z.shape
        if len(z.shape) == 3:
            batch_size, num_players, latent_dim = z.shape
            # Reshape to [batch*num_players, latent_dim]
            z = z.view(batch_size * num_players, latent_dim)

        # Extract first dimension for classification
        z = z[:, 0:1]  # Shape: [batch, 1] or [batch*num_players, 1]
        output = self.cls_sq(z)

        # Reshape back to 3D if input was 3D
        if len(original_shape) == 3:
            output = output.view(batch_size, num_players, 1)

        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, self.cls(z)


class Classifier(nn.Module):
    def __init__(self, n_vae_dis=16):
        super(Classifier, self).__init__()

        self.cls_sq = nn.Sequential(
            nn.Linear(n_vae_dis - 1, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Handle both 2D [batch, latent_dim-1] and 3D [batch, num_players, latent_dim-1]
        original_shape = x.shape
        if len(x.shape) == 3:
            batch_size, num_players, latent_dim = x.shape
            # Reshape to [batch*num_players, latent_dim]
            x = x.view(batch_size * num_players, latent_dim)

        output = self.cls_sq(x)

        # Reshape back to 3D if input was 3D
        if len(original_shape) == 3:
            output = output.view(batch_size, num_players, 1)

        return output
