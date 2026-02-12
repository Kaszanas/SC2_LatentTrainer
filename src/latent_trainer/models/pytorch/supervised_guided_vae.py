import torch
from torch import nn
import torch.nn.functional as F


class View(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class SupervisedGuidedVAE(nn.Module):
    def __init__(self, n_latent_dimensions: int = 16):
        super().__init__()

        self.n_latent_dimensions = n_latent_dimensions

        # TODO: Change this to fit the model for SC2 data:
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            View((-1, 256 * 1 * 1)),
            nn.Linear(256, n_latent_dimensions * 2),
        )

        # TODO: Change this to fit the model for SC2 data:
        self.decoder = nn.Sequential(
            nn.Linear(n_latent_dimensions, 256),
            View((-1, 256, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = x[:, : self.n_latent_dimensions]
        logvar = x[:, self.n_latent_dimensions :]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def cls(self, z):
        z = torch.split(z, 1, 1)[0]
        return self.classification_head(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, self.cls(z)


class Classifier(nn.Module):
    def __init__(self, n_vae_dis=16):
        super(Classifier, self).__init__()

        self.cls_sq = nn.Sequential(
            nn.Linear(n_vae_dis - 1, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.cls_sq(x)
