"""Tests for the LitGuidedVAE Lightning module."""

import torch

from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE


class TestLitGuidedVAE:
    """Shape and smoke tests for LitGuidedVAE."""

    def _make_model(self, input_dim: int = 39, n_vae_dis: int = 8) -> LitGuidedVAE:
        return LitGuidedVAE(n_vae_dis=n_vae_dis, input_dim=input_dim)

    def test_forward_shape_3d(self):
        model = self._make_model()
        x = torch.randn(4, 2, 39)
        recon, mu, logvar, cls = model(x)
        assert recon.shape == (4, 2, 39)
        assert cls.shape == (4, 2, 1)

    def test_forward_shape_2d(self):
        model = self._make_model(input_dim=10, n_vae_dis=4)
        x = torch.randn(4, 10)
        recon, mu, logvar, cls = model(x)
        assert recon.shape == (4, 10)
        assert cls.shape == (4, 1)

    def test_configure_optimizers_returns_three(self):
        model = self._make_model()
        optimizers, schedulers = model.configure_optimizers()
        assert len(optimizers) == 3
        assert schedulers == []

    def test_prepare_labels_filters_invalid(self):
        """Invalid labels (-1) should be filtered out."""
        data = torch.randn(5, 10)
        labels = torch.tensor([1.0, 0.0, -1.0, 1.0, -1.0])
        result = LitGuidedVAE._prepare_labels(data, labels)
        assert result is not None
        valid_data, valid_label = result
        assert valid_data.shape[0] == 3  # 2 invalid removed

    def test_prepare_labels_returns_none_if_all_invalid(self):
        data = torch.randn(3, 10)
        labels = torch.tensor([-1.0, -1.0, -1.0])
        result = LitGuidedVAE._prepare_labels(data, labels)
        assert result is None

    def test_slice_latent_2d(self):
        z = torch.randn(4, 8)
        sliced = LitGuidedVAE._slice_latent(z)
        assert sliced.shape == (4, 7)  # first dim excluded

    def test_slice_latent_3d(self):
        z = torch.randn(4, 2, 8)
        sliced = LitGuidedVAE._slice_latent(z)
        assert sliced.shape == (4, 2, 7)
