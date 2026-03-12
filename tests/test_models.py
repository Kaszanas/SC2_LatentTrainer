"""Tests for model architectures — forward pass shape checks.

These tests verify that models produce the expected output shapes
without requiring any training data or GPU.
"""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# SimpleVAE (from train_two_stage.py — kept as root-level model)
# ---------------------------------------------------------------------------

class TestSimpleVAE:
    @pytest.fixture
    def vae(self):
        from train_two_stage import SimpleVAE
        return SimpleVAE(input_dim=203, latent_dim=32)

    def test_encode_shape(self, vae):
        x = torch.randn(4, 203)
        mu, logvar = vae.encode(x)
        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)

    def test_decode_shape(self, vae):
        z = torch.randn(4, 32)
        recon = vae.decode(z)
        assert recon.shape == (4, 203)

    def test_forward_shape(self, vae):
        x = torch.randn(4, 203)
        recon, mu, logvar = vae(x)
        assert recon.shape == (4, 203)
        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)

    def test_roundtrip_shape(self, vae):
        """Encode → reparameterize → decode preserves shape."""
        x = torch.randn(4, 203)
        mu, logvar = vae.encode(x)
        z = vae.reparameterize(mu, logvar)
        recon = vae.decode(z)
        assert recon.shape == x.shape


# ---------------------------------------------------------------------------
# LatentClassifier (from train_two_stage.py)
# ---------------------------------------------------------------------------

class TestLatentClassifier:
    def test_forward_shape(self):
        from train_two_stage import LatentClassifier
        clf = LatentClassifier(latent_dim=32)
        z = torch.randn(4, 64)  # 2 * latent_dim
        out = clf(z)
        assert out.shape == (4, 1)
        assert (out >= 0).all() and (out <= 1).all()  # Sigmoid output


# ---------------------------------------------------------------------------
# LitVAE (Lightning module)
# ---------------------------------------------------------------------------

class TestLitVAE:
    @pytest.fixture
    def lit_vae(self):
        from latent_trainer.models.lightning.lit_vae import LitVAE
        return LitVAE(input_dim=203, latent_dim=16, hidden_dims=[128, 64], lr=1e-3)

    def test_encode_shape(self, lit_vae):
        x = torch.randn(4, 203)
        mu, logvar = lit_vae.encode(x)
        assert mu.shape == (4, 16)
        assert logvar.shape == (4, 16)

    def test_forward_shape(self, lit_vae):
        x = torch.randn(4, 203)
        recon, mu, logvar = lit_vae(x)
        assert recon.shape == (4, 203)

    def test_custom_hidden_dims(self):
        from latent_trainer.models.lightning.lit_vae import LitVAE
        vae = LitVAE(input_dim=100, latent_dim=8, hidden_dims=[64, 32, 16])
        x = torch.randn(2, 100)
        recon, mu, logvar = vae(x)
        assert recon.shape == (2, 100)
        assert mu.shape == (2, 8)


# ---------------------------------------------------------------------------
# LitClassifier (Lightning module)
# ---------------------------------------------------------------------------

class TestLitClassifier:
    def test_forward_shape(self):
        from latent_trainer.models.lightning.lit_classifier import LitClassifier
        clf = LitClassifier(latent_dim=16, hidden_dims=[64, 32], lr=1e-3, dropout=0.3)
        z = torch.randn(4, 32)  # 2 * latent_dim
        out = clf(z)
        assert out.shape == (4, 1)
        assert (out >= 0).all() and (out <= 1).all()


# ---------------------------------------------------------------------------
# suGuidedVAE + Classifier (guided_vae.py)
# ---------------------------------------------------------------------------

class TestSuGuidedVAE:
    @pytest.fixture
    def model(self):
        from latent_trainer.models.guided_vae import suGuidedVAE
        return suGuidedVAE(n_vae_dis=16, input_dim=39)

    def test_encode_2d(self, model):
        x = torch.randn(4, 39)
        mu, logvar = model.encode(x)
        assert mu.shape == (4, 16)
        assert logvar.shape == (4, 16)

    def test_encode_3d(self, model):
        x = torch.randn(4, 2, 39)
        mu, logvar = model.encode(x)
        assert mu.shape == (4, 2, 16)

    def test_forward_3d(self, model):
        x = torch.randn(4, 2, 39)
        recon, mu, logvar, cls_out = model(x)
        assert recon.shape == (4, 2, 39)
        assert cls_out.shape == (4, 2, 1)

    def test_logvar_is_clamped(self, model):
        """Logvar should be clamped to [-20, 2]."""
        x = torch.randn(4, 39) * 100  # large input
        mu, logvar = model.encode(x)
        assert logvar.min() >= -20
        assert logvar.max() <= 2


class TestAdversarialClassifier:
    def test_forward_2d(self):
        from latent_trainer.models.guided_vae import Classifier
        clf = Classifier(n_vae_dis=16)
        x = torch.randn(4, 15)  # n_vae_dis - 1
        out = clf(x)
        assert out.shape == (4, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_forward_3d(self):
        from latent_trainer.models.guided_vae import Classifier
        clf = Classifier(n_vae_dis=16)
        x = torch.randn(4, 2, 15)
        out = clf(x)
        assert out.shape == (4, 2, 1)
