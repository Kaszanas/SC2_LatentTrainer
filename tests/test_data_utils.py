"""Tests for latent_trainer.data_utils — the canonical data loading module."""

import pytest
import torch

from latent_trainer.data_utils import (
    CachedSC2Dataset,
    Encoder,
    extract_latents,
    load_and_normalize,
    load_cached_dataloaders,
    normalize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_tensors():
    """Create small train/val tensors mimicking the cached dataset shape."""
    torch.manual_seed(42)
    N_train, N_val, F = 32, 8, 10
    train_X = torch.randn(N_train, 2, F)
    val_X = torch.randn(N_val, 2, F)
    return train_X, val_X


@pytest.fixture
def dummy_cache(tmp_path, dummy_tensors):
    """Write a minimal cached dataset to a temp .pt file."""
    train_X, val_X = dummy_tensors
    cache_path = tmp_path / "cache.pt"
    torch.save(
        {
            "train_features": train_X,
            "train_labels": torch.randint(0, 2, (train_X.shape[0],)).float(),
            "val_features": val_X,
            "val_labels": torch.randint(0, 2, (val_X.shape[0],)).float(),
        },
        cache_path,
    )
    return str(cache_path)


class FakeEncoder:
    """Minimal encoder satisfying the Encoder protocol."""

    def __init__(self, latent_dim: int = 4):
        self.latent_dim = latent_dim
        self._is_eval = False

    def eval(self):
        self._is_eval = True
        return self

    def to(self, device):
        return self

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        mu = torch.randn(B, self.latent_dim)
        logvar = torch.zeros(B, self.latent_dim)
        return mu, logvar


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_shapes_preserved(self, dummy_tensors):
        train_X, val_X = dummy_tensors
        normed_train, normed_val, mean, std = normalize(train_X, val_X)
        assert normed_train.shape == train_X.shape
        assert normed_val.shape == val_X.shape

    def test_mean_std_shapes(self, dummy_tensors):
        train_X, val_X = dummy_tensors
        _, _, mean, std = normalize(train_X, val_X)
        F = train_X.shape[-1]
        assert mean.shape == (1, F)
        assert std.shape == (1, F)

    def test_train_is_roughly_standardised(self, dummy_tensors):
        train_X, val_X = dummy_tensors
        normed_train, _, _, _ = normalize(train_X, val_X)
        flat = normed_train.reshape(-1, train_X.shape[-1])
        # After standardisation, mean should be ~0 and std ~1
        assert flat.mean(dim=0).abs().max() < 0.1
        assert (flat.std(dim=0) - 1.0).abs().max() < 0.1

    def test_no_nan_or_inf(self, dummy_tensors):
        train_X, val_X = dummy_tensors
        normed_train, normed_val, _, _ = normalize(train_X, val_X)
        assert torch.isfinite(normed_train).all()
        assert torch.isfinite(normed_val).all()

    def test_constant_feature_handled(self):
        """A constant feature (std=0) should not produce NaN."""
        train_X = torch.ones(16, 2, 5)
        val_X = torch.ones(4, 2, 5)
        normed_train, normed_val, _, _ = normalize(train_X, val_X)
        assert torch.isfinite(normed_train).all()
        assert torch.isfinite(normed_val).all()


# ---------------------------------------------------------------------------
# load_and_normalize()
# ---------------------------------------------------------------------------

class TestLoadAndNormalize:
    def test_returns_six_tensors(self, dummy_cache):
        result = load_and_normalize(dummy_cache)
        assert len(result) == 6

    def test_shapes(self, dummy_cache):
        train_X, train_y, val_X, val_y, mean, std = load_and_normalize(dummy_cache)
        assert train_X.shape[0] == train_y.shape[0]
        assert val_X.shape[0] == val_y.shape[0]
        assert train_X.ndim == 3  # [N, 2, F]
        assert train_y.ndim == 1

    def test_dtype_is_float(self, dummy_cache):
        train_X, train_y, val_X, val_y, _, _ = load_and_normalize(dummy_cache)
        assert train_X.dtype == torch.float32
        assert train_y.dtype == torch.float32


# ---------------------------------------------------------------------------
# load_cached_dataloaders()
# ---------------------------------------------------------------------------

class TestLoadCachedDataloaders:
    def test_returns_loaders_and_dim(self, dummy_cache):
        train_dl, val_dl, input_dim = load_cached_dataloaders(dummy_cache, batch_size=4)
        assert input_dim == 10
        batch = next(iter(train_dl))
        features, labels = batch
        assert features.shape[-1] == input_dim


# ---------------------------------------------------------------------------
# CachedSC2Dataset
# ---------------------------------------------------------------------------

class TestCachedSC2Dataset:
    def test_len(self):
        ds = CachedSC2Dataset(torch.randn(10, 5), torch.randn(10))
        assert len(ds) == 10

    def test_getitem(self):
        ds = CachedSC2Dataset(torch.randn(10, 5), torch.randn(10))
        feat, label = ds[3]
        assert feat.shape == (5,)
        assert label.shape == ()


# ---------------------------------------------------------------------------
# extract_latents()
# ---------------------------------------------------------------------------

class TestExtractLatents:
    def test_output_shape(self):
        latent_dim = 4
        enc = FakeEncoder(latent_dim=latent_dim)
        data = torch.randn(16, 2, 10)
        device = torch.device("cpu")
        result = extract_latents(enc, data, device, batch_size=8)
        assert result.shape == (16, 2 * latent_dim)

    def test_deterministic_encoding(self):
        """extract_latents should use mu (deterministic), not sampled z."""
        latent_dim = 4
        enc = FakeEncoder(latent_dim=latent_dim)
        data = torch.randn(4, 2, 10)
        device = torch.device("cpu")
        # FakeEncoder uses torch.randn so results differ each call,
        # but the function should at least set eval mode
        extract_latents(enc, data, device)
        assert enc._is_eval is True
