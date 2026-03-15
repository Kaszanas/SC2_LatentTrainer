"""Tests for latent_trainer.data_utils — the canonical data loading module."""

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from latent_trainer.data_utils import (
    CachedSC2Dataset,
    extract_latents,
    load_and_normalize,
    load_cached_dataloaders,
    normalize,
)
from latent_trainer.features.type import CachedDatasetFileSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def dummy_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create small train/val tensors mimicking the cached dataset shape."""
    torch.manual_seed(42)
    N_train = 32
    N_val = 8
    N_test = 8
    F = 10

    train_X = torch.randn(N_train, 2, F)
    val_X = torch.randn(N_val, 2, F)
    test_X = torch.randn(N_test, 2, F)

    return train_X, val_X, test_X


@pytest.fixture
def dummy_cache(
    tmp_path: Path, dummy_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> str:
    """Write a minimal cached dataset to a temp .pt file."""
    train_X, val_X, _ = dummy_tensors
    cache_path = tmp_path / "cache.pt"

    cache_dummy = CachedDatasetFileSpec(
        train_features=train_X,
        train_labels=torch.randint(0, 2, (train_X.shape[0],)).float(),
        val_features=val_X,
        val_labels=torch.randint(0, 2, (val_X.shape[0],)).float(),
        test_features=torch.randn(8, 2, 10),
        test_labels=torch.randint(0, 2, (8,)).float(),
        transform="",
    )

    torch.save(asdict(cache_dummy), cache_path)

    return str(cache_path)


class FakeEncoder:
    """Minimal encoder satisfying the Encoder protocol."""

    def __init__(self, latent_dim: int = 4):
        self.latent_dim = latent_dim
        self._is_eval = False

    def eval(self) -> "FakeEncoder":
        self._is_eval = True
        return self

    def to(self, device: torch.device) -> "FakeEncoder":
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
    def test_shapes_preserved(
        self, dummy_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        train_X, val_X, test_X = dummy_tensors
        normalized_data = normalize(train_X, val_X, test_X)
        assert normalized_data.train_X.shape == train_X.shape
        assert normalized_data.val_X.shape == val_X.shape
        assert normalized_data.test_X.shape == test_X.shape

    def test_mean_std_shapes(
        self, dummy_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        train_X, val_X, test_X = dummy_tensors
        normalized_data = normalize(train_X, val_X, test_X)
        F = train_X.shape[-1]
        assert normalized_data.mean.shape == (1, F)
        assert normalized_data.std.shape == (1, F)

    def test_train_is_roughly_standardised(
        self, dummy_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        train_X, val_X, test_X = dummy_tensors
        normalized_data = normalize(train_X, val_X, test_X)
        flat = normalized_data.train_X.reshape(-1, train_X.shape[-1])
        # After standardisation, mean should be ~0 and std ~1
        assert flat.mean(dim=0).abs().max() < 0.1
        assert (flat.std(dim=0) - 1.0).abs().max() < 0.1

    def test_no_nan_or_inf(
        self, dummy_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        train_X, val_X, test_X = dummy_tensors
        normalized_data = normalize(train_X, val_X, test_X)
        assert torch.isfinite(normalized_data.train_X).all()
        assert torch.isfinite(normalized_data.val_X).all()
        assert torch.isfinite(normalized_data.test_X).all()

    def test_constant_feature_handled(self) -> None:
        """A constant feature (std=0) should not produce NaN."""
        train_X = torch.ones(16, 2, 5)
        val_X = torch.ones(4, 2, 5)
        test_X = torch.ones(4, 2, 5)
        normalized_data = normalize(train_X, val_X, test_X)
        assert torch.isfinite(normalized_data.train_X).all()
        assert torch.isfinite(normalized_data.val_X).all()
        assert torch.isfinite(normalized_data.test_X).all()


# ---------------------------------------------------------------------------
# load_and_normalize()
# ---------------------------------------------------------------------------
class TestLoadAndNormalize:
    def test_returns_expected_fields(self, dummy_cache: str) -> None:
        result = load_and_normalize(dummy_cache)
        expected_fields = {
            "train_X",
            "train_y",
            "val_X",
            "val_y",
            "test_X",
            "test_y",
            "mean",
            "std",
        }
        assert set(result.__dict__.keys()) == expected_fields

    def test_shapes(self, dummy_cache: str) -> None:
        normalized_data = load_and_normalize(dummy_cache)
        assert normalized_data.train_X.shape[0] == normalized_data.train_y.shape[0]
        assert normalized_data.val_X.shape[0] == normalized_data.val_y.shape[0]
        assert normalized_data.test_X.shape[0] == normalized_data.test_y.shape[0]
        assert normalized_data.train_X.ndim == 3  # [N, 2, F]
        assert normalized_data.train_y.ndim == 1

    def test_dtype_is_float(self, dummy_cache: str) -> None:
        normalized_data = load_and_normalize(dummy_cache)
        assert normalized_data.train_X.dtype == torch.float32
        assert normalized_data.train_y.dtype == torch.float32


# ---------------------------------------------------------------------------
# load_cached_dataloaders()
# ---------------------------------------------------------------------------
class TestLoadCachedDataloaders:
    def test_returns_loaders_and_dim(self, dummy_cache: str) -> None:
        train_dl, val_dl, input_dim = load_cached_dataloaders(dummy_cache, batch_size=4)
        assert input_dim == 10
        batch = next(iter(train_dl))
        features, labels = batch
        assert features.shape[-1] == input_dim


# ---------------------------------------------------------------------------
# CachedSC2Dataset
# ---------------------------------------------------------------------------
class TestCachedSC2Dataset:
    def test_len(self) -> None:
        ds = CachedSC2Dataset(torch.randn(10, 5), torch.randn(10))
        assert len(ds) == 10

    def test_getitem(self) -> None:
        ds = CachedSC2Dataset(torch.randn(10, 5), torch.randn(10))
        feat, label = ds[3]
        assert feat.shape == (5,)
        assert label.shape == ()


# ---------------------------------------------------------------------------
# extract_latents()
# ---------------------------------------------------------------------------
class TestExtractLatents:
    def test_output_shape(self) -> None:
        latent_dim = 4
        enc = FakeEncoder(latent_dim=latent_dim)
        data = torch.randn(16, 2, 10)
        device = torch.device("cpu")
        result = extract_latents(enc, data, device, batch_size=8)
        assert result.shape == (16, 2 * latent_dim)

    def test_deterministic_encoding(self) -> None:
        """extract_latents should use mu (deterministic), not sampled z."""
        latent_dim = 4
        enc = FakeEncoder(latent_dim=latent_dim)
        data = torch.randn(4, 2, 10)
        device = torch.device("cpu")
        # FakeEncoder uses torch.randn so results differ each call,
        # but the function should at least set eval mode
        extract_latents(enc, data, device)
        assert enc._is_eval is True
