"""Tests for the model's per-channel input scaling."""

import numpy as np
import pytest
import torch

from latent_trainer.features.channel_transform import (
    ChannelTransform,
    channel_kind,
    symexp,
    symlog,
)
from latent_trainer.features.event_transform import STAT_CHANNEL_NAMES

NAMES = [
    "foodUsed",
    "pac_latency_s",
    "mineralsCurrent",
    "mineralsLostTechnology",
    "vespeneFriendlyFireEconomy",
]


def _raw(n: int = 20_000, seed: int = 0) -> torch.Tensor:
    """Heavy-tailed raw columns in ``NAMES`` order."""
    rng = np.random.default_rng(seed)
    columns = [
        rng.uniform(0, 200, n),  # foodUsed
        rng.lognormal(-1.8, 0.6, n),  # pac_latency_s
        rng.lognormal(5.0, 1.5, n),  # mineralsCurrent
        rng.normal(100, 400, n),  # mineralsLostTechnology (negatives)
        np.zeros(n),  # vespeneFriendlyFireEconomy
    ]
    return torch.tensor(np.stack(columns, axis=1), dtype=torch.float32)


def test_every_cache_channel_has_a_plan():
    """All 44 cache channels map to a treatment; only one is dropped."""
    kinds = {name: channel_kind(name) for name in STAT_CHANNEL_NAMES}
    assert [n for n, k in kinds.items() if k == "drop"] == [
        "vespeneFriendlyFireEconomy"
    ]
    assert kinds["pac_latency_s"] == "clip" and kinds["pac_actions"] == "clip"
    assert kinds["workersActiveCount"] == "z"
    assert kinds["mineralsCurrent"] == "symlog"


def test_unknown_channel_is_refused():
    """A new channel must be added to the plan explicitly."""
    with pytest.raises(ValueError, match="no scaling plan"):
        channel_kind("brandNewChannel")


def test_fit_standardises_and_drops():
    """After fitting, each kept training channel has mean 0 and std 1."""
    raw = _raw()
    transform = ChannelTransform(NAMES).fit(raw)
    assert transform.output_names == NAMES[:4]
    out = transform(raw)
    assert out.shape == (len(raw), 4)
    assert torch.allclose(out.mean(0), torch.zeros(4), atol=1e-4)
    assert torch.allclose(out.std(0), torch.ones(4), atol=1e-3)


def test_clip_uses_the_training_quantile():
    """Clipped channels are capped at their 99.9th training percentile."""
    raw = _raw()
    transform = ChannelTransform(NAMES).fit(raw)
    limit = float(np.quantile(raw[:, 1].double().numpy(), 0.999))
    assert float(transform.clip_max[1]) == pytest.approx(limit, rel=1e-5)
    assert torch.isinf(transform.clip_max[[0, 2, 3]]).all()
    spike = raw[:3].clone()
    spike[:, 1] = 1000.0
    assert torch.allclose(transform(spike)[:, 1], transform(spike)[0, 1])


def test_inverse_recovers_unclipped_values_including_negatives():
    """inverse(transform(x)) == x for values inside the clip limit."""
    raw = _raw()
    transform = ChannelTransform(NAMES).fit(raw)
    sample = raw[:500]
    inside = sample[:, 1] <= transform.clip_max[1]
    back = transform.inverse(transform(sample))
    kept = sample[:, :4]
    assert (kept[:, 3] < 0).any()
    assert torch.allclose(back[inside], kept[inside], rtol=1e-3, atol=1e-2)


def test_symlog_round_trip():
    """The signed exp inverts the signed log for both signs."""
    x = torch.tensor([-5000.0, -1.0, 0.0, 0.5, 12345.0])
    assert torch.allclose(symexp(symlog(x)), x, rtol=1e-5)


def test_use_before_fit_is_refused():
    """An unfitted transform cannot silently pass raw values through."""
    with pytest.raises(RuntimeError, match="before fit"):
        ChannelTransform(NAMES)(_raw(10))


def test_state_dict_round_trip():
    """A fresh transform loaded from the state dict gives identical outputs."""
    raw = _raw()
    fitted = ChannelTransform(NAMES).fit(raw)
    loaded = ChannelTransform(NAMES)
    loaded.load_state_dict(fitted.state_dict())
    assert torch.equal(loaded(raw[:100]), fitted(raw[:100]))
