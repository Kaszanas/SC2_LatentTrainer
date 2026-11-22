"""Tests for the load-time layout check of the ``events`` cache."""

import numpy as np
import pytest
import torch

from latent_trainer.features.data_utils import load_event_cache
from latent_trainer.features.event_transform import (
    EVENT_TYPE_NAMES,
    PlayerEventStreams,
    ReplayEventSample,
)
from latent_trainer.features.preprocess_events import _flatten_split
from latent_trainer.features.rich_transform import SORTED_PLAYERSTATS_KEYS
from latent_trainer.features.type import EventSequenceDataset

N_CHANNELS = len(SORTED_PLAYERSTATS_KEYS)
UNIT_VOCAB = ["<PAD>", "<UNK>", "Marine"]


def _player(n_ticks: int, n_events: int, value: float) -> PlayerEventStreams:
    """Return a stream whose stat rows are all ``value``."""
    ticks = np.arange(n_ticks, dtype=np.int32)
    return PlayerEventStreams(
        stat_values=np.full((n_ticks, N_CHANNELS), value, dtype=np.float32),
        stat_loops=ticks * 160,
        stat_tick=ticks,
        evt_type=np.ones(n_events, dtype=np.int8),
        evt_unit_local=np.zeros(n_events, dtype=np.int32),
        evt_loops=np.zeros(n_events, dtype=np.int32),
        evt_tick=np.zeros(n_events, dtype=np.int32),
    )


def _samples() -> list[ReplayEventSample]:
    """Two replays with distinct per-player lengths and values."""
    return [
        ReplayEventSample(
            players=(_player(3, 2, 10.0), _player(4, 1, 20.0)),
            local_vocab=["Marine"],
            label=1,
        ),
        ReplayEventSample(
            players=(_player(5, 0, 30.0), _player(2, 3, 40.0)),
            local_vocab=["Marine"],
            label=0,
        ),
    ]


def _cache() -> dict:
    """Build a cache dict with the real producer, same two replays per split."""
    cache: dict = {}
    for split in ("train", "val", "test"):
        for key, value in _flatten_split(_samples(), UNIT_VOCAB).items():
            cache[f"{split}_{key}"] = value
        cache[f"{split}_ids"] = ["t/a.json", "t/b.json"]
    cache.update(
        stat_channel_names=list(SORTED_PLAYERSTATS_KEYS),
        unit_vocab=UNIT_VOCAB,
        event_type_names=list(EVENT_TYPE_NAMES),
        tick_interval_loops=160,
        transform="events",
    )
    return cache


def _load(tmp_path, cache: dict):
    path = tmp_path / "cache.pt"
    torch.save(cache, path)
    return load_event_cache(path)


def test_valid_cache_loads_and_pairs_streams_with_their_replay(tmp_path):
    """A producer-built cache loads, and replay i reads streams 2i and 2i + 1."""
    bundle = _load(tmp_path, _cache())
    dataset = EventSequenceDataset(bundle.val, drop_tail_frac=0.0)
    mean, std = bundle.mean[0, 0], bundle.std[0, 0]

    second = dataset[1]
    p1, p2 = second["players"]
    assert int(second["label"]) == 0
    assert p1["stat_values"].shape[0] == 5 and p2["stat_values"].shape[0] == 2
    assert torch.allclose(p1["stat_values"][:, 0] * std + mean, torch.tensor(30.0))
    assert torch.allclose(p2["stat_values"][:, 0] * std + mean, torch.tensor(40.0))
    assert p1["evt_type"].numel() == 0 and p2["evt_type"].numel() == 3


def test_missing_stream_is_rejected(tmp_path):
    """Dropping one player stream (one replay with a single stream) fails."""
    cache = _cache()
    offsets = cache["train_stat_offsets"]
    last = int(offsets[-2])
    cache["train_stat_offsets"] = offsets[:-1]
    for key in ("train_stat_values", "train_stat_loops", "train_stat_tick"):
        cache[key] = cache[key][:last]
    with pytest.raises(ValueError, match="train stat_offsets has 4 entries"):
        _load(tmp_path, cache)


def test_label_count_mismatch_is_rejected(tmp_path):
    """One label too many no longer matches 2N + 1 offsets."""
    cache = _cache()
    cache["val_labels"] = torch.cat([cache["val_labels"], torch.tensor([1])])
    with pytest.raises(ValueError, match="val stat_offsets"):
        _load(tmp_path, cache)


def test_offsets_not_covering_values_are_rejected(tmp_path):
    """Offsets that end before the end of the value arrays fail."""
    cache = _cache()
    cache["test_evt_type"] = torch.cat([cache["test_evt_type"], cache["test_evt_type"]])
    with pytest.raises(ValueError, match="test evt_offsets ends at"):
        _load(tmp_path, cache)


def test_decreasing_offsets_are_rejected(tmp_path):
    """Offsets must not decrease."""
    cache = _cache()
    offsets = cache["train_evt_offsets"].clone()
    offsets[1], offsets[2] = offsets[2].clone(), offsets[1].clone()
    cache["train_evt_offsets"] = offsets
    with pytest.raises(ValueError, match="train evt_offsets decreases"):
        _load(tmp_path, cache)


def test_replay_id_count_mismatch_is_rejected(tmp_path):
    """Replay ids, when present, must be one per replay."""
    cache = _cache()
    cache["val_ids"] = ["t/a.json"]
    with pytest.raises(ValueError, match="val has 1 replay ids for 2 labels"):
        _load(tmp_path, cache)


def test_old_cache_without_ids_still_loads(tmp_path):
    """Caches built before the split manifest have empty ids and still load."""
    cache = _cache()
    for split in ("train", "val", "test"):
        del cache[f"{split}_ids"]
    assert len(_load(tmp_path, cache).train.labels) == 2
