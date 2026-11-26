"""Tests for the pair tail, prefix crops, collate grouping and train-only vocab."""

import math

import numpy as np
import pytest
import torch

from latent_trainer.features.data_utils import collate_event_sequences
from latent_trainer.features.event_transform import (
    N_STAT_CHANNELS,
    PlayerEventStreams,
    ReplayEventSample,
)
from latent_trainer.features.perception_action_cycle import GAME_LOOPS_PER_SECOND
from latent_trainer.features.preprocess_events import (
    _flatten_split,
    build_unit_vocab,
    unseen_unit_counts,
)
from latent_trainer.features.type import EventSequenceDataset, EventSplit

VOCAB = ["<PAD>", "<UNK>", "Marine"]


def _seconds(loop: int) -> float:
    """Game seconds of a loop, so a fixed crop lands exactly on it."""
    return loop / GAME_LOOPS_PER_SECOND


def _player(n_ticks: int, event_loops=(), race: int = 0) -> PlayerEventStreams:
    """Ticks at loops 1, 160, 320, ...; channel 0 holds the tick index."""
    ticks = np.arange(n_ticks, dtype=np.int32)
    values = np.zeros((n_ticks, N_STAT_CHANNELS), dtype=np.float32)
    values[:, 0] = ticks
    loops = np.asarray(event_loops, dtype=np.int32)
    return PlayerEventStreams(
        stat_values=values,
        stat_loops=np.maximum(ticks * 160, 1).astype(np.int32),
        stat_tick=ticks,
        evt_type=np.ones(len(loops), dtype=np.int8),
        evt_unit_local=np.zeros(len(loops), dtype=np.int32),
        evt_loops=loops,
        evt_tick=np.minimum(-(-loops // 160), n_ticks - 1).astype(np.int32),
        race=race,
    )


def _sample(p1: PlayerEventStreams, p2: PlayerEventStreams, label: int = 1):
    return ReplayEventSample(players=(p1, p2), local_vocab=["Marine"], label=label)


def _split(samples: list[ReplayEventSample]) -> EventSplit:
    return EventSplit(**_flatten_split(samples, VOCAB))


# Pair tail:
def test_tail_drop_is_shared_by_the_pair():
    """The tail is cut on the shorter stream and applied to both players."""
    split = _split([_sample(_player(10), _player(8))])
    dataset = EventSequenceDataset(split, drop_tail_frac=0.25)
    p1, p2 = dataset[0]["players"]
    keep = math.ceil(8 * 0.75)
    assert p1["stat_slice"].tolist() == list(range(keep))
    assert p2["stat_slice"].tolist() == list(range(keep))
    assert int(dataset.horizon_loops[0]) == (keep - 1) * 160
    assert int(dataset[0]["crop_loop"]) == (keep - 1) * 160


def test_tail_drop_counts_slices_of_several_ticks():
    """With slice_ticks=2 the tail is cut in whole slices."""
    split = _split([_sample(_player(10), _player(10))])
    p1, _ = EventSequenceDataset(split, slice_ticks=2, drop_tail_frac=0.2)[0]["players"]
    # 5 slices, keep 4, i.e. ticks 0..7.
    assert p1["stat_values"][:, 0].tolist() == list(range(8))
    assert p1["stat_slice"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


# Prefix crops:
def test_fixed_crop_keeps_nothing_after_the_cutoff():
    """Ticks after the cutoff go, and so do events of a dropped tick."""
    events = [150, 170, 330, 470, 900]
    split = _split([_sample(_player(20, events), _player(20, events))])
    item = EventSequenceDataset(split, drop_tail_frac=0.0, crop=_seconds(400))[0]
    assert int(item["crop_loop"]) == 400
    for player in item["players"]:
        assert player["stat_loops"].tolist() == [1, 160, 320]
        # 330 <= 400 but belongs to the tick at 480, which is cut.
        assert player["evt_loops"].tolist() == [150, 170]
        assert int(player["evt_loops"].max()) <= 400


def test_fixed_crop_excludes_replays_that_end_before_it():
    """Short replays are excluded and indices map to the remaining replays."""
    split = _split(
        [
            _sample(_player(10), _player(10), label=0),
            _sample(_player(30), _player(30), label=1),
        ]
    )
    dataset = EventSequenceDataset(split, drop_tail_frac=0.0, crop=_seconds(2000))
    assert len(dataset) == 1 and dataset.n_excluded == 1
    assert int(dataset[0]["label"]) == 1


def test_random_crop_is_reproducible_and_within_bounds():
    """Random cutoffs lie in [min_crop_s, horizon] and follow torch's seed."""
    split = _split([_sample(_player(100), _player(100))])
    dataset = EventSequenceDataset(
        split, drop_tail_frac=0.0, crop="random", min_crop_s=60.0
    )
    horizon = int(dataset.horizon_loops[0])

    def draws() -> list[int]:
        torch.manual_seed(0)
        return [int(dataset[0]["crop_loop"]) for _ in range(50)]

    first = draws()
    assert first == draws()
    assert len(set(first)) > 1
    assert all(60.0 * GAME_LOOPS_PER_SECOND <= c <= horizon for c in first)
    torch.manual_seed(1)
    item = dataset[0]
    for player in item["players"]:
        assert int(player["stat_loops"].max()) <= int(item["crop_loop"])


def test_random_crop_serves_games_shorter_than_min_crop_whole():
    """A game shorter than min_crop_s is not cropped."""
    split = _split([_sample(_player(10), _player(10))])
    dataset = EventSequenceDataset(
        split, drop_tail_frac=0.0, crop="random", min_crop_s=600.0
    )
    assert int(dataset[0]["crop_loop"]) == int(dataset.horizon_loops[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crop": "later"},
        {"crop": 0},
        {"crop": True},
        {"min_crop_s": -1.0},
        {"slice_ticks": 0},
        {"drop_tail_frac": 1.0},
    ],
)
def test_invalid_arguments_raise(kwargs):
    """Bad policy arguments are rejected."""
    split = _split([_sample(_player(5), _player(5))])
    with pytest.raises(ValueError):
        EventSequenceDataset(split, **kwargs)


# Collate:
def _batch(max_events_per_slice: int | None = 2) -> dict:
    split = _split(
        [
            _sample(
                _player(4, [10, 20, 30, 40, 50, 200], race=2),
                _player(6, [170], race=1),
                label=0,
            )
        ]
    )
    dataset = EventSequenceDataset(split, drop_tail_frac=0.0)
    return collate_event_sequences([dataset[0]], max_events_per_slice)


def test_collate_groups_events_by_slice_with_cap():
    """Slices keep K evenly spaced events and report the true count."""
    batch = _batch(max_events_per_slice=2)
    # 4 and 6 ticks: the pair horizon cuts both players to 4 slices.
    assert batch["evt_group"].shape == (1, 2, 4, 2)
    # Player 1: five events in slice 1, one in slice 2.
    assert batch["evt_count"][0, 0].tolist() == [0, 5, 1, 0]
    assert batch["evt_count"][0, 1].tolist() == [0, 0, 1, 0]
    assert batch["evt_group"][0, 0, 1].tolist() == [0, 2]
    assert batch["evt_group_mask"][0, 0, 1].tolist() == [True, True]
    assert batch["evt_group"][0, 0, 2].tolist() == [5, 0]
    assert batch["evt_group_mask"][0, 0, 2].tolist() == [True, False]
    assert not batch["evt_group_mask"][0, 0, 0].any()


def test_collate_groups_point_at_events_of_their_own_slice():
    """Every unmasked group entry indexes an event in that slice."""
    batch = _batch(max_events_per_slice=3)
    group, mask = batch["evt_group"], batch["evt_group_mask"]
    for player in range(2):
        for slice_index in range(group.shape[2]):
            for k in range(group.shape[3]):
                if mask[0, player, slice_index, k]:
                    event = group[0, player, slice_index, k]
                    assert batch["evt_slice"][0, player, event] == slice_index


def test_collate_slice_loops_race_and_crop():
    """Slice loops, slice mask, race and crop loop come through per player."""
    batch = _batch()
    assert batch["slice_loops"][0].tolist() == [[1, 160, 320, 480]] * 2
    assert batch["slice_mask"][0].sum(dim=1).tolist() == [4, 4]
    assert batch["race"].tolist() == [[2, 1]]
    assert batch["crop_loop"].tolist() == [480]


def test_event_slots_invert_the_groups():
    """Every kept event's slot points back at it; the others get the spare."""
    batch = _batch(max_events_per_slice=2)
    group, mask, slot = batch["evt_group"], batch["evt_group_mask"], batch["evt_slot"]
    spare = group.shape[2] * group.shape[3]
    for player in range(2):
        flat_group = group[0, player].reshape(-1)
        flat_mask = mask[0, player].reshape(-1)
        kept = set(flat_group[flat_mask].tolist())
        for event, s in enumerate(slot[0, player].tolist()):
            if s == spare:
                assert event not in kept
            else:
                assert flat_mask[s] and flat_group[s] == event
        assert sum(s != spare for s in slot[0, player].tolist()) == len(kept)


def test_slice_valid_index_lists_the_real_slices():
    """The index form equals the mask's positions."""
    batch = _batch()
    expected = batch["slice_mask"].reshape(-1).nonzero()[:, 0]
    assert torch.equal(batch["slice_valid_index"], expected)


def test_collate_without_a_cap_keeps_every_event():
    """None sizes K to the batch's busiest slice, so nothing is dropped."""
    batch = _batch(max_events_per_slice=None)
    assert batch["evt_group"].shape[-1] == 5
    assert (batch["evt_group_mask"].sum(-1) == batch["evt_count"]).all()


def test_flatten_refuses_ids_that_overflow_their_dtype():
    """A vocabulary beyond int16 raises instead of wrapping around."""
    big = [f"U{i}" for i in range(40_000)]
    player = _player(3, [10])
    player.evt_unit_local = np.asarray([0], dtype=np.int32)
    sample = ReplayEventSample(
        players=(player, _player(3)), local_vocab=[big[-1]], label=1
    )
    with pytest.raises(OverflowError, match="int16"):
        _flatten_split([sample], ["<PAD>", "<UNK>", *big])


def test_collate_rejects_a_zero_cap():
    """K must be at least 1."""
    split = _split([_sample(_player(4), _player(4))])
    with pytest.raises(ValueError):
        collate_event_sequences([EventSequenceDataset(split)[0]], 0)


# Vocabulary:
def test_vocab_comes_from_train_and_unseen_names_map_to_unk():
    """Names first seen outside train become <UNK> (id 1)."""
    train = [_sample(_player(3, [10]), _player(3))]
    vocab = build_unit_vocab(train)
    assert vocab == VOCAB

    unseen = _player(3, [10, 20])
    unseen.evt_unit_local = np.asarray([0, 1], dtype=np.int32)
    val = [
        ReplayEventSample(
            players=(unseen, _player(3)), local_vocab=["Marine", "Thor"], label=1
        )
    ]
    assert _flatten_split(val, vocab)["evt_unit"].tolist() == [2, 1]
    assert unseen_unit_counts(val, vocab) == {"Thor": 1}
