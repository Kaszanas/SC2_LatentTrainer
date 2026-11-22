"""Tests for the per-tick PAC channels and race handling of the event transform."""

import random
from types import SimpleNamespace

import pytest

from latent_trainer.features.event_transform import (
    N_STAT_CHANNELS,
    RACE_IDS,
    STAT_CHANNEL_NAMES,
    event_transform,
)
from latent_trainer.features.perception_action_cycle import (
    GAME_LOOPS_PER_SECOND,
    PAC_CHANNEL_NAMES,
    extract_perception_action_cycles,
    extract_timed_pac_facts,
    per_tick_pac_channels,
)
from latent_trainer.features.rich_transform import SORTED_PLAYERSTATS_KEYS

LPS = GAME_LOOPS_PER_SECOND


# Fake events: the code dispatches on the class name, so the names matter.
class CameraUpdate:
    """Camera event with a target."""

    def __init__(self, loop: int, x: float, user: int = 0):
        self.loop, self.userid = loop, user
        self.target = SimpleNamespace(x=x, y=0.0)


class Cmd:
    """Command (an action)."""

    def __init__(self, loop: int, user: int = 0):
        self.loop, self.userid = loop, user


class PlayerStats:
    """Tracker stat snapshot with every counter set to ``value``."""

    def __init__(self, loop: int, player: int, value: float = 1.0):
        self.loop, self.playerId = loop, player
        self.stats = SimpleNamespace(**{k: value for k in SORTED_PLAYERSTATS_KEYS})


def _scenario_a() -> list:
    """
    Return three cycles; a brief flick at 200 merges into the first.

    Fixations: 100 (stands), 200 (ends at 202: brief, merges into 100),
    202 (stands, known at 207), 400 (last, known at 405).
    """
    return [
        CameraUpdate(100, 0.0),
        Cmd(103),
        Cmd(110),
        CameraUpdate(200, 50.0),
        Cmd(201),
        CameraUpdate(202, 100.0),
        Cmd(250),
        CameraUpdate(400, 200.0),
        Cmd(420),
    ]


# Timed facts:
def test_facts_have_known_values_and_settle_loops():
    """Each fact carries its value and the loop at which it became certain."""
    facts = extract_timed_pac_facts(_scenario_a(), user_id=0)
    assert facts.cycle_loops == [100, 207, 405]
    assert facts.action_loops == [103, 110, 201, 250, 420]
    assert facts.latencies == [(103, 3), (250, 48), (420, 20)]
    assert facts.gaps == [(207, 1), (405, 150)]


def test_latency_in_brief_fixation_waits_for_its_merge():
    """A first action inside a brief fixation is only certain once it merges."""
    events = [
        CameraUpdate(100, 0.0),
        CameraUpdate(300, 50.0),
        Cmd(301),
        CameraUpdate(302, 100.0),
    ]
    facts = extract_timed_pac_facts(events, user_id=0)
    # Fixation 300 ends at 302 (brief), so its action belongs to cycle 100.
    assert facts.latencies[0] == (302, 201)


def test_other_users_and_unparsed_events_are_ignored():
    """Events of other users and ``None`` entries do not contribute."""
    events = [*_scenario_a(), Cmd(105, user=1), None, CameraUpdate(150, 9.0, user=1)]
    assert extract_timed_pac_facts(events, user_id=0) == extract_timed_pac_facts(
        _scenario_a(), user_id=0
    )


# Per-tick channels:
def test_channels_bin_each_fact_at_the_first_tick_at_or_after_it():
    """Rows hold counts, mean latency/gap in seconds and the latency flag."""
    rows = per_tick_pac_channels(
        extract_timed_pac_facts(_scenario_a(), user_id=0), stat_loops=[1, 160, 320, 480]
    )
    expected = [
        [0, 0, 0, 0, 0],
        [1, 2, 3 / LPS, 0, 1],
        [1, 2, 48 / LPS, 1 / LPS, 1],
        [1, 1, 20 / LPS, 150 / LPS, 1],
    ]
    assert len(rows) == len(expected)
    for row, want in zip(rows, expected, strict=True):
        assert row == pytest.approx(want)
    assert len(PAC_CHANNEL_NAMES) == len(rows[0])


def test_facts_after_the_last_tick_are_dropped():
    """Nothing past the observation horizon is counted."""
    rows = per_tick_pac_channels(
        extract_timed_pac_facts(_scenario_a(), user_id=0), stat_loops=[1, 160]
    )
    assert rows[1][:2] == [1.0, 2.0]
    assert len(rows) == 2


def _random_events(rng: random.Random, n: int = 120) -> list:
    """Return a random stream of near and far camera moves, flicks and commands."""
    events, loop, x = [], 0, 0.0
    for _ in range(n):
        loop += rng.choice([0, 1, 2, 3, 4, 5, 6, 10, 30, 80])
        if rng.random() < 0.5:
            x += rng.choice([1.0, 3.0, 20.0, 60.0])  # some stay in the fixation
            events.append(CameraUpdate(loop, x))
        else:
            events.append(Cmd(loop))
    return events


@pytest.mark.parametrize("seed", range(200))
def test_channels_are_causal_under_truncation(seed):
    """Cutting the events at any loop leaves every earlier tick unchanged."""
    rng = random.Random(seed)
    events = _random_events(rng)
    end = max(e.loop for e in events)
    stat_loops = [1, *range(10, end + 40, 10)]
    full = per_tick_pac_channels(extract_timed_pac_facts(events, 0), stat_loops)
    for cut in stat_loops:
        part_events = [e for e in events if e.loop <= cut]
        part = per_tick_pac_channels(
            extract_timed_pac_facts(part_events, 0), stat_loops
        )
        for tick, loop in enumerate(stat_loops):
            if loop <= cut:
                assert part[tick] == full[tick], (seed, cut, tick)


@pytest.mark.parametrize("seed", range(50))
def test_facts_agree_with_the_whole_game_cycles(seed):
    """Facts reproduce the batch extractor's cycles, latencies and gaps."""
    rng = random.Random(seed)
    # A long final fixation with an action keeps the two final-cycle rules equal.
    events = _random_events(rng)
    end = max(e.loop for e in events)
    events += [CameraUpdate(end + 100, 1e6), Cmd(end + 150)]

    cycles = extract_perception_action_cycles(events, user_id=0)
    facts = extract_timed_pac_facts(events, user_id=0)

    assert len(facts.cycle_loops) == len(cycles)
    assert facts.action_loops == [loop for c in cycles for loop in c.action_loops]
    assert [v for _, v in facts.latencies] == [
        c.latency_loops for c in cycles if c.latency_loops is not None
    ]
    expected_gaps = [
        cycles[i + 1].start_loop - cycles[i].action_loops[-1]
        for i in range(len(cycles) - 1)
        if cycles[i].action_loops
    ]
    assert [v for _, v in facts.gaps] == expected_gaps


# Event transform:
def _replay(
    races=("Terr", "Zerg"),
    selected=("Terr", "Rand"),
    user_ids=(1, 0),
    game_events=None,
    game_events_err=False,
):
    """Return a minimal replay: 31 stat ticks per player, camera for userid 1."""
    infos = [
        SimpleNamespace(
            playerID=pid,
            userID=user_ids[pid - 1],
            race=races[pid - 1],
            selectedRace=selected[pid - 1],
            result="Win" if pid == 1 else "Loss",
        )
        for pid in (1, 2)
    ]
    tracker = [
        PlayerStats(loop, pid)
        for pid in (1, 2)
        for loop in [1, *range(160, 31 * 160 + 1, 160)]
    ]
    if game_events is None:
        game_events = [CameraUpdate(100, 0.0, user=1), Cmd(110, user=1)]
    return SimpleNamespace(
        header=SimpleNamespace(elapsedGameLoops=31 * 160),
        gameEventsErr=game_events_err,
        gameEvents=game_events,
        trackerEvents=tracker,
        toonPlayerDescMap=[SimpleNamespace(toon_player_info=info) for info in infos],
    )


def test_race_is_the_race_played_not_the_lobby_pick():
    """A Random pick resolves to the race actually played."""
    sample = event_transform(_replay(races=("Terr", "Zerg"), selected=("Terr", "Rand")))
    assert [p.race for p in sample.players] == [RACE_IDS["Terr"], RACE_IDS["Zerg"]]


def test_pac_channels_follow_user_id_not_player_id():
    """Player 1 has userID 1 here, so only player 1 gets the camera's PAC."""
    sample = event_transform(_replay(user_ids=(1, 0)))
    p1, p2 = sample.players
    assert p1.stat_values.shape[1] == N_STAT_CHANNELS
    assert list(STAT_CHANNEL_NAMES[-5:]) == list(PAC_CHANNEL_NAMES)
    assert p1.stat_values[:, -5:].sum() > 0
    assert p2.stat_values[:, -5:].sum() == 0
    # Cycle at loop 100 and its action at 110 land on the tick at loop 160.
    assert p1.stat_values[1, -5] == 1 and p1.stat_values[1, -4] == 1
    assert p1.stat_values[0, -5:].sum() == 0


@pytest.mark.parametrize(
    "kwargs",
    [{"races": ("Terr", "Rand")}, {"races": ("Terr", "")}, {"game_events_err": True}],
)
def test_replays_without_played_race_or_with_broken_game_events_are_skipped(kwargs):
    """Unknown race or unparsed game events make the replay unusable."""
    assert event_transform(_replay(**kwargs)) is None
