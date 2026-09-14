"""Raw event-stream transform for SC2 replays (sequence pipeline).

Where ``rich_transform`` collapses a replay into one flat vector per player,
this transform keeps the game as what it is: a regular signal plus a set of
scattered events, both at full temporal resolution.

Per player it emits two streams:

- **stat stream** — the 39 ``SORTED_PLAYERSTATS_KEYS`` at every ``PlayerStats``
  tick, unaveraged, each carrying its true game ``loop``.
- **event stream** — every ``UnitBorn`` / ``UnitInit`` / ``UnitDied`` /
  ``Upgrade``, as ``(loop, type_id, unit_id)``.

Both streams are indexed onto a canonical tick grid.  The data study
(``docs/event_stream_study.md``) established that ``PlayerStats`` sits on an
exact ``loop = 160 * k`` grid (7.143 s/tick) once the stream is deduped and the
trailing partial tick is dropped, so the tick index *is* wall-clock time.  That
is what lets decoded feedback quote real game seconds instead of bin fractions.

Two properties this module guarantees, because the implementation it replaces
violated both: extraction is **pure** (no model, no RNG, no ``hash()``), and it
is **deterministic** across processes and interpreter restarts.

Unit-type strings are not resolved to global ids here.  Each replay carries a
small local vocabulary and local indices; the assembly step unions those into
one sorted global vocabulary and remaps.  This keeps workers independent
without ever needing a shared mutable table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sc2_datasets.replay_data.sc2_replay_data import SC2ReplayData

from latent_trainer.features.rich_transform import (
    SORTED_PLAYERSTATS_KEYS,
    _get_outcome,
    _get_stats_values,
)

# Measured in Phase 0: the PlayerStats cadence is exactly 160 game loops,
# i.e. 7.143 s at the 22.4 loops/s "Faster" game speed every SC2EGSet replay
# is recorded at.  99.9999% of ticks conform after cleaning.
TICK_INTERVAL_LOOPS = 160

# Index in this tuple *is* the event type id, so id 0 is reserved for padding
# and never collides with a real type.  The rejected implementation numbered
# its types from 0 and silently aliased them onto PAD/CLS/SEP.
EVENT_TYPE_NAMES: tuple[str, ...] = (
    "<PAD>",
    "UnitBorn",
    "UnitInit",
    "UnitDied",
    "Upgrade",
)
EVENT_TYPE_IDS = {name: idx for idx, name in enumerate(EVENT_TYPE_NAMES)}

# Reserved head of the global unit vocabulary.  Real type names start at 2.
PAD_UNIT_TOKEN = "<PAD>"
UNK_UNIT_TOKEN = "<UNK>"
RESERVED_UNIT_TOKENS: tuple[str, ...] = (PAD_UNIT_TOKEN, UNK_UNIT_TOKEN)

PLAYER_IDS: tuple[int, int] = (1, 2)

# ---------------------------------------------------------------------------
# Non-gameplay entities.
#
# The tracker stream attributes a number of engine, UI and cosmetic entities to
# a controlling player exactly as if they were units.  Measured over a 500-replay
# sample they are ~16% of all event tokens, and they are race-skewed
# (AdeptPhaseShift is Protoss-only, BroodlingEscort Zerg-only), so leaving them
# in would spend a sixth of every slice's attention budget on bookkeeping and
# hand the model a race cue dressed up as behaviour.
#
# The line drawn here is "did a player decide this, and can they act on it".
# Deployed ability units that a player aims and that affect the game are KEPT:
# KD8Charge, OracleStasisTrap, Changeling*, Interceptor, LocustMP, Broodling,
# CreepTumor, MULE.  Larva is emphatically kept -- larva births encode inject
# cadence, which is exactly the kind of coachable macro this project exists to
# give feedback on.
# ---------------------------------------------------------------------------
NON_GAMEPLAY_PREFIXES: tuple[str, ...] = (
    "Beacon",  # rally / control-group UI markers, spawned once per player
    "Spray",  # cosmetic emote sprays
    "RewardDance",  # cosmetic victory dances
    "LoadOut",  # cosmetic loadouts, e.g. LoadOutSpray@1
)
# Substring rather than suffix: Blizzard is not consistent about placement, so
# "Skin" catches AdeptSkin and PylonSkin but also GhostSkinNova.
NON_GAMEPLAY_SUBSTRINGS: tuple[str, ...] = ("Skin",)
# Suffix: the engine's placeholder targets all end this way -- InvisibleTargetDummy
# (~10.6% of raw tokens on its own), ParasiticBombDummy, ParasiticBombRelayDummy.
NON_GAMEPLAY_SUFFIXES: tuple[str, ...] = ("Dummy",)
NON_GAMEPLAY_NAMES: frozenset[str] = frozenset(
    {
        "GameHeartActive",  # map-script entity
        "AdeptPhaseShift",  # ability placeholder, not the Adept itself
        "AdeptPiercingAttack",  # ability placeholder
        "DisruptorPhased",  # Purification Nova projectile entity
        "LocustMPPrecursor",  # spawn artifact; the Locust itself is kept
        "BroodlingEscort",  # attack visual; the Broodling itself is kept
        "GhostAlternate",  # alternate-model skin variant
        "ColossusTal",  # Tal'darim skin variant
    }
)


def is_gameplay_entity(type_name: str) -> bool:
    """Whether a unit or upgrade name represents something a player actually did.

    Applies to ``Upgrade`` events as well as unit events, and must: SC2 records
    cosmetics as *upgrades*, so ``PylonSkin``, ``SprayZerg`` and
    ``RewardDanceColossus`` arrive through the upgrade branch rather than as
    units.  Real upgrades are mostly lower-case (``zerglingmovementspeed``), but
    plenty are not (``ProtossGroundWeaponsLevel1``, ``Stimpack``, ``BlinkTech``),
    so case is a tell and not a rule.
    """
    if type_name in NON_GAMEPLAY_NAMES:
        return False
    if type_name.startswith(NON_GAMEPLAY_PREFIXES):
        return False
    if type_name.endswith(NON_GAMEPLAY_SUFFIXES):
        return False
    return not any(token in type_name for token in NON_GAMEPLAY_SUBSTRINGS)


# Same minimum-duration guard as rich_transform: 4032 loops = 3 minutes.
MIN_GAME_LOOPS = 4032

N_STAT_CHANNELS = len(SORTED_PLAYERSTATS_KEYS)


@dataclass
class PlayerEventStreams:
    """One player's two streams, with unit ids local to the parent replay."""

    stat_values: np.ndarray  # [T_stat, 39] float32
    stat_loops: np.ndarray  # [T_stat]     int32
    stat_tick: np.ndarray  # [T_stat]     int32 — canonical grid index
    evt_type: np.ndarray  # [T_evt]      int8
    evt_unit_local: np.ndarray  # [T_evt]      int32 — index into local_vocab
    evt_loops: np.ndarray  # [T_evt]      int32
    evt_tick: np.ndarray  # [T_evt]      int32 — slice this event belongs to


@dataclass
class ReplayEventSample:
    """The transform's per-replay output."""

    players: tuple[PlayerEventStreams, PlayerEventStreams]
    local_vocab: list[str]
    label: int


def stat_tick_index(loop: int) -> int:
    """Canonical grid index of a ``PlayerStats`` tick recorded at *loop*.

    Rounding rather than flooring is deliberate: the opening tick sits at
    loop 1, not loop 0, and must still land on index 0.
    """
    return int((loop + TICK_INTERVAL_LOOPS // 2) // TICK_INTERVAL_LOOPS)


def event_tick_index(loop: int) -> int:
    """Index of the slice an irregular event belongs to.

    A slice is "the stat tick at loop ``160k`` plus everything that happened
    since the previous tick", so an event is attributed forward to the next
    tick boundary: loops ``(160(k-1), 160k]`` all belong to slice ``k``.
    """
    return int(-(-loop // TICK_INTERVAL_LOOPS))


def _clean_stat_events(events: list) -> list:
    """Apply the Phase 0 cleaning rule: dedupe by loop, drop the final tick.

    Duplicate loops occur in ~6k of 1.1M ticks and are exact repeats.  The
    trailing tick is dropped because it is both off-grid (the game ended
    mid-interval) and the game-end state — the leakage-prone ``final`` block
    the published pipeline already excludes.  Dropping it is correctness and
    leakage hygiene at once.
    """
    seen_loops: set[int] = set()
    deduped = []
    for event in events:
        if event.loop in seen_loops:
            continue
        seen_loops.add(event.loop)
        deduped.append(event)

    return deduped[:-1]


def _sorted_tracker_events(sc2_replay: SC2ReplayData) -> list:
    """Tracker events in causal order.

    ``sorted`` is stable, so events sharing a loop keep the parser's original
    ordering — which is the true within-loop causal order, and matters because
    a morph and a death can land on the same loop.
    """
    return sorted(sc2_replay.trackerEvents, key=lambda event: event.loop)


def _extract_streams(
    sc2_replay: SC2ReplayData,
) -> tuple[dict[int, list], dict[int, list[tuple[int, int, str]]]] | None:
    """Single pass over the replay, building both streams for both players.

    Returns ``(stats_by_player, events_by_player)`` where each event is
    ``(loop, type_id, type_name)``, or ``None`` if the replay is unusable.

    The single pass is what makes death attribution correct.  ``UnitDied``
    carries only ``(unitTagIndex, unitTagRecycle)`` — neither owner nor type —
    so both must come from a tag map built as the game plays out.  Phase 0
    measured that joining against ``UnitBorn`` alone resolves 91.7% of deaths
    and adding ``UnitInit`` reaches 100%.  ``UnitTypeChange`` is replayed onto
    the map in order so a Zerg morph is attributed to what the unit *became* if
    it dies after morphing, and to what it *was* if it dies before.
    """
    stats_by_player: dict[int, list] = {pid: [] for pid in PLAYER_IDS}
    events_by_player: dict[int, list[tuple[int, int, str]]] = {
        pid: [] for pid in PLAYER_IDS
    }

    # (unitTagIndex, unitTagRecycle) -> (current type name, owning player id)
    spawn_by_tag: dict[tuple[int, int], tuple[str, int]] = {}

    for event in _sorted_tracker_events(sc2_replay):
        name = type(event).__name__

        if name == "PlayerStats":
            if event.playerId in stats_by_player:
                stats_by_player[event.playerId].append(event)

        elif name in ("UnitBorn", "UnitInit"):
            key = (event.unitTagIndex, event.unitTagRecycle)
            # Recorded for every entity, filtered or not: the tag map is what
            # later resolves a death's type and owner, so pruning it would
            # reintroduce the orphaned-death problem it exists to solve.
            spawn_by_tag[key] = (event.unitTypeName, event.controlPlayerId)
            if event.controlPlayerId in events_by_player and is_gameplay_entity(
                event.unitTypeName
            ):
                events_by_player[event.controlPlayerId].append(
                    (event.loop, EVENT_TYPE_IDS[name], event.unitTypeName)
                )

        elif name == "UnitTypeChange":
            key = (event.unitTagIndex, event.unitTagRecycle)
            existing = spawn_by_tag.get(key)
            if existing is not None:
                # Keep the owner; only the type changes.
                spawn_by_tag[key] = (event.unitTypeName, existing[1])

        elif name == "UnitDied":
            key = (event.unitTagIndex, event.unitTagRecycle)
            spawned = spawn_by_tag.get(key)
            if spawned is None:
                # Phase 0 measured this at 0% with both spawn sources joined;
                # skip rather than emit an untyped, unattributed death.
                continue
            type_name, owner_id = spawned
            if owner_id in events_by_player and is_gameplay_entity(type_name):
                events_by_player[owner_id].append(
                    (event.loop, EVENT_TYPE_IDS["UnitDied"], type_name)
                )

        elif name == "Upgrade":
            # Cosmetics (skins, sprays, dances) are recorded as upgrades, so
            # the same filter has to run on this branch too.
            if event.playerId in events_by_player and is_gameplay_entity(
                event.upgradeTypeName
            ):
                events_by_player[event.playerId].append(
                    (event.loop, EVENT_TYPE_IDS["Upgrade"], event.upgradeTypeName)
                )

    for player_id in PLAYER_IDS:
        if not stats_by_player[player_id]:
            return None

    return stats_by_player, events_by_player


def _build_player_streams(
    stat_events: list,
    raw_events: Iterable[tuple[int, int, str]],
    vocab_index: dict[str, int],
    local_vocab: list[str],
) -> PlayerEventStreams | None:
    """Turn one player's raw events into arrays, interning type names locally."""

    cleaned = _clean_stat_events(sorted(stat_events, key=lambda e: e.loop))
    if not cleaned:
        return None

    stat_values = np.asarray(
        [_get_stats_values(stats_obj=event.stats) for event in cleaned],
        dtype=np.float32,
    )
    stat_loops = np.asarray([event.loop for event in cleaned], dtype=np.int32)
    stat_tick = np.asarray(
        [stat_tick_index(event.loop) for event in cleaned], dtype=np.int32
    )

    # Everything after the last kept stat tick is past the observation horizon.
    # Dropping it keeps the two streams on the same clock and prevents
    # post-final-tick events from leaking the ending back in.
    horizon = int(stat_loops[-1])
    last_tick = int(stat_tick[-1])

    evt_type: list[int] = []
    evt_unit_local: list[int] = []
    evt_loops: list[int] = []
    evt_tick: list[int] = []

    for loop, type_id, type_name in sorted(raw_events, key=lambda item: item[0]):
        if loop > horizon:
            continue
        slice_index = min(event_tick_index(loop), last_tick)
        local_id = vocab_index.get(type_name)
        if local_id is None:
            local_id = len(local_vocab)
            vocab_index[type_name] = local_id
            local_vocab.append(type_name)
        evt_type.append(type_id)
        evt_unit_local.append(local_id)
        evt_loops.append(loop)
        evt_tick.append(slice_index)

    return PlayerEventStreams(
        stat_values=stat_values,
        stat_loops=stat_loops,
        stat_tick=stat_tick,
        evt_type=np.asarray(evt_type, dtype=np.int8),
        evt_unit_local=np.asarray(evt_unit_local, dtype=np.int32),
        evt_loops=np.asarray(evt_loops, dtype=np.int32),
        evt_tick=np.asarray(evt_tick, dtype=np.int32),
    )


def event_transform(sc2_replay: SC2ReplayData) -> ReplayEventSample | None:
    """Extract both event streams for both players, or ``None`` to skip.

    Mirrors the ``rich_transform`` contract (skip on missing outcome, missing
    stats, or a game shorter than three minutes) so the same preprocessing
    bookkeeping applies, but returns a structured sample rather than a dense
    tensor — the streams are ragged and cannot be stacked.
    """
    label = _get_outcome(sc2_replay=sc2_replay)
    if label is None:
        return None

    try:
        game_duration = float(sc2_replay.header.elapsedGameLoops)
        if game_duration < MIN_GAME_LOOPS:
            return None
    except (AttributeError, ValueError, TypeError):
        return None

    extracted = _extract_streams(sc2_replay=sc2_replay)
    if extracted is None:
        return None
    stats_by_player, events_by_player = extracted

    local_vocab: list[str] = []
    vocab_index: dict[str, int] = {}

    players: list[PlayerEventStreams] = []
    for player_id in PLAYER_IDS:
        streams = _build_player_streams(
            stat_events=stats_by_player[player_id],
            raw_events=events_by_player[player_id],
            vocab_index=vocab_index,
            local_vocab=local_vocab,
        )
        if streams is None:
            return None
        players.append(streams)

    return ReplayEventSample(
        players=(players[0], players[1]),
        local_vocab=local_vocab,
        label=int(label),
    )
