"""
Raw event-stream transform for SC2 replays (sequence pipeline).

Where ``rich_transform`` collapses a replay into one flat vector per player,
this transform keeps the game as what it is: a regular signal plus a set of
scattered events, both at full temporal resolution.

Per player it emits two streams:

- **stat stream** — the 39 ``SORTED_PLAYERSTATS_KEYS`` at every ``PlayerStats``
  tick, unaveraged, each carrying its true game ``loop``.
- **event stream** — every ``UnitBorn`` / ``UnitInit`` / ``UnitDied`` /
  ``Upgrade``, as ``(loop, type_id, unit_id)``.

Both streams are indexed onto a canonical tick grid.  The Phase 0 data study
(``scripts/analyze_event_streams.py``) established that ``PlayerStats`` sits on an
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

from latent_trainer.features.perception_action_cycle import (
    PAC_CHANNEL_NAMES,
    extract_timed_pac_facts,
    per_tick_pac_channels,
)
from latent_trainer.features.rich_transform import (
    SORTED_PLAYERSTATS_KEYS,
    _get_outcome,
    _get_player_info,
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
    "UnitTypeChange",
)
EVENT_TYPE_IDS = {name: idx for idx, name in enumerate(EVENT_TYPE_NAMES)}

# Reserved head of the global unit vocabulary.  Real type names start at 2.
PAD_UNIT_TOKEN = "<PAD>"
UNK_UNIT_TOKEN = "<UNK>"
RESERVED_UNIT_TOKENS: tuple[str, ...] = (PAD_UNIT_TOKEN, UNK_UNIT_TOKEN)

PLAYER_IDS: tuple[int, int] = (1, 2)

# Race actually played (``ToonPlayerInfo.race``), never ``selectedRace``: a
# Random pick resolves to the race the player ended up with. Index is the id.
RACE_NAMES: tuple[str, ...] = ("Terr", "Prot", "Zerg")
RACE_IDS = {name: idx for idx, name in enumerate(RACE_NAMES)}

# Non-gameplay entities:
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
# KD8Charge, OracleStasisTrap, Changeling*, LocustMP, CreepTumor, MULE.
# Interceptor, Broodling and Larva are real units but spawn automatically; they
# are removed separately, see AUTO_SPAWNED_NAMES below.
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

# Automatically-spawned entities:
#
# Distinct from the artifacts above: these are real gameplay objects that fight
# and die, but no player *decided* to create them. The structure spawns them on
# its own. Including them spends the token budget on the engine's behaviour
# rather than the player's, which is the opposite of what a coaching model wants
# to attend to -- and it is race-skewed, since all three are Zerg or Protoss.
#
# The contrast that defines the line: a MULE costs Orbital energy the player
# chose to spend, a CreepTumor is a deliberate spread decision, a Changeling is
# a cast -- all kept. A Larva is simply regenerated by a Hatchery.
#
# Larva was measured before removal at 25.2% of all event tokens. It was
# initially kept on the theory that Queen injects (which spawn 3 at once) would
# leave a recoverable burst signature; measurement refuted that -- 98.4% of
# larva births are singletons and only 0.5% arrive in threes, so inject cadence
# is not actually present in this signal. Larva deaths are additionally
# redundant, since morphing a larva already emits the resulting unit's birth.
AUTO_SPAWNED_NAMES: frozenset[str] = frozenset(
    {
        "Larva",  # Hatchery regenerates automatically
        "Broodling",  # byproduct of every Brood Lord attack
        "Interceptor",  # Carrier rebuilds these for free on a timer
    }
)
# Transitional morph states (Larva -> Egg, Zergling -> BanelingCocoon). A unit
# hatched from an egg or cocoon gets its own UnitBorn, so these duplicate it.
# Morphs in place (Hatchery -> Lair, CommandCenter -> OrbitalCommand, Roach ->
# Ravager, Gateway -> WarpGate) have no birth and are emitted as UnitTypeChange.
AUTO_SPAWNED_SUFFIXES: tuple[str, ...] = ("Egg", "Cocoon")


def is_gameplay_entity(type_name: str) -> bool:
    """
    Whether a unit or upgrade name represents something a player actually did.

    Also applied to ``Upgrade`` events: SC2 records cosmetics (``PylonSkin``,
    ``SprayZerg``, ``RewardDanceColossus``) as upgrades. Engine placeholders and
    automatically spawned entities (Larva, Broodling, Interceptor, eggs and
    cocoons) are excluded too.

    Parameters
    ----------
    type_name : str
        Unit type or upgrade name from the tracker event.

    Returns
    -------
    bool
        False for non-gameplay and automatically spawned entities, else True.
    """
    if type_name in NON_GAMEPLAY_NAMES or type_name in AUTO_SPAWNED_NAMES:
        return False
    if type_name.startswith(NON_GAMEPLAY_PREFIXES):
        return False
    if type_name.endswith(NON_GAMEPLAY_SUFFIXES + AUTO_SPAWNED_SUFFIXES):
        return False
    return not any(token in type_name for token in NON_GAMEPLAY_SUBSTRINGS)


# Same minimum-duration guard as rich_transform: 4032 loops = 3 minutes.
MIN_GAME_LOOPS = 4032

# Stat channels per tick: the PlayerStats counters, then the per-tick PAC channels.
STAT_CHANNEL_NAMES: tuple[str, ...] = (*SORTED_PLAYERSTATS_KEYS, *PAC_CHANNEL_NAMES)
N_STAT_CHANNELS = len(STAT_CHANNEL_NAMES)


@dataclass
class PlayerEventStreams:
    """
    One player's two streams, with unit ids local to the parent replay.

    Parameters
    ----------
    stat_values : np.ndarray
        Stat channels per cleaned tick (:data:`STAT_CHANNEL_NAMES`: PlayerStats
        counters, then PAC), ``[T, C]`` float32.
    stat_loops : np.ndarray
        Game loop of each tick, ``[T]`` int32.
    stat_tick : np.ndarray
        Grid index of each tick (:func:`stat_tick_index`), ``[T]`` int32.
    evt_type : np.ndarray
        Event type id into ``EVENT_TYPE_NAMES``, ``[E]``.
    evt_unit_local : np.ndarray
        Unit/upgrade id into the replay's ``local_vocab``, ``[E]``.
    evt_loops : np.ndarray
        Game loop of each event, ``[E]`` int32.
    evt_tick : np.ndarray
        Slice index of each event (:func:`event_tick_index`), ``[E]`` int32.
    race : int
        Race actually played, an id into :data:`RACE_NAMES`.
    """

    stat_values: np.ndarray  # [T_stat, C] float32
    stat_loops: np.ndarray  # [T_stat]     int32
    stat_tick: np.ndarray  # [T_stat]     int32 — canonical grid index
    evt_type: np.ndarray  # [T_evt]      int8
    evt_unit_local: np.ndarray  # [T_evt]      int32 — index into local_vocab
    evt_loops: np.ndarray  # [T_evt]      int32
    evt_tick: np.ndarray  # [T_evt]      int32 — slice this event belongs to
    race: int


@dataclass
class ReplayEventSample:
    """
    The event transform's per-replay output.

    Parameters
    ----------
    players : tuple[PlayerEventStreams, PlayerEventStreams]
        Streams of player 1 and player 2.
    local_vocab : list[str]
        Unit/upgrade names indexed by ``evt_unit_local``.
    label : int
        1 if player 1 won, 0 if player 2 won.
    """

    players: tuple[PlayerEventStreams, PlayerEventStreams]
    local_vocab: list[str]
    label: int


def stat_tick_index(loop: int) -> int:
    """
    Return the canonical grid index of a PlayerStats tick.

    Rounds rather than floors: the opening tick sits at loop 1, not 0, and must
    still land on index 0.

    Parameters
    ----------
    loop : int
        Game loop the tick was recorded at.

    Returns
    -------
    int
        ``round(loop / 160)``.
    """
    return int((loop + TICK_INTERVAL_LOOPS // 2) // TICK_INTERVAL_LOOPS)


def event_tick_index(loop: int) -> int:
    """
    Return the index of the slice an irregular event belongs to.

    A slice is the stat tick at loop ``160k`` plus everything since the previous
    tick, so loops ``(160(k-1), 160k]`` all belong to slice ``k``.

    Parameters
    ----------
    loop : int
        Game loop of the event.

    Returns
    -------
    int
        ``ceil(loop / 160)``.
    """
    return int(-(-loop // TICK_INTERVAL_LOOPS))


def _clean_stat_events(events: list) -> list:
    """
    Apply the Phase 0 cleaning rule: dedupe by loop, drop the final tick.

    Duplicate loops are exact repeats. The trailing tick is off-grid (the game
    ended mid-interval) and is the game-end state, the leakage-prone ``final``
    block the published pipeline excludes.

    Parameters
    ----------
    events : list
        One player's PlayerStats events, sorted by loop.

    Returns
    -------
    list
        Deduplicated events without the last one.
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
    """
    Return the replay's tracker events in causal order.

    The sort is stable, so events sharing a loop keep the parser's order, which
    is the true within-loop order (a morph and a death can share a loop).

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed replay.

    Returns
    -------
    list
        Tracker events sorted by loop.
    """
    return sorted(sc2_replay.trackerEvents, key=lambda event: event.loop)


def _extract_streams(
    sc2_replay: SC2ReplayData,
) -> tuple[dict[int, list], dict[int, list[tuple[int, int, str]]]]:
    """
    Build both streams for both players in a single causal pass.

    ``UnitDied`` carries only ``(unitTagIndex, unitTagRecycle)``, so owner and
    type come from a tag map built as the game plays out, joining ``UnitBorn``
    and ``UnitInit`` (100% of deaths resolved) and replaying ``UnitTypeChange``
    so a morph is attributed to what the unit was when it died. Each gameplay
    ``UnitTypeChange`` is also an event of its own, since in-place morphs
    (Lair, OrbitalCommand, Ravager, WarpGate) have no ``UnitBorn``.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed replay.

    Returns
    -------
    tuple[dict[int, list], dict[int, list[tuple[int, int, str]]]]
        ``(stats_by_player, events_by_player)`` keyed by player id, events as
        ``(loop, type_id, type_name)``; a player's list may be empty.
    """
    stats_by_player: dict[int, list] = {pid: [] for pid in PLAYER_IDS}
    events_by_player: dict[int, list[tuple[int, int, str]]] = {
        pid: [] for pid in PLAYER_IDS
    }

    # (unitTagIndex, unitTagRecycle) -> (current type name, owning player id)
    spawn_by_tag: dict[tuple[int, int], tuple[str, int]] = {}

    for event in _sorted_tracker_events(sc2_replay):
        name = type(event).__name__

        match name:
            case "PlayerStats":
                if event.playerId in stats_by_player:
                    stats_by_player[event.playerId].append(event)

            case "UnitBorn" | "UnitInit":
                key = (event.unitTagIndex, event.unitTagRecycle)
                # Recorded for every entity, filtered or not: the tag map is
                # what later resolves a death's type and owner, so pruning it
                # would reintroduce the orphaned-death problem it exists to solve.
                spawn_by_tag[key] = (event.unitTypeName, event.controlPlayerId)
                if event.controlPlayerId in events_by_player and is_gameplay_entity(
                    event.unitTypeName
                ):
                    events_by_player[event.controlPlayerId].append(
                        (event.loop, EVENT_TYPE_IDS[name], event.unitTypeName)
                    )

            case "UnitTypeChange":
                key = (event.unitTagIndex, event.unitTagRecycle)
                existing = spawn_by_tag.get(key)
                if existing is not None:
                    # Keep the owner; only the type changes.
                    owner_id = existing[1]
                    spawn_by_tag[key] = (event.unitTypeName, owner_id)
                    # Tech morphs (Lair, OrbitalCommand, Ravager) and mode
                    # toggles (SiegeTankSieged) are emitted under the new type.
                    if owner_id in events_by_player and is_gameplay_entity(
                        event.unitTypeName
                    ):
                        events_by_player[owner_id].append(
                            (
                                event.loop,
                                EVENT_TYPE_IDS["UnitTypeChange"],
                                event.unitTypeName,
                            )
                        )

            case "UnitDied":
                key = (event.unitTagIndex, event.unitTagRecycle)
                spawned = spawn_by_tag.get(key)
                if spawned is None:
                    # Phase 0 measured this at 0% with both spawn sources
                    # joined; skip rather than emit an untyped, unattributed death.
                    continue
                type_name, owner_id = spawned
                if owner_id in events_by_player and is_gameplay_entity(type_name):
                    events_by_player[owner_id].append(
                        (event.loop, EVENT_TYPE_IDS["UnitDied"], type_name)
                    )

            case "Upgrade":
                # Cosmetics (skins, sprays, dances) are recorded as upgrades, so
                # the same filter has to run on this branch too.
                if event.playerId in events_by_player and is_gameplay_entity(
                    event.upgradeTypeName
                ):
                    events_by_player[event.playerId].append(
                        (event.loop, EVENT_TYPE_IDS["Upgrade"], event.upgradeTypeName)
                    )

    return stats_by_player, events_by_player


def _build_player_streams(
    stat_events: list,
    raw_events: Iterable[tuple[int, int, str]],
    game_events: list,
    user_id: int,
    race: int,
    vocab_index: dict[str, int],
    local_vocab: list[str],
) -> PlayerEventStreams | None:
    """
    Turn one player's raw events into arrays, interning type names locally.

    Stat ticks are cleaned with :func:`_clean_stat_events`; events after the
    last kept tick are past the observation horizon and dropped. The PAC
    channels are appended to each tick's PlayerStats counters; each PAC fact is
    counted at the first tick at or after the loop where it became certain
    (:func:`per_tick_pac_channels`), so no tick reflects anything later.

    Parameters
    ----------
    stat_events : list
        The player's PlayerStats events.
    raw_events : Iterable[tuple[int, int, str]]
        The player's ``(loop, type_id, type_name)`` events.
    game_events : list
        The replay's game events, for the PAC channels.
    user_id : int
        The player's ``userID`` in the game events (differs from ``playerID``).
    race : int
        Race id of the player, see :data:`RACE_NAMES`.
    vocab_index : dict[str, int]
        Replay-local name-to-id map, extended in place.
    local_vocab : list[str]
        Replay-local vocabulary, extended in place.

    Returns
    -------
    PlayerEventStreams | None
        The player's arrays, or None if no stat tick survives cleaning.
    """
    cleaned = _clean_stat_events(sorted(stat_events, key=lambda e: e.loop))
    if not cleaned:
        return None

    loops = [int(event.loop) for event in cleaned]
    pac = per_tick_pac_channels(
        extract_timed_pac_facts(game_events=game_events, user_id=user_id),
        stat_loops=loops,
    )
    stat_values = np.asarray(
        [
            [*_get_stats_values(stats_obj=event.stats), *pac_row]
            for event, pac_row in zip(cleaned, pac, strict=True)
        ],
        dtype=np.float32,
    )
    stat_loops = np.asarray(loops, dtype=np.int32)
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
        race=race,
    )


def event_transform(sc2_replay: SC2ReplayData) -> ReplayEventSample | None:
    """
    Extract both event streams for both players, or None to skip the replay.

    Mirrors the ``rich_transform`` skip rules (undecided outcome, missing stats,
    game shorter than 4032 loops) but returns a ragged structured sample. Also
    skipped: replays whose game events failed to parse (the PAC channels would
    be computed from a partial stream) and players without a known ``userID``
    or played race.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed replay.

    Returns
    -------
    ReplayEventSample | None
        Streams for both players and the label, or None to skip.
    """
    sample, _ = _transform_or_skip(sc2_replay)
    return sample


def event_skip_reason(sc2_replay: SC2ReplayData) -> tuple[str, str] | None:
    """
    Explain why :func:`event_transform` skips a replay.

    Runs the same code path as the transform, so the reason is by construction
    the branch that rejected the replay.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed replay.

    Returns
    -------
    tuple[str, str] | None
        ``(code, detail)`` with ``code`` from :data:`SKIP_REASONS` and the
        values that triggered it, or None if the replay is kept.
    """
    _, reason = _transform_or_skip(sc2_replay)
    return reason


# Skip reasons, in the order they are checked:
SKIP_REASONS: dict[str, str] = {
    "no_outcome": "result is undecided/draw/tie, unrecognised, or player 1 missing",
    "bad_header": "header.elapsedGameLoops missing or not a number",
    "too_short": f"game shorter than {MIN_GAME_LOOPS} loops (3 min)",
    "game_events_error": "the parser flagged the game events as broken",
    "no_player_stats": "a player has no PlayerStats tracker events",
    "no_player_info": "a player is missing from toonPlayerDescMap",
    "unknown_race": "a player's played race is not Terr/Prot/Zerg",
    "no_stat_ticks": "a player has no PlayerStats tick left after cleaning",
}


def _results(sc2_replay: SC2ReplayData) -> str:
    """
    Describe every player's result, for skip details.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed replay.

    Returns
    -------
    str
        ``playerID=result`` pairs.
    """
    return ", ".join(
        f"{toon.toon_player_info.playerID}={toon.toon_player_info.result!r}"
        for toon in sc2_replay.toonPlayerDescMap
    )


def _transform_or_skip(
    sc2_replay: SC2ReplayData,
) -> tuple[ReplayEventSample | None, tuple[str, str] | None]:
    """
    Build the sample, or return the reason the replay is skipped.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed replay.

    Returns
    -------
    tuple[ReplayEventSample | None, tuple[str, str] | None]
        ``(sample, None)`` if kept, else ``(None, (code, detail))``.
    """
    label = _get_outcome(sc2_replay=sc2_replay)
    if label is None:
        return None, ("no_outcome", f"results: {_results(sc2_replay)}")

    try:
        game_duration = float(sc2_replay.header.elapsedGameLoops)
    except (AttributeError, ValueError, TypeError) as error:
        return None, ("bad_header", repr(error))
    if game_duration < MIN_GAME_LOOPS:
        return None, ("too_short", f"elapsedGameLoops={game_duration:.0f}")

    if sc2_replay.gameEventsErr:
        return None, ("game_events_error", "gameEventsErr=True")

    stats_by_player, events_by_player = _extract_streams(sc2_replay=sc2_replay)
    for player_id in PLAYER_IDS:
        if not stats_by_player[player_id]:
            return None, ("no_player_stats", f"player {player_id}")

    local_vocab: list[str] = []
    vocab_index: dict[str, int] = {}

    players: list[PlayerEventStreams] = []
    for player_id in PLAYER_IDS:
        info = _get_player_info(sc2_replay=sc2_replay, player_id=player_id)
        if info is None:
            return None, ("no_player_info", f"player {player_id}")
        if info.race not in RACE_IDS:
            return None, (
                "unknown_race",
                f"player {player_id}: race={info.race!r}, "
                f"selectedRace={info.selectedRace!r}",
            )
        streams = _build_player_streams(
            stat_events=stats_by_player[player_id],
            raw_events=events_by_player[player_id],
            game_events=sc2_replay.gameEvents,
            user_id=info.userID,
            race=RACE_IDS[info.race],
            vocab_index=vocab_index,
            local_vocab=local_vocab,
        )
        if streams is None:
            n_stats = len(stats_by_player[player_id])
            return None, ("no_stat_ticks", f"player {player_id}: {n_stats} raw ticks")
        players.append(streams)

    sample = ReplayEventSample(
        players=(players[0], players[1]),
        local_vocab=local_vocab,
        label=int(label),
    )
    return sample, None
