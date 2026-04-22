"""Rich feature transform for SC2 replays.

Extracts a comprehensive feature vector per player represented as a nested TensorDict:
- Temporal economy snapshots (early/mid/late game) — 39 Stats fields × 3 time windows
- Final economy state — 39 Stats fields
- Economy rate-of-change (late minus early) — 39 Stats fields
- Player meta stats: APM, MMR, SQ, supply_capped_percent
- Unit activity: units_born, units_killed
- upgrade_count

Output: TensorDict with batch_size=[2] (player 0 = player 1, player 1 = player 2).
Each player entry is a nested TensorDict with keys:
  early, mid, late, final, delta  → nested TensorDict of 39 Stats fields (float32)
  meta                            → nested TensorDict {APM, MMR, SQ, supply_capped_percent}
  units_born, units_killed, upgrade_count  → scalar float32 tensors

TODO: preprocess_dataset.py must be updated in a follow-up to handle TensorDict output.
"""

from typing import Tuple

import torch
from sc2_datasets.replay_data.sc2_replay_data import SC2ReplayData, ToonPlayerDesc
from sc2_datasets.replay_parser.tracker_events.events.player_stats.player_stats import (
    PlayerStats,
)
from sc2_datasets.replay_parser.tracker_events.events.player_stats.stats import Stats
from tensordict import TensorDict


def _get_stats_values(stats: Stats) -> TensorDict:
    """Convert a Stats dataclass to a float32 TensorDict with one key per field."""
    td = TensorDict.from_dataclass(stats)
    return td.apply(lambda t: t.float())


def _get_player_stats_timeseries(
    sc2_replay: SC2ReplayData,
    player_id: int,
) -> list[PlayerStats]:
    """
    Collect all PlayerStats events for a given player, sorted by loop.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed SC2 replay data containing tracker events.
    player_id : int
        ID of the player to extract stats for.

    Returns
    -------
    list[PlayerStats]
        List of PlayerStats events for the specified player, sorted by game loop.
    """
    events = []
    for event in sc2_replay.trackerEvents:
        if type(event).__name__ == "PlayerStats" and event.playerId == player_id:
            events.append(event)
    events.sort(key=lambda e: e.loop)
    return events


def _temporal_snapshot(
    events: list[PlayerStats],
    start_frac: float,
    end_frac: float,
) -> TensorDict:
    """
    Extract a temporal snapshot of player stats averaged over a fractional time window.

    Parameters
    ----------
    events : list[PlayerStats]
        List of PlayerStats events.
    start_frac : float
        Fractional start time of the window.
    end_frac : float
        Fractional end time of the window.

    Returns
    -------
    TensorDict
        Averaged Stats fields over the specified time window.
    """
    n = len(events)
    start_idx = int(start_frac * n)
    end_idx = max(int(end_frac * n), start_idx + 1)

    window = events[start_idx:end_idx]
    values = [_get_stats_values(e.stats) for e in window]
    stacked = torch.stack(values, dim=0)
    return stacked.mean(dim=0, dtype=torch.float32)


def _count_units_born(sc2_replay: SC2ReplayData, player_id: int) -> int:
    """
    Counts the UnitBorn events for a given player.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed SC2 replay data containing tracker events.
    player_id : int
        ID of the player to count units for.

    Returns
    -------
    int
        Number of units born for the specified player.
    """
    count = 0
    for event in sc2_replay.trackerEvents:
        if type(event).__name__ == "UnitBorn":
            if hasattr(event, "controlPlayerId") and event.controlPlayerId == player_id:
                count += 1
    return count


def _count_units_died_by_opponent(sc2_replay: SC2ReplayData, player_id: int) -> int:
    """
    Counts the UnitDied events where the opponent killed this player's units.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed SC2 replay data containing tracker events.
    player_id : int
        ID of the player to count units for.

    Returns
    -------
    int
        Number of units killed by the opponent for the specified player.
    """
    count = 0
    for event in sc2_replay.trackerEvents:
        if type(event).__name__ == "UnitDied":
            if hasattr(event, "killerPlayerId") and event.killerPlayerId == player_id:
                # This player KILLED an enemy unit (good for this player)
                count += 1
    return count


def _count_upgrades(sc2_replay: SC2ReplayData, player_id: int) -> int:
    """
    Counts the Upgrade events for a given player.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed SC2 replay data containing tracker events.
    player_id : int
        ID of the player to count upgrades for.

    Returns
    -------
    int
        Number of upgrades for the specified player.
    """
    count = 0
    for event in sc2_replay.trackerEvents:
        if type(event).__name__ == "Upgrade" and event.playerId == player_id:
            count += 1
    return count


def _get_player_info(
    sc2_replay: SC2ReplayData, player_id: int
) -> ToonPlayerDesc | None:
    """
    Get the ToonPlayerInfo for a specific player from the replay data.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed SC2 replay data containing player information.
    player_id : int
        ID of the player to retrieve info for.

    Returns
    -------
    ToonPlayerDesc | None
        ToonPlayerDesc object containing player information, or None if not found.
    """
    for toon_desc in sc2_replay.toonPlayerDescMap:
        if str(toon_desc.toon_player_info.playerID) == str(player_id):
            return toon_desc.toon_player_info
    return None


def _get_outcome(sc2_replay: SC2ReplayData) -> int | None:
    """
    Get the game outcome for player 1 (0=loss, 1=win), or None to skip if undecided/draw.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed SC2 replay data containing player information.

    Returns
    -------
    int | None
        Game outcome for player 1 (0=loss, 1=win), or None to skip if undecided/draw.
    """
    result_map = {"Loss": 0, "Win": 1, "Victory": 1, "Defeat": 0}
    skip_results = {"Undecided", "Draw", "Tie"}

    for toon_desc in sc2_replay.toonPlayerDescMap:
        info = toon_desc.toon_player_info
        if info.result in skip_results:
            return None

    # Get player 1's result
    for toon_desc in sc2_replay.toonPlayerDescMap:
        info = toon_desc.toon_player_info
        if str(info.playerID) == "1":
            return result_map.get(info.result)

    return None


def prepare_player_features(
    sc2_replay: SC2ReplayData,
    player_id: int,
) -> TensorDict | None:
    """
    Build a nested TensorDict of features for one player.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed SC2 replay data.
    player_id : int
        ID of the player (1 or 2).

    Returns
    -------
    TensorDict | None
        Nested TensorDict with keys: early, mid, late, final, delta, meta,
        units_born, units_killed, upgrade_count. Returns None to skip the replay.
    """
    player_stats_events = _get_player_stats_timeseries(
        sc2_replay=sc2_replay,
        player_id=player_id,
    )
    if not player_stats_events:
        return None

    early = _temporal_snapshot(player_stats_events, 0.0, 0.33)
    mid = _temporal_snapshot(player_stats_events, 0.33, 0.67)
    late = _temporal_snapshot(player_stats_events, 0.67, 1.0)
    final = _get_stats_values(player_stats_events[-1].stats)
    delta = late - early

    player_info = _get_player_info(sc2_replay=sc2_replay, player_id=player_id)
    if player_info is None:
        return None

    meta = TensorDict(
        {
            "APM": torch.tensor(float(player_info.APM)),
            "MMR": torch.tensor(float(player_info.MMR) if player_info.MMR else 0.0),
            "SQ": torch.tensor(float(player_info.SQ) if player_info.SQ else 0.0),
            "supply_capped_percent": torch.tensor(
                float(player_info.supplyCappedPercent)
                if player_info.supplyCappedPercent
                else 0.0
            ),
        },
    )

    return_tensordict = TensorDict(
        {
            "early": early,
            "mid": mid,
            "late": late,
            "final": final,
            "delta": delta,
            "meta": meta,
            "units_born": torch.tensor(
                float(_count_units_born(sc2_replay=sc2_replay, player_id=player_id)),
                dtype=torch.float32,
            ),
            "units_killed": torch.tensor(
                float(
                    _count_units_died_by_opponent(
                        sc2_replay=sc2_replay, player_id=player_id
                    ),
                ),
                dtype=torch.float32,
            ),
            "upgrade_count": torch.tensor(
                float(_count_upgrades(sc2_replay=sc2_replay, player_id=player_id)),
                dtype=torch.float32,
            ),
        },
    )

    return return_tensordict


def rich_transform(sc2_replay: SC2ReplayData) -> Tuple[TensorDict, int] | None:
    """Extract rich features from an SC2 replay.

    Returns:
        Tuple of (features TensorDict batch_size=[2], label) or None to skip.
        Index 0 = player 1, index 1 = player 2.
    """
    label = _get_outcome(sc2_replay=sc2_replay)
    if label is None:
        return None

    # Games need to be at least 180s * 22.4 game_speed = 4032 gameloops to be included
    try:
        game_duration = float(sc2_replay.header.elapsedGameLoops)
        if game_duration < 4032:
            return None
    except (AttributeError, ValueError, TypeError):
        return None

    player_tds = []
    for player_id in [1, 2]:
        player_td = prepare_player_features(
            sc2_replay=sc2_replay,
            player_id=player_id,
        )
        if player_td is None:
            return None
        player_tds.append(player_td)

    features = torch.stack(player_tds, dim=0)  # TensorDict, batch_size=[2]

    return features, label
