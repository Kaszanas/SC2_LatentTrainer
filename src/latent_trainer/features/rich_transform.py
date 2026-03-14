"""Rich feature transform for SC2 replays.

Extracts a comprehensive feature vector per player including:
- Temporal economy snapshots (early/mid/late game) — 39 features × 3 time windows
- Final economy state — 39 features
- Economy rate-of-change (late minus early) — 39 features
- Player meta stats: APM, MMR, SQ, supplyCappedPercent
- Unit activity: units born count, units lost count (killed by opponent)
- Upgrade count
- Game duration (shared)

Total: per player = 39*3 + 39 + 39 + 4 + 2 + 1 + 1 = 203 features
Output shape: [2, 203] per replay
"""

from typing import Optional, Tuple

import numpy as np
import torch
from sc2_datasets.replay_data.sc2_replay_data import SC2ReplayData

# Race encoding: map race name to float
RACE_MAP = {"Zerg": 0.0, "Protoss": 1.0, "Terran": 2.0}


def _get_stats_values(stats_obj) -> list:
    """Extract float values from a Stats object."""
    return [float(v) for v in stats_obj.__dict__.values()]


def _get_player_stats_timeseries(sc2_replay, player_id: int):
    """Collect all PlayerStats events for a given player, sorted by loop."""
    events = []
    for event in sc2_replay.trackerEvents:
        if type(event).__name__ == "PlayerStats" and event.playerId == player_id:
            events.append(event)
    events.sort(key=lambda e: e.loop)
    return events


def _temporal_snapshot(events, start_frac: float, end_frac: float) -> np.ndarray:
    """Average stats within a fractional time window of the game."""
    if not events:
        return np.zeros(39)

    n = len(events)
    start_idx = int(start_frac * n)
    end_idx = max(int(end_frac * n), start_idx + 1)

    window = events[start_idx:end_idx]
    if not window:
        return np.zeros(39)

    values = [_get_stats_values(e.stats) for e in window]
    return np.mean(values, axis=0)


def _count_units_born(sc2_replay, player_id: int) -> int:
    """Count UnitBorn events for a player."""
    count = 0
    for event in sc2_replay.trackerEvents:
        if type(event).__name__ == "UnitBorn":
            if hasattr(event, "controlPlayerId") and event.controlPlayerId == player_id:
                count += 1
    return count


def _count_units_died_by_opponent(sc2_replay, player_id: int) -> int:
    """Count UnitDied events where opponent killed this player's units."""
    count = 0
    for event in sc2_replay.trackerEvents:
        if type(event).__name__ == "UnitDied":
            if hasattr(event, "killerPlayerId") and event.killerPlayerId == player_id:
                # This player KILLED an enemy unit (good for this player)
                count += 1
    return count


def _count_upgrades(sc2_replay, player_id: int) -> int:
    """Count Upgrade events for a player."""
    count = 0
    for event in sc2_replay.trackerEvents:
        if type(event).__name__ == "Upgrade" and event.playerId == player_id:
            count += 1
    return count


def _get_player_info(sc2_replay, player_id: int):
    """Get ToonPlayerInfo for a specific player."""
    for toon_desc in sc2_replay.toonPlayerDescMap:
        if str(toon_desc.toon_player_info.playerID) == str(player_id):
            return toon_desc.toon_player_info
    return None


def _get_outcome(sc2_replay) -> Optional[int]:
    """Get game outcome. Returns label for player 1 (0=loss, 1=win), or None for skip."""
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


def rich_transform(sc2_replay: SC2ReplayData) -> Optional[Tuple[torch.Tensor, int]]:
    """Extract rich features from an SC2 replay.

    Returns:
        Tuple of (features_tensor [2, N_features], label) or None to skip.
    """
    # Get outcome
    label = _get_outcome(sc2_replay)
    if label is None:
        return None

    # Game duration in loops
    try:
        game_duration = float(sc2_replay.header.elapsedGameLoops)
    except (AttributeError, ValueError, TypeError):
        game_duration = 0.0

    player_features = []

    for player_id in [1, 2]:
        # --- 1. Temporal economy snapshots ---
        events = _get_player_stats_timeseries(sc2_replay, player_id)

        if not events:
            return None  # Skip replays without economy data

        early_stats = _temporal_snapshot(events, 0.0, 0.33)  # first third
        mid_stats = _temporal_snapshot(events, 0.33, 0.67)  # middle third
        late_stats = _temporal_snapshot(events, 0.67, 1.0)  # last third

        # --- 2. Final economy state ---
        final_stats = _get_stats_values(events[-1].stats)
        final_stats = np.array(final_stats, dtype=np.float32)

        # --- 3. Economy rate of change (late - early) ---
        econ_delta = late_stats - early_stats

        # --- 4. Player meta stats ---
        player_info = _get_player_info(sc2_replay, player_id)
        if player_info is None:
            return None

        meta_features = np.array(
            [
                float(player_info.APM),
                float(player_info.MMR) if player_info.MMR else 0.0,
                float(player_info.SQ) if player_info.SQ else 0.0,
                float(player_info.supplyCappedPercent)
                if player_info.supplyCappedPercent
                else 0.0,
            ],
            dtype=np.float32,
        )

        # --- 5. Unit activity ---
        units_born = float(_count_units_born(sc2_replay, player_id))
        units_killed = float(_count_units_died_by_opponent(sc2_replay, player_id))

        # --- 6. Upgrades ---
        upgrade_count = float(_count_upgrades(sc2_replay, player_id))

        # --- 7. Game duration (same for both, but included) ---
        duration = np.array([game_duration], dtype=np.float32)

        # Concatenate all features for this player
        player_feat = np.concatenate(
            [
                early_stats,  # 39
                mid_stats,  # 39
                late_stats,  # 39
                final_stats,  # 39
                econ_delta,  # 39
                meta_features,  # 4
                [units_born],  # 1
                [units_killed],  # 1
                [upgrade_count],  # 1
                duration,  # 1
            ]
        )

        player_features.append(player_feat)

    features = torch.tensor(np.stack(player_features), dtype=torch.float32)
    return features, label
