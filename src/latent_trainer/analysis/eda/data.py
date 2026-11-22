"""
Tidy pandas views of the events cache (plus optional metadata) for the EDA.

``replays`` and ``players`` cover all splits; ``ticks`` and ``units`` are built
from the training split only, because everything computed from them can inform
a modelling choice. Stat values are in raw units, as stored in the cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from latent_trainer.features.event_transform import TICK_INTERVAL_LOOPS
from latent_trainer.features.perception_action_cycle import (
    GAME_LOOPS_PER_SECOND,
    PAC_CHANNEL_NAMES,
)
from latent_trainer.features.type import EventDataBundle, EventSplit

SECONDS_PER_TICK = TICK_INTERVAL_LOOPS / GAME_LOOPS_PER_SECOND

# Raw stat channels kept per tick (name in the tables -> channel in the cache).
TICK_CHANNELS: dict[str, tuple[str, ...]] = {
    "workers": ("workersActiveCount",),
    "minerals_rate": ("mineralsCollectionRate",),
    "gas_rate": ("vespeneCollectionRate",),
    "supply": ("foodUsed",),
    "army_value": ("mineralsUsedCurrentArmy", "vespeneUsedCurrentArmy"),
    **{name: (name,) for name in PAC_CHANNEL_NAMES},
}

# Conventional matchup names, independent of player order.
_MATCHUPS: dict[frozenset, str] = {
    frozenset({"Terr"}): "TvT",
    frozenset({"Prot"}): "PvP",
    frozenset({"Zerg"}): "ZvZ",
    frozenset({"Terr", "Prot"}): "PvT",
    frozenset({"Terr", "Zerg"}): "TvZ",
    frozenset({"Prot", "Zerg"}): "PvZ",
}
MATCHUP_ORDER: tuple[str, ...] = ("TvT", "PvP", "ZvZ", "PvT", "TvZ", "PvZ")


def matchup_name(race_a: str, race_b: str) -> str:
    """
    Return the conventional matchup name of two races.

    Parameters
    ----------
    race_a : str
        One race (``Terr``, ``Prot`` or ``Zerg``).
    race_b : str
        The other race.

    Returns
    -------
    str
        E.g. ``"TvZ"``; the same for either order.
    """
    return _MATCHUPS[frozenset({race_a, race_b})]


@dataclass
class EDAData:
    """
    Every table the EDA outputs are computed from.

    Parameters
    ----------
    replays : pd.DataFrame
        One row per replay, all splits.
    players : pd.DataFrame
        One row per player stream, all splits.
    ticks : pd.DataFrame
        One row per stat tick of the training split, raw units.
    units : pd.DataFrame
        Training event counts per ``unit`` and ``race``.
    stat_values : torch.Tensor
        Raw training stat values ``[sum_t, C]``.
    stat_channel_names : list[str]
        Names of the ``C`` channels.
    has_metadata : bool
        Whether replay metadata (patch, players, true length) was joined.
    skip_report : dict | None
        The skipped-replay report, if one was found.
    """

    replays: pd.DataFrame
    players: pd.DataFrame
    ticks: pd.DataFrame
    units: pd.DataFrame
    stat_values: torch.Tensor
    stat_channel_names: list[str]
    has_metadata: bool
    skip_report: dict | None

    def train_only(self) -> EDAData:
        """
        Return a view restricted to the training split.

        Returns
        -------
        EDAData
            Same object with ``replays``/``players`` filtered to train and no
            skip report; the tick-level tables are train-only already.
        """
        return replace(
            self,
            replays=self.replays[self.replays["split"] == "train"],
            players=self.players[self.players["split"] == "train"],
            skip_report=None,
        )


def _stream_index(offsets: torch.Tensor) -> np.ndarray:
    """
    Map every row of a CSR array to its stream.

    Parameters
    ----------
    offsets : torch.Tensor
        CSR offsets ``[2N + 1]``.

    Returns
    -------
    np.ndarray
        Stream index per row.
    """
    lengths = (offsets[1:] - offsets[:-1]).numpy()
    return np.repeat(np.arange(len(lengths)), lengths)


def _replay_table(
    split_name: str, split: EventSplit, ids: list[str], race_names: list[str]
) -> pd.DataFrame:
    """
    Build the per-replay rows of one split.

    Parameters
    ----------
    split_name : str
        ``train``, ``val`` or ``test``.
    split : EventSplit
        The split.
    ids : list[str]
        Replay ids aligned with the split's labels.
    race_names : list[str]
        Race vocabulary.

    Returns
    -------
    pd.DataFrame
        ``split``, ``row``, ``replay_id``, ``label``, ``p1_race``, ``p2_race``,
        ``matchup``, ``horizon_s`` (last stat tick kept in the cache).
    """
    n = len(split.labels)
    races = split.race.long().numpy().reshape(n, 2)
    last_loop = split.stat_loops[split.stat_offsets[2::2] - 1].numpy()
    table = pd.DataFrame(
        {
            "split": split_name,
            "row": np.arange(n),
            "replay_id": ids if ids else [f"{split_name}/{i}" for i in range(n)],
            "label": split.labels.numpy(),
            "p1_race": [race_names[r] for r in races[:, 0]],
            "p2_race": [race_names[r] for r in races[:, 1]],
            "horizon_s": last_loop / GAME_LOOPS_PER_SECOND,
        }
    )
    table["matchup"] = [
        matchup_name(a, b) for a, b in zip(table["p1_race"], table["p2_race"])
    ]
    return table


def _player_table(replays: pd.DataFrame, split: EventSplit) -> pd.DataFrame:
    """
    Build the per-player rows of one split.

    Parameters
    ----------
    replays : pd.DataFrame
        The split's replay rows.
    split : EventSplit
        The split.

    Returns
    -------
    pd.DataFrame
        ``split``, ``replay_id``, ``player``, ``race``, ``opp_race``,
        ``matchup``, ``won``, ``n_ticks``, ``n_events``, ``events_per_min``.
    """
    n_ticks = (split.stat_offsets[1:] - split.stat_offsets[:-1]).numpy()
    n_events = (split.evt_offsets[1:] - split.evt_offsets[:-1]).numpy()
    rows = []
    for player in (1, 2):
        stream = 2 * replays["row"].to_numpy() + (player - 1)
        own = replays[f"p{player}_race"].to_numpy()
        other = replays[f"p{3 - player}_race"].to_numpy()
        won = replays["label"].to_numpy() == (1 if player == 1 else 0)
        frame = pd.DataFrame(
            {
                "split": replays["split"].to_numpy(),
                "replay_id": replays["replay_id"].to_numpy(),
                "player": player,
                "race": own,
                "opp_race": other,
                "matchup": replays["matchup"].to_numpy(),
                "won": won,
                "n_ticks": n_ticks[stream],
                "n_events": n_events[stream],
            }
        )
        rows.append(frame)
    players = pd.concat(rows, ignore_index=True)
    minutes = players["n_ticks"] * SECONDS_PER_TICK / 60.0
    players["events_per_min"] = players["n_events"] / minutes.where(minutes > 0)
    return players


def _tick_table(bundle: EventDataBundle, replays: pd.DataFrame) -> pd.DataFrame:
    """
    Build the per-tick rows of the training split in raw units.

    Parameters
    ----------
    bundle : EventDataBundle
        The loaded cache.
    replays : pd.DataFrame
        Training replay rows.

    Returns
    -------
    pd.DataFrame
        ``stream``, ``replay_id``, ``player``, ``race``, ``matchup``, ``won``,
        ``tick``, ``minute``, ``n_events``, the :data:`TICK_CHANNELS`, and the
        replay's ``horizon_s``.
    """
    split = bundle.train
    stream = _stream_index(split.stat_offsets)
    tick = split.stat_tick.long().numpy()
    names = bundle.stat_channel_names
    raw = split.stat_values.numpy()

    # Events per (stream, tick): events of tick k belong to the slice ending at k.
    evt_stream = _stream_index(split.evt_offsets)
    width = int(tick.max()) + 1 if len(tick) else 1
    evt_key = evt_stream * width + split.evt_tick.long().numpy()
    counts = np.bincount(evt_key, minlength=(int(stream.max()) + 1) * width)

    replay_row = replays.set_index("row").sort_index()
    replay_of_stream = stream // 2
    player_of_stream = stream % 2 + 1
    labels = replay_row["label"].to_numpy()
    table = pd.DataFrame(
        {
            "stream": stream,
            "replay_id": replay_row["replay_id"].to_numpy()[replay_of_stream],
            "player": player_of_stream,
            "tick": tick,
            "minute": split.stat_loops.numpy() / GAME_LOOPS_PER_SECOND / 60.0,
            "n_events": counts[stream * width + tick],
            "won": np.where(
                player_of_stream == 1,
                labels[replay_of_stream] == 1,
                labels[replay_of_stream] == 0,
            ),
            "horizon_s": replay_row["horizon_s"].to_numpy()[replay_of_stream],
        }
    )
    races = np.stack(
        [replay_row["p1_race"].to_numpy(), replay_row["p2_race"].to_numpy()], axis=1
    )
    table["race"] = races[replay_of_stream, player_of_stream - 1]
    table["matchup"] = replay_row["matchup"].to_numpy()[replay_of_stream]
    for column, channels in TICK_CHANNELS.items():
        table[column] = sum(raw[:, names.index(c)] for c in channels)
    return table


def _unit_table(bundle: EventDataBundle, race_names: list[str]) -> pd.DataFrame:
    """
    Count training events per unit and race of the acting player.

    Parameters
    ----------
    bundle : EventDataBundle
        The loaded cache.
    race_names : list[str]
        Race vocabulary.

    Returns
    -------
    pd.DataFrame
        ``unit``, ``race``, ``count``.
    """
    split = bundle.train
    stream = _stream_index(split.evt_offsets)
    race = split.race.long().numpy()[stream]
    unit = split.evt_unit.long().numpy()
    key = unit * len(race_names) + race
    counts = np.bincount(key, minlength=len(bundle.unit_vocab) * len(race_names))
    counts = counts.reshape(len(bundle.unit_vocab), len(race_names))
    rows = [
        {
            "unit": bundle.unit_vocab[u],
            "race": race_names[r],
            "count": int(counts[u, r]),
        }
        for u in range(len(bundle.unit_vocab))
        for r in range(len(race_names))
        if counts[u, r] > 0
    ]
    return pd.DataFrame(rows, columns=["unit", "race", "count"])


def build_eda_data(
    bundle: EventDataBundle,
    metadata: pd.DataFrame | None = None,
    skip_report_path: Path | None = None,
) -> EDAData:
    """
    Assemble all EDA tables from a loaded events cache.

    Parameters
    ----------
    bundle : EventDataBundle
        Output of ``load_event_cache``.
    metadata : pd.DataFrame | None
        Output of ``load_metadata``, joined on ``replay_id``, by default None.
    skip_report_path : Path | None
        ``scripts/report_skipped_replays.py`` output, by default None.

    Returns
    -------
    EDAData
        The tables.

    Raises
    ------
    ValueError
        If metadata is given but lacks a replay of the cache.
    """
    race_names = bundle.race_names
    replay_tables = []
    player_tables = []
    for split_name in ("train", "val", "test"):
        split = getattr(bundle, split_name)
        replays = _replay_table(
            split_name, split, bundle.split_ids.get(split_name, []), race_names
        )
        replay_tables.append(replays)
        player_tables.append(_player_table(replays, split))
    replays = pd.concat(replay_tables, ignore_index=True)
    players = pd.concat(player_tables, ignore_index=True)

    if metadata is not None:
        joined = replays.merge(
            metadata.drop(columns=["split"]),
            on="replay_id",
            how="left",
            indicator=True,
            suffixes=("", "_meta"),
        )
        missing = int((joined["_merge"] != "both").sum())
        if missing:
            raise ValueError(f"metadata lacks {missing} replays of the cache")
        # The cache's race is authoritative; the metadata copy must agree.
        for player in (1, 2):
            column = f"p{player}_race"
            if f"{column}_meta" in joined:
                differ = int((joined[column] != joined[f"{column}_meta"]).sum())
                if differ:
                    raise ValueError(f"{differ} replays disagree on {column}")
        meta_copies = [c for c in joined.columns if c.endswith("_meta")]
        replays = joined.drop(columns=["_merge", *meta_copies])
        replays["duration_s"] = replays["elapsed_loops"] / GAME_LOOPS_PER_SECOND
        replays["patch"] = replays["game_version"].str.split(".").str[:2].str.join(".")
        pack = replays["replay_id"].str.split("/").str[0]
        replays["tournament"] = pack
        replays["year"] = pd.to_numeric(pack.str[:4], errors="coerce")
        for player in (1, 2):
            mask = players["player"] == player
            lookup = replays.set_index("replay_id")
            players.loc[mask, "toon"] = lookup.loc[
                players.loc[mask, "replay_id"], f"p{player}_toon"
            ].to_numpy()
            players.loc[mask, "apm"] = lookup.loc[
                players.loc[mask, "replay_id"], f"p{player}_apm"
            ].to_numpy()
    else:
        replays["duration_s"] = replays["horizon_s"]

    train_replays = replays[replays["split"] == "train"]
    skip_report = None
    if skip_report_path is not None and skip_report_path.exists():
        skip_report = json.loads(skip_report_path.read_text(encoding="utf-8"))

    return EDAData(
        replays=replays,
        players=players,
        ticks=_tick_table(bundle, train_replays),
        units=_unit_table(bundle, race_names),
        stat_values=bundle.train.stat_values,
        stat_channel_names=list(bundle.stat_channel_names),
        has_metadata=metadata is not None,
        skip_report=skip_report,
    )
