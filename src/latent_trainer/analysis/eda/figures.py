"""
The EDA outputs: each one a ``compute`` (data -> table) and optional ``plot``.

``plot`` functions only read the table their ``compute`` returned, so a saved
table is enough to redraw a figure. Outputs with ``scope="train"`` receive the
training split only (see :meth:`EDAData.train_only`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from latent_trainer.analysis.eda import style
from latent_trainer.analysis.eda.data import MATCHUP_ORDER, SECONDS_PER_TICK, EDAData
from latent_trainer.features.event_transform import SKIP_REASONS

RACES = ("Terr", "Prot", "Zerg")
OUTCOMES = ("winner", "loser")
CUTOFFS_MIN = (5, 10, 15)
K_CANDIDATES = (32, 64, 128)
MIN_STREAMS = 100  # fewest streams behind a plotted time bin
MAX_MINUTE = 25.0
# De-normalising (z * std + mean) turns exact zeros into float noise.
ZERO_TOLERANCE = 1e-3
# Whole-game PAC means from the PAC module's validation (299 replays, commit dd49983).
PAC_REFERENCE = {
    "winner": {"ppm": 46.3, "pal_s": 0.174, "gap_s": 0.185},
    "loser": {"ppm": 44.6, "pal_s": 0.182, "gap_s": 0.193},
}
TRAJECTORY_CHANNELS = {
    "workers": "Workers",
    "minerals_rate": "Mineral income (per min)",
    "gas_rate": "Gas income (per min)",
    "supply": "Supply used",
    "army_value": "Army value (resources)",
}
PAC_PANELS = {
    "pac_cycles": "PAC cycles per min",
    "pac_actions": "Actions per min",
    "pac_latency_s": "Action latency (s)",
    "pac_gap_s": "Gap to next look (s)",
}


@dataclass(frozen=True)
class Output:
    """
    One EDA output.

    Parameters
    ----------
    name : str
        File stem in the output directory.
    scope : str
        ``"all"`` (descriptive, every split) or ``"train"``.
    compute : Callable[[EDAData], pd.DataFrame]
        Builds the table.
    plot : Callable[[pd.DataFrame], plt.Figure] | None
        Draws the figure from the table; None for table-only outputs.
    caption : str
        Caption for the report and the LaTeX table.
    latex : bool
        Also write the table as LaTeX.
    needs_metadata : bool
        Requires replay metadata.
    needs_skip_report : bool
        Requires the skipped-replay report.
    """

    name: str
    scope: str
    compute: Callable[[EDAData], pd.DataFrame]
    plot: Callable[[pd.DataFrame], plt.Figure] | None
    caption: str
    latex: bool = False
    needs_metadata: bool = False
    needs_skip_report: bool = False


# Helpers:
def wilson_interval(successes: np.ndarray, n: np.ndarray, z: float = 1.96):
    """
    Return the Wilson score interval of a binomial proportion.

    Parameters
    ----------
    successes : np.ndarray
        Successes per group.
    n : np.ndarray
        Trials per group.
    z : float
        Normal quantile, by default 1.96 (95%).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower and upper bounds.
    """
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def _quantile_band(
    frame: pd.DataFrame, value: str, by: list[str], x: str = "minute_bin"
) -> pd.DataFrame:
    """
    Summarise a value per group and time bin as median and interquartile range.

    Each stream is first averaged within the bin, so every game counts once
    per bin and small integer counts do not produce staircase quantiles.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows with ``value``, ``by`` columns, ``x`` and ``stream``.
    value : str
        Column to summarise.
    by : list[str]
        Grouping columns besides ``x``.
    x : str
        Time-bin column, by default ``"minute_bin"``.

    Returns
    -------
    pd.DataFrame
        ``by``, ``x``, ``median``, ``q25``, ``q75``, ``n_streams``; bins with
        fewer than :data:`MIN_STREAMS` streams are dropped.
    """
    per_stream = frame.groupby([*by, x, "stream"])[value].mean().reset_index()
    grouped = per_stream.groupby([*by, x])
    table = grouped[value].quantile([0.25, 0.5, 0.75]).unstack()
    table.columns = ["q25", "median", "q75"]
    table["n_streams"] = grouped.size()
    table = table.reset_index()
    return table[table["n_streams"] >= MIN_STREAMS]


def _binned_ticks(data: EDAData) -> pd.DataFrame:
    """
    Return training ticks with a half-minute bin and an outcome label.

    Parameters
    ----------
    data : EDAData
        EDA tables.

    Returns
    -------
    pd.DataFrame
        Ticks up to :data:`MAX_MINUTE` with ``minute_bin`` and ``outcome``.
    """
    ticks = data.ticks[data.ticks["minute"] <= MAX_MINUTE].copy()
    ticks["minute_bin"] = np.floor(ticks["minute"] * 2) / 2
    ticks["outcome"] = np.where(ticks["won"], "winner", "loser")
    return ticks


def _band_axes(
    ax: plt.Axes,
    table: pd.DataFrame,
    group: str,
    colors: dict[str, str],
    labels: dict[str, str] | None = None,
    linestyles: dict[str, str] | None = None,
) -> None:
    """
    Draw one median line with an IQR band per group.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    table : pd.DataFrame
        ``group``, ``minute_bin``, ``median``, ``q25``, ``q75``.
    group : str
        Column naming the series.
    colors : dict[str, str]
        Colour per series.
    labels : dict[str, str] | None
        Legend label per series, by default the series value.
    linestyles : dict[str, str] | None
        Line style per series, by default solid.
    """
    for key, color in colors.items():
        part = table[table[group] == key].sort_values("minute_bin")
        if part.empty:
            continue
        ax.fill_between(
            part["minute_bin"],
            part["q25"],
            part["q75"],
            color=color,
            alpha=style.BAND_ALPHA,
            linewidth=0,
        )
        ax.plot(
            part["minute_bin"],
            part["median"],
            color=color,
            linestyle=(linestyles or {}).get(key, "-"),
            label=(labels or {}).get(key, key),
        )


# 1. Composition:
def compute_composition(data: EDAData) -> pd.DataFrame:
    """
    Count replays, races and matchups per split.

    Parameters
    ----------
    data : EDAData
        EDA tables.

    Returns
    -------
    pd.DataFrame
        One row per split plus ``All``.
    """
    rows = []
    for split in ("train", "val", "test", "All"):
        replays = (
            data.replays
            if split == "All"
            else data.replays[data.replays["split"] == split]
        )
        players = (
            data.players
            if split == "All"
            else data.players[data.players["split"] == split]
        )
        row = {"Split": split, "Replays": len(replays)}
        for race in RACES:
            row[style.RACE_LABELS[race]] = int((players["race"] == race).sum())
        for matchup in MATCHUP_ORDER:
            row[matchup] = int((replays["matchup"] == matchup).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def compute_patches(data: EDAData) -> pd.DataFrame:
    """
    Count replays per game patch (major.minor).

    Parameters
    ----------
    data : EDAData
        EDA tables.

    Returns
    -------
    pd.DataFrame
        ``patch``, ``replays``, ``share``, in version order.
    """
    counts = data.replays["patch"].value_counts()
    table = counts.rename_axis("patch").reset_index(name="replays")
    table["key"] = table["patch"].map(lambda v: tuple(int(p) for p in v.split(".")))
    table = table.sort_values("key").drop(columns="key")
    table["share"] = table["replays"] / table["replays"].sum()
    return table.reset_index(drop=True)


def plot_patches(table: pd.DataFrame) -> plt.Figure:
    """
    Plot replays per patch as horizontal bars.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compute_patches`.

    Returns
    -------
    plt.Figure
        The figure.
    """
    fig, ax = plt.subplots(figsize=style.figure_size("single", 0.2 + 0.07 * len(table)))
    ax.barh(table["patch"], table["replays"], color=style.NEUTRAL_COLOR, height=0.7)
    ax.invert_yaxis()
    ax.set_xlabel("Replays")
    ax.set_ylabel("Game patch")
    ax.grid(axis="y", visible=False)
    return fig


def compute_years(data: EDAData) -> pd.DataFrame:
    """
    Count replays and tournaments per year.

    Parameters
    ----------
    data : EDAData
        EDA tables.

    Returns
    -------
    pd.DataFrame
        ``year``, ``tournaments``, ``replays``.
    """
    grouped = data.replays.dropna(subset=["year"]).groupby("year")
    table = pd.DataFrame(
        {"tournaments": grouped["tournament"].nunique(), "replays": grouped.size()}
    ).reset_index()
    table["year"] = table["year"].astype(int)
    return table


def plot_years(table: pd.DataFrame) -> plt.Figure:
    """
    Plot replays per year.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compute_years`.

    Returns
    -------
    plt.Figure
        The figure.
    """
    fig, ax = plt.subplots(figsize=style.figure_size("single", 0.55))
    ax.bar(
        table["year"].astype(str),
        table["replays"],
        color=style.NEUTRAL_COLOR,
        width=0.7,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Replays")
    ax.grid(axis="x", visible=False)
    return fig


def compute_skip_reasons(data: EDAData) -> pd.DataFrame:
    """
    Tabulate why source replays were skipped.

    Parameters
    ----------
    data : EDAData
        EDA tables with a skip report.

    Returns
    -------
    pd.DataFrame
        ``Reason``, ``Description``, ``Replays``.
    """
    counts = data.skip_report["counts"]
    return pd.DataFrame(
        [
            {"Reason": code, "Description": SKIP_REASONS.get(code, ""), "Replays": n}
            for code, n in counts.items()
        ]
    )


# 2. Label balance:
def compute_label_balance(data: EDAData) -> pd.DataFrame:
    """
    Compute the player-1 win rate per matchup and race win rates.

    Parameters
    ----------
    data : EDAData
        EDA tables.

    Returns
    -------
    pd.DataFrame
        ``Measure``, ``Group``, ``n``, ``Rate``, ``CI low``, ``CI high``.
    """
    rows = []
    replays = data.replays
    for group in ("All", *MATCHUP_ORDER):
        part = replays if group == "All" else replays[replays["matchup"] == group]
        rows.append(("Player 1 wins", group, len(part), int(part["label"].sum())))
    players = data.players
    for matchup in ("PvT", "TvZ", "PvZ"):
        first = {"PvT": "Prot", "TvZ": "Terr", "PvZ": "Prot"}[matchup]
        part = players[(players["matchup"] == matchup) & (players["race"] == first)]
        label = f"{style.RACE_LABELS[first]} wins {matchup}"
        rows.append(("Race wins", label, len(part), int(part["won"].sum())))
    table = pd.DataFrame(rows, columns=["Measure", "Group", "n", "wins"])
    low, high = wilson_interval(table["wins"], table["n"])
    table["Rate"] = table["wins"] / table["n"]
    table["CI low"], table["CI high"] = low, high
    return table.drop(columns="wins")


def plot_label_balance(table: pd.DataFrame) -> plt.Figure:
    """
    Plot win rates with 95% Wilson intervals against parity.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compute_label_balance`.

    Returns
    -------
    plt.Figure
        The figure.
    """
    fig, ax = plt.subplots(figsize=style.figure_size("single", 0.75))
    y = np.arange(len(table))[::-1]
    ax.errorbar(
        table["Rate"],
        y,
        xerr=[table["Rate"] - table["CI low"], table["CI high"] - table["Rate"]],
        fmt="o",
        color=style.NEUTRAL_COLOR,
        ecolor=style.NEUTRAL_COLOR,
        elinewidth=1,
        capsize=0,
    )
    ax.axvline(0.5, color=style.INK_SECONDARY, linewidth=0.8, linestyle=":")
    ax.set_yticks(y, table["Group"])
    ax.set_xlabel("Win rate (95% Wilson CI)")
    ax.grid(axis="y", visible=False)
    return fig


# 3. Game length:
def compute_game_length(data: EDAData) -> pd.DataFrame:
    """
    Return every replay's length in minutes with its matchup and split.

    Parameters
    ----------
    data : EDAData
        EDA tables.

    Returns
    -------
    pd.DataFrame
        ``split``, ``matchup``, ``minutes``.
    """
    return pd.DataFrame(
        {
            "split": data.replays["split"],
            "matchup": data.replays["matchup"],
            "minutes": data.replays["duration_s"] / 60.0,
        }
    )


def plot_game_length(table: pd.DataFrame) -> plt.Figure:
    """
    Plot length per matchup (left) and the share of games still running (right).

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compute_game_length`.

    Returns
    -------
    plt.Figure
        The figure.
    """
    fig, (left, right) = plt.subplots(
        1, 2, figsize=style.figure_size("double", 0.32), gridspec_kw={"wspace": 0.35}
    )
    groups = [table.loc[table["matchup"] == m, "minutes"] for m in MATCHUP_ORDER]
    left.boxplot(
        groups,
        vert=False,
        tick_labels=MATCHUP_ORDER,
        widths=0.6,
        showfliers=False,
        medianprops={"color": style.INK},
        boxprops={"color": style.NEUTRAL_COLOR},
        whiskerprops={"color": style.NEUTRAL_COLOR},
        capprops={"color": style.NEUTRAL_COLOR},
    )
    left.invert_yaxis()
    left.set_xlabel("Game length (min)")
    left.grid(axis="y", visible=False)

    grid = np.linspace(0, min(40.0, table["minutes"].max()), 321)
    minutes = np.sort(table["minutes"].to_numpy())
    running = 1.0 - np.searchsorted(minutes, grid, side="left") / len(minutes)
    right.plot(grid, running, color=style.NEUTRAL_COLOR)
    for cutoff in CUTOFFS_MIN:
        share = float((table["minutes"] >= cutoff).mean())
        right.axvline(cutoff, color=style.INK_SECONDARY, linewidth=0.6, linestyle=":")
        right.annotate(
            f"{share:.0%}",
            (cutoff, share),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=plt.rcParams["xtick.labelsize"],
            color=style.INK_SECONDARY,
        )
    right.set_xlabel("Game time (min)")
    right.set_ylabel("Games still running")
    right.set_ylim(0, 1.02)
    return fig


def compute_survival(data: EDAData) -> pd.DataFrame:
    """
    Count games still running at each evaluation cutoff, per split.

    Parameters
    ----------
    data : EDAData
        EDA tables.

    Returns
    -------
    pd.DataFrame
        ``Cutoff (min)`` and, per split, games running and their share.
    """
    rows = []
    for cutoff in (3, *CUTOFFS_MIN, 20, 25):
        row = {"Cutoff (min)": cutoff}
        for split in ("train", "val", "test"):
            minutes = (
                data.replays.loc[data.replays["split"] == split, "duration_s"] / 60
            )
            row[f"{split} n"] = int((minutes >= cutoff).sum())
            row[f"{split} share"] = float((minutes >= cutoff).mean())
        rows.append(row)
    return pd.DataFrame(rows)


# 4. Event density and the per-slice cap:
def compute_event_rate(data: EDAData) -> pd.DataFrame:
    """
    Summarise tracker events per minute over game time, by race.

    Parameters
    ----------
    data : EDAData
        EDA tables (train only).

    Returns
    -------
    pd.DataFrame
        ``race``, ``minute_bin``, ``median``, ``q25``, ``q75``, ``n_streams``.
    """
    ticks = _binned_ticks(data)
    ticks["events_per_min"] = ticks["n_events"] * 60.0 / SECONDS_PER_TICK
    return _quantile_band(ticks, "events_per_min", ["race"])


def plot_event_rate(table: pd.DataFrame) -> plt.Figure:
    """
    Plot events per minute over game time by race.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compute_event_rate`.

    Returns
    -------
    plt.Figure
        The figure.
    """
    fig, ax = plt.subplots(figsize=style.figure_size("single", 0.7))
    _band_axes(ax, table, "race", style.RACE_COLORS, style.RACE_LABELS)
    ax.set_xlabel("Game time (min)")
    ax.set_ylabel("Events per min")
    ax.legend(loc="upper left")
    return fig


def compute_slice_events(data: EDAData) -> pd.DataFrame:
    """
    Count slices (one tick each) by their number of events.

    Parameters
    ----------
    data : EDAData
        EDA tables (train only).

    Returns
    -------
    pd.DataFrame
        ``events``, ``slices``.
    """
    counts = data.ticks["n_events"].value_counts().sort_index()
    return counts.rename_axis("events").reset_index(name="slices")


def plot_slice_events(table: pd.DataFrame) -> plt.Figure:
    """
    Plot the share of slices above each event count, with the candidate caps.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compute_slice_events`.

    Returns
    -------
    plt.Figure
        The figure.
    """
    fig, ax = plt.subplots(figsize=style.figure_size("single", 0.65))
    events = table["events"].to_numpy()
    slices = table["slices"].to_numpy()
    above = 1.0 - np.cumsum(slices) / slices.sum()
    ax.step(events, above, where="post", color=style.NEUTRAL_COLOR)
    ax.set_yscale("log")
    for k in K_CANDIDATES:
        share = float(slices[events > k].sum() / slices.sum())
        ax.axvline(k, color=style.INK_SECONDARY, linewidth=0.6, linestyle=":")
        ax.annotate(
            f"K={k}: {share * 100:.2g}%",
            (k, max(share, 1e-6)),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=plt.rcParams["xtick.labelsize"],
            color=style.INK_SECONDARY,
        )
    ax.set_xlim(0, max(K_CANDIDATES) * 1.4)
    ax.set_xlabel("Events in a slice (7.14 s)")
    ax.set_ylabel("Share of slices with more")
    return fig


def compute_k_cap(data: EDAData) -> pd.DataFrame:
    """
    Quantify what each candidate per-slice cap K would drop.

    Parameters
    ----------
    data : EDAData
        EDA tables (train only).

    Returns
    -------
    pd.DataFrame
        ``K``, share of slices above K, share of events beyond K.
    """
    n = data.ticks["n_events"].to_numpy()
    rows = []
    for k in K_CANDIDATES:
        rows.append(
            {
                "K": k,
                "Slices above K": float((n > k).mean()),
                "Events dropped": float(np.clip(n - k, 0, None).sum() / n.sum()),
            }
        )
    return pd.DataFrame(rows)


# 5. Vocabulary:
def compute_vocab_rank(data: EDAData) -> pd.DataFrame:
    """
    Rank units by their training event count.

    Parameters
    ----------
    data : EDAData
        EDA tables (train only).

    Returns
    -------
    pd.DataFrame
        ``rank``, ``unit``, ``count``.
    """
    totals = data.units.groupby("unit")["count"].sum().sort_values(ascending=False)
    table = totals.reset_index()
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def plot_vocab_rank(table: pd.DataFrame) -> plt.Figure:
    """
    Plot the rank-frequency curve on log-log axes.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compute_vocab_rank`.

    Returns
    -------
    plt.Figure
        The figure.
    """
    fig, ax = plt.subplots(figsize=style.figure_size("single", 0.65))
    ax.loglog(
        table["rank"],
        table["count"],
        color=style.NEUTRAL_COLOR,
        marker="o",
        markersize=2,
        linewidth=0.8,
    )
    ax.set_xlabel("Unit rank")
    ax.set_ylabel("Training events")
    return fig


def compute_vocab_top(data: EDAData, top: int = 10) -> pd.DataFrame:
    """
    Return each race's most frequent units.

    Parameters
    ----------
    data : EDAData
        EDA tables (train only).
    top : int
        Units per race, by default 10.

    Returns
    -------
    pd.DataFrame
        ``race``, ``unit``, ``count``, ``share`` (of the race's events).
    """
    parts = []
    for race in RACES:
        part = data.units[data.units["race"] == race].sort_values(
            "count", ascending=False
        )
        part = part.assign(share=part["count"] / part["count"].sum()).head(top)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def plot_vocab_top(table: pd.DataFrame) -> plt.Figure:
    """
    Plot the top units of each race in one panel per race.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compute_vocab_top`.

    Returns
    -------
    plt.Figure
        The figure.
    """
    fig, axes = plt.subplots(
        1, 3, figsize=style.figure_size("double", 0.36), gridspec_kw={"wspace": 0.9}
    )
    for ax, race in zip(axes, RACES, strict=True):
        part = table[table["race"] == race]
        ax.barh(part["unit"], part["share"], color=style.RACE_COLORS[race], height=0.7)
        ax.invert_yaxis()
        ax.set_title(style.RACE_LABELS[race])
        ax.set_xlabel("Share of events")
        ax.grid(axis="y", visible=False)
    return fig


# 6. Stat trajectories, winner vs loser:
def compute_trajectories(data: EDAData) -> pd.DataFrame:
    """
    Summarise key stats over game time for winners and losers.

    Parameters
    ----------
    data : EDAData
        EDA tables (train only).

    Returns
    -------
    pd.DataFrame
        ``channel``, ``outcome``, ``minute_bin``, ``median``, ``q25``, ``q75``,
        ``n_streams``.
    """
    ticks = _binned_ticks(data)
    parts = [
        _quantile_band(ticks, channel, ["outcome"]).assign(channel=channel)
        for channel in TRAJECTORY_CHANNELS
    ]
    return pd.concat(parts, ignore_index=True)


def _outcome_panels(
    table: pd.DataFrame, panels: dict[str, str], n_cols: int, aspect: float
) -> plt.Figure:
    """
    Draw one winner/loser panel per channel.

    Parameters
    ----------
    table : pd.DataFrame
        ``channel``, ``outcome`` and band columns.
    panels : dict[str, str]
        Channel -> y-axis label.
    n_cols : int
        Panels per row.
    aspect : float
        Figure height as a fraction of the double-column width.

    Returns
    -------
    plt.Figure
        The figure.
    """
    n_rows = -(-len(panels) // n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=style.figure_size("double", aspect),
        squeeze=False,
        gridspec_kw={"wspace": 0.4, "hspace": 0.45},
    )
    flat = axes.ravel()
    for ax, (channel, label) in zip(flat, panels.items()):
        _band_axes(
            ax,
            table[table["channel"] == channel],
            "outcome",
            style.OUTCOME_COLORS,
            {"winner": "Winner", "loser": "Loser"},
            style.OUTCOME_LINESTYLES,
        )
        ax.set_ylabel(label)
        ax.set_xlabel("Game time (min)")
    for ax in flat[len(panels) :]:
        ax.set_visible(False)
    flat[0].legend(loc="upper left")
    return fig


def plot_trajectories(table: pd.DataFrame) -> plt.Figure:
    """
    Plot key stats over game time, winner vs loser.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compute_trajectories`.

    Returns
    -------
    plt.Figure
        The figure.
    """
    return _outcome_panels(table, TRAJECTORY_CHANNELS, n_cols=3, aspect=0.55)


def compute_army_by_matchup(data: EDAData) -> pd.DataFrame:
    """
    Summarise army value over game time per matchup, winner vs loser.

    Parameters
    ----------
    data : EDAData
        EDA tables (train only).

    Returns
    -------
    pd.DataFrame
        ``matchup``, ``outcome``, ``minute_bin`` and band columns.
    """
    return _quantile_band(_binned_ticks(data), "army_value", ["matchup", "outcome"])


def plot_army_by_matchup(table: pd.DataFrame) -> plt.Figure:
    """
    Plot army value over game time per matchup, winner vs loser.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compute_army_by_matchup`.

    Returns
    -------
    plt.Figure
        The figure.
    """
    renamed = table.rename(columns={"matchup": "channel"})
    return _outcome_panels(
        renamed, {m: f"{m}: army value" for m in MATCHUP_ORDER}, n_cols=3, aspect=0.55
    )


# 7. PAC:
def compute_pac_trajectories(data: EDAData) -> pd.DataFrame:
    """
    Summarise the per-tick PAC channels over game time, winner vs loser.

    Counts are converted to per-minute rates. Latency uses only ticks where a
    latency settled; the gap uses ticks where it is positive, because a tick
    without a settled gap also reads 0.

    Parameters
    ----------
    data : EDAData
        EDA tables (train only).

    Returns
    -------
    pd.DataFrame
        ``channel``, ``outcome``, ``minute_bin`` and band columns.
    """
    ticks = _binned_ticks(data)
    per_min = 60.0 / SECONDS_PER_TICK
    ticks["pac_cycles"] = ticks["pac_cycles"] * per_min
    ticks["pac_actions"] = ticks["pac_actions"] * per_min
    parts = []
    for channel in PAC_PANELS:
        part = ticks
        if channel == "pac_latency_s":
            part = ticks[ticks["pac_has_latency"] > 0.5]
        elif channel == "pac_gap_s":
            part = ticks[ticks["pac_gap_s"] > 0]
        parts.append(_quantile_band(part, channel, ["outcome"]).assign(channel=channel))
    return pd.concat(parts, ignore_index=True)


def plot_pac_trajectories(table: pd.DataFrame) -> plt.Figure:
    """
    Plot PAC over game time, winner vs loser.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compute_pac_trajectories`.

    Returns
    -------
    plt.Figure
        The figure.
    """
    return _outcome_panels(table, PAC_PANELS, n_cols=2, aspect=0.62)


def compute_pac_whole_game(data: EDAData) -> pd.DataFrame:
    """
    Rebuild whole-game PAC statistics from the per-tick channels.

    PPM and APP come from summed counts; PAL and GAP are averaged over ticks
    where they are defined (see :func:`compute_pac_trajectories`), so they are
    tick-weighted approximations of the per-cycle means.

    Parameters
    ----------
    data : EDAData
        EDA tables (train only).

    Returns
    -------
    pd.DataFrame
        One row per statistic (PPM, APP, PAL, GAP); per outcome the mean, the
        median and the reference mean (none for APP).
    """
    ticks = data.ticks
    grouped = ticks.groupby("stream")
    minutes = grouped["tick"].count() * SECONDS_PER_TICK / 60.0
    cycles = grouped["pac_cycles"].sum()
    per_stream = pd.DataFrame(
        {
            "won": grouped["won"].first(),
            "ppm": cycles / minutes,
            "app": grouped["pac_actions"].sum() / cycles.where(cycles > 0),
            "pal_s": ticks[ticks["pac_has_latency"] > 0.5]
            .groupby("stream")["pac_latency_s"]
            .mean(),
            "gap_s": ticks[ticks["pac_gap_s"] > 0]
            .groupby("stream")["pac_gap_s"]
            .mean(),
        }
    )
    labels = {"ppm": "PPM", "app": "APP", "pal_s": "PAL (s)", "gap_s": "GAP (s)"}
    rows = []
    for stat, label in labels.items():
        row = {"Statistic": label}
        for outcome in OUTCOMES:
            part = per_stream.loc[per_stream["won"] == (outcome == "winner"), stat]
            name = outcome.capitalize()
            row[f"{name} mean"] = float(part.mean())
            row[f"{name} median"] = float(part.median())
            row[f"{name} ref."] = PAC_REFERENCE[outcome].get(stat, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# 8. Normalisation:
def compute_normalisation(data: EDAData) -> pd.DataFrame:
    """
    Describe every stat channel's distribution before and after z-scoring.

    Parameters
    ----------
    data : EDAData
        EDA tables (train only).

    Returns
    -------
    pd.DataFrame
        Per channel: mean, std, whether the std floor of 1 applies, skewness,
        excess kurtosis, share of zeros, 99th percentile and maximum of |z|.
    """
    z = data.stat_values.numpy().astype(np.float64)
    mean = data.mean.numpy()[0]
    std = data.std.numpy()[0]
    raw = z * std + mean
    abs_z = np.abs(z)
    raw_std = raw.std(axis=0)
    return pd.DataFrame(
        {
            "Channel": data.stat_channel_names,
            "Mean": mean,
            "Std": raw_std,
            "Floored": np.where(raw_std < 1.0, "yes", ""),
            "Skew": stats.skew(raw, axis=0),
            "Kurtosis": stats.kurtosis(raw, axis=0),
            "Zero share": (np.abs(raw) < ZERO_TOLERANCE).mean(axis=0),
            "p99 |z|": np.percentile(abs_z, 99, axis=0),
            "Max |z|": abs_z.max(axis=0),
        }
    )


TRANSFORMS = ("z-score", "log1p, z-score", "clip p99.9, z-score")


def _transformed(raw: np.ndarray, transform: str) -> np.ndarray:
    """
    Apply a candidate transform and z-score the result.

    Parameters
    ----------
    raw : np.ndarray
        Raw channel values.
    transform : str
        One of :data:`TRANSFORMS`.

    Returns
    -------
    np.ndarray
        Standardised values (std floored at 1e-12).
    """
    if transform == "log1p, z-score":
        raw = np.log1p(np.clip(raw, 0, None))
    elif transform == "clip p99.9, z-score":
        raw = np.clip(raw, None, np.percentile(raw, 99.9))
    return (raw - raw.mean()) / max(raw.std(), 1e-12)


def _heavy_tail_channels(data: EDAData, n_other: int = 3) -> list[str]:
    """
    Return the PAC channels plus the most skewed informative other channels.

    Channels that are zero in more than half the ticks (e.g. friendly fire)
    are skipped: their extremes are rare events, not a tail to transform.

    Parameters
    ----------
    data : EDAData
        EDA tables (train only).
    n_other : int
        Non-PAC channels to add, by largest skewness, by default 3.

    Returns
    -------
    list[str]
        Channel names.
    """
    names = data.stat_channel_names
    z = data.stat_values.numpy()
    raw = z * data.std.numpy()[0] + data.mean.numpy()[0]
    mostly_zero = (np.abs(raw) < ZERO_TOLERANCE).mean(axis=0) > 0.5
    skew = np.nan_to_num(stats.skew(raw, axis=0), nan=0.0)
    others = [
        i
        for i in np.argsort(-skew)
        if not names[i].startswith("pac_") and not mostly_zero[i]
    ]
    pac = [n for n in names if n.startswith("pac_") and n != "pac_has_latency"]
    return pac + [names[i] for i in others[:n_other]]


def compute_heavy_tails(data: EDAData) -> pd.DataFrame:
    """
    Compare candidate transforms on the heavy-tailed channels.

    Parameters
    ----------
    data : EDAData
        EDA tables (train only).

    Returns
    -------
    pd.DataFrame
        ``Channel``, ``Transform``, skewness, 99th percentile and maximum of the
        standardised values.
    """
    names = data.stat_channel_names
    z = data.stat_values.numpy().astype(np.float64)
    raw = z * data.std.numpy()[0] + data.mean.numpy()[0]
    rows = []
    for channel in _heavy_tail_channels(data):
        column = raw[:, names.index(channel)]
        for transform in TRANSFORMS:
            values = _transformed(column, transform)
            rows.append(
                {
                    "Channel": channel,
                    "Transform": transform,
                    "Skew": float(stats.skew(values)),
                    "p99": float(np.percentile(values, 99)),
                    "Max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def compute_heavy_tail_hist(data: EDAData, bins: int = 60) -> pd.DataFrame:
    """
    Histogram each heavy-tailed channel under each candidate transform.

    Parameters
    ----------
    data : EDAData
        EDA tables (train only).
    bins : int
        Bins per histogram, by default 60.

    Returns
    -------
    pd.DataFrame
        ``channel``, ``transform``, ``left``, ``right``, ``density``.
    """
    names = data.stat_channel_names
    z = data.stat_values.numpy().astype(np.float64)
    raw = z * data.std.numpy()[0] + data.mean.numpy()[0]
    rows = []
    for channel in _heavy_tail_channels(data):
        column = raw[:, names.index(channel)]
        for transform in TRANSFORMS:
            values = _transformed(column, transform)
            high = np.percentile(values, 99.9)
            density, edges = np.histogram(
                values, bins=bins, range=(values.min(), high), density=True
            )
            rows.extend(
                {
                    "channel": channel,
                    "transform": transform,
                    "left": edges[i],
                    "right": edges[i + 1],
                    "density": density[i],
                }
                for i in range(bins)
            )
    return pd.DataFrame(rows)


def plot_heavy_tail_hist(table: pd.DataFrame) -> plt.Figure:
    """
    Plot one row per channel and one column per candidate transform.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compute_heavy_tail_hist`.

    Returns
    -------
    plt.Figure
        The figure.
    """
    channels = list(dict.fromkeys(table["channel"]))
    fig, axes = plt.subplots(
        len(channels),
        len(TRANSFORMS),
        squeeze=False,
        figsize=style.figure_size("double", 0.17 * len(channels) + 0.05),
        gridspec_kw={"hspace": 0.7, "wspace": 0.25},
    )
    for row, channel in enumerate(channels):
        for col, transform in enumerate(TRANSFORMS):
            ax = axes[row, col]
            part = table[
                (table["channel"] == channel) & (table["transform"] == transform)
            ]
            ax.bar(
                part["left"],
                part["density"],
                width=part["right"] - part["left"],
                align="edge",
                color=style.NEUTRAL_COLOR,
            )
            ax.set_yticks([])
            ax.grid(visible=False)
            if row == 0:
                ax.set_title(transform)
            if col == 0:
                ax.set_ylabel(channel, rotation=0, ha="right", va="center")
    for ax in axes[-1]:
        ax.set_xlabel("Standardised value (to p99.9)")
    return fig


# 9. Duration confound:
LEADER_FEATURES = {
    "army_value": "Army value",
    "workers": "Workers",
    "income": "Income",
}
LENGTH_BUCKETS = ((0, 10), (10, 15), (15, 20), (20, np.inf))


def _bucket_label(low: float, high: float) -> str:
    """
    Name a game-length bucket.

    Parameters
    ----------
    low : float
        Lower bound in minutes (inclusive).
    high : float
        Upper bound in minutes (exclusive), may be infinite.

    Returns
    -------
    str
        E.g. ``"10-15"`` or ``"20+"``.
    """
    return f"{low:g}+" if np.isinf(high) else f"{low:g}-{high:g}"


def compute_duration_confound(data: EDAData) -> pd.DataFrame:
    """
    Score a "leader at the cutoff wins" rule by total game length.

    For each cutoff and feature, each game that runs past the cutoff is
    predicted for the player with the larger value at the last tick before
    the cutoff (ties dropped). Accuracy is reported per total-length bucket:
    if short games are decided early, accuracy falls with game length.

    Parameters
    ----------
    data : EDAData
        EDA tables (train only).

    Returns
    -------
    pd.DataFrame
        ``feature``, ``cutoff_min``, ``length``, ``n``, ``accuracy``,
        ``ci_low``, ``ci_high``.
    """
    ticks = data.ticks.assign(
        income=data.ticks["minerals_rate"] + data.ticks["gas_rate"]
    )
    length = data.replays.set_index("replay_id")["duration_s"] / 60.0
    rows = []
    for cutoff in CUTOFFS_MIN:
        before = ticks[ticks["minute"] <= cutoff]
        last = before.loc[before.groupby("stream")["tick"].idxmax()]
        last = last[last["horizon_s"] / 60.0 >= cutoff]
        for feature in LEADER_FEATURES:
            wide = last.pivot_table(index="replay_id", columns="player", values=feature)
            winners = last.pivot_table(
                index="replay_id", columns="player", values="won"
            )
            wide = wide.dropna()
            diff = wide[1] - wide[2]
            keep = diff != 0
            predicted_p1 = diff[keep] > 0
            p1_won = winners.loc[predicted_p1.index, 1].astype(bool)
            correct = predicted_p1 == p1_won
            minutes = length.loc[correct.index]
            for low, high in LENGTH_BUCKETS:
                in_bucket = (minutes >= low) & (minutes < high)
                n = int(in_bucket.sum())
                if n == 0:
                    continue
                hits = int(correct[in_bucket].sum())
                ci_low, ci_high = wilson_interval(np.array([hits]), np.array([n]))
                rows.append(
                    {
                        "feature": feature,
                        "cutoff_min": cutoff,
                        "length": _bucket_label(low, high),
                        "n": n,
                        "accuracy": hits / n,
                        "ci_low": float(ci_low[0]),
                        "ci_high": float(ci_high[0]),
                    }
                )
    return pd.DataFrame(rows)


def plot_duration_confound(table: pd.DataFrame) -> plt.Figure:
    """
    Plot leader-rule accuracy by game length, one panel per feature.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compute_duration_confound`.

    Returns
    -------
    plt.Figure
        The figure.
    """
    fig, axes = plt.subplots(
        1,
        len(LEADER_FEATURES),
        figsize=style.figure_size("double", 0.3),
        sharey=True,
        gridspec_kw={"wspace": 0.15},
    )
    order = [_bucket_label(low, high) for low, high in LENGTH_BUCKETS]
    for ax, (feature, label) in zip(axes, LEADER_FEATURES.items(), strict=True):
        for color, cutoff in zip(style.SEQUENTIAL_BLUES, CUTOFFS_MIN, strict=True):
            part = table[
                (table["feature"] == feature) & (table["cutoff_min"] == cutoff)
            ]
            x = [order.index(b) for b in part["length"]]
            ax.errorbar(
                x,
                part["accuracy"],
                yerr=[
                    part["accuracy"] - part["ci_low"],
                    part["ci_high"] - part["accuracy"],
                ],
                color=color,
                marker="o",
                capsize=0,
                label=f"{cutoff} min",
            )
        ax.set_xticks(range(len(order)), order)
        ax.set_xlabel("Game length (min)")
        ax.set_title(label)
        ax.axhline(0.5, color=style.INK_SECONDARY, linewidth=0.6, linestyle=":")
    axes[0].set_ylabel("Leader-wins accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Cutoff",
        loc="lower center",
        ncol=len(CUTOFFS_MIN),
        bbox_to_anchor=(0.5, 1.0),
    )
    return fig


# 10. Split comparability:
def compute_split_comparability(data: EDAData) -> pd.DataFrame:
    """
    Test whether val and test look like train.

    Kolmogorov-Smirnov tests for continuous variables, chi-square tests for
    matchup and patch counts. Descriptive only: with ~20k replays even tiny
    differences reach significance, so read the statistics, not just p.

    Parameters
    ----------
    data : EDAData
        EDA tables.

    Returns
    -------
    pd.DataFrame
        ``Variable``, ``Comparison``, ``Test``, ``Statistic``, ``p``.
    """
    replays, players = data.replays, data.players
    continuous = {
        "Game length": (replays, replays["duration_s"] / 60),
        "Events per min": (players, players["events_per_min"]),
    }
    if "apm" in players:
        continuous["APM"] = (players, players["apm"].astype(float))
    categorical = {"Matchup": (replays, "matchup")}
    if "patch" in replays:
        categorical["Patch"] = (replays, "patch")

    rows = []
    for other in ("val", "test"):
        comparison = f"train vs {other}"
        for name, (frame, values) in continuous.items():
            a = values[frame["split"] == "train"].dropna()
            b = values[frame["split"] == other].dropna()
            result = stats.ks_2samp(a, b)
            rows.append((name, comparison, "KS", result.statistic, result.pvalue))
        for name, (frame, column) in categorical.items():
            part = frame[frame["split"].isin(["train", other])]
            counts = pd.crosstab(part[column], part["split"])
            statistic, p, _, _ = stats.chi2_contingency(counts)
            rows.append((name, comparison, "chi-square", statistic, p))
    return pd.DataFrame(
        rows, columns=["Variable", "Comparison", "Test", "Statistic", "p"]
    )


# 11. Player overlap:
def compute_player_overlap(data: EDAData) -> pd.DataFrame:
    """
    Measure how many val/test games involve players or series seen in train.

    Players are identified by their battle.net ``toon``; a series is the same
    pair of players in the same tournament.

    Parameters
    ----------
    data : EDAData
        EDA tables with metadata.

    Returns
    -------
    pd.DataFrame
        Per split: players, games, games with 0/1/2 players seen in train, and
        games whose series also has games in train.
    """
    replays = data.replays
    train = replays[replays["split"] == "train"]
    train_players = set(train["p1_toon"]) | set(train["p2_toon"])

    def series(frame: pd.DataFrame) -> pd.Series:
        return (
            frame["tournament"]
            + "|"
            + frame[["p1_toon", "p2_toon"]].apply(
                lambda r: "|".join(sorted(map(str, r))), axis=1
            )
        )

    train_series = set(series(train))
    rows = []
    for split in ("train", "val", "test"):
        part = replays[replays["split"] == split]
        seen = part["p1_toon"].isin(train_players).astype(int) + part["p2_toon"].isin(
            train_players
        ).astype(int)
        row = {
            "Split": split,
            "Players": len(set(part["p1_toon"]) | set(part["p2_toon"])),
            "Games": len(part),
        }
        for k in (0, 1, 2):
            row[f"{k} seen"] = float((seen == k).mean()) if split != "train" else np.nan
        row["Series in train"] = (
            float(series(part).isin(train_series).mean())
            if split != "train"
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


OUTPUTS: tuple[Output, ...] = (
    Output(
        "composition",
        "all",
        compute_composition,
        None,
        "Replays per split, players per race and replays per matchup.",
        latex=True,
    ),
    Output(
        "skip_reasons",
        "all",
        compute_skip_reasons,
        None,
        "Source replays excluded by the events transform, by reason.",
        latex=True,
        needs_skip_report=True,
    ),
    Output(
        "patches",
        "all",
        compute_patches,
        plot_patches,
        "Replays per game patch.",
        needs_metadata=True,
    ),
    Output(
        "years",
        "all",
        compute_years,
        plot_years,
        "Replays per year.",
        needs_metadata=True,
    ),
    Output(
        "label_balance",
        "all",
        compute_label_balance,
        plot_label_balance,
        "Player-1 win rate per matchup and race win rates in non-mirror "
        "matchups, with 95% Wilson intervals.",
        latex=True,
    ),
    Output(
        "game_length",
        "all",
        compute_game_length,
        plot_game_length,
        "Game length per matchup (left; boxes are quartiles, whiskers 1.5 IQR) "
        "and share of games still running (right; dotted: evaluation cutoffs).",
    ),
    Output(
        "survival",
        "all",
        compute_survival,
        None,
        "Games still running at each cutoff, per split.",
        latex=True,
    ),
    Output(
        "event_rate",
        "train",
        compute_event_rate,
        plot_event_rate,
        "Tracker events per minute over game time by race (median and IQR; "
        "training split).",
    ),
    Output(
        "slice_events",
        "train",
        compute_slice_events,
        plot_slice_events,
        "Share of 7.14 s slices with more than a given number of events, with "
        "the candidate caps K (training split).",
    ),
    Output(
        "k_cap",
        "train",
        compute_k_cap,
        None,
        "Slices and events affected by each candidate per-slice cap K "
        "(training split).",
        latex=True,
    ),
    Output(
        "vocab_rank",
        "train",
        compute_vocab_rank,
        plot_vocab_rank,
        "Rank-frequency of unit and upgrade types (training split).",
    ),
    Output(
        "vocab_top",
        "train",
        compute_vocab_top,
        plot_vocab_top,
        "Most frequent unit and upgrade types per race (training split).",
    ),
    Output(
        "trajectories",
        "train",
        compute_trajectories,
        plot_trajectories,
        "Key statistics over game time, winners vs losers (median and IQR; "
        "time bins with at least 100 streams; training split).",
    ),
    Output(
        "army_by_matchup",
        "train",
        compute_army_by_matchup,
        plot_army_by_matchup,
        "Army value over game time per matchup, winners vs losers (training split).",
    ),
    Output(
        "pac_trajectories",
        "train",
        compute_pac_trajectories,
        plot_pac_trajectories,
        "Perception-action cycle channels over game time, winners vs losers "
        "(training split).",
    ),
    Output(
        "pac_whole_game",
        "train",
        compute_pac_whole_game,
        None,
        "Whole-game PAC statistics rebuilt from the per-tick channels, against "
        "the reference means of the PAC module's validation (training split).",
        latex=True,
    ),
    Output(
        "normalisation",
        "train",
        compute_normalisation,
        None,
        "Stat channel distributions: raw mean and std, the std floor, shape, "
        "and the extremes of the z-scores (training split).",
        latex=True,
    ),
    Output(
        "heavy_tails",
        "train",
        compute_heavy_tails,
        None,
        "Candidate transforms on the heavy-tailed channels (training split).",
        latex=True,
    ),
    Output(
        "heavy_tail_hist",
        "train",
        compute_heavy_tail_hist,
        plot_heavy_tail_hist,
        "Distributions of the heavy-tailed channels under each candidate "
        "transform, up to the 99.9th percentile (training split).",
    ),
    Output(
        "duration_confound",
        "train",
        compute_duration_confound,
        plot_duration_confound,
        "Accuracy of predicting the player ahead at the cutoff as the winner, "
        "by total game length (95% Wilson intervals; training split).",
        latex=True,
    ),
    Output(
        "split_comparability",
        "all",
        compute_split_comparability,
        None,
        "Distribution tests of val and test against train.",
        latex=True,
    ),
    Output(
        "player_overlap",
        "all",
        compute_player_overlap,
        None,
        "Players per split and val/test games whose players or series appear in train.",
        latex=True,
        needs_metadata=True,
    ),
)
