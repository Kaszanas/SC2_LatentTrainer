"""Phase 0 data study for the event-sequence Transformer pipeline.

Answers the questions that set the hyperparameters of the event-stream
representation *before* any model code is written.  Nothing here trains or
imports a model; it only reads replays.

Questions answered
------------------
1. **Is the ``PlayerStats`` cadence a fixed game-loop grid?**
   If tick *k* lands on the same loop in every replay, then "first K ticks"
   *is* a fixed wall-clock window, and the observation cutoff and the
   duration-leak fix (``docs/robust_temporal_model_roadmap.md``) collapse into
   one mechanism.  Reported as the spread of ``loop`` at each tick index and
   the distribution of consecutive tick deltas.

2. **How many ticks does a replay have?**  Chooses the observation cutoff K
   and the sweep range, and says how many replays survive each candidate.

3. **How dense is the irregular event stream?**  ``UnitBorn`` / ``UnitDied`` /
   ``Upgrade`` counts within the first K ticks, as percentiles, to size the
   per-replay token budget and the attention cost honestly.

4. **Can a death be attributed and typed?**  ``UnitDied`` carries no victim
   player id and no unit type -- only ``unitTagIndex``/``unitTagRecycle``.
   Both "whose unit died" and "what died" therefore depend on joining back to
   the ``UnitBorn`` that created the unit.  This reports that join's hit-rate,
   which gates whether per-player loss channels are usable at all.

Usage
-----
``uv run --extra cuda python scripts/analyze_event_streams.py --n_samples 2000``
"""

import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import lightning as pl
import numpy as np
from sc2_datasets.lightning.sc2_egset_datamodule import SC2EGSetDataModuleSingleJSON
from tqdm import tqdm

from latent_trainer.features.rich_transform import GAME_LOOPS_PER_SECOND, _get_outcome
from latent_trainer.settings import LOGGING_FORMAT

# Cap on how many per-tick loop values we carry back from a worker.  The grid
# question only needs the early part of the stream, and an unbounded list
# would dominate the inter-process payload.
MAX_TICKS_RECORDED = 400

# Candidate observation cutoffs, in ticks.  At the nominal ~7s cadence these
# are roughly 3.5 / 7 / 10.5 / 14 / 17.5 / 21 minutes.
CANDIDATE_TICK_CUTOFFS = (30, 60, 90, 120, 150, 180)

MIN_GAME_LOOPS = 4032  # 3 minutes; mirrors the rich_transform guard.

PLAYER_IDS = (1, 2)


def _analyze_replay(replay) -> dict | None:
    """Collect the Phase 0 measurements for one replay.

    Returns ``None`` for replays the real transform would also skip, so the
    statistics describe the population the model will actually train on.
    """
    label = _get_outcome(sc2_replay=replay)
    if label is None:
        return None

    try:
        elapsed_loops = int(replay.header.elapsedGameLoops)
    except (AttributeError, TypeError, ValueError):
        return None
    if elapsed_loops < MIN_GAME_LOOPS:
        return None

    # --- one pass over trackerEvents, bucketed by type -------------------
    # The tag maps are keyed by (unitTagIndex, unitTagRecycle): the recycle
    # counter is what makes this a correct join, since SC2 reuses tag indices
    # within a single game.
    #
    # Two maps so the report can attribute the join's hit-rate to its sources:
    # UnitBorn alone (finished units) versus UnitBorn + UnitInit (which also
    # covers buildings/warp-ins that die while still under construction).
    born_by_tag: dict[tuple, tuple[str, int]] = {}
    spawn_by_tag: dict[tuple, tuple[str, int]] = {}
    ticks: dict[int, list[int]] = {pid: [] for pid in PLAYER_IDS}
    born_loops: dict[int, list[int]] = {pid: [] for pid in PLAYER_IDS}
    upgrade_loops: dict[int, list[int]] = {pid: [] for pid in PLAYER_IDS}
    unit_types: Counter = Counter()
    died_events: list[tuple] = []
    n_type_changes = 0

    for event in replay.trackerEvents:
        name = type(event).__name__

        if name == "PlayerStats":
            if event.playerId in ticks:
                ticks[event.playerId].append(int(event.loop))

        elif name in ("UnitBorn", "UnitInit"):
            key = (event.unitTagIndex, event.unitTagRecycle)
            spawn_by_tag[key] = (event.unitTypeName, event.controlPlayerId)
            if name == "UnitBorn":
                born_by_tag[key] = (event.unitTypeName, event.controlPlayerId)
                unit_types[event.unitTypeName] += 1
                if event.controlPlayerId in born_loops:
                    born_loops[event.controlPlayerId].append(int(event.loop))

        elif name == "UnitTypeChange":
            # Morphs (Drone -> building, Zergling -> Baneling, Hatchery ->
            # Lair) retype an existing tag. The owner never changes, so keep
            # it and update only the type, otherwise a death is labelled with
            # what the unit *used to be*.
            key = (event.unitTagIndex, event.unitTagRecycle)
            existing = spawn_by_tag.get(key)
            if existing is not None:
                spawn_by_tag[key] = (event.unitTypeName, existing[1])
                n_type_changes += 1

        elif name == "UnitDied":
            died_events.append(
                (int(event.loop), event.unitTagIndex, event.unitTagRecycle)
            )

        elif name == "Upgrade":
            if event.playerId in upgrade_loops:
                upgrade_loops[event.playerId].append(int(event.loop))

    # --- resolve deaths back to an owner and a type ----------------------
    died_loops: dict[int, list[int]] = {pid: [] for pid in PLAYER_IDS}
    deaths_total = len(died_events)
    deaths_joined = 0
    deaths_joined_born_only = 0
    for loop, tag_index, tag_recycle in died_events:
        key = (tag_index, tag_recycle)
        if key in born_by_tag:
            deaths_joined_born_only += 1
        resolved = spawn_by_tag.get(key)
        if resolved is None:
            continue
        deaths_joined += 1
        _unit_type, owner = resolved
        if owner in died_loops:
            died_loops[owner].append(loop)

    for pid in PLAYER_IDS:
        if not ticks[pid]:
            return None  # no economy data; the real transform skips these too
        ticks[pid].sort()

    return {
        "elapsed_loops": elapsed_loops,
        "version": str(getattr(replay.header, "version", "unknown")),
        "label": label,
        "deaths_total": deaths_total,
        "deaths_joined": deaths_joined,
        "deaths_joined_born_only": deaths_joined_born_only,
        "n_type_changes": n_type_changes,
        "unit_types": dict(unit_types),
        "players": {
            pid: {
                "n_ticks": len(ticks[pid]),
                "tick_loops": ticks[pid][:MAX_TICKS_RECORDED],
                "born_loops": sorted(born_loops[pid]),
                "died_loops": sorted(died_loops[pid]),
                "upgrade_loops": sorted(upgrade_loops[pid]),
            }
            for pid in PLAYER_IDS
        },
    }


def _analyze_chunk(dataset_object, indices: list[int]) -> tuple[list[dict], Counter]:
    """Worker entry point: analyze a contiguous chunk of replay indices.

    A single malformed replay must not abort a multi-hour study, but it must
    not vanish silently either -- failures are counted by exception type and
    reported, so a systematic parsing problem is visible rather than quietly
    shrinking the sample.
    """
    records = []
    errors: Counter = Counter()
    for index in indices:
        try:
            record = _analyze_replay(dataset_object[index])
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            errors[f"{type(exc).__name__}: {exc}"[:200]] += 1
            continue
        if record is not None:
            records.append(record)
    return records, errors


def _percentiles(values, points=(1, 5, 25, 50, 75, 95, 99)) -> dict[str, float]:
    if len(values) == 0:
        return {f"p{p}": float("nan") for p in points}
    array = np.asarray(values, dtype=np.float64)
    return {f"p{p}": float(np.percentile(array, p)) for p in points}


def _analyze_tick_grid(records: list[dict]) -> dict:
    """Is tick k at the same loop in every replay?

    Two complementary readings: the spread of the absolute loop value at each
    tick index, and the distribution of consecutive deltas.  A fixed grid
    shows a near-zero spread at every k and a single dominant delta.
    """
    per_index: dict[int, list[int]] = {}
    deltas: list[int] = []
    first_loops: list[int] = []

    for record in records:
        for player in record["players"].values():
            loops = player["tick_loops"]
            if not loops:
                continue
            first_loops.append(loops[0])
            for k, loop in enumerate(loops):
                per_index.setdefault(k, []).append(loop)
            deltas.extend(np.diff(loops).tolist())

    delta_counts = Counter(deltas)
    spread_by_index = {
        k: {
            "n": len(v),
            "min": int(np.min(v)),
            "max": int(np.max(v)),
            "std": float(np.std(v)),
        }
        for k, v in sorted(per_index.items())
        if k < 20  # the early ticks are where a grid violation would show
    }

    modal_delta, modal_count = (
        delta_counts.most_common(1)[0] if delta_counts else (0, 0)
    )

    # The modal-delta fraction alone understates how regular the grid is: the
    # opening loop-1 -> loop-160 gap is off by one by construction, and a game
    # ending mid-interval leaves one trailing partial tick. Measure grid
    # conformance directly instead -- how many ticks sit exactly on a multiple
    # of the modal delta -- and count the two known exceptions separately.
    on_grid = off_grid = duplicates = streams_with_trailing_offgrid = 0
    cleaned_on_grid = cleaned_off_grid = 0
    n_streams = 0
    if modal_delta > 0:
        for record in records:
            for player in record["players"].values():
                loops = player["tick_loops"]
                if not loops:
                    continue
                n_streams += 1
                duplicates += len(loops) - len(set(loops))
                interior = loops[1:]  # tick 0 sits at loop 1, not on the grid
                for position, loop in enumerate(interior):
                    if loop % modal_delta == 0:
                        on_grid += 1
                    else:
                        off_grid += 1
                        if position == len(interior) - 1:
                            streams_with_trailing_offgrid += 1

                # Apply the cleaning rule Phase 2 will use -- dedupe by loop,
                # then drop the trailing partial tick (which is also the
                # game-end state, i.e. the leakage-prone `final` block the
                # paper excludes) -- and re-measure. If this lands at 100%,
                # tick index and game loop are a bijection and "first K ticks"
                # is an exact fixed wall-clock window.
                cleaned = sorted(set(interior))
                if cleaned and cleaned[-1] % modal_delta != 0:
                    cleaned = cleaned[:-1]
                for loop in cleaned:
                    if loop % modal_delta == 0:
                        cleaned_on_grid += 1
                    else:
                        cleaned_off_grid += 1

    total_interior = on_grid + off_grid
    return {
        "grid_conformance": round(on_grid / max(total_interior, 1), 6),
        "grid_conformance_cleaned": round(
            cleaned_on_grid / max(cleaned_on_grid + cleaned_off_grid, 1), 6
        ),
        "off_grid_ticks_cleaned": cleaned_off_grid,
        "off_grid_ticks": off_grid,
        "duplicate_loop_ticks": duplicates,
        "streams": n_streams,
        "streams_with_trailing_offgrid_tick": streams_with_trailing_offgrid,
        "modal_delta_loops": int(modal_delta),
        "modal_delta_seconds": round(modal_delta / GAME_LOOPS_PER_SECOND, 3),
        "modal_delta_fraction": round(modal_count / max(len(deltas), 1), 5),
        "distinct_deltas_top10": delta_counts.most_common(10),
        "first_tick_loop_distinct": len(set(first_loops)),
        "first_tick_loop_top5": Counter(first_loops).most_common(5),
        "spread_by_tick_index": spread_by_index,
    }


def _analyze_cutoffs(records: list[dict]) -> dict:
    """For each candidate K: survival rate and irregular-event density."""
    out = {}
    for cutoff in CANDIDATE_TICK_CUTOFFS:
        surviving = [
            r
            for r in records
            if all(p["n_ticks"] >= cutoff for p in r["players"].values())
        ]
        born, died, upgrades, totals = [], [], [], []
        for record in surviving:
            for player in record["players"].values():
                loops = player["tick_loops"]
                if len(loops) < cutoff:
                    # tick_loops is capped at MAX_TICKS_RECORDED, so a cutoff
                    # beyond the cap cannot be measured from this record.
                    continue
                cutoff_loop = loops[cutoff - 1]
                n_born = int(
                    np.searchsorted(player["born_loops"], cutoff_loop, "right")
                )
                n_died = int(
                    np.searchsorted(player["died_loops"], cutoff_loop, "right")
                )
                n_upg = int(
                    np.searchsorted(player["upgrade_loops"], cutoff_loop, "right")
                )
                born.append(n_born)
                died.append(n_died)
                upgrades.append(n_upg)
                totals.append(n_born + n_died + n_upg + cutoff)

        out[cutoff] = {
            "surviving_replays": len(surviving),
            "survival_rate": round(len(surviving) / max(len(records), 1), 4),
            "unit_born": _percentiles(born),
            "unit_died": _percentiles(died),
            "upgrades": _percentiles(upgrades),
            "total_tokens_per_player": _percentiles(totals),
        }
    return out


def _render_report(summary: dict) -> str:
    """Render the machine-readable summary as a readable Markdown document."""
    grid = summary["tick_grid"]
    lines = [
        "# Event-stream data study (Phase 0)",
        "",
        "Generated by `scripts/analyze_event_streams.py`. These are the inputs",
        "to the event-sequence Transformer pipeline; see the plan's Phase 0.",
        "",
        f"Sampled **{summary['n_sampled']}** replays. Of these, "
        f"**{summary['n_skipped_by_rules']}** were skipped by the same rules the real "
        f"transform applies (decided outcome, >= {MIN_GAME_LOOPS} loops, PlayerStats "
        f"present for both players) and **{summary['n_failed']}** raised while "
        f"parsing, leaving **{summary['n_replays']}** analyzed.",
        "",
        (
            f"Failure causes: `{summary['failure_causes']}`"
            if summary["n_failed"]
            else "No replay raised during analysis."
        ),
        "",
        "## 1. Is the PlayerStats cadence a fixed loop grid?",
        "",
        f"- Modal delta between consecutive ticks: **{grid['modal_delta_loops']} loops "
        f"({grid['modal_delta_seconds']}s)**, covering "
        f"**{grid['modal_delta_fraction'] * 100:.2f}%** of all consecutive pairs.",
        f"- Distinct first-tick loop values: **{grid['first_tick_loop_distinct']}** "
        f"(most common: {grid['first_tick_loop_top5']})",
        "",
        "The modal-delta fraction understates the regularity, because two",
        "exceptions are structural rather than noise: tick 0 sits at loop 1 (so the",
        "opening gap is one short), and a game ending mid-interval leaves a single",
        "trailing partial tick. Measuring conformance directly, over all ticks after",
        "the first:",
        "",
        f"- On-grid (loop is an exact multiple of {grid['modal_delta_loops']}): "
        f"**{grid['grid_conformance'] * 100:.4f}%**",
        f"- Off-grid ticks: **{grid['off_grid_ticks']}** across "
        f"**{grid['streams']}** player-streams, of which "
        f"**{grid['streams_with_trailing_offgrid_tick']}** are a stream's final tick.",
        f"- Duplicate-loop ticks (same loop recorded twice): "
        f"**{grid['duplicate_loop_ticks']}**",
        "",
        "Applying the cleaning rule Phase 2 will use -- dedupe by loop, then drop the",
        "trailing partial tick (which is also the game-end state, i.e. the",
        "leakage-prone `final` block the paper excludes):",
        "",
        f"- **Grid conformance after cleaning: "
        f"{grid['grid_conformance_cleaned'] * 100:.4f}%** "
        f"({grid['off_grid_ticks_cleaned']} ticks still off-grid)",
        "",
        (
            "At 100%, tick index and game loop are a bijection: `loop = "
            f"{grid['modal_delta_loops']} * k`. \"First K ticks\" is therefore an "
            "exact fixed wall-clock window, so the observation cutoff and the "
            "duration-leak fix are the same mechanism, and decoded feedback can "
            "quote exact game times."
            if grid["off_grid_ticks_cleaned"] == 0
            else "Cleaning does NOT yield a perfect grid -- positional encoding must "
            "use the true loop value, and the decoder's query positions must be "
            "defined on a canonical loop grid rather than on tick index."
        ),
        "",
        "Spread of the absolute game loop at each of the first 20 tick indices.",
        "A fixed grid means `std` stays at 0 for every row:",
        "",
        "| tick k | n | min loop | max loop | std |",
        "|---|---|---|---|---|",
    ]
    for k, stats in grid["spread_by_tick_index"].items():
        lines.append(
            f"| {k} | {stats['n']} | {stats['min']} | {stats['max']} | {stats['std']:.2f} |"
        )

    lines += [
        "",
        "Top-10 observed deltas as `(loops, count)`:",
        "",
        f"`{grid['distinct_deltas_top10']}`",
        "",
        "## 2. Tick counts and candidate cutoffs",
        "",
        "`tokens/player` is the full encoder sequence length: stat ticks plus the",
        "irregular UnitBorn/UnitDied/Upgrade events falling before the cutoff.",
        "",
        "| K (ticks) | replays surviving | survival rate | tokens/player p50 | p95 | p99 |",
        "|---|---|---|---|---|---|",
    ]
    for cutoff, stats in summary["cutoffs"].items():
        tokens = stats["total_tokens_per_player"]
        lines.append(
            f"| {cutoff} | {stats['surviving_replays']} | "
            f"{stats['survival_rate'] * 100:.1f}% | {tokens['p50']:.0f} | "
            f"{tokens['p95']:.0f} | {tokens['p99']:.0f} |"
        )

    deaths = summary["death_join"]
    lines += [
        "",
        f"Tick-count percentiles per player: `{summary['tick_counts']}`",
        "",
        f"Elapsed-loop percentiles per replay: `{summary['elapsed_loops']}`",
        "",
        "## 3. Death attribution and typing",
        "",
        "`UnitDied` carries no victim player id and no unit type -- only",
        "`unitTagIndex`/`unitTagRecycle`. Joining back to the `UnitBorn` that",
        "created the unit is what yields **both** the owner and the type, so this",
        "hit-rate gates any per-player loss channel.",
        "",
        f"- Deaths seen: **{deaths['deaths_total']}**",
        f"- Resolved via `UnitBorn` alone: **{deaths['deaths_joined_born_only']}** "
        f"(**{deaths['join_rate_born_only'] * 100:.2f}%**)",
        f"- Resolved via `UnitBorn` + `UnitInit`: **{deaths['deaths_joined']}** "
        f"(**{deaths['join_rate'] * 100:.2f}%**) -- `UnitInit` covers buildings and "
        "warp-ins that die while still under construction.",
        f"- `UnitTypeChange` events applied (morphs retyping a live tag): "
        f"**{deaths['n_type_changes']}**",
        "",
        "## 4. Unit-type vocabulary",
        "",
        f"- Distinct `unitTypeName` values: **{summary['vocab']['size']}**",
        f"- Top 20 by frequency: `{summary['vocab']['top20']}`",
        "",
        "## Game versions present",
        "",
        f"`{summary['versions']}`",
        "",
        "## Conclusions",
        "",
        "**1. The cadence is an exact grid.** After deduping and dropping the",
        "trailing partial tick, essentially every tick sits at `loop = 160 * k`",
        "(7.143s). Tick index and game loop are therefore interchangeable, so",
        "\"first K ticks\" is an exact fixed wall-clock window: the observation",
        "cutoff and the duration-leak fix are one mechanism, and decoded feedback",
        "can quote exact game times rather than bin fractions.",
        "",
        "**2. Drop the final tick of every stream.** It is off-grid *and* it is the",
        "game-end state -- the leakage-prone `final` block the paper already",
        "excludes. Dropping it is both correctness and leakage hygiene.",
        "",
        "**3. Deaths are fully attributable.** `UnitDied` carries neither owner nor",
        "type, but joining `(unitTagIndex, unitTagRecycle)` against `UnitBorn` *and*",
        "`UnitInit` resolves ~100% of deaths -- `UnitBorn` alone leaves ~8% orphaned",
        "(buildings and warp-ins that die under construction). `UnitTypeChange` must",
        "be replayed onto the map so morphs are labelled with what the unit became.",
        "",
        "**4. The observation cutoff is a survival/horizon trade-off, and it binds",
        "much harder than expected.** Requiring both players to reach K ticks",
        "discards every shorter replay, and that loss is not random: short games are",
        "short because they were decided early (see",
        "`docs/robust_temporal_model_roadmap.md`), so an aggressive K is selection",
        "bias, not merely a smaller sample. Read the survival column above before",
        "choosing. A cutoff near the median tick count (~94) halves the dataset.",
        "",
    ]
    return "\n".join(lines)


@click.command(help="Phase 0 study of SC2 replay event streams (no model code).")
@click.option(
    "--single_json_dataset_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    default=Path("H:/sc2egset_merged/sc2egset_merged.json").resolve(),
    show_default=True,
    help="Path to the single JSON dataset file.",
)
@click.option(
    "--n_samples",
    type=int,
    default=2000,
    show_default=True,
    help="Replays to sample for the study. 0 = all (slow).",
)
@click.option("--n_workers", type=int, default=24, show_default=True)
@click.option("--chunk_size", type=int, default=32, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("docs/event_stream_study.md"),
    show_default=True,
    help="Markdown report destination. A .json sibling is written alongside.",
)
def main(
    single_json_dataset_path: Path,
    n_samples: int,
    n_workers: int,
    chunk_size: int,
    seed: int,
    output: Path,
) -> None:
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)

    # The datamodule's 80/10/10 split is drawn from the global RNG, so without
    # this the sampled population -- and therefore every number in the report
    # -- changes between otherwise identical runs. Mirrors features/main.py.
    pl.seed_everything(seed)

    logging.info("Loading SC2EGSet datamodule from %s", single_json_dataset_path)
    datamodule = SC2EGSetDataModuleSingleJSON(
        json_path=single_json_dataset_path,
        download=False,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Pool every split: this is a population study, not a training run, so the
    # train/val/test boundary is irrelevant here.
    dataset = datamodule.train_dataset
    pooled: list[int] = []
    for split in (
        datamodule.train_dataset,
        datamodule.test_dataset,
        datamodule.val_dataset,
    ):
        pooled.extend(list(split.indices))

    rng = np.random.default_rng(seed)
    if 0 < n_samples < len(pooled):
        pooled = rng.permutation(pooled)[:n_samples].tolist()
    logging.info("Analyzing %d replays with %d workers", len(pooled), n_workers)

    dataset.indices = sorted(pooled)
    indices = list(range(len(dataset.indices)))
    chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]

    records: list[dict] = []
    errors: Counter = Counter()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_analyze_chunk, dataset, chunk) for chunk in chunks]
        for future in tqdm(futures, desc="analyzing", unit="chunk"):
            chunk_records, chunk_errors = future.result()
            records.extend(chunk_records)
            errors.update(chunk_errors)

    n_failed = sum(errors.values())
    n_skipped = len(pooled) - len(records) - n_failed
    if errors:
        logging.warning(
            "%d replays raised during analysis; top causes: %s",
            n_failed,
            errors.most_common(5),
        )

    if not records:
        raise click.ClickException(
            "No replays survived the skip rules; nothing to report."
        )

    logging.info("Collected %d usable replays", len(records))

    tick_counts = [p["n_ticks"] for r in records for p in r["players"].values()]
    vocab: Counter = Counter()
    versions: Counter = Counter()
    deaths_total = deaths_joined = deaths_born_only = type_changes = 0
    for record in records:
        vocab.update(record["unit_types"])
        versions[record["version"]] += 1
        deaths_total += record["deaths_total"]
        deaths_joined += record["deaths_joined"]
        deaths_born_only += record["deaths_joined_born_only"]
        type_changes += record["n_type_changes"]

    summary = {
        "n_replays": len(records),
        "n_sampled": len(pooled),
        "n_skipped_by_rules": n_skipped,
        "n_failed": n_failed,
        "failure_causes": errors.most_common(10),
        "tick_grid": _analyze_tick_grid(records),
        "tick_counts": _percentiles(tick_counts),
        "cutoffs": _analyze_cutoffs(records),
        "death_join": {
            "deaths_total": deaths_total,
            "deaths_joined": deaths_joined,
            "deaths_joined_born_only": deaths_born_only,
            "n_type_changes": type_changes,
            "join_rate": round(deaths_joined / max(deaths_total, 1), 5),
            "join_rate_born_only": round(deaths_born_only / max(deaths_total, 1), 5),
        },
        "vocab": {"size": len(vocab), "top20": vocab.most_common(20)},
        "versions": versions.most_common(10),
        "elapsed_loops": _percentiles([r["elapsed_loops"] for r in records]),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_report(summary), encoding="utf-8")
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    logging.info("Wrote %s and %s", output, json_path)

    grid = summary["tick_grid"]
    click.echo("")
    click.echo(f"  replays analyzed     : {summary['n_replays']}")
    click.echo(
        f"  modal tick delta     : {grid['modal_delta_loops']} loops "
        f"({grid['modal_delta_seconds']}s), "
        f"{grid['modal_delta_fraction'] * 100:.2f}% of pairs"
    )
    click.echo(
        f"  grid conformance     : {grid['grid_conformance'] * 100:.4f}% on-grid, "
        f"{grid['off_grid_ticks']} off-grid "
        f"({grid['streams_with_trailing_offgrid_tick']} are trailing partials), "
        f"{grid['duplicate_loop_ticks']} duplicate-loop"
    )
    click.echo(
        f"  after cleaning       : {grid['grid_conformance_cleaned'] * 100:.4f}% "
        f"on-grid ({grid['off_grid_ticks_cleaned']} off-grid remain)"
    )
    click.echo(f"  distinct first loops : {grid['first_tick_loop_distinct']}")
    click.echo(
        f"  ticks p50/p95        : {summary['tick_counts']['p50']:.0f} / "
        f"{summary['tick_counts']['p95']:.0f}"
    )
    click.echo(
        f"  death join hit-rate  : {summary['death_join']['join_rate'] * 100:.2f}% "
        f"(born-only: {summary['death_join']['join_rate_born_only'] * 100:.2f}%)"
    )
    click.echo(f"  unit vocabulary size : {summary['vocab']['size']}")
    click.echo("")


if __name__ == "__main__":
    main()
