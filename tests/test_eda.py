"""Tests for the EDA package: scope rule, known values, replot and LaTeX output."""

import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from latent_trainer.analysis.eda import figures, style
from latent_trainer.analysis.eda.data import build_eda_data, matchup_name
from latent_trainer.analysis.eda.metadata import (
    SIDECAR_FILE,
    load_metadata,
)
from latent_trainer.features.event_transform import (
    EVENT_TYPE_NAMES,
    N_STAT_CHANNELS,
    RACE_NAMES,
    STAT_CHANNEL_NAMES,
    PlayerEventStreams,
    ReplayEventSample,
)
from latent_trainer.features.preprocess_events import _flatten_split
from latent_trainer.features.type import EventDataBundle, EventSplit

ARMY = STAT_CHANNEL_NAMES.index("mineralsUsedCurrentArmy")
N_TICKS = 150  # ~18 min


def _player(seed: int, race: int, strength: float) -> PlayerEventStreams:
    """Return a stream whose army grows with ``strength``; events every tick."""
    rng = np.random.default_rng(seed)
    ticks = np.arange(N_TICKS, dtype=np.int32)
    values = rng.random((N_TICKS, N_STAT_CHANNELS)).astype(np.float32) * 10
    values[:, ARMY] = ticks * strength
    loops = ticks[1:] * 160 - 5
    return PlayerEventStreams(
        stat_values=values,
        stat_loops=np.maximum(ticks * 160, 1).astype(np.int32),
        stat_tick=ticks,
        evt_type=np.ones(len(loops), dtype=np.int8),
        evt_unit_local=np.zeros(len(loops), dtype=np.int32),
        evt_loops=loops.astype(np.int32),
        evt_tick=ticks[1:],
        race=race,
    )


def _split(n: int, seed: int) -> EventSplit:
    """Return a split whose winner always has the larger army."""
    samples = []
    for i in range(n):
        label = i % 2
        p1 = _player(seed + 2 * i, i % 3, 2.0 if label else 1.0)
        p2 = _player(seed + 2 * i + 1, (i + 1) % 3, 1.0 if label else 2.0)
        samples.append(
            ReplayEventSample(players=(p1, p2), local_vocab=["Marine"], label=label)
        )
    return EventSplit(**_flatten_split(samples, ["<PAD>", "<UNK>", "Marine"]))


def _bundle(val_seed: int = 100) -> EventDataBundle:
    """Return a small raw bundle; ``val_seed`` changes val and test only."""
    train, val, test = _split(12, 0), _split(4, val_seed), _split(4, val_seed + 50)
    return EventDataBundle(
        train=train,
        val=val,
        test=test,
        stat_channel_names=list(STAT_CHANNEL_NAMES),
        race_names=list(RACE_NAMES),
        unit_vocab=["<PAD>", "<UNK>", "Marine"],
        event_type_names=list(EVENT_TYPE_NAMES),
        tick_interval_loops=160,
        split_ids={
            s: [f"20{18 + i % 3}_Cup/{s}{i}" for i in range(n)]
            for s, n in (("train", 12), ("val", 4), ("test", 4))
        },
    )


def _metadata(bundle: EventDataBundle) -> pd.DataFrame:
    """Return metadata in which val replay 0 reuses train players."""
    rows = []
    for split, ids in bundle.split_ids.items():
        for i, replay_id in enumerate(ids):
            p1, p2 = f"{split}-a{i}", f"{split}-b{i}"
            if split == "val" and i == 0:
                p1, p2 = "train-a0", "train-b0"
            rows.append(
                {
                    "split": split,
                    "replay_id": replay_id,
                    "elapsed_loops": 150 * 160,
                    "game_version": "5.0.11.1234",
                    "base_build": "1",
                    "map_name": "M",
                    "time_utc": "t",
                    "p1_toon": p1,
                    "p2_toon": p2,
                    "p1_apm": 300,
                    "p2_apm": 280,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _few_streams(monkeypatch):
    """Let tiny data produce time bands."""
    monkeypatch.setattr(figures, "MIN_STREAMS", 1)
    style.apply_style("default")


# Scope rule:
def test_train_scope_outputs_ignore_val_and_test():
    """Changing val/test leaves every train-scope table identical."""
    first = build_eda_data(_bundle(val_seed=100))
    second = build_eda_data(_bundle(val_seed=999))
    for output in figures.OUTPUTS:
        if output.scope != "train" or output.needs_metadata:
            continue
        a = output.compute(first.train_only())
        b = output.compute(second.train_only())
        pd.testing.assert_frame_equal(a, b, obj=output.name)


def test_every_output_computes_and_plots(tmp_path):
    """All outputs run on synthetic data and write their files."""
    bundle = _bundle()
    data = build_eda_data(bundle, _metadata(bundle))
    data.skip_report = {"counts": {"too_short": 3}}
    for output in figures.OUTPUTS:
        view = data.train_only() if output.scope == "train" else data
        table = output.compute(view)
        assert isinstance(table, pd.DataFrame), output.name
        if output.plot is not None:
            paths = style.save_figure(output.plot(table), tmp_path, output.name)
            assert all(p.stat().st_size > 0 for p in paths), output.name


def test_replot_from_saved_table_matches(tmp_path):
    """A table read back from Parquet plots the same as the fresh one."""
    table = figures.compute_trajectories(build_eda_data(_bundle()).train_only())
    table.to_parquet(tmp_path / "t.parquet", index=False)
    loaded = pd.read_parquet(tmp_path / "t.parquet")
    pd.testing.assert_frame_equal(table, loaded)
    style.save_figure(figures.plot_trajectories(loaded), tmp_path, "t")


# Known values:
def test_wilson_interval_known_value():
    """50 of 100 gives the textbook interval."""
    low, high = figures.wilson_interval(np.array([50]), np.array([100]))
    assert low[0] == pytest.approx(0.4038, abs=1e-4)
    assert high[0] == pytest.approx(0.5962, abs=1e-4)


def test_leader_rule_is_perfect_when_the_winner_leads():
    """Every synthetic winner has the larger army, so army accuracy is 1."""
    table = figures.compute_duration_confound(build_eda_data(_bundle()).train_only())
    army = table[table["feature"] == "army_value"]
    assert not army.empty
    assert (army["accuracy"] == 1.0).all()


def test_player_overlap_counts_seen_players():
    """One of four val games has both players seen in train."""
    bundle = _bundle()
    table = figures.compute_player_overlap(build_eda_data(bundle, _metadata(bundle)))
    val = table.set_index("Split").loc["val"]
    assert val["2 seen"] == pytest.approx(0.25)
    assert val["0 seen"] == pytest.approx(0.75)


def test_matchup_name_ignores_order():
    """Matchups are named the same for either player order."""
    assert matchup_name("Zerg", "Terr") == matchup_name("Terr", "Zerg") == "TvZ"


# Metadata:
def test_metadata_from_another_manifest_is_refused(tmp_path):
    """Metadata built for one manifest cannot be used with another."""
    (tmp_path / SIDECAR_FILE).write_text('{"splits_sha256": "abc", "n_rows": 1}')

    class Manifest:
        def sha256(self):
            return "def"

    with pytest.raises(ValueError, match="rebuild"):
        load_metadata(tmp_path, Manifest())
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path / "missing", Manifest())


# Style and LaTeX:
def test_presets_and_overrides_set_the_figure_size():
    """Presets set widths; overrides replace single fields."""
    active = style.apply_style("ieee", font_size=9)
    assert active.font_size == 9 and active.double_width_in == 7.16
    assert style.figure_size("double", 0.5) == (7.16, 3.58)
    with pytest.raises(ValueError):
        style.figure_size("triple")


def test_latex_table_is_escaped_booktabs(tmp_path):
    """Special characters are escaped and the table uses booktabs rules."""
    table = pd.DataFrame({"a_b": ["x&y", "50%"], "n": [1.23456, 2.0]})
    path = style.write_latex_table(table, tmp_path / "t.tex", "Cap_tion 50%", "tab:t")
    text = path.read_text()
    assert r"\toprule" in text and r"\bottomrule" in text
    assert r"a\_b" in text and r"x\&y" in text and r"50\%" in text
    assert r"Cap\_tion 50\%" in text


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX installed")
def test_latex_table_compiles(tmp_path):
    """A generated table compiles in a minimal document."""
    data = build_eda_data(_bundle())
    style.write_latex_table(
        figures.compute_composition(data), tmp_path / "t.tex", "Composition", "tab:c"
    )
    (tmp_path / "doc.tex").write_text(
        "\\documentclass{article}\\usepackage{booktabs}"
        "\\begin{document}\\input{t.tex}\\end{document}"
    )
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "doc.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout[-2000:]


def test_replace_keeps_train_only_view_cheap():
    """The train-only view filters replays and players and drops the skip report."""
    data = build_eda_data(_bundle())
    data = replace(data, skip_report={"counts": {}})
    view = data.train_only()
    assert set(view.replays["split"]) == {"train"}
    assert set(view.players["split"]) == {"train"}
    assert view.skip_report is None
    assert torch.equal(view.stat_values, data.stat_values)
