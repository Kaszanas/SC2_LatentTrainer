"""Tests for event-model training runs, validation settings and the search space."""

import numpy as np
import optuna
import pytest
from mlflow import MlflowClient

from latent_trainer.configs.experiment_config import ExperimentConfig
from latent_trainer.configs.search_space import (
    EVENT_DATA_KEYS,
    EVENT_MODEL_KEYS,
    event_params_from_flat,
    get_event_guided_vae_search_space,
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
from latent_trainer.hyperparameter_search import event_guided_vae as hpo
from latent_trainer.models import train_event_guided as tev

VOCAB = ["<PAD>", "<UNK>", "Marine", "SCV"]
PARAMS = {
    "d_model": 32,
    "n_heads": 4,
    "n_layers": 1,
    "ffn_mult": 2,
    "dropout": 0.0,
    "latent_dim": 8,
    "supervised_dim": 3,
    "decoder_width": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "learning_rate_cls": 1e-4,
    "weight_decay_cls": 1e-4,
    "classification_weight": 1.0,
    "kl_beta": 0.1,
    "batch_size": 4,
    "min_crop_s": 30.0,
    "max_events_per_slice": 8,
}


def _player(n_ticks: int, race: int, seed: int) -> PlayerEventStreams:
    rng = np.random.default_rng(seed)
    ticks = np.arange(n_ticks, dtype=np.int32)
    loops = np.sort(rng.integers(1, (n_ticks - 1) * 160, 20)).astype(np.int32)
    return PlayerEventStreams(
        stat_values=rng.gamma(2.0, 50.0, (n_ticks, N_STAT_CHANNELS)).astype(np.float32),
        stat_loops=np.maximum(ticks * 160, 1).astype(np.int32),
        stat_tick=ticks,
        evt_type=rng.integers(1, len(EVENT_TYPE_NAMES), 20).astype(np.int8),
        evt_unit_local=rng.integers(0, 2, 20).astype(np.int32),
        evt_loops=loops,
        evt_tick=np.minimum(-(-loops // 160), n_ticks - 1).astype(np.int32),
        race=race,
    )


def _split(n: int, seed: int) -> EventSplit:
    samples = [
        ReplayEventSample(
            players=(
                _player(12 + i, i % 3, seed + 2 * i),
                _player(12 + i, (i + 1) % 3, seed + 2 * i + 1),
            ),
            local_vocab=["Marine", "SCV"],
            label=i % 2,
        )
        for i in range(n)
    ]
    return EventSplit(**_flatten_split(samples, VOCAB))


def _bundle() -> EventDataBundle:
    """Train and val only; test is None, so any access to it fails."""
    return EventDataBundle(
        train=_split(8, 0),
        val=_split(4, 100),
        test=None,
        stat_channel_names=list(STAT_CHANNEL_NAMES),
        race_names=list(RACE_NAMES),
        unit_vocab=VOCAB,
        event_type_names=list(EVENT_TYPE_NAMES),
        tick_interval_loops=160,
        splits_sha256="abc",
    )


def _config(tmp_path, **overrides) -> ExperimentConfig:
    kwargs = dict(
        sweep=False,
        experiment_name="unit_event",
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        val_cutoffs_min=(0.5, 1.0),
        guided_vae_epochs=2,
        search_max_epochs=2,
        precision="32-true",
        cache_sha256="def",
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


# Validation settings and the search subset:
def test_validation_tags_name_the_whole_game_then_each_cutoff():
    """Tags follow the order of the validation loaders."""
    assert tev.validation_tags((5.0, 10.0, 15.0)) == ["full", "5min", "10min", "15min"]


def test_search_subset_is_seeded_and_sized():
    """The same seed gives the same sorted subset of the requested size."""
    a = tev.search_subset(1000, 0.3, seed=42)
    assert a == tev.search_subset(1000, 0.3, seed=42)
    assert len(a) == 300 and a == sorted(a) and len(set(a)) == 300
    assert a != tev.search_subset(1000, 0.3, seed=43)
    with pytest.raises(ValueError):
        tev.search_subset(10, 0.0, seed=1)


def test_loaders_use_train_and_val_only_with_fixed_cutoffs(tmp_path):
    """No loader touches the test split; cutoff loaders crop at exactly their loop."""
    config = _config(tmp_path)
    train, vals, tags = tev.make_loaders(_bundle(), PARAMS, config, seed=0)
    assert tags == ["full", "0.5min", "1min"]
    assert len(vals) == 3
    assert vals[0].dataset.crop is None
    for loader, minutes in zip(vals[1:], config.val_cutoffs_min, strict=True):
        assert loader.dataset.crop == minutes * 60
        for batch in loader:
            assert (batch["crop_loop"] == round(minutes * 60 * 22.4)).all()
    assert train.dataset.crop == "random"


# Search space:
def test_search_space_only_produces_valid_configs():
    """Heads divide the width and supervised dims leave free dims."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(300):
        trial = study.ask()
        params = get_event_guided_vae_search_space(trial)
        assert set(params) == set(EVENT_MODEL_KEYS) | set(EVENT_DATA_KEYS)
        assert params["d_model"] % params["n_heads"] == 0
        assert 0 < params["supervised_dim"] < params["latent_dim"]
        study.tell(trial, 0.0)


def test_stored_trial_params_rebuild_the_same_config():
    """Raw Optuna params rebuild exactly the config the trial trained."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    trial = study.ask()
    sampled = get_event_guided_vae_search_space(trial)
    assert event_params_from_flat(trial.params) == sampled


def test_top_search_configs_are_the_best_completed_trials(tmp_path):
    """Confirm reads the k lowest-objective completed trials, best first."""
    config = _config(
        tmp_path,
        optuna_db=f"sqlite:///{(tmp_path / 'o.db').as_posix()}",
        confirm_top_k=2,
    )
    study = optuna.create_study(
        study_name=hpo.search_study_name(config),
        storage=config.optuna_db,
        sampler=optuna.samplers.RandomSampler(seed=2),
    )
    for value in (0.9, 0.3, 0.5, 0.1):
        trial = study.ask()
        get_event_guided_vae_search_space(trial)
        study.tell(trial, value)
    failed = study.ask()
    get_event_guided_vae_search_space(failed)
    study.tell(failed, state=optuna.trial.TrialState.FAIL)

    top = hpo.top_search_configs(config)
    assert [c["search_objective"] for c in top] == [0.1, 0.3]
    assert set(EVENT_MODEL_KEYS) <= set(top[0])


# One run:
@pytest.mark.parametrize("mode", ["search", "final"])
def test_training_run_logs_per_setting_metrics_objective_and_provenance(
    tmp_path, monkeypatch, mode
):
    """A run logs every setting, the averaged objective and its provenance."""
    monkeypatch.setattr(tev, "CHECKPOINTS_DIR", tmp_path / "ckpt")
    config = _config(tmp_path)
    metrics = tev.train_event_guided_run(
        bundle=_bundle(),
        config=config,
        params=PARAMS,
        seed=7,
        run_name=f"unit_{mode}",
        mode=mode,
        cache_path=tmp_path / "cache.pt",
    )
    tags = ["full", "0.5min", "1min"]
    for tag in tags:
        assert f"val_cls_loss@{tag}" in metrics and f"val_acc@{tag}" in metrics
    expected = np.mean(
        [
            0.5 * metrics[f"val_vae_loss@{t}"] + 0.5 * metrics[f"val_cls_loss@{t}"]
            for t in tags
        ]
    )
    assert metrics["val_objective"] == pytest.approx(expected, rel=1e-5)
    assert np.isfinite(metrics["best_val_objective"])

    run = MlflowClient(tracking_uri=config.mlflow_tracking_uri).get_run(
        metrics["run_id"]
    )
    assert run.data.params["prov_splits_sha256"] == "abc"
    assert run.data.params["prov_cache_sha256"] == "def"
    assert run.data.params["prov_seed"] == "7"
    assert run.data.params["mode"] == mode
    if mode == "final":
        assert metrics["checkpoint"].endswith(".ckpt")
        assert (tmp_path / "ckpt" / "unit_event" / "unit_final").is_dir()
    else:
        assert "checkpoint" not in metrics


def test_unknown_mode_is_refused(tmp_path):
    """Only search, confirm and final exist."""
    with pytest.raises(ValueError, match="mode"):
        tev.train_event_guided_run(
            bundle=_bundle(),
            config=_config(tmp_path),
            params=PARAMS,
            seed=0,
            run_name="x",
            mode="train",
            cache_path=tmp_path / "c.pt",
        )
