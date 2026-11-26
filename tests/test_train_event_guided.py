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


# Resuming a stage:
def _logged_run(client, experiment_id, name, seed, params, status="FINISHED", **kw):
    """Create an MLflow run shaped like a finished confirm/final run."""
    run = client.create_run(experiment_id, run_name=name)
    logged = {
        "mode": kw.get("mode", "confirm"),
        "prov_seed": str(seed),
        "prov_cache_sha256": kw.get("cache", "def"),
        **{key: str(value) for key, value in params.items()},
    }
    for key, value in logged.items():
        client.log_param(run.info.run_id, key, value)
    if kw.get("objective", True):
        client.log_metric(run.info.run_id, "best_val_objective", 0.4)
    if "checkpoint" in kw:
        client.set_tag(run.info.run_id, "checkpoint", kw["checkpoint"])
    if status != "RUNNING":
        client.set_terminated(run.info.run_id, status=status)
    return run.info.run_id


def test_resume_reuses_only_identical_finished_runs(tmp_path):
    """Seed, parameters, cache hash, status and the objective must all match."""
    config = _config(tmp_path, resume=True)
    client = MlflowClient(tracking_uri=config.mlflow_tracking_uri)
    exp = client.create_experiment(config.experiment_name)
    other = {**PARAMS, "learning_rate": 5e-4}
    runs = [{"params": PARAMS, "seed": s, "name": f"confirm_c0_s{s}"} for s in range(6)]
    reused = _logged_run(client, exp, "confirm_c0_s0", 0, PARAMS)
    _logged_run(client, exp, "confirm_c0_s1", 99, PARAMS)  # other seed
    _logged_run(client, exp, "confirm_c0_s2", 2, other)  # other parameters
    _logged_run(client, exp, "confirm_c0_s3", 3, PARAMS, cache="zzz")  # other cache
    _logged_run(client, exp, "confirm_c0_s4", 4, PARAMS, objective=False)
    stale = _logged_run(client, exp, "confirm_c0_s5", 5, PARAMS, status="RUNNING")

    found = hpo._finished_runs(config, "confirm", runs)
    assert list(found) == ["confirm_c0_s0"]
    assert found["confirm_c0_s0"] == {"best_val_objective": 0.4, "run_id": reused}
    assert client.get_run(stale).info.status == "KILLED"


def test_resume_of_final_runs_needs_their_checkpoint(tmp_path):
    """A final run is reused only with its tagged checkpoint on disk."""
    config = _config(tmp_path, resume=True)
    client = MlflowClient(tracking_uri=config.mlflow_tracking_uri)
    exp = client.create_experiment(config.experiment_name)
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("x")
    runs = [{"params": PARAMS, "seed": s, "name": f"final_s{s}"} for s in (0, 1)]
    _logged_run(client, exp, "final_s0", 0, PARAMS, mode="final", checkpoint=str(ckpt))
    missing = str(tmp_path / "gone.ckpt")
    _logged_run(client, exp, "final_s1", 1, PARAMS, mode="final", checkpoint=missing)
    found = hpo._finished_runs(config, "final", runs)
    assert list(found) == ["final_s0"]
    assert found["final_s0"]["checkpoint"] == str(ckpt)


def test_fully_resumed_stage_starts_no_training(tmp_path, monkeypatch):
    """When every run is reusable Ray is never started."""
    import mlflow

    config = _config(tmp_path, resume=True)
    client = MlflowClient(tracking_uri=config.mlflow_tracking_uri)
    exp = client.create_experiment(config.experiment_name)
    runs = [{"params": PARAMS, "seed": 0, "name": "confirm_c0_s0"}]
    _logged_run(client, exp, "confirm_c0_s0", 0, PARAMS)

    def no_ray(*args, **kwargs):
        raise AssertionError("Ray must not start")

    monkeypatch.setattr(hpo.tune, "Tuner", no_ray)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    with mlflow.start_run(experiment_id=exp) as parent:
        results = hpo._grid_runs(
            config, tmp_path / "c.pt", runs, "confirm", parent.info.run_id, "confirm"
        )
    assert results[0]["best_val_objective"] == 0.4
    tags = client.get_run(parent.info.run_id).data.tags
    assert tags["resumed_run_ids"] == results[0]["run_id"]


# Compilation:
def _small_model(tmp_path):
    config = _config(tmp_path)
    return tev.build_model(_bundle(), PARAMS, config, steps_per_epoch=2)


def test_compile_is_off_by_default_and_leaves_the_model_alone(tmp_path):
    """Without the flag nothing is compiled."""
    model = _small_model(tmp_path)
    tev.compile_if_requested(model, _config(tmp_path))
    assert model.model.encoder._compiled_call_impl is None


def test_compile_without_triton_fails_clearly(tmp_path, monkeypatch):
    """A missing Triton is reported, not silently ignored."""
    import torch.utils._triton as triton_utils

    monkeypatch.setattr(triton_utils, "has_triton", lambda: False)
    with pytest.raises(RuntimeError, match="Triton"):
        tev.compile_if_requested(
            _small_model(tmp_path), _config(tmp_path, compile_model=True)
        )


def test_compiling_in_place_keeps_checkpoint_keys(tmp_path):
    """Parameter names do not change, so checkpoints stay loadable."""
    model = _small_model(tmp_path)
    keys = list(model.state_dict())
    model.model.compile_submodules()
    assert list(model.state_dict()) == keys
    assert model.model.encoder._compiled_call_impl is not None


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
