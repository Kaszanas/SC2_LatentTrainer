"""Tests for the event-stream guided VAE and its Lightning module."""

import functools

import lightning as pl
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from latent_trainer.features.data_utils import collate_event_sequences
from latent_trainer.features.event_transform import (
    EVENT_TYPE_NAMES,
    N_STAT_CHANNELS,
    RACE_NAMES,
    STAT_CHANNEL_NAMES,
    PlayerEventStreams,
    ReplayEventSample,
)
from latent_trainer.features.preprocess_events import _flatten_split
from latent_trainer.features.type import EventSequenceDataset, EventSplit
from latent_trainer.models.lightning.lit_event_guided_vae import LitEventGuidedVAE

VOCAB = ["<PAD>", "<UNK>", "Marine", "SCV", "Zergling"]
SMALL = dict(
    d_model=32,
    n_heads=4,
    n_layers=2,
    latent_dim=8,
    supervised_dim=3,
    decoder_width=32,
    decoder_layers=2,
    dropout=0.0,
)


def _player(n_ticks: int, n_events: int, race: int, seed: int) -> PlayerEventStreams:
    rng = np.random.default_rng(seed)
    ticks = np.arange(n_ticks, dtype=np.int32)
    loops = np.sort(rng.integers(1, (n_ticks - 1) * 160, n_events)).astype(np.int32)
    return PlayerEventStreams(
        stat_values=rng.gamma(2.0, 50.0, (n_ticks, N_STAT_CHANNELS)).astype(np.float32),
        stat_loops=np.maximum(ticks * 160, 1).astype(np.int32),
        stat_tick=ticks,
        evt_type=rng.integers(1, len(EVENT_TYPE_NAMES), n_events).astype(np.int8),
        evt_unit_local=rng.integers(0, 3, n_events).astype(np.int32),
        evt_loops=loops,
        evt_tick=np.minimum(-(-loops // 160), n_ticks - 1).astype(np.int32),
        race=race,
    )


def _split(n: int = 8, seed: int = 0, n_events: int = 30) -> EventSplit:
    samples = [
        ReplayEventSample(
            players=(
                _player(10 + i, n_events, i % 3, seed + 2 * i),
                _player(10 + i, n_events, (i + 1) % 3, seed + 2 * i + 1),
            ),
            local_vocab=["Marine", "SCV", "Zergling"],
            label=i % 2,
        )
        for i in range(n)
    ]
    return EventSplit(**_flatten_split(samples, VOCAB))


def _module(split: EventSplit, **overrides) -> LitEventGuidedVAE:
    kwargs = dict(
        channel_names=list(STAT_CHANNEL_NAMES),
        unit_vocab=VOCAB,
        event_type_names=list(EVENT_TYPE_NAMES),
        race_names=list(RACE_NAMES),
        **SMALL,
    )
    kwargs.update(overrides)
    return LitEventGuidedVAE(**kwargs).fit_input_transform(split.stat_values)


def _batch(split: EventSplit, indices) -> dict:
    dataset = EventSequenceDataset(split, drop_tail_frac=0.0)
    return collate_event_sequences(
        [dataset[i] for i in indices], max_events_per_slice=8
    )


def _swap(batch: dict) -> dict:
    """Swap the two players in every per-player tensor and flip the label."""
    swapped = {}
    for key, value in batch.items():
        if (
            value.dim() >= 2
            and value.shape[1] == 2
            and key not in ("label", "crop_loop")
        ):
            swapped[key] = value.flip(1)
        else:
            swapped[key] = value
    swapped["label"] = 1 - batch["label"]
    return swapped


def test_output_shapes():
    """Latents per player, a reconstruction per observed slice, one logit per pair."""
    split = _split()
    module = _module(split).eval()
    batch = _batch(split, range(4))
    out = module(batch)
    slices = batch["slice_mask"].shape[2]
    assert out["mu"].shape == out["logvar"].shape == (4, 2, SMALL["latent_dim"])
    assert (
        out["recon"].shape == out["target"].shape == (4, 2, slices, N_STAT_CHANNELS - 1)
    )
    assert out["logit"].shape == (4, 1)


def test_padding_does_not_change_the_encoding():
    """A replay encodes the same alone and next to a longer, busier replay."""
    split = _split()
    module = _module(split).eval()
    with torch.no_grad():
        alone, _ = module.model.encode(_batch(split, [0]))
        padded, _ = module.model.encode(_batch(split, [0, 7]))
    assert torch.allclose(alone[0], padded[0], atol=1e-5)


def test_batches_without_index_keys_encode_identically():
    """Older batches lacking evt_slot/slice_valid_index take the fallback path."""
    split = _split()
    module = _module(split).eval()
    batch = _batch(split, range(4))
    bare = {
        k: v for k, v in batch.items() if k not in ("evt_slot", "slice_valid_index")
    }
    with torch.no_grad():
        fast, _ = module.model.encode(batch)
        fallback, _ = module.model.encode(bare)
    assert torch.equal(fast, fallback)


def test_swapping_players_negates_the_logit_exactly():
    """Slot independence by construction: P(swapped) = 1 - P."""
    split = _split()
    module = _module(split).eval()
    batch = _batch(split, range(6))
    with torch.no_grad():
        mu, _ = module.model.encode(batch)
        mu_swapped, _ = module.model.encode(_swap(batch))
        logit = module.model.cls(mu)
        logit_swapped = module.model.cls(mu_swapped)
    assert torch.allclose(mu_swapped, mu.flip(1), atol=1e-5)
    assert torch.allclose(logit_swapped, -logit, atol=1e-5)


def test_heads_are_antisymmetric_for_any_latent():
    """Both the outcome head and the adversary negate under a swap."""
    module = _module(_split())
    z = torch.randn(5, 2, SMALL["latent_dim"])
    head, adversary = module.model.cls, module.adversarial_classifier
    assert torch.allclose(head(z.flip(1)), -head(z), atol=1e-6)
    free = z[:, :, SMALL["supervised_dim"] :]
    assert torch.allclose(adversary(free.flip(1)), -adversary(free), atol=1e-6)


def test_replays_without_events_encode():
    """Streams with no events still produce finite latents."""
    split = _split(n=2, n_events=0)
    module = _module(split).eval()
    with torch.no_grad():
        mu, logvar = module.model.encode(_batch(split, [0, 1]))
    assert torch.isfinite(mu).all() and torch.isfinite(logvar).all()


def _loader(split: EventSplit) -> DataLoader:
    return DataLoader(
        EventSequenceDataset(split, drop_tail_frac=0.0),
        batch_size=4,
        collate_fn=functools.partial(collate_event_sequences, max_events_per_slice=8),
    )


def _trainer(tmp_path, max_epochs: int = 1) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )


def test_training_updates_all_three_parameter_groups(tmp_path):
    """One epoch changes the encoder, decoder, outcome head and adversary."""
    split = _split()
    module = _module(split, kl_warmup_steps=4)
    before = {k: v.detach().clone() for k, v in module.state_dict().items()}
    trainer = _trainer(tmp_path)
    trainer.fit(module, _loader(split), _loader(split))
    after = module.state_dict()
    for prefix in (
        "model.encoder.",
        "model.decoder.",
        "model.outcome_head.",
        "adversarial_classifier.",
    ):
        changed = [
            k
            for k in before
            if k.startswith(prefix) and not torch.equal(before[k], after[k])
        ]
        assert changed, prefix
    # The fitted input scaling is not trained.
    assert torch.equal(before["model.transform.mean"], after["model.transform.mean"])
    assert int(module.kl_step) == 2  # two training batches
    for name in ("val_loss", "val_vae_loss", "val_cls_loss", "val_acc", "val_kl"):
        assert name in trainer.callback_metrics


def test_checkpoint_round_trip_keeps_vocab_and_scaling(tmp_path):
    """A saved checkpoint restores hyperparameters, scaling and outputs."""
    split = _split()
    module = _module(split, data_config={"slice_ticks": 1})
    trainer = _trainer(tmp_path)
    trainer.fit(module, _loader(split))
    path = tmp_path / "event.ckpt"
    trainer.save_checkpoint(path)

    loaded = LitEventGuidedVAE.load_from_checkpoint(path, map_location="cpu").eval()
    module.eval()
    assert loaded.hparams.unit_vocab == VOCAB
    assert loaded.hparams.data_config == {"slice_ticks": 1}
    assert bool(loaded.model.transform.fitted)
    batch = _batch(split, range(3))
    with torch.no_grad():
        assert torch.allclose(
            loaded.model.encode(batch)[0], module.model.encode(batch)[0]
        )


def test_bf16_autocast_forward_and_loss():
    """The forward pass and the logit losses run under bf16 autocast."""
    split = _split()
    module = _module(split)
    batch = _batch(split, range(4))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        vae_loss, logit, _ = module._vae_forward(batch)
        loss = vae_loss + module._outcome_loss(
            logit, module._prepare_label(batch["label"])
        )
    loss.backward()
    assert torch.isfinite(loss)


def test_invalid_dimensions_are_refused():
    """The supervised dims must leave free dims; heads must divide the width."""
    split = _split()
    with pytest.raises(ValueError, match="supervised_dim"):
        _module(split, supervised_dim=8)
    with pytest.raises(ValueError, match="n_heads"):
        _module(split, n_heads=5)
