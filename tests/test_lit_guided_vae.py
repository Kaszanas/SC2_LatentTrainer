"""Regression tests pinning the current LitGuidedVAE / suGuidedVAE behaviour."""

import lightning as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE

INPUT_DIM = 12
LATENT_DIM = 6
SUPERVISED_DIM = 2


def _make_model(**overrides) -> LitGuidedVAE:
    kwargs = dict(
        input_dim=INPUT_DIM,
        encoder_hidden_dims=[16, 8],
        supervised_dim=SUPERVISED_DIM,
        latent_dim=LATENT_DIM,
        mean=torch.zeros(1, INPUT_DIM),
        std=torch.ones(1, INPUT_DIM),
    )
    kwargs.update(overrides)
    return LitGuidedVAE(**kwargs)


def _loader(n: int = 16, batch_size: int = 8) -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    data = torch.randn(n, 2, INPUT_DIM, generator=generator)
    labels = torch.randint(0, 2, (n,), generator=generator)
    return DataLoader(TensorDataset(data, labels), batch_size=batch_size)


def _trainer(max_epochs: int = 1) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def test_forward_shapes_pair_input():
    """Pair input gives per-player outputs and one win probability per pair."""
    model = _make_model()
    recon, mu, logvar, cls = model(torch.randn(4, 2, INPUT_DIM))
    assert recon.shape == (4, 2, INPUT_DIM)
    assert mu.shape == logvar.shape == (4, 2, LATENT_DIM)
    assert cls.shape == (4, 1)


def test_encode_flat_input():
    """Encoding ``[B, F]`` works and clamps logvar to ``[-20, 2]``."""
    model = _make_model()
    mu, logvar = model.model.encode(torch.randn(5, INPUT_DIM))
    assert mu.shape == logvar.shape == (5, LATENT_DIM)
    assert logvar.min() >= -20 and logvar.max() <= 2


def test_classifier_reads_only_supervised_dims_in_player_order():
    """The win classifier sees ``cat(p1_sup, p2_sup)`` and ignores the free dims."""
    model = _make_model().eval()
    z = torch.randn(3, 2, LATENT_DIM)

    expected = model.model.classifier(
        torch.cat([z[:, 0, :SUPERVISED_DIM], z[:, 1, :SUPERVISED_DIM]], dim=1)
    )
    torch.testing.assert_close(model.model.cls(z), expected)

    # Free dims must not influence the supervised classifier.
    z_free_changed = z.clone()
    z_free_changed[:, :, SUPERVISED_DIM:] += 5.0
    torch.testing.assert_close(model.model.cls(z_free_changed), expected)


def test_adversary_input_is_free_dims():
    """The adversary receives exactly the free (non-supervised) dims."""
    model = _make_model()
    z = torch.randn(3, 2, LATENT_DIM)
    free = model._slice_free_dims(z)
    assert free.shape == (3, 2, LATENT_DIM - SUPERVISED_DIM)
    torch.testing.assert_close(free, z[:, :, SUPERVISED_DIM:])


def test_prepare_label_shape_and_validation():
    """Labels become ``[B, 1]`` float32; values outside 0/1 raise."""
    label = LitGuidedVAE._prepare_label(torch.tensor([0, 1, 1]))
    assert label.dtype == torch.float32 and label.shape == (3, 1)
    with pytest.raises(ValueError):
        LitGuidedVAE._prepare_label(torch.tensor([0, -1, 1]))


def test_training_runs_three_optimizers_and_updates_weights():
    """Training steps three optimizers, updates both models and logs val metrics."""
    pl.seed_everything(0)
    model = _make_model()
    before_vae = [p.detach().clone() for p in model.model.parameters()]
    before_adv = [p.detach().clone() for p in model.adversarial_classifier.parameters()]

    trainer = _trainer(max_epochs=2)
    trainer.fit(model, train_dataloaders=_loader(), val_dataloaders=_loader())

    assert len(trainer.optimizers) == 3
    assert len(trainer.lr_scheduler_configs) == 3
    assert any(
        not torch.equal(a, b) for a, b in zip(before_vae, model.model.parameters())
    )
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before_adv, model.adversarial_classifier.parameters())
    )
    for key in ("val_loss", "val_vae_loss", "val_cls_loss", "val_acc"):
        assert key in trainer.callback_metrics


def test_checkpoint_round_trip_keeps_normalisation(tmp_path):
    """A checkpoint reload keeps ``mean``/``std`` and the encoder outputs."""
    mean = torch.arange(INPUT_DIM, dtype=torch.float32).unsqueeze(0)
    std = torch.full((1, INPUT_DIM), 3.0)
    model = _make_model(mean=mean, std=std)

    trainer = _trainer()
    trainer.fit(model, train_dataloaders=_loader(), val_dataloaders=_loader())
    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(path)

    loaded = LitGuidedVAE.load_from_checkpoint(path).eval()
    torch.testing.assert_close(loaded.mean, mean)
    torch.testing.assert_close(loaded.std, std)

    x = torch.randn(2, 2, INPUT_DIM)
    torch.testing.assert_close(loaded.model.encode(x)[0], model.eval().model.encode(x)[0])
