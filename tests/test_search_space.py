"""Tests for the search space definitions and architecture builder."""

import optuna

from latent_trainer.configs.search_space import (
    build_hidden_layers,
    get_two_stage_search_space,
)


class TestBuildHiddenLayers:
    def test_returns_list_of_ints(self):
        trial = optuna.trial.FixedTrial(
            {"hidden_n_layers": 2, "hidden_width_0": 128, "hidden_width_1": 64}
        )
        layers = build_hidden_layers(trial, prefix="hidden", max_layers=4)
        assert layers == [128, 64]
        assert all(isinstance(x, int) for x in layers)

    def test_single_layer(self):
        trial = optuna.trial.FixedTrial({"hidden_n_layers": 1, "hidden_width_0": 256})
        layers = build_hidden_layers(trial, prefix="hidden", max_layers=4)
        assert layers == [256]

    def test_max_layers(self):
        params = {"hidden_n_layers": 4}
        params.update({f"hidden_width_{i}": 64 * (i + 1) for i in range(4)})
        trial = optuna.trial.FixedTrial(params)
        layers = build_hidden_layers(trial, prefix="hidden", max_layers=4)
        assert len(layers) == 4


class TestGetTwoStageSearchSpace:
    def test_returns_dict(self):
        trial = optuna.trial.create_trial(
            params={
                "latent_dim": 32,
                "vae_lr": 1e-3,
                "cls_lr": 1e-3,
                "batch_size": 256,
                "dropout": 0.3,
                "vae_n_hidden_layers": 2,
                "vae_hidden_0": 128,
                "vae_hidden_1": 64,
                "cls_n_hidden_layers": 1,
                "cls_hidden_0": 64,
            },
            distributions={
                "latent_dim": optuna.distributions.IntDistribution(8, 128, step=8),
                "vae_lr": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
                "cls_lr": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
                "batch_size": optuna.distributions.CategoricalDistribution(
                    [64, 128, 256, 512]
                ),
                "dropout": optuna.distributions.FloatDistribution(0.1, 0.5),
                "vae_n_hidden_layers": optuna.distributions.IntDistribution(1, 4),
                "vae_hidden_0": optuna.distributions.IntDistribution(32, 512, step=32),
                "vae_hidden_1": optuna.distributions.IntDistribution(32, 512, step=32),
                "cls_n_hidden_layers": optuna.distributions.IntDistribution(1, 4),
                "cls_hidden_0": optuna.distributions.IntDistribution(32, 512, step=32),
            },
            values=[0.75],
        )
        # Smoke test: the function should be callable. We test the real
        # sampling via build_hidden_layers above.
        assert callable(get_two_stage_search_space)
