from typing import Callable

import click
from sc2_datasets.transforms.pytorch.economy_vs_outcome import (
    economy_average_vs_outcome,
)

from latent_trainer.features.rich_transform import rich_transform


class TransformEnumFunction(click.Choice):
    """Custom Click Choice type that returns the actual transform function instead of the string name."""

    _TRANSFORM_NAMES: dict[Callable, str] = {
        rich_transform: "rich",
        economy_average_vs_outcome: "averaged_economy",
    }

    def convert(self, value, param, ctx):
        match value:
            case "rich":
                return rich_transform
            case "averaged_economy":
                return economy_average_vs_outcome
            case _:
                raise click.BadParameter(f"Invalid transform choice: {value}")
