from pathlib import Path
from typing import Callable

import click
import lightning as pl

from latent_trainer.features.preprocess_dataset import (
    TransformEnumFunction,
    preprocess_dataset,
)


@click.command(
    help="Pre-process the SC2_Dataset in a JSON format and cache transformed tensors to disk."
)
@click.option(
    "--transform",
    type=TransformEnumFunction(["rich", "averaged_economy"]),
    default="rich",
    show_default=True,
    help="Transform to use: 'rich' (temporal+meta+units, 204 features) or 'averaged_economy' (averaged economy, 39 features)",
)
@click.option(
    "--output-directory",
    type=click.Path(
        file_okay=False,
        writable=True,
        path_type=Path,
        resolve_path=True,
    ),
    default=Path("./data").resolve(),
    show_default=True,
    help="Output directory for cached dataset (default: data/)",
)
@click.option(
    "--single_json_dataset_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    default=Path("H:/sc2egset_merged/sc2egset_merged.json").resolve(),
    help="Path to the single JSON dataset file.",
)
@click.option(
    "--n-workers",
    type=int,
    default=4,
    show_default=True,
    help="Number of parallel workers for processing replays.",
)
def main(
    transform: Callable,
    output_directory: Path,
    single_json_dataset_path: Path,
    n_workers: int,
) -> None:
    """Pre-process Single JSON SC2_Dataset and cache the transformed tensors to drive."""
    transform_name = TransformEnumFunction._TRANSFORM_NAMES[transform]
    output_path = output_directory or Path("data")

    # Consistent seeding for reproducibility of the splits.
    pl.seed_everything(seed=42)

    preprocess_dataset(
        output_directory=output_path,
        single_json_dataset_path=single_json_dataset_path,
        transform_fn=transform,
        transform_name=transform_name,
        n_workers=n_workers,
    )


if __name__ == "__main__":
    main()
