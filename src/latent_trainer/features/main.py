import logging
from pathlib import Path
from typing import Callable

import click
import lightning as pl

from latent_trainer.features.available_transforms import TransformEnumFunction
from latent_trainer.features.preprocess_dataset import (
    preprocess_dataset_chunked,
)
from latent_trainer.settings import DATA_DIR, SEED


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
    "--single_json_dataset_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    default=Path("H:/sc2egset_merged/sc2egset_merged.json").resolve(),
    help="Path to the single JSON dataset file.",
)
@click.option(
    "--n_workers",
    type=int,
    default=24,
    show_default=True,
    help="Number of parallel workers for processing replays.",
)
def main(
    transform: Callable,
    single_json_dataset_path: Path,
    n_workers: int,
) -> None:
    """Pre-process Single JSON SC2_Dataset and cache the transformed tensors to drive."""
    transform_name = TransformEnumFunction._TRANSFORM_NAMES[transform]

    # profiler = cProfile.Profile()
    # profiler.enable()

    pl.seed_everything(SEED)

    try:
        # preprocess_dataset(
        #     output_directory=DATA_DIR,
        #     single_json_dataset_path=single_json_dataset_path,
        #     transform_fn=transform,
        #     transform_name=transform_name,
        #     n_workers=n_workers,
        # )

        # debug_preprocess_dataset(
        #     single_json_dataset_path=single_json_dataset_path,
        #     transform_fn=transform,
        # )

        preprocess_dataset_chunked(
            output_directory=DATA_DIR,
            single_json_dataset_path=single_json_dataset_path,
            transform_fn=transform,
            transform_name=transform_name,
            n_workers=n_workers,
        )
    except Exception as e:
        logging.error(f"Error during dataset preprocessing: {e}")
    # finally:
    # profiler.disable()

    # profile_output = Path("preprocess_dataset_profile_no_chunk.prof").resolve()

    # if profile_output is not None:
    #     profiler.dump_stats(str(profile_output))
    #     click.echo(f"Saved cProfile stats to: {profile_output}")

    # stats_output = io.StringIO()
    # stats = pstats.Stats(profiler, stream=stats_output).sort_stats("cumulative")
    # stats.print_stats(30)
    # click.echo("cProfile results (top 30 by cumulative time):")
    # click.echo(stats_output.getvalue())


if __name__ == "__main__":
    main()
