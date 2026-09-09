import functools
import logging
from pathlib import Path
from typing import Callable

import click
import lightning as pl

from latent_trainer.features.preprocess_dataset import (
    TransformEnumFunction,
    preprocess_dataset_chunked_profile,
    preprocess_dataset_test_only,
)
from latent_trainer.features.rich_transform import ALL_FEATURE_BLOCKS, rich_transform
from latent_trainer.settings import DATA_DIR, LOGGING_FORMAT, SEED


@click.command(
    help="Pre-process the SC2_Dataset in a JSON format and cache transformed tensors to disk."
)
@click.option(
    "--transform",
    type=TransformEnumFunction(["rich", "averaged_economy", "granular"]),
    default="rich",
    show_default=True,
    help=(
        "Transform to use: 'rich' (temporal+meta, 196 features/player), "
        "'averaged_economy' (averaged economy, 39 features/player), or "
        "'granular' (20 x 5%-of-game bins + meta, 781 features/player -- "
        "for leakage-localization sweeps; slice the cache at training time "
        "instead of reprocessing per sweep point)."
    ),
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
@click.option(
    "--n_samples",
    type=int,
    default=0,
    show_default=True,
    help="Total games to randomly sample before processing. 0 = use all.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducible sampling.",
)
@click.option(
    "--test_only",
    is_flag=True,
    default=False,
    help=(
        "Extract a test-only dataset. Pools all split indices, samples n_samples from "
        "the full pool, and stores everything in test_features/test_labels with empty "
        "train/val. Output: cached_dataset_<transform>_test_<n>.pt. "
        "Use this for path-charting evaluation on an independent dataset."
    ),
)
@click.option(
    "--feature_blocks",
    default=None,
    help=(
        "Comma-separated subset of the 'rich' transform's feature blocks to include: "
        f"{','.join(ALL_FEATURE_BLOCKS)}. Only applies to '--transform rich'. "
        "Default (omit this flag): all blocks, i.e. unchanged 196-feature/player output. "
        "Use e.g. 'early,mid,meta' to drop the leakage-prone final/late/econDelta blocks "
        "for a leakage-audit cache; the output filename gets a matching suffix."
    ),
)
def main(
    transform: Callable,
    single_json_dataset_path: Path,
    n_workers: int,
    n_samples: int,
    seed: int,
    test_only: bool,
    feature_blocks: str | None,
) -> None:
    """Pre-process Single JSON SC2_Dataset and cache the transformed tensors to drive."""
    transform_name = TransformEnumFunction._TRANSFORM_NAMES[transform]

    if feature_blocks is not None:
        if transform is not rich_transform:
            raise click.BadParameter(
                "--feature_blocks only applies to '--transform rich'."
            )
        blocks = tuple(b.strip() for b in feature_blocks.split(","))
        unknown = set(blocks) - set(ALL_FEATURE_BLOCKS)
        if unknown:
            raise click.BadParameter(
                f"Unknown feature block(s) {sorted(unknown)}; "
                f"valid blocks: {ALL_FEATURE_BLOCKS}"
            )
        transform = functools.partial(rich_transform, feature_blocks=blocks)
        transform_name = f"{transform_name}_{'-'.join(blocks)}"

    logging.basicConfig(
        level=logging.INFO,
        format=LOGGING_FORMAT,
    )

    pl.seed_everything(SEED)

    try:
        if test_only:
            preprocess_dataset_test_only(
                output_directory=DATA_DIR,
                single_json_dataset_path=single_json_dataset_path,
                transform_fn=transform,
                transform_name=transform_name,
                n_workers=n_workers,
                n_samples=n_samples,
                seed=seed,
            )
        else:
            preprocess_dataset_chunked_profile(
                output_directory=DATA_DIR,
                single_json_dataset_path=single_json_dataset_path,
                transform_fn=transform,
                transform_name=transform_name,
                n_workers=n_workers,
                n_samples=n_samples,
                seed=seed,
            )
    except Exception as e:
        logging.error(f"Error during dataset preprocessing: {e}")


if __name__ == "__main__":
    main()
