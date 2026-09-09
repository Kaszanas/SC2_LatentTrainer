"""CLI for the single-replay hosting pipeline.

    python -m latent_trainer_serve build-reference-pack \\
        --model output/checkpoints/.../best.ckpt \\
        --dataset cached_dataset_granular.pt \\
        --output reference_pack.pt

    python -m latent_trainer_serve predict \\
        --replay game.SC2Replay \\
        --model output/checkpoints/.../best.ckpt \\
        --reference-pack reference_pack.pt
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from latent_trainer.inference.extract import DEFAULT_DOCKER_IMAGE
from latent_trainer.inference.pipeline import DEFAULT_CACHE_DIR, predict_replay
from latent_trainer.inference.reference_pack import (
    build_reference_pack,
    save_reference_pack,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@click.group()
def cli() -> None:
    """SC2 Latent Trainer -- single-replay hosting pipeline."""


@cli.command("build-reference-pack")
@click.option(
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    help="Trained guided-VAE checkpoint (.ckpt).",
)
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    help="Cached dataset (.pt) the checkpoint was trained on -- its test "
    "split becomes the win/loss reference pool.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path, resolve_path=True),
    help="Where to save the reference pack.",
)
def cmd_build_reference_pack(model_path: Path, dataset_path: Path, output_path: Path) -> None:
    """One-time offline step: precompute the win/loss latent pool for a checkpoint."""
    pack = build_reference_pack(model_path=model_path, dataset_path=dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_reference_pack(pack, output_path)
    click.echo(
        f"Saved reference pack ({len(pack.labels_tensor)} games) to {output_path}"
    )


@cli.command("predict")
@click.option(
    "--replay",
    "replay_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    help="Path to a .SC2Replay file.",
)
@click.option(
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    help="Trained guided-VAE checkpoint (.ckpt).",
)
@click.option(
    "--reference-pack",
    "reference_pack_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    help="Reference pack built via 'build-reference-pack' for this checkpoint.",
)
@click.option(
    "--player",
    type=click.IntRange(0, 1),
    default=None,
    help="Which player (0 or 1) to analyze. Default: whichever player lost.",
)
@click.option(
    "--strategy",
    type=click.Choice(["linear", "gradient_ascent", "optimal_transport"]),
    default="linear",
    show_default=True,
    help="Path-generation strategy. 'neural_flow' needs a separately "
    "trained flow checkpoint not available for this feature set.",
)
@click.option(
    "--method",
    type=click.Choice(["centroid", "nearest"]),
    default="centroid",
    show_default=True,
    help="Linear-strategy target: centroid of wins, or nearest-k-NN mean.",
)
@click.option("--k-neighbours", type=int, default=5, show_default=True)
@click.option("--n-steps", type=int, default=50, show_default=True)
@click.option("--top-k", type=int, default=15, show_default=True)
@click.option(
    "--docker-image",
    default=DEFAULT_DOCKER_IMAGE,
    show_default=True,
    help="SC2InfoExtractorGo Docker image used to parse the replay "
    "(ignored if --extractor-binary is set).",
)
@click.option(
    "--extractor-binary",
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    default=None,
    help="Run this local SC2InfoExtractorGo binary directly instead of via "
    "Docker -- e.g. the binary baked into this container's own image.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path, resolve_path=True),
    default=DEFAULT_CACHE_DIR,
    show_default=True,
    help="Where to cache extracted features per replay (by content hash).",
)
def cmd_predict(
    replay_path: Path,
    model_path: Path,
    reference_pack_path: Path,
    player: int | None,
    strategy: str,
    method: str,
    k_neighbours: int,
    n_steps: int,
    top_k: int,
    docker_image: str,
    extractor_binary: Path | None,
    cache_dir: Path,
) -> None:
    """Analyze one replay and generate a counterfactual improvement-path report."""
    predict_replay(
        replay_path=replay_path,
        model_path=model_path,
        reference_pack_path=reference_pack_path,
        player=player,
        strategy=strategy,
        method=method,
        k_neighbours=k_neighbours,
        n_steps=n_steps,
        top_k=top_k,
        docker_image=docker_image,
        extractor_binary=extractor_binary,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    cli()
