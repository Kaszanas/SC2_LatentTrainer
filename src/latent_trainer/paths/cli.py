# ---------------------------------------------------------------------------
# CLI group + sub-commands
# ---------------------------------------------------------------------------
from functools import partial
from pathlib import Path

import click
import torch

from latent_trainer.paths.data import (
    encode_player,
    load_model_and_data,
    nearest_winning_target,
    opponent_aware_logit,
    opponent_aware_score,
)
from latent_trainer.paths.pipeline import run_path_charting_pipeline
from latent_trainer.paths.strategies import path_linear
from latent_trainer.paths.strategies.geodesic import path_geodesic
from latent_trainer.paths.strategies.gradient_ascent import path_gradient_ascent
from latent_trainer.paths.strategies.optimal_transport import path_optimal_transport
from latent_trainer.settings import DATA_DIR

PATH_CHARTING_CLI_COMMON_OPTIONS = [
    click.option(
        "--model_path",
        help="Filename of the trained model.",
        type=click.Path(
            exists=True,
            dir_okay=False,
            path_type=Path,
            resolve_path=True,
        ),
    ),
    click.option(
        "--dataset_filename",
        default="cached_dataset_rich.pt",
        show_default=True,
        help="Filename of the cached dataset placed in the DATA_DIR (set in settings.py).",
        type=str,
    ),
    click.option(
        "--sample_idx",
        type=int,
        default=None,
        help="Index of the game within the dataset for which to find the improvement path. If not provided a random losing sample will be chosen.",
    ),
    click.option(
        "--n_steps",
        type=int,
        default=20,
        show_default=True,
        help="Number of waypoints along the path.",
    ),
    click.option(
        "--top_k",
        type=int,
        default=10,
        show_default=True,
        help="Number of features requiring improvement to display.",
    ),
]


def global_options(fn):
    """Decorator that attaches all global options to a sub-command."""
    for option in reversed(PATH_CHARTING_CLI_COMMON_OPTIONS):
        fn = option(fn)
    return fn


@click.group()
def cli() -> None:
    """Latent-space improvement path finder for SC2 players."""


@cli.command("linear")
@global_options
@click.option(
    "--method",
    type=click.Choice(["centroid", "nearest"]),
    default="centroid",
    show_default=True,
    help="Target selection: centroid of wins or nearest k-NN mean.",
)
@click.option(
    "--k_neighbours",
    type=int,
    default=5,
    show_default=True,
    help="k for nearest-neighbour target.",
)
def cmd_linear(
    model_path: Path,
    dataset_filename: str,
    sample_idx: int | None,
    n_steps: int,
    top_k: int,
    method: str,
    k_neighbours: int,
):
    """Linear interpolation toward a winning target."""

    print("Loading model and data...")
    vae, val_X, val_y, norm_mean, norm_std, _ = load_model_and_data(
        model_path=model_path,
        cached_dataset_filepath=DATA_DIR / dataset_filename,
    )
    labels = val_y.numpy()
    labels_tensor = torch.tensor(labels)
    print(f"Validation: {len(val_X)}")

    print("Encoding into latent space...")
    latents_p0 = encode_player(vae=vae, data=val_X[:, 0, :])
    latents_p1 = encode_player(vae=vae, data=val_X[:, 1, :])

    # Win cloud: label=1 → p0 won; label=0 → p1 won.
    win_latents = torch.where(
        (labels_tensor == 1).unsqueeze(1),
        latents_p0,
        latents_p1,
    )
    win_centroid = win_latents.mean(dim=0)

    n = len(labels)
    chosen = (
        sample_idx
        if (sample_idx is not None and sample_idx < n)
        else torch.randint(n, (1,)).item()
    )
    player_idx = int(labels[chosen])  # 0 if p0 lost, 1 if p1 lost
    sample_z = latents_p0[chosen] if player_idx == 0 else latents_p1[chosen]
    print(
        f"Sample idx: {chosen} (label={int(labels[chosen])}, loser=player {player_idx})"
    )

    match method:
        case "centroid":
            print("  Target: centroid")
            target_z = win_centroid.numpy()
        case "nearest":
            print(f"  Target: nearest (k={k_neighbours})")
            target_z = nearest_winning_target(
                sample_z=sample_z,
                win_latents=win_latents,
                k=k_neighbours,
            ).numpy()
        case _:
            raise click.ClickException(f"Invalid method: {method}")

    path_z_np = path_linear(
        z_start=sample_z.detach().cpu().numpy(),
        target_z=target_z,
        n_waypoints=n_steps,
    )
    run_path_charting_pipeline(
        model=model_path,
        cache=dataset_filename,
        chosen=int(chosen),
        player_idx=player_idx,
        n_steps=n_steps,
        top_k=top_k,
        strategy="linear",
        path_z_np=path_z_np,
    )


@cli.command("gradient_ascent")
@global_options
@click.option(
    "--ga_steps",
    type=int,
    default=500,
    show_default=True,
    help="Max gradient ascent steps.",
)
@click.option(
    "--ga_lr",
    type=float,
    default=0.02,
    show_default=True,
    help="Learning rate.",
)
@click.option(
    "--ga_momentum",
    type=float,
    default=0.9,
    show_default=True,
    help="Momentum.",
)
@click.option(
    "--density_weight",
    type=float,
    default=0.3,
    show_default=True,
    help="Weight of the KDE density gradient.",
)
@click.option(
    "--kde_bandwidth",
    type=float,
    default=0.5,
    show_default=True,
    help="Gaussian KDE bandwidth.",
)
@click.option(
    "--convergence_threshold",
    type=float,
    default=0.95,
    show_default=True,
    help="P(win) threshold for early stopping.",
)
def cmd_gradient_ascent(
    model_path: Path,
    dataset_filename: str,
    sample_idx: int | None,
    n_steps: int,
    top_k: int,
    ga_steps: int,
    ga_lr: float,
    ga_momentum: float,
    density_weight: float,
    kde_bandwidth: float,
    convergence_threshold: float,
):
    """Gradient ascent on P(win) regularised by a KDE density prior."""

    print("Loading model and data...")
    vae, val_X, val_y, norm_mean, norm_std, _ = load_model_and_data(
        model_path=model_path,
        cached_dataset_filepath=DATA_DIR / dataset_filename,
    )
    labels = val_y.numpy()
    labels_tensor = torch.tensor(labels)
    print(f"Validation: {len(val_X)}")

    print("Encoding into latent space...")
    latents_p0 = encode_player(vae=vae, data=val_X[:, 0, :])
    latents_p1 = encode_player(vae=vae, data=val_X[:, 1, :])

    n = len(labels)
    chosen = (
        sample_idx
        if (sample_idx is not None and sample_idx < n)
        else torch.randint(n, (1,)).item()
    )
    player_idx = int(labels[chosen])  # 0 if p0 lost, 1 if p1 lost
    sample_z = latents_p0[chosen] if player_idx == 0 else latents_p1[chosen]
    opponent_z = latents_p1[chosen] if player_idx == 0 else latents_p0[chosen]
    print(
        f"Sample idx: {chosen} (label={int(labels[chosen])}, loser=player {player_idx})"
    )

    # All loser latents as reference distribution for KDE.
    all_loss_latents = torch.where(
        (labels_tensor == 0).unsqueeze(1),
        latents_p0,
        latents_p1,
    )

    score_fn = partial(
        opponent_aware_score,
        classifier=vae.model.classifier,
        opponent_z=opponent_z,
        player_idx=player_idx,
    )
    logit_fn = partial(
        opponent_aware_logit,
        classifier=vae.model.classifier,
        opponent_z=opponent_z,
        player_idx=player_idx,
    )

    print("Running gradient ascent with KDE density regularisation...")
    path_z_np = path_gradient_ascent(
        z_start=sample_z.detach().cpu().numpy(),
        score_fn=score_fn,
        logit_fn=logit_fn,
        Z_all=all_loss_latents.detach().cpu().numpy(),
        steps=ga_steps,
        lr=ga_lr,
        momentum=ga_momentum,
        density_weight=density_weight,
        kde_bandwidth=kde_bandwidth,
        n_waypoints=n_steps,
        convergence_threshold=convergence_threshold,
    )
    run_path_charting_pipeline(
        model=model_path,
        cache=dataset_filename,
        chosen=int(chosen),
        player_idx=player_idx,
        n_steps=n_steps,
        top_k=top_k,
        strategy="gradient_ascent",
        path_z_np=path_z_np,
    )


@cli.command("optimal_transport")
@global_options
@click.option(
    "--ot_reg",
    type=float,
    default=0.0,
    show_default=True,
    help="Entropic regularisation (0 = exact EMD).",
)
def cmd_optimal_transport(
    model_path: Path,
    dataset_filename: str,
    sample_idx: int,
    n_steps: int,
    top_k: int,
    ot_reg: float,
):
    """Wasserstein-barycentric path into the winning distribution."""

    print("Loading model and data...")
    vae, val_X, val_y, norm_mean, norm_std, _ = load_model_and_data(
        model_path=model_path,
        cached_dataset_filepath=DATA_DIR / dataset_filename,
    )
    labels = val_y.numpy()
    labels_tensor = torch.tensor(labels)
    print(f"Validation: {len(val_X)}")

    print("Encoding into latent space...")
    latents_p0 = encode_player(vae=vae, data=val_X[:, 0, :])
    latents_p1 = encode_player(vae=vae, data=val_X[:, 1, :])

    # Win cloud: label=1 → p0 won; label=0 → p1 won.
    win_latents = torch.where(
        (labels_tensor == 1).unsqueeze(1),
        latents_p0,
        latents_p1,
    )

    n = len(labels)
    chosen = (
        sample_idx
        if (sample_idx is not None and sample_idx < n)
        else torch.randint(n, (1,)).item()
    )
    player_idx = int(labels[chosen])  # 0 if p0 lost, 1 if p1 lost
    sample_z = latents_p0[chosen] if player_idx == 0 else latents_p1[chosen]
    print(
        f"Sample idx: {chosen} (label={int(labels[chosen])}, loser=player {player_idx})"
    )

    print("Computing optimal transport path...")
    path_z_np = path_optimal_transport(
        z_start=sample_z.detach().cpu().numpy(),
        Z_win=win_latents.detach().cpu().numpy(),
        reg=ot_reg,
        n_waypoints=n_steps,
    )
    if not torch.isfinite(torch.as_tensor(path_z_np)).all():
        raise click.ClickException(
            "Optimal transport produced non-finite values (NaN/Inf). "
            "Try '--ot-reg 0.0' for exact EMD or a larger regularization "
            "such as '--ot-reg 0.05' or '--ot-reg 0.1'."
        )
    run_path_charting_pipeline(
        model=model_path,
        cache=dataset_filename,
        chosen=int(chosen),
        player_idx=player_idx,
        n_steps=n_steps,
        top_k=top_k,
        strategy="optimal_transport",
        path_z_np=path_z_np,
    )


@cli.command("geodesic")
@global_options
@click.option(
    "--geodesic-k",
    type=int,
    default=12,
    show_default=True,
    help="Number of neighbours for the kNN graph.",
)
def cmd_geodesic(
    model_path: Path,
    dataset_filename: str,
    sample_idx: int,
    n_steps: int,
    top_k: int,
    geodesic_k: int,
):
    """Shortest path on a kNN latent-space graph."""

    print("Loading model and data...")
    vae, val_X, val_y, norm_mean, norm_std, _ = load_model_and_data(
        model_path=model_path,
        cached_dataset_filepath=DATA_DIR / dataset_filename,
    )
    labels = val_y.numpy()
    labels_tensor = torch.tensor(labels)
    print(f"Validation: {len(val_X)}")

    print("Encoding into latent space...")
    latents_p0 = encode_player(vae=vae, data=val_X[:, 0, :])
    latents_p1 = encode_player(vae=vae, data=val_X[:, 1, :])

    # Win cloud: label=1 → p0 won; label=0 → p1 won.
    win_latents = torch.where(
        (labels_tensor == 1).unsqueeze(1),
        latents_p0,
        latents_p1,
    )
    # All latents for kNN graph.
    all_latents = torch.cat([latents_p0, latents_p1], dim=0)

    n = len(labels)
    chosen = (
        sample_idx
        if (sample_idx is not None and sample_idx < n)
        else torch.randint(n, (1,)).item()
    )
    player_idx = int(labels[chosen])  # 0 if p0 lost, 1 if p1 lost
    sample_z = latents_p0[chosen] if player_idx == 0 else latents_p1[chosen]
    print(
        f"Sample idx: {chosen} (label={int(labels[chosen])}, loser=player {player_idx})"
    )

    print("Computing geodesic path on kNN graph...")
    path_z_np = path_geodesic(
        z_start=sample_z.detach().cpu().numpy(),
        Z_win=win_latents.detach().cpu().numpy(),
        Z_all=all_latents.detach().cpu().numpy(),
        k=geodesic_k,
        n_waypoints=n_steps,
    )
    run_path_charting_pipeline(
        model=model_path,
        cache=dataset_filename,
        chosen=int(chosen),
        player_idx=player_idx,
        n_steps=n_steps,
        top_k=top_k,
        strategy="geodesic",
        path_z_np=path_z_np,
    )
