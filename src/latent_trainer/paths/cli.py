# ---------------------------------------------------------------------------
# CLI group + sub-commands
# ---------------------------------------------------------------------------
from functools import partial

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

_PATH_CHARTING_CLI_COMMON_OPTIONS = [
    click.option(
        "--model",
        default="two_stage_model.pth",
        show_default=True,
        help="Filename of the trained model.",
    ),
    click.option(
        "--cache",
        default="cached_dataset_rich.pt",
        show_default=True,
        help="Filename of the cached dataset.",
    ),
    click.option(
        "--sample-idx",
        type=int,
        default=None,
        help="Dataset index of the game to analyse (default: random).",
    ),
    click.option(
        "--n-steps",
        type=int,
        default=20,
        show_default=True,
        help="Number of waypoints along the path.",
    ),
    click.option(
        "--top-k",
        type=int,
        default=10,
        show_default=True,
        help="Number of top features to display.",
    ),
]


def global_options(fn):
    """Decorator that attaches all global options to a sub-command."""
    for option in reversed(_PATH_CHARTING_CLI_COMMON_OPTIONS):
        fn = option(fn)
    return fn


@click.group()
def cli() -> None:
    """Latent-space improvement path finder for SC2 players.

    Choose a sub-command for the path-finding strategy:

    \b
      linear            Linear interpolation toward centroid / k-NN target.
      gradient-ascent   Gradient ascent with KDE density regularisation.
      optimal-transport Wasserstein-barycentric transport into win cloud.
      geodesic          Shortest path on a kNN latent-space graph.

    Global options (--model, --cache, --sample-idx, --n-steps, --top-k)
    must be placed BEFORE the sub-command name.
    The losing player is detected automatically from the game label.
    """


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
    "--k-neighbours",
    type=int,
    default=5,
    show_default=True,
    help="k for nearest-neighbour target.",
)
def cmd_linear(model, cache, sample_idx, n_steps, top_k, method, k_neighbours):
    """Linear interpolation toward a winning target."""

    print("Loading model and data...")
    vae, classifier, val_X, val_y, norm_mean, norm_std, _ = load_model_and_data(
        model_path=model,
        cache_path=cache,
    )
    labels = val_y.numpy()
    labels_tensor = torch.tensor(labels)
    print(f"  Validation: {len(val_X)}")

    print("Encoding into latent space...")
    latents_p0 = encode_player(vae, val_X[:, 0, :])
    latents_p1 = encode_player(vae, val_X[:, 1, :])

    # Win cloud: label=1 → p0 won; label=0 → p1 won.
    win_latents = torch.where((labels_tensor == 1).unsqueeze(1), latents_p0, latents_p1)
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
        f"  Sample idx: {chosen} (label={int(labels[chosen])}, loser=player {player_idx})"
    )

    if method == "centroid":
        target_z = win_centroid.numpy()
        print("  Target: centroid")
    else:
        target_z = nearest_winning_target(sample_z, win_latents, k=k_neighbours).numpy()
        print(f"  Target: nearest (k={k_neighbours})")

    path_z_np = path_linear(
        z_start=sample_z.detach().cpu().numpy(),
        target_z=target_z,
        n_waypoints=n_steps,
    )
    run_path_charting_pipeline(
        model=model,
        cache=cache,
        chosen=int(chosen),
        player_idx=player_idx,
        n_steps=n_steps,
        top_k=top_k,
        strategy="linear",
        path_z_np=path_z_np,
    )


@cli.command("gradient-ascent")
@global_options
@click.option(
    "--ga-steps",
    type=int,
    default=500,
    show_default=True,
    help="Max gradient ascent steps.",
)
@click.option(
    "--ga-lr", type=float, default=0.02, show_default=True, help="Learning rate."
)
@click.option(
    "--ga-momentum", type=float, default=0.9, show_default=True, help="Momentum."
)
@click.option(
    "--density-weight",
    type=float,
    default=0.3,
    show_default=True,
    help="Weight of the KDE density gradient.",
)
@click.option(
    "--kde-bandwidth",
    type=float,
    default=0.5,
    show_default=True,
    help="Gaussian KDE bandwidth.",
)
@click.option(
    "--convergence-threshold",
    type=float,
    default=0.95,
    show_default=True,
    help="P(win) threshold for early stopping.",
)
def cmd_gradient_ascent(
    model,
    cache,
    sample_idx,
    n_steps,
    top_k,
    ga_steps,
    ga_lr,
    ga_momentum,
    density_weight,
    kde_bandwidth,
    convergence_threshold,
):
    """Gradient ascent on P(win) regularised by a KDE density prior."""

    print("Loading model and data...")
    vae, classifier, val_X, val_y, norm_mean, norm_std, _ = load_model_and_data(
        model, cache
    )
    labels = val_y.numpy()
    labels_tensor = torch.tensor(labels)
    print(f"  Validation: {len(val_X)}")

    print("Encoding into latent space...")
    latents_p0 = encode_player(vae, val_X[:, 0, :])
    latents_p1 = encode_player(vae, val_X[:, 1, :])

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
        f"  Sample idx: {chosen} (label={int(labels[chosen])}, loser=player {player_idx})"
    )

    # All loser latents as reference distribution for KDE.
    all_loss_latents = torch.where(
        (labels_tensor == 0).unsqueeze(1), latents_p0, latents_p1
    )

    score_fn = partial(
        opponent_aware_score,
        classifier=classifier,
        opponent_z=opponent_z,
        player_idx=player_idx,
    )
    logit_fn = partial(
        opponent_aware_logit,
        classifier=classifier,
        opponent_z=opponent_z,
        player_idx=player_idx,
    )

    print("Running gradient ascent with KDE density regularisation...")
    path_z_np = path_gradient_ascent(
        sample_z.detach().cpu().numpy(),
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
        model=model,
        cache=cache,
        chosen=int(chosen),
        player_idx=player_idx,
        n_steps=n_steps,
        top_k=top_k,
        strategy="gradient_ascent",
        path_z_np=path_z_np,
    )


@cli.command("optimal-transport")
@global_options
@click.option(
    "--ot-reg",
    type=float,
    default=0.0,
    show_default=True,
    help="Entropic regularisation (0 = exact EMD).",
)
def cmd_optimal_transport(model, cache, sample_idx, n_steps, top_k, ot_reg):
    """Wasserstein-barycentric path into the winning distribution."""

    print("Loading model and data...")
    vae, classifier, val_X, val_y, norm_mean, norm_std, _ = load_model_and_data(
        model, cache
    )
    labels = val_y.numpy()
    labels_tensor = torch.tensor(labels)
    print(f"  Validation: {len(val_X)}")

    print("Encoding into latent space...")
    latents_p0 = encode_player(vae, val_X[:, 0, :])
    latents_p1 = encode_player(vae, val_X[:, 1, :])

    # Win cloud: label=1 → p0 won; label=0 → p1 won.
    win_latents = torch.where((labels_tensor == 1).unsqueeze(1), latents_p0, latents_p1)

    n = len(labels)
    chosen = (
        sample_idx
        if (sample_idx is not None and sample_idx < n)
        else torch.randint(n, (1,)).item()
    )
    player_idx = int(labels[chosen])  # 0 if p0 lost, 1 if p1 lost
    sample_z = latents_p0[chosen] if player_idx == 0 else latents_p1[chosen]
    print(
        f"  Sample idx: {chosen} (label={int(labels[chosen])}, loser=player {player_idx})"
    )

    print("Computing optimal transport path...")
    path_z_np = path_optimal_transport(
        sample_z.detach().cpu().numpy(),
        win_latents.detach().cpu().numpy(),
        reg=ot_reg,
        n_waypoints=n_steps,
    )
    run_path_charting_pipeline(
        model=model,
        cache=cache,
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
def cmd_geodesic(model, cache, sample_idx, n_steps, top_k, geodesic_k):
    """Shortest path on a kNN latent-space graph."""

    print("Loading model and data...")
    vae, classifier, val_X, val_y, norm_mean, norm_std, _ = load_model_and_data(
        model, cache
    )
    labels = val_y.numpy()
    labels_tensor = torch.tensor(labels)
    print(f"  Validation: {len(val_X)}")

    print("Encoding into latent space...")
    latents_p0 = encode_player(vae, val_X[:, 0, :])
    latents_p1 = encode_player(vae, val_X[:, 1, :])

    # Win cloud: label=1 → p0 won; label=0 → p1 won.
    win_latents = torch.where((labels_tensor == 1).unsqueeze(1), latents_p0, latents_p1)
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
        f"  Sample idx: {chosen} (label={int(labels[chosen])}, loser=player {player_idx})"
    )

    print("Computing geodesic path on kNN graph...")
    path_z_np = path_geodesic(
        sample_z.detach().cpu().numpy(),
        win_latents.detach().cpu().numpy(),
        all_latents.detach().cpu().numpy(),
        k=geodesic_k,
        n_waypoints=n_steps,
    )
    run_path_charting_pipeline(
        model=model,
        cache=cache,
        chosen=int(chosen),
        player_idx=player_idx,
        n_steps=n_steps,
        top_k=top_k,
        strategy="geodesic",
        path_z_np=path_z_np,
    )
