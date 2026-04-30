# ---------------------------------------------------------------------------
# CLI group + sub-commands
# ---------------------------------------------------------------------------
from functools import partial
from pathlib import Path

import click
import torch

from latent_trainer.paths.data import (
    compute_win_latents,
    nearest_winning_target,
    opponent_aware_logit,
    opponent_aware_score,
    prepare_path_context,
)
from latent_trainer.paths.flow import LitOTFlowMatching
from latent_trainer.paths.options import (
    global_options,
)
from latent_trainer.paths.pipeline import run_path_charting_pipeline
from latent_trainer.paths.strategies import path_linear
from latent_trainer.paths.strategies.geodesic import path_geodesic
from latent_trainer.paths.strategies.gradient_ascent import path_gradient_ascent
from latent_trainer.paths.strategies.neural_flow import path_neural_flow
from latent_trainer.paths.strategies.optimal_transport import path_optimal_transport
from latent_trainer.settings import DATA_DIR


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
    path_context = prepare_path_context(
        model_path=model_path,
        dataset_path=DATA_DIR / dataset_filename,
        sample_idx=sample_idx,
    )
    win_latents = compute_win_latents(
        labels_tensor=path_context.labels_tensor,
        latents_p0=path_context.latents_p0,
        latents_p1=path_context.latents_p1,
    )

    win_centroid = win_latents.mean(dim=0)

    match method:
        case "centroid":
            print("Target: centroid")
            target_z = win_centroid.cpu().numpy()
        case "nearest":
            print(f"Target: nearest (k={k_neighbours})")
            target_z = (
                nearest_winning_target(
                    sample_z=path_context.sample_z,
                    win_latents=win_latents,
                    k=k_neighbours,
                )
                .cpu()
                .numpy()
            )
        case _:
            raise click.ClickException(f"Invalid method: {method}")

    path_z_np = path_linear(
        z_start=path_context.sample_z.detach().cpu().numpy(),
        z_target=target_z,
        n_waypoints=n_steps,
    )
    run_path_charting_pipeline(
        path_context=path_context,
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
    default=1000,
    show_default=True,
    help="Max gradient ascent steps.",
)
@click.option(
    "--ga_lr",
    type=float,
    default=0.005,
    show_default=True,
    help="Learning rate.",
)
@click.option(
    "--ga_momentum",
    type=float,
    default=0.5,
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
    path_context = prepare_path_context(
        model_path=model_path,
        dataset_path=DATA_DIR / dataset_filename,
        sample_idx=sample_idx,
    )
    opponent_z = (
        path_context.latents_p1[path_context.chosen]
        if path_context.player_idx == 0
        else path_context.latents_p0[path_context.chosen]
    )

    win_latents = compute_win_latents(
        labels_tensor=path_context.labels_tensor,
        latents_p0=path_context.latents_p0,
        latents_p1=path_context.latents_p1,
    )

    score_fn = partial(
        opponent_aware_score,
        guided_vae=path_context.guided_vae,
        opponent_z=opponent_z,
        player_idx=path_context.player_idx,
    )
    logit_fn = partial(
        opponent_aware_logit,
        guided_vae=path_context.guided_vae,
        opponent_z=opponent_z,
        player_idx=path_context.player_idx,
    )

    print("Running gradient ascent with KDE density regularisation...")
    path_z_np = path_gradient_ascent(
        z_start=path_context.sample_z.detach().cpu().numpy(),
        score_fn=score_fn,
        logit_fn=logit_fn,
        Z_all=win_latents.detach().cpu().numpy(),
        steps=ga_steps,
        lr=ga_lr,
        momentum=ga_momentum,
        density_weight=density_weight,
        kde_bandwidth=kde_bandwidth,
        n_waypoints=n_steps,
        convergence_threshold=convergence_threshold,
    )
    run_path_charting_pipeline(
        path_context=path_context,
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
    default=0.01,
    show_default=True,
    help="Entropic regularisation (0 = exact EMD).",
)
def cmd_optimal_transport(
    model_path: Path,
    dataset_filename: str,
    sample_idx: int | None,
    n_steps: int,
    top_k: int,
    ot_reg: float,
):
    """Wasserstein-barycentric path into the winning distribution."""
    path_context = prepare_path_context(
        model_path=model_path,
        dataset_path=DATA_DIR / dataset_filename,
        sample_idx=sample_idx,
    )
    win_latents = compute_win_latents(
        labels_tensor=path_context.labels_tensor,
        latents_p0=path_context.latents_p0,
        latents_p1=path_context.latents_p1,
    )

    print("Computing optimal transport path...")
    path_z_np = path_optimal_transport(
        z_start=path_context.sample_z.detach().cpu().numpy(),
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
        path_context=path_context,
        n_steps=n_steps,
        top_k=top_k,
        strategy="optimal_transport",
        path_z_np=path_z_np,
    )


@cli.command("geodesic")
@global_options
@click.option(
    "--geodesic_k",
    type=int,
    default=12,
    show_default=True,
    help="Number of neighbours for the kNN graph.",
)
def cmd_geodesic(
    model_path: Path,
    dataset_filename: str,
    sample_idx: int | None,
    n_steps: int,
    top_k: int,
    geodesic_k: int,
):
    """Shortest path on a kNN latent-space graph."""
    path_context = prepare_path_context(
        model_path=model_path,
        dataset_path=DATA_DIR / dataset_filename,
        sample_idx=sample_idx,
    )
    win_latents = compute_win_latents(
        labels_tensor=path_context.labels_tensor,
        latents_p0=path_context.latents_p0,
        latents_p1=path_context.latents_p1,
    )
    all_latents = torch.cat([path_context.latents_p0, path_context.latents_p1], dim=0)

    print("Computing geodesic path on kNN graph...")
    path_z_np = path_geodesic(
        z_start=path_context.sample_z.detach().cpu().numpy(),
        Z_win=win_latents.detach().cpu().numpy(),
        Z_all=all_latents.detach().cpu().numpy(),
        k=geodesic_k,
        n_waypoints=n_steps,
    )
    run_path_charting_pipeline(
        path_context=path_context,
        n_steps=n_steps,
        top_k=top_k,
        strategy="geodesic",
        path_z_np=path_z_np,
    )


@cli.command("neural_flow")
@global_options
@click.option(
    "--flow_checkpoint",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    help="Path to the trained OT-Flow Matching checkpoint (.ckpt).",
)
@click.option(
    "--guidance_scale",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "Classifier-guidance strength added to each Euler step. "
        "0 = pure flow (default). Try 0.05–0.5 to steer toward high-P(win) latents."
    ),
)
@click.option(
    "--diagnose",
    is_flag=True,
    default=False,
    help="Print diagnostics: reachable P(win) ceiling, path P(win) trace, flow endpoint distance.",
)
def cmd_neural_flow(
    model_path: Path,
    dataset_filename: str,
    sample_idx: int | None,
    n_steps: int,
    top_k: int,
    flow_checkpoint: Path,
    guidance_scale: float,
    diagnose: bool,
):
    """Counterfactual path via OT-Flow Matching (learned velocity field)."""
    path_context = prepare_path_context(
        model_path=model_path,
        dataset_path=DATA_DIR / dataset_filename,
        sample_idx=sample_idx,
    )

    opponent_z = (
        path_context.latents_p1[path_context.chosen]
        if path_context.player_idx == 0
        else path_context.latents_p0[path_context.chosen]
    )
    score_fn = partial(
        opponent_aware_score,
        guided_vae=path_context.guided_vae,
        opponent_z=opponent_z,
        player_idx=path_context.player_idx,
    )

    if diagnose:
        _print_diagnostics(path_context=path_context, score_fn=score_fn)

    print(f"Loading flow model from {flow_checkpoint}...")
    flow_model = LitOTFlowMatching.load_from_checkpoint(flow_checkpoint)
    flow_model.eval()
    flow_model = flow_model.to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Integrating velocity field...")
    z_start_np = path_context.sample_z.detach().cpu().numpy()
    if guidance_scale > 0.0:
        print(f"  Using classifier guidance (scale={guidance_scale})")
        path_z_np = flow_model.predict_path_guided(
            z_start=z_start_np,
            score_fn=score_fn,
            guidance_scale=guidance_scale,
            steps=n_steps - 1,
        )
    else:
        path_z_np = path_neural_flow(
            z_start=z_start_np,
            flow_model=flow_model,
            n_waypoints=n_steps,
        )

    if diagnose:
        _print_path_trace(
            path_z_np=path_z_np, score_fn=score_fn, path_context=path_context
        )

    run_path_charting_pipeline(
        path_context=path_context,
        n_steps=n_steps,
        top_k=top_k,
        strategy="neural_flow",
        path_z_np=path_z_np,
    )


def _print_diagnostics(path_context, score_fn) -> None:
    """Report the P(win) ceiling and opponent context for the chosen sample."""
    import numpy as np

    win_latents = compute_win_latents(
        labels_tensor=path_context.labels_tensor,
        latents_p0=path_context.latents_p0,
        latents_p1=path_context.latents_p1,
    )

    with torch.no_grad():
        win_scores = score_fn(win_latents).cpu().numpy()

    pct_above_50 = (win_scores > 0.5).mean() * 100
    print("\n" + "═" * 60)
    print("  DIAGNOSTICS")
    print("═" * 60)
    print(f"  Winning latents vs. this opponent (N={len(win_scores)}):")
    print(f"    max  P(win) = {win_scores.max():.3f}")
    print(f"    mean P(win) = {win_scores.mean():.3f}")
    print(f"    med  P(win) = {np.median(win_scores):.3f}")
    print(f"    min  P(win) = {win_scores.min():.3f}")
    print(f"    % > 0.50   = {pct_above_50:.1f}%")
    if win_scores.max() < 0.5:
        print("  ⚠  No winning latent in the dataset scores > 50% against this")
        print("     opponent — the ceiling is below the decision boundary.")
        print("     The flow cannot cross 50% regardless of where it goes.")
    print("═" * 60 + "\n")


def _print_path_trace(path_z_np, score_fn, path_context) -> None:
    """Print P(win) at 10 evenly-spaced waypoints and nearest-win distance at endpoint."""
    import numpy as np

    path_z = torch.tensor(path_z_np, dtype=torch.float32)
    with torch.no_grad():
        win_probs = score_fn(path_z).cpu().numpy()

    n = len(win_probs)
    indices = np.linspace(0, n - 1, min(10, n), dtype=int)
    print("  P(win) trace along path:")
    for i in indices:
        bar = "█" * int(win_probs[i] * 20)
        print(f"    step {i:>4d}/{n - 1}  P(win)={win_probs[i]:.3f}  {bar}")

    win_latents = compute_win_latents(
        labels_tensor=path_context.labels_tensor,
        latents_p0=path_context.latents_p0,
        latents_p1=path_context.latents_p1,
    )
    endpoint = path_z[-1].unsqueeze(0)
    dists = torch.cdist(endpoint, win_latents).squeeze(0)
    start = torch.tensor(path_z_np[0], dtype=torch.float32).unsqueeze(0)
    start_dists = torch.cdist(start, win_latents).squeeze(0)
    print("\n  Distance to nearest winning latent:")
    print(f"    start:    {start_dists.min():.3f}")
    print(f"    endpoint: {dists.min():.3f}")
    print(
        f"    win centroid dist (endpoint): {torch.norm(path_z[-1] - win_latents.mean(0)):.3f}"
    )
    print()


# Register comparison sub-commands (imported here to avoid circular imports)
from latent_trainer.paths.compare.cli import (  # noqa: E402
    cmd_compare,
    cmd_compare_datasets,
    cmd_tune,
)

cli.add_command(cmd_compare)
cli.add_command(cmd_tune)
cli.add_command(cmd_compare_datasets)
