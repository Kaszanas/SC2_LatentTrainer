"""End-to-end single-replay counterfactual prediction pipeline.

This is the single end-user-facing entry point: extraction -> features ->
encode -> path-generation -> feedback report, for one .SC2Replay file.
Both the CLI and (later) the API call only :func:`predict_replay`.

Currently only the ``"linear"`` strategy is wired up end-to-end.
``gradient_ascent``/``optimal_transport``/``neural_flow`` build the exact
same PathContext (see paths/cli.py's cmd_gradient_ascent /
cmd_optimal_transport / cmd_neural_flow for the pattern) -- extending this
function to support them is mechanical, not a redesign.
"""

from __future__ import annotations

import hashlib
import logging
from functools import partial
from pathlib import Path

import numpy as np
import torch

from latent_trainer.features.rich_transform import (
    GAME_LOOPS_PER_SECOND,
    N_GRANULAR_BINS,
    rich_transform_granular,
)
from latent_trainer.inference.extract import DEFAULT_DOCKER_IMAGE, extract_replay
from latent_trainer.inference.reference_pack import ReferencePack, load_reference_pack
from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE
from latent_trainer.paths.data import (
    PathContext,
    build_granular_feature_names,
    build_path_context,
    compute_win_latents,
    encode_player,
    get_supervised_dim,
    nearest_winning_target,
    opponent_aware_logit,
    opponent_aware_score,
)
from latent_trainer.paths.pipeline import run_path_charting_pipeline
from latent_trainer.paths.strategies import path_linear
from latent_trainer.paths.strategies.gradient_ascent import path_gradient_ascent
from latent_trainer.paths.strategies.optimal_transport import path_optimal_transport

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "latent_trainer_serve"


def _replay_hash(replay_path: Path) -> str:
    h = hashlib.sha256()
    with open(replay_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _load_model(model_path: Path) -> LitGuidedVAE:
    vae_model = LitGuidedVAE.load_from_checkpoint(checkpoint_path=model_path)
    vae_model.eval()
    # Defense-in-depth: a near-zero std for a near-constant feature turns any
    # nonzero outlier value into an astronomical z-score, which the encoder
    # propagates into a huge latent vector that can poison downstream
    # centroid/mean computations over the reference pool (root-caused and
    # fixed at training time in features/data_utils.py::normalize, but floor
    # it again here too in case an older/not-yet-retrained checkpoint is
    # served).
    vae_model.std.clamp_(min=1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return vae_model.to(device)


def _extract_features_cached(
    replay_path: Path,
    cache_dir: Path,
    docker_image: str,
    extractor_binary: Path | None = None,
) -> tuple[torch.Tensor, int, float | None]:
    """Extract + transform a replay, caching the result by content hash.

    Re-analyzing the same replay (e.g. with a different --strategy) then
    skips both the extraction and the feature computation.

    Returns (features, label, duration_seconds). duration_seconds is the
    replay's real-world length (elapsedGameLoops / GAME_LOOPS_PER_SECOND),
    used to translate each bin's event-index-fraction window into an
    approximate wall-clock time range for display. ``None`` if the header's
    elapsedGameLoops isn't available for some reason.
    """
    replay_hash = _replay_hash(replay_path)
    cache_path = cache_dir / f"{replay_hash}_granular_features.pt"

    if cache_path.exists():
        logger.info("Using cached features for %s", replay_path.name)
        cached = torch.load(str(cache_path), weights_only=True)
        return cached["features"], int(cached["label"]), cached.get("duration_seconds")

    logger.info("Extracting %s...", replay_path.name)
    sc2_replay = extract_replay(
        replay_path=replay_path,
        docker_image=docker_image,
        extractor_binary=extractor_binary,
    )

    result = rich_transform_granular(sc2_replay, n_bins=N_GRANULAR_BINS)
    if result is None:
        raise ValueError(
            f"Could not extract features from {replay_path.name} -- the game "
            "may be too short (<3 minutes), have an undecided/draw outcome, "
            "or be missing PlayerStats/player info."
        )
    features, label = result
    try:
        duration_seconds = (
            float(sc2_replay.header.elapsedGameLoops) / GAME_LOOPS_PER_SECOND
        )
    except (AttributeError, ValueError, TypeError):
        duration_seconds = None
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"features": features, "label": label, "duration_seconds": duration_seconds},
        str(cache_path),
    )
    return features, label, duration_seconds


def predict_replay(
    replay_path: Path,
    model_path: Path,
    reference_pack_path: Path,
    player: int | None = None,
    strategy: str = "linear",
    method: str = "centroid",
    k_neighbours: int = 5,
    n_steps: int = 50,
    top_k: int = 15,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    extractor_binary: Path | None = None,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    # gradient_ascent-only tunables (see paths/cli.py::cmd_gradient_ascent):
    ga_steps: int = 1000,
    ga_lr: float = 0.005,
    ga_momentum: float = 0.5,
    density_weight: float = 0.3,
    kde_bandwidth: float = 0.5,
    convergence_threshold: float = 0.95,
    # optimal_transport-only tunable (see paths/cli.py::cmd_optimal_transport):
    ot_reg: float = 0.01,
) -> dict:
    """Run extraction -> features -> encode -> path -> feedback for one replay.

    Parameters
    ----------
    player:
        Which player (0 or 1) to analyze. ``None`` -> analyze whichever
        player actually lost the match (the natural choice: every game has
        exactly one loser, and that's who the "improvement path" is for).
    strategy:
        ``"linear"``, ``"gradient_ascent"``, or ``"optimal_transport"``.
        ``"neural_flow"`` is not supported here -- it needs a separately
        trained OT-flow-matching checkpoint (paths/cli.py's
        ``cmd_neural_flow``) that doesn't exist for the granular feature
        set this pipeline serves.
    extractor_binary:
        Path to a locally available SC2InfoExtractorGo binary. When set,
        extraction runs that binary directly -- no Docker involved. When
        ``None`` (default), falls back to ``docker run <docker_image>``.
        A containerized server should bake the binary into its own image
        (multi-stage ``COPY --from=kaszanas/sc2infoextractorgo:dev``) and
        set this, avoiding Docker-out-of-Docker entirely.
    output_dir:
        Where to write this call's plots/report. ``None`` -> the process-wide
        default (today's CLI behavior). Callers handling concurrent requests
        (e.g. an API) must pass a request-scoped directory so plots from
        different replays/requests don't overwrite each other.

    Returns
    -------
    dict
        ``run_path_charting_pipeline``'s result: ``{"feedback": ...,
        "saved_files": {plot_key: {"pdf": Path, "png": Path}}}``.
    """
    replay_path = Path(replay_path)
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR

    features, true_label, duration_seconds = _extract_features_cached(
        replay_path=replay_path,
        cache_dir=cache_dir,
        docker_image=docker_image,
        extractor_binary=extractor_binary,
    )

    guided_vae = _load_model(model_path)
    reference_pack: ReferencePack = load_reference_pack(reference_pack_path)
    device = next(guided_vae.parameters()).device

    mean, std = guided_vae.mean.cpu(), guided_vae.std.cpu()
    input_dim = mean.shape[-1]
    if features.shape[-1] != input_dim:
        # The checkpoint was trained on a truncated feature set (e.g. a k<20
        # bin-count cutoff from the leakage sweep) -- keep only the matching
        # leading columns, same slice-invariant truncation used at training
        # time and in paths/data.py::load_model_and_data for the reference
        # pack build.
        features = features[..., :input_dim]
    normalized = (features.float() - mean) / std
    new_latents_p0 = encode_player(
        vae=guided_vae.model, data=normalized[0:1, :].to(device)
    )
    new_latents_p1 = encode_player(
        vae=guided_vae.model, data=normalized[1:2, :].to(device)
    )

    # Slot the new replay into the reference pool as one extra row, so the
    # existing latents_pX[chosen] opponent-lookup pattern (used unchanged
    # throughout paths/pipeline.py and paths/cli.py) resolves to *this*
    # match's actual opponent instead of an unrelated training-set game.
    latents_p0 = torch.cat([reference_pack.latents_p0.to(device), new_latents_p0])
    latents_p1 = torch.cat([reference_pack.latents_p1.to(device), new_latents_p1])
    labels_tensor = torch.cat([
        reference_pack.labels_tensor.to(device),
        torch.tensor([float(true_label)], device=device),
    ])
    chosen = len(reference_pack.labels_tensor)

    # player_idx (the analyzed player) equals the label's own convention:
    # label encodes which side lost (see rich_transform._get_outcome), and
    # build_path_context reads player_idx = int(labels[chosen]) directly.
    labels_np = labels_tensor.cpu().numpy().copy()
    if player is not None:
        labels_np[chosen] = player

    path_context: PathContext = build_path_context(
        guided_vae=guided_vae,
        X=torch.empty(0),  # unused downstream (see paths/pipeline.py, cli.py)
        labels=labels_np,
        labels_tensor=labels_tensor,
        latents_p0=latents_p0,
        latents_p1=latents_p1,
        chosen=chosen,
    )

    if strategy not in ("linear", "gradient_ascent", "optimal_transport"):
        raise NotImplementedError(
            f"strategy={strategy!r} isn't supported. Choices: 'linear', "
            "'gradient_ascent', 'optimal_transport'. 'neural_flow' needs a "
            "separately trained OT-flow-matching checkpoint that doesn't "
            "exist for the granular feature set this pipeline serves -- see "
            "paths/cli.py's cmd_neural_flow for that pattern if one is "
            "trained later."
        )

    win_latents = compute_win_latents(
        labels_tensor=path_context.labels_tensor,
        latents_p0=path_context.latents_p0,
        latents_p1=path_context.latents_p1,
    )
    sup_dim = get_supervised_dim(path_context.guided_vae)
    z_start_full = path_context.sample_z.detach().cpu().numpy()
    z_free = z_start_full[sup_dim:]
    win_latents_sup = win_latents[:, :sup_dim]

    if strategy == "linear":
        if method == "centroid":
            # Robust to a stray outlier latent (e.g. from a poisoned/near-zero-std
            # training feature -- see _load_model's std floor): drop points whose
            # norm is many median-absolute-deviations away from the median norm
            # before averaging, instead of a plain mean that a single
            # astronomically large outlier could otherwise dominate completely.
            norms = win_latents_sup.norm(dim=1)
            median_norm = norms.median()
            mad = (norms - median_norm).abs().median() + 1e-6
            inliers = (norms - median_norm).abs() <= 10 * mad
            if not inliers.any():
                inliers = torch.ones_like(inliers)
            target_z = win_latents_sup[inliers].mean(dim=0).cpu().numpy()
        elif method == "nearest":
            target_z = (
                nearest_winning_target(
                    sample_z=path_context.sample_z[:sup_dim],
                    win_latents=win_latents_sup,
                    k=k_neighbours,
                )
                .cpu()
                .numpy()
            )
        else:
            raise ValueError(
                f"Unknown method: {method!r}, expected 'centroid' or 'nearest'."
            )
        path_z_sup = path_linear(
            z_start=z_start_full[:sup_dim],
            z_target=target_z,
            n_waypoints=n_steps,
        )

    elif strategy == "gradient_ascent":
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
        logit_fn = partial(
            opponent_aware_logit,
            guided_vae=path_context.guided_vae,
            opponent_z=opponent_z,
            player_idx=path_context.player_idx,
        )
        path_z_sup = path_gradient_ascent(
            z_start=z_start_full[:sup_dim],
            score_fn=score_fn,
            logit_fn=logit_fn,
            Z_all=win_latents_sup.cpu().numpy(),
            steps=ga_steps,
            lr=ga_lr,
            momentum=ga_momentum,
            density_weight=density_weight,
            kde_bandwidth=kde_bandwidth,
            n_waypoints=n_steps,
            convergence_threshold=convergence_threshold,
        )

    else:  # optimal_transport
        path_z_sup = path_optimal_transport(
            z_start=z_start_full[:sup_dim],
            Z_win=win_latents_sup.cpu().numpy(),
            reg=ot_reg,
            n_waypoints=n_steps,
        )
        if not torch.isfinite(torch.as_tensor(path_z_sup)).all():
            raise ValueError(
                "Optimal transport produced non-finite values (NaN/Inf). "
                "Try ot_reg=0.0 for exact EMD or a larger regularization "
                "such as ot_reg=0.05 or ot_reg=0.1."
            )

    path_z_np = np.concatenate(
        [path_z_sup, np.tile(z_free, (len(path_z_sup), 1))], axis=1
    )

    n_bins = (features.shape[-1] - 1) // 39
    result = run_path_charting_pipeline(
        path_context=path_context,
        n_steps=n_steps,
        top_k=top_k,
        strategy=strategy,
        path_z_np=path_z_np,
        feature_names=build_granular_feature_names(n_bins=n_bins),
        output_dir=output_dir,
        plots_dir=output_dir,
    )
    # For translating each "binNN_..." feature name into an approximate
    # wall-clock time range: bins are always 1/N_GRANULAR_BINS-of-the-game
    # windows regardless of how many leading bins a truncated checkpoint
    # actually uses (see build_granular_feature_names/N_GRANULAR_BINS).
    result["game_duration_seconds"] = duration_seconds
    result["n_bins_total"] = N_GRANULAR_BINS
    return result
