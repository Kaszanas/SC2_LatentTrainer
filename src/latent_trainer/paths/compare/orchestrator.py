"""Outer loop: runs all methods across N samples and returns a ComparisonReport."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.neighbors import KernelDensity

from latent_trainer.paths.compare.results import (
    ComparisonReport,
    MethodSpec,
    PathRunResult,
)
from latent_trainer.paths.compare.runners import run_method
from latent_trainer.paths.data import (
    build_path_context,
    compute_win_latents,
    encode_player,
    load_model_and_data,
)


def run_comparison(
    *,
    model_path: Path,
    dataset_path: Path,
    method_specs: Sequence[MethodSpec],
    n_samples: int,
    n_steps: int,
    top_k: int,
    seed: int,
    flow_checkpoint: Path | None,
) -> ComparisonReport:
    """Compare all methods across N samples.

    Parameters
    ----------
    n_samples:
        Number of samples to analyse. 0 means use all samples in the dataset.
    """
    # --- Load model and data once ---
    print("Loading model and data...")
    guided_vae, X, y = load_model_and_data(
        model_path=model_path,
        cached_dataset_filepath=dataset_path,
    )
    labels = y.cpu().numpy()
    labels_tensor = y
    print(f"  Dataset size: {len(X)} samples")

    # --- Encode both players once ---
    print("Encoding latent space...")
    latents_p0 = encode_player(vae=guided_vae.model, data=X[:, 0, :])
    latents_p1 = encode_player(vae=guided_vae.model, data=X[:, 1, :])

    # --- Compute shared tensors once ---
    win_latents = compute_win_latents(
        labels_tensor=labels_tensor,
        latents_p0=latents_p0,
        latents_p1=latents_p1,
    )
    all_latents = torch.cat([latents_p0, latents_p1], dim=0)

    # Fit KDE on winning latents once (used for geometry metrics in all runs)
    win_np = win_latents.detach().cpu().numpy()
    win_kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(win_np)
    print(f"  Winning latents: {len(win_np)}")

    # --- Filter / sample indices ---
    all_indices = np.arange(len(labels))
    if n_samples == 0 or n_samples >= len(all_indices):
        sample_indices = all_indices
    else:
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(all_indices, size=n_samples, replace=False)
        sample_indices = np.sort(sample_indices)

    n_evaluated = len(sample_indices)
    print(
        f"  Evaluating {n_evaluated} samples × {len(method_specs)} methods "
        f"= {n_evaluated * len(method_specs)} runs"
    )

    # --- Load flow model once if needed ---
    flow_model = None
    needs_flow = any(s.requires_flow_checkpoint for s in method_specs)
    if needs_flow:
        if flow_checkpoint is None:
            print(
                "  WARNING: neural_flow spec present but no --flow_checkpoint provided — skipping."
            )
            method_specs = [s for s in method_specs if not s.requires_flow_checkpoint]
        else:
            from latent_trainer.paths.flow import LitOTFlowMatching

            print(f"  Loading flow model from {flow_checkpoint}...")
            flow_model = LitOTFlowMatching.load_from_checkpoint(flow_checkpoint)
            flow_model.eval()
            flow_model = flow_model.to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

    # --- Main loop ---
    results: list[PathRunResult] = []
    for run_idx, chosen in enumerate(sample_indices):
        chosen = int(chosen)
        ctx = build_path_context(
            guided_vae=guided_vae,
            X=X,
            labels=labels,
            labels_tensor=labels_tensor,
            latents_p0=latents_p0,
            latents_p1=latents_p1,
            chosen=chosen,
        )
        for spec in method_specs:
            result = run_method(
                spec=spec,
                ctx=ctx,
                n_steps=n_steps,
                top_k=top_k,
                flow_model=flow_model,
                win_latents=win_latents,
                all_latents=all_latents,
                win_kde=win_kde,
            )
            results.append(result)
            status = "OK" if result.error is None else "FAIL"
            print(
                f"  [{run_idx + 1}/{n_evaluated}] sample={chosen} "
                f"method={spec.name:<22} {status}  "
                f"P(win): {result.p_win_start:.3f} → {result.p_win_end:.3f}"
                + (
                    f"  crossover@{result.crossover_alpha:.2f}"
                    if result.success
                    else "  no crossover"
                )
            )

    return ComparisonReport(
        n_samples_requested=n_samples,
        n_samples_evaluated=n_evaluated,
        methods=tuple(method_specs),
        results=tuple(results),
        seed=seed,
        model_path=model_path,
        dataset_path=dataset_path,
    )


def save_report(report: ComparisonReport, output_dir: Path) -> Path:
    path = output_dir / "report.pkl"
    with open(path, "wb") as f:
        pickle.dump(report, f)
    print(f"  Report saved to {path}")
    return path


def load_report(path: Path) -> ComparisonReport:
    with open(path, "rb") as f:
        return pickle.load(f)
