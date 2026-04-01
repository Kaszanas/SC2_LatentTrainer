"""Three-signal feedback analysis for latent-space paths.

Given a path through latent space (from any strategy) and a decoder +
score function, computes three complementary feedback signals:

1. **Raw delta** (start → end) — overall direction of change.
2. **Minimum-viable delta** — change only up to the P(win)=0.5 crossover.
3. **P(win)-gain-weighted delta** — features weighted by how much
   P(win) improved at each step.

The module is model-agnostic: it accepts callables for decoding and
scoring so it works with any model architecture.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def compute_feedback(
    path_z: np.ndarray,
    *,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    norm_mean: np.ndarray | torch.Tensor,
    norm_std: np.ndarray | torch.Tensor,
    feature_names: list[str],
    top_k: int = 10,
    method_name: str = "",
) -> dict:
    """Compute three-signal feedback along a latent-space path.

    Parameters
    ----------
    path_z:
        Path waypoints, shape ``(n_waypoints, latent_dim)``.
    decode_fn:
        ``z (N, D) → x_hat (N, F)`` — decodes latent codes to normalised
        feature space.  Called inside ``torch.no_grad()``.
    score_fn:
        ``z (N, D) → P(win) (N,)`` or ``(N, 1)`` — win probability.
        Called inside ``torch.no_grad()``.
    norm_mean:
        Per-feature mean used during normalisation, shape ``(1, F)`` or ``(F,)``.
    norm_std:
        Per-feature std used during normalisation, shape ``(1, F)`` or ``(F,)``.
    feature_names:
        Human-readable feature names, length ``F``.
    top_k:
        Number of top features to include in the ranked tables.
    method_name:
        Label for this path method (e.g. ``"OT"``, ``"GA"``).

    Returns
    -------
    dict
        Keys include ``raw``, ``minimum_viable``, ``gain_weighted``
        (ranked tables), raw arrays (``_raw_delta``, etc.), and
        P(win) values (``_p_vals``).
    """
    if isinstance(norm_mean, torch.Tensor):
        norm_mean = norm_mean.cpu().numpy()
    if isinstance(norm_std, torch.Tensor):
        norm_std = norm_std.cpu().numpy()

    norm_mean = np.asarray(norm_mean).reshape(1, -1)
    norm_std = np.asarray(norm_std).reshape(1, -1)

    path_tensor = torch.tensor(path_z, dtype=torch.float32)

    with torch.no_grad():
        x_decoded = decode_fn(path_tensor).cpu().numpy()
        p_vals = score_fn(path_tensor).cpu().numpy().squeeze()

    # Inverse-transform to original feature scale
    x_orig = x_decoded * norm_std + norm_mean
    x_start = x_orig[0]
    x_end = x_orig[-1]

    # ── Signal 1: Raw start → end delta
    raw_delta = x_end - x_start

    # ── Signal 2: Minimum-viable delta (stop at P(win) ≥ 0.5 crossover)
    cross_idx = np.where(np.diff(np.sign(p_vals - 0.5)))[0]
    if len(cross_idx) > 0:
        threshold_wp = cross_idx[0] + 1
        mv_delta = x_orig[threshold_wp] - x_start
        mv_p = p_vals[threshold_wp]
        mv_label = f"waypoint {threshold_wp}/{len(p_vals) - 1}  (P(win)={mv_p:.3f})"
    else:
        mid = len(p_vals) // 2
        mv_delta = x_orig[mid] - x_start
        mv_p = p_vals[mid]
        mv_label = f"midpoint (P(win)={mv_p:.3f}, no crossing found)"

    # ── Signal 3: P(win)-gain-weighted delta
    p_gains = np.diff(p_vals).clip(min=0)
    x_steps = np.diff(x_orig, axis=0)
    if p_gains.sum() > 0:
        weighted_delta = (x_steps * p_gains[:, None]).sum(0) / p_gains.sum()
    else:
        weighted_delta = raw_delta.copy()

    # ── Build ranked feedback tables
    def _rank_table(delta: np.ndarray, label: str):
        ranked = np.argsort(np.abs(delta))[::-1]
        rows = []
        for i in ranked[:top_k]:
            rows.append(
                {
                    "priority": int(np.where(ranked == i)[0][0]) + 1,
                    "feature": feature_names[i],
                    "current": float(x_start[i]),
                    "target": float(x_start[i] + delta[i]),
                    "delta": float(delta[i]),
                    "direction": "▲" if delta[i] > 0 else "▼",
                }
            )
        return rows, label

    raw_rows, raw_lbl = _rank_table(raw_delta, "Full path  (start → end)")
    mv_rows, mv_lbl = _rank_table(mv_delta, f"Minimum viable  ({mv_label})")
    wgt_rows, wgt_lbl = _rank_table(weighted_delta, "P(win)-gain weighted")

    return {
        "method": method_name,
        "raw": raw_rows,
        "raw_label": raw_lbl,
        "minimum_viable": mv_rows,
        "mv_label": mv_lbl,
        "gain_weighted": wgt_rows,
        "wgt_label": wgt_lbl,
        "_raw_delta": raw_delta,
        "_mv_delta": mv_delta,
        "_weighted_delta": weighted_delta,
        "_x_start": x_start,
        "_x_orig": x_orig,
        "_p_vals": p_vals,
        "_mv_crossover_wp": int(cross_idx[0]) if len(cross_idx) > 0 else None,
    }


def print_feedback_report(feedback: dict, *, top_k: int = 10) -> None:
    """Pretty-print the three-signal feedback table to stdout."""
    method = feedback.get("method", "")
    header = f"  {'Feature':<22}  {'Current':>9}  {'Target':>9}  {'Δ':>9}"
    sep = "  " + "─" * 54

    print(f"\n{'═' * 60}")
    print(f"  [{method}] FEEDBACK REPORT  —  top {top_k} features")
    print(f"{'═' * 60}")

    for key, lbl_key in [
        ("raw", "raw_label"),
        ("minimum_viable", "mv_label"),
        ("gain_weighted", "wgt_label"),
    ]:
        print(f"\n  ── {feedback[lbl_key]}")
        print(header)
        print(sep)
        for r in feedback[key]:
            print(
                f"  {r['feature']:<22}  {r['current']:>9.3f}"
                f"  {r['target']:>9.3f}  {r['direction']} {abs(r['delta']):>7.3f}"
            )
