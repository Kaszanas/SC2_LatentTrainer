"""Optuna-based hyperparameter search for path-charting methods."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import optuna

from latent_trainer.paths.compare.aggregate import summarise
from latent_trainer.paths.compare.orchestrator import run_comparison
from latent_trainer.paths.compare.results import MethodSpec

optuna.logging.set_verbosity(optuna.logging.WARNING)

_METRIC_ATTR: dict[str, str] = {
    "auc": "auc_mean",
    "success_rate": "success_rate",
    "p_win_gain": "p_win_gain_mean",
}


def _suggest_params(trial: optuna.Trial, spec: MethodSpec) -> dict:
    match spec.strategy:
        case "linear":
            return {
                "method": spec.params["method"],  # fixed per spec, not tuned
                "k_neighbours": trial.suggest_int("k_neighbours", 3, 15),
                "k_opponents": trial.suggest_int("k_opponents", 10, 200),
            }
        case "gradient_ascent":
            return {
                "steps": 2000,
                "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
                "momentum": trial.suggest_float("momentum", 0.0, 0.95),
                "density_weight": trial.suggest_float("density_weight", 0.0, 1.0),
                "kde_bandwidth": trial.suggest_float("kde_bandwidth", 0.1, 2.0),
                "convergence_threshold": 0.95,
            }
        case "optimal_transport":
            return {
                "reg": trial.suggest_float("reg", 0.01, 0.5, log=True),
                "step_size": trial.suggest_float("step_size", 0.05, 0.5),
                "k_opponents": trial.suggest_int("k_opponents", 10, 200),
            }
        case "neural_flow":
            return {
                "guidance_scale": trial.suggest_float("guidance_scale", 1.0, 1.0),
            }
        case _:
            return {}


def tune_method(
    *,
    spec: MethodSpec,
    model_path: Path,
    dataset_path: Path,
    n_trials: int,
    n_samples: int,
    n_steps: int,
    top_k: int,
    metric: str,
    seed: int,
    flow_checkpoint: Path | None,
) -> tuple[dict, float]:
    """Run Optuna to find best hyperparameters for a single method.

    Returns
    -------
    best_params : dict
        Best hyperparameter values found.
    best_value : float
        Best metric value achieved.
    """
    metric_attr = _METRIC_ATTR.get(metric, "auc_mean")

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, spec)
        trial_spec = MethodSpec(
            name=spec.name,
            display_name=spec.display_name,
            strategy=spec.strategy,
            params=params,
            requires_flow_checkpoint=spec.requires_flow_checkpoint,
        )
        report = run_comparison(
            model_path=model_path,
            dataset_path=dataset_path,
            method_specs=[trial_spec],
            n_samples=n_samples,
            n_steps=n_steps,
            top_k=top_k,
            seed=seed,
            flow_checkpoint=flow_checkpoint,
        )
        stats = summarise(report)
        if not stats:
            return math.nan
        value = getattr(stats[0], metric_attr, math.nan)
        return value if not math.isnan(value) else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def tune_all_methods(
    *,
    method_specs: Sequence[MethodSpec],
    model_path: Path,
    dataset_path: Path,
    n_trials: int,
    n_samples: int,
    n_steps: int,
    top_k: int,
    metric: str,
    seed: int,
    flow_checkpoint: Path | None,
) -> dict[str, dict]:
    """Tune all methods and return a dict of {method_name: best_params}."""
    best: dict[str, dict] = {}
    for spec in method_specs:
        print(
            f"\nTuning {spec.display_name} ({n_trials} trials × {n_samples} samples)..."
        )
        try:
            params, value = tune_method(
                spec=spec,
                model_path=model_path,
                dataset_path=dataset_path,
                n_trials=n_trials,
                n_samples=n_samples,
                n_steps=n_steps,
                top_k=top_k,
                metric=metric,
                seed=seed,
                flow_checkpoint=flow_checkpoint,
            )
            print(f"  Best {metric}: {value:.4f}  params: {params}")
            best[spec.name] = params
        except Exception as exc:
            print(f"  WARNING: {spec.name} tuning failed — {exc}")
    return best


def save_config(best: dict[str, dict], output_path: Path) -> None:
    """Save best hyperparameters as a JSON config file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"Best configs saved to {output_path}")


def load_config(config_path: Path) -> dict[str, dict]:
    """Load hyperparameter config from a JSON file."""
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def apply_config(
    specs: Sequence[MethodSpec], config: dict[str, dict]
) -> list[MethodSpec]:
    """Override MethodSpec params with values from a loaded config."""
    result = []
    for spec in specs:
        if spec.name in config:
            merged = {**spec.params, **config[spec.name]}
            result.append(
                MethodSpec(
                    name=spec.name,
                    display_name=spec.display_name,
                    strategy=spec.strategy,
                    params=merged,
                    requires_flow_checkpoint=spec.requires_flow_checkpoint,
                )
            )
        else:
            result.append(spec)
    return result
