# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SC2 Latent Trainer — a research pipeline that trains VAE-based models on StarCraft II replay data (SC2EGSet dataset) to learn latent representations of player behavior, then uses those representations to predict match outcomes and generate counterfactual "improvement paths" for losing players.

## Environment Setup

Uses `uv` for dependency management. PyTorch must be installed separately via extras:

```bash
# CUDA build (default for training):
uv sync --extra cuda

# CPU-only build:
uv sync --extra cpu
```

## Key Commands

**Preprocess the SC2EGSet dataset into cached tensors:**
```bash
uv run python -m latent_trainer.features \
    --transform rich \
    --single_json_dataset_path H:/sc2egset_merged/sc2egset_merged.json \
    --output-directory data/ \
    --n-workers 24
```

**Train a model (single run):**
```bash
uv run python -m latent_trainer --pipeline two_stage --cache data/cached_dataset_rich_sc2egset.pt --mode single
uv run python -m latent_trainer --pipeline guided_vae --cache data/cached_dataset_rich_sc2egset.pt
```

**Hyperparameter sweep (Ray Tune + Optuna):**
```bash
uv run python -m latent_trainer --pipeline two_stage --cache data/cached_dataset_rich_sc2egset.pt --mode sweep --n-trials 50
```

**Generate improvement paths (feedback):**
```bash
uv run python src/latent_trainer/paths/main.py linear --method centroid
uv run python src/latent_trainer/paths/main.py gradient-ascent --top-k 15
uv run python src/latent_trainer/paths/main.py optimal-transport
```

**Linting and type checking:**
```bash
uv run ruff check src/
uv run mypy src/
```

**Run tests:**
```bash
uv run pytest
uv run pytest tests/path/to/test_file.py::test_name  # single test
```

**Serve MLFlow UI:**
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Optuna dashboard:**
```bash
uv run optuna-dashboard sqlite:///optuna_study.db
```

## Architecture

The pipeline has three distinct phases:

### 1. Feature Extraction (`src/latent_trainer/features/`)

`preprocess_dataset.py` reads raw SC2 replays from the SC2EGSet single-JSON dataset, applies a transform, and saves `(features, labels)` tensors to a `.pt` cache file.

Two transforms are available:
- **`rich`** — 203 features per player: temporal economy snapshots (early/mid/late game at 25%/50%/75% of game duration), final state, rate-of-change delta, plus APM/MMR/SQ/supply-capped-%, unit activity, upgrade count, game duration. Output shape: `[2, 203]` per replay.
- **`averaged_economy`** — 39 averaged economy features per player (from `sc2_datasets`).

Processing is parallelized via `ProcessPoolExecutor` using a chunked strategy (`process_set_chunked_single_pool`) for throughput. The dataset is expected to be split 80/10/10 (train/test/val); `check_split` validates this.

### 2. Model Training (`src/latent_trainer/models/`)

Two training pipelines:

**Two-stage pipeline** (`hyperparameter_search/ray_optuna_search.py`):
1. Train a VAE (`LitVAE` in `models/lightning/lit_vae.py`) to encode player feature vectors into a latent space.
2. Freeze the VAE, encode all training data, train a `LitClassifier` on the concatenated `(player0_z, player1_z)` latent pair to predict match outcome.

**Guided VAE pipeline** (`models/guided_vae.py`, `models/train_model.py`):
- `suGuidedVAE` — a single model that jointly reconstructs input features and predicts outcome via an adversarial classifier.

All Lightning models log to **MLFlow** (default: `sqlite:///mlflow.db`) and support **TensorBoard** (`output/tensorboard_logs/`). HPO uses **Ray Tune** as the trial scheduler with **Optuna** as the sampler.

### 3. Latent Path Charting (`src/latent_trainer/paths/`)

After training, `paths/main.py` (CLI) loads the saved two-stage model and finds counterfactual improvement paths for a losing player in latent space. Four strategies in `paths/strategies.py`:

| Strategy | Description |
|---|---|
| `linear` | Linear interpolation toward win centroid or k-NN mean |
| `gradient_ascent` | Gradient ascent on P(win) + KDE density regularization |
| `optimal_transport` | Wasserstein-barycentric path (uses POT library) |
| `neural_flow` | OT-paired flow matching from losing → winning distribution |

`paths/feedback.py` decodes the latent path back to feature space and computes three feedback signals: raw delta, minimum-viable delta, and P(win)-gain-weighted delta.

All strategies are **opponent-aware**: P(win) uses `(player_z, opponent_z)` as classifier input.

### Configuration

`ExperimentConfig` (`configs/experiment_config.py`) is a frozen dataclass that centralizes all training hyperparameters. It is constructed from CLI flags in `train.py` and passed through the entire pipeline — never use ad-hoc dicts for parameters.

`config.py` holds global constants: `LOGGING_FORMAT` and `DEFAULT_MLFLOW_URI` (SQLite, absolute path so Ray workers with changed CWD still write to the correct DB).

## Data Flow

```
SC2EGSet JSON → features/preprocess_dataset.py → data/cached_dataset_<transform>.pt
                                                            ↓
                                               train.py (ExperimentConfig)
                                                    ↓
                                         Two-stage: VAE → LitClassifier
                                         Guided VAE: suGuidedVAE
                                                    ↓
                                    output/two_stage_model.pth / checkpoints
                                                    ↓
                                         paths/main.py → output/feedback_*.png
```

## Notes

- The `experiments/` subdirectory is a separate standalone project (dimensionality reduction experiments with UMAP/t-SNE/PCA) with its own `pyproject.toml` and `uv.lock`. It is not part of the main `latent_trainer` package.
- Raw replay data (SC2EGSet) is expected at `H:/sc2egset_merged/sc2egset_merged.json` by default; override with `--single_json_dataset_path`.
- Cached datasets are saved to `data/` by default; training commands expect `data/cached_dataset_rich_sc2egset.pt`.
- Model outputs (checkpoints, plots) go to `output/`.
- The `sc2-datasets` dependency is sourced from the `dev` branch of `github.com/Kaszanas/SC2_Datasets`.
