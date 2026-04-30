# SC2 Latent Trainer

An approach to try and create a traversal through the latent space of a model that has been optimized for classifying player skill in StarCraft 2.

## Dependencies

Please make sure to install PyTorch with CUDA support before running `uv sync`.

## Quick Start Guide

### Step 1: Install dependencies

```bash
uv sync
```
### Step 2: Cache the dataset

Activate the new environment and run the feature extraction script to cache the dataset. Please look at the help message for details on the available options:

```bash
python ./src/latent_trainer/features/main.py --help
```

### Step 3: Start the training

**Unified entrypoint (two-stage or guided VAE):**

As an example you can run the hyperparamter and architecture sweep using the following commands:

```bash
# Two-stage pipeline (default):
python src/latent_trainer/train.py --experiment_name two_stage_sweep --pipeline two_stage --dataset_filename cached_dataset_rich.pt

# Guided VAE pipeline:
python src/latent_trainer/train.py --experiment_name guided_vae_sweep --pipeline guided_vae --dataset_filename cached_dataset_rich.pt

# Hyperparameter sweep (Ray Tune + Optuna):
python src/latent_trainer/train.py --pipeline two_stage --mode sweep --n_trials 30
```

**Standalone guided-VAE training:**

```bash
uv run python -m latent_trainer.models.train_model --dataset_filename cached_dataset_rich.pt
```

### Step 4: Monitor training

**TensorBoard** (in a separate terminal):

```bash
tensorboard --logdir=output/tensorboard_logs
```

**MLFlow UI** (in a separate terminal):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

All training runs are logged to a local SQLite database (`mlflow.db`) by
default.  To use a remote tracking server, pass `--mlflow-uri`:

```bash
python src/latent_trainer/train.py --mlflow_uri http://localhost:5000
```

## Feedback Path Finder

After training, use the feedback path finder to generate improvement
guidance for a losing sample:

```bash
# Linear interpolation (fast, deterministic):
python src/latent_trainer/paths/main.py linear --method centroid

# Gradient ascent + KDE density (manifold-following, 3-signal feedback):
python src/latent_trainer/paths/main.py gradient_kde --top-k 15
```

The `gradient_kde` strategy computes:
1. **Raw delta** — overall direction of change (start → end)
2. **Minimum-viable delta** — change needed just to cross P(win) = 0.5
3. **P(win)-gain-weighted delta** — features that moved while winning probability rose

Both strategies are opponent-aware — P(win) is computed using both
players' latent codes.

