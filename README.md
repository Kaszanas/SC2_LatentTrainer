# SC2 Latent Trainer

An approach to create a traversal through the latent space of a model optimized for classifying player skill in StarCraft 2. Trains VAE-based models on SC2EGSet replay data to learn latent representations of player behaviour, then uses those representations to predict match outcomes and generate counterfactual improvement paths for losing players.

## Setup

### 1. Install dependencies

```bash
uv sync --extra cuda   # GPU (recommended)
uv sync --extra cpu    # CPU-only
```

Activate the environment before running any command below.

---

### 2. Preprocess the dataset

Reads the SC2EGSet single-JSON file, applies a feature transform, and writes a cached `.pt` tensor file used by all downstream steps.

```bash
python src/latent_trainer/features/main.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--transform` | `rich` | Feature transform: `rich` (204 features) or `averaged_economy` (39 features) |
| `--single_json_dataset_path` | `H:/sc2egset_merged/sc2egset_merged.json` | Path to the SC2EGSet JSON file |
| `--n_workers` | `24` | Parallel workers |
| `--n_samples` | `0` (all) | Randomly sample N games before processing; `0` = use all |
| `--seed` | `42` | Random seed for sampling |

**Examples:**

```bash
# Full dataset, rich transform:
python src/latent_trainer/features/main.py \
    --single_json_dataset_path H:/sc2egset_merged/sc2egset_merged.json

# Random subset of 2000 games from a large dataset:
python src/latent_trainer/features/main.py \
    --single_json_dataset_path H:/sc2egset_large/sc2egset_large.json \
    --n_samples 2000 \
    --seed 42
```

Output: `data/cached_dataset_rich_sc2egset.pt` (or `data/cached_dataset_rich_sc2egset_2000.pt` when `--n_samples` is set).

---

### 3. Train a model

#### Unified entry point (HPO sweep + best-trial retraining)

Supports both the two-stage pipeline (VAE → classifier) and the guided VAE pipeline. Run a Ray+Optuna sweep to find hyperparameters, then retrain with the best trial.

```bash
python src/latent_trainer/train.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pipeline` | `guided_vae` | `guided_vae` |
| `--dataset_filename` | `cached_dataset_rich_sc2egset.pt` | Cached dataset file in `data/` |
| `--sweep` | off | Enable Ray+Optuna hyperparameter search |
| `--n_trials` | `20` | Number of Optuna trials (sweep mode) |
| `--experiment_name` | — | MLFlow experiment name (**required**) |
| `--mlflow_uri` | local SQLite | MLFlow tracking URI |
| `--gpus_per_trial` | `0.1` | Fractional GPU per Ray trial |
| `--cpus_per_trial` | `2` | CPUs per Ray trial |
| `--optuna_db` | `sqlite:///optuna_study.db` | Optuna storage URL |
| `--source_experiment` | same as `--experiment_name` | MLFlow experiment to load best params from |
| `--source_run` | most recent best trial | MLFlow run to load params from |
| `--run_name` | auto-generated | Explicit MLFlow run name for retraining |

**Examples:**

```bash
# Hyperparameter sweep — guided VAE pipeline, 50 trials:
python src/latent_trainer/train.py \
    --pipeline guided_vae \
    --experiment_name guided_vae_sweep \
    --sweep \
    --n_trials 50

```

```bash
# Retrain with best params from a previous sweep:
python src/latent_trainer/train.py \
    --pipeline guided_vae \
    --experiment_name guided_vae_retrain \
    --source_experiment guided_vae_sweep
```


#### Standalone guided-VAE retraining (fixed best-known hyperparameters)

Skips the HPO sweep and trains directly with pre-set defaults. Edit the defaults in the script to match your best-found values.

```bash
python src/latent_trainer/retrain_guided.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset_filename` | `cached_dataset_rich_sc2egset.pt` | Cached dataset file in `data/` |
| `--experiment_name` | — | MLFlow experiment name (**required**) |
| `--run_name` | `guided_vae_best` | MLFlow run name |
| `--latent_dim` | `32` | VAE latent dimensionality |
| `--encoder_hidden_dims` | `256,128,64` | Encoder layer sizes, comma-separated |
| `--supervised_dim` | `2` | Supervised head output dimension |
| `--batch_size` | `64` | Batch size |
| `--learning_rate` | `1e-4` | VAE learning rate |
| `--weight_decay` | `1e-5` | VAE weight decay |
| `--learning_rate_cls` | `1e-4` | Classifier learning rate |
| `--weight_decay_cls` | `1e-4` | Classifier weight decay |
| `--classification_weight` | `0.5` | Classification loss weight |
| `--epochs` | `100` | Max training epochs |
| `--mlflow_uri` | local SQLite | MLFlow tracking URI |

**Example:**

```bash
python src/latent_trainer/retrain_guided.py \
    --experiment_name guided_vae_best \
    --latent_dim 32 \
    --encoder_hidden_dims 256,128,64 \
    --epochs 150
```

---

### 4. Monitor training

```bash
# TensorBoard (separate terminal):
tensorboard --logdir=output/tensorboard_logs

# MLFlow UI (separate terminal):
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Optuna dashboard (separate terminal):
optuna-dashboard sqlite:///optuna_study.db
```

---

## Path Charting

After training, use the path-charting CLI to generate counterfactual improvement guidance for a losing player.

All subcommands share these global options:

| Option | Default | Description |
|--------|---------|-------------|
| `--model_path` | — | Path to trained model checkpoint (`.ckpt`) |
| `--dataset_filename` | `cached_dataset_rich_sc2egset.pt` | Cached dataset in `data/` |
| `--sample_idx` | random | Index of the game to analyse |
| `--n_steps` | `20` | Number of waypoints along the path |
| `--top_k` | `10` | Top-K features to display in feedback |

```bash
python src/latent_trainer/paths/main.py --help
```

### Single-sample strategies

#### Linear interpolation

Fast, deterministic path toward the winning centroid or nearest winning neighbours.

```bash
python src/latent_trainer/paths/main.py linear \
    --model_path best.ckpt \
    --method centroid

python src/latent_trainer/paths/main.py linear \
    --model_path best.ckpt \
    --method nearest \
    --k_neighbours 5
```

#### Gradient ascent

Climbs P(win) via gradient ascent regularised by a KDE density prior, keeping the path on the data manifold.

```bash
python src/latent_trainer/paths/main.py gradient_ascent \
    --model_path best.ckpt \
    --ga_steps 2000 \
    --ga_lr 0.01 \
    --density_weight 0.3 \
    --kde_bandwidth 0.5
```

#### Optimal transport

Iterative Wasserstein-barycentric flow into the winning distribution.

```bash
python src/latent_trainer/paths/main.py optimal_transport \
    --model_path best.ckpt \
    --ot_reg 0.05
```

#### Neural flow

Counterfactual path via a learned OT-Flow Matching velocity field. Requires a separately trained flow checkpoint.

```bash
# Pure flow (no guidance):
python src/latent_trainer/paths/main.py neural_flow \
    --model_path best.ckpt \
    --flow_checkpoint best_flow.ckpt

# With classifier guidance:
python src/latent_trainer/paths/main.py neural_flow \
    --model_path best.ckpt \
    --flow_checkpoint best_flow.ckpt \
    --guidance_scale 0.2

# Print P(win) ceiling and path trace diagnostics:
python src/latent_trainer/paths/main.py neural_flow \
    --model_path best.ckpt \
    --flow_checkpoint best_flow.ckpt \
    --diagnose
```

---

### Bulk comparison

#### `compare` — evaluate all methods across N samples

```bash
python src/latent_trainer/paths/main.py compare \
    --model_path best.ckpt \
    --n_samples 1000 \
    --methods all \
    --save_raw \
    --output_dir output/compare/

# With a tuned config file:
python src/latent_trainer/paths/main.py compare \
    --model_path best.ckpt \
    --n_samples 1000 \
    --config output/compare/best_configs.json
```

#### `tune` — find best hyperparameters via Optuna

```bash
python src/latent_trainer/paths/main.py tune \
    --model_path best.ckpt \
    --methods gradient_ascent,optimal_transport \
    --n_trials 50 \
    --n_samples 200 \
    --metric auc \
    --output_config output/compare/best_configs.json
```

#### `replot` — regenerate plots from a saved report (no recomputation)

Requires a `report.pkl` produced by `compare --save_raw`.

```bash
python src/latent_trainer/paths/main.py replot output/compare/report.pkl

# Write plots to a different directory:
python src/latent_trainer/paths/main.py replot output/compare/report.pkl \
    --output_dir output/compare_replot/
```

#### `compare-datasets` — cross-dataset comparison

```bash
python src/latent_trainer/paths/main.py compare-datasets \
    --reports output/compare_ds1/report.pkl \
    --reports output/compare_ds2/report.pkl \
    --labels "Dataset A" "Dataset B" \
    --output_dir output/cross_dataset/
```
