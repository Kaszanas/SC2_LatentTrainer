# Usage

## Training

### Unified Entrypoint

The main training CLI supports two pipelines and two modes:

```bash
# Two-stage pipeline, single run:
uv run python -m latent_trainer --pipeline two_stage --cache data/cached_dataset_rich.pt

# Guided VAE pipeline:
uv run python -m latent_trainer --pipeline guided_vae --cache data/cached_dataset_rich.pt

# Hyperparameter sweep (Ray Tune + Optuna):
uv run python -m latent_trainer --mode sweep --n-trials 30
```

See all options:

```bash
uv run python -m latent_trainer --help
```

### Standalone Guided VAE

```bash
uv run python -m latent_trainer.models.train_model --cached data/cached_dataset_rich.pt
```

## MLFlow Tracking

Runs are logged to `sqlite:///mlflow.db` by default.  Launch the UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Override with `--mlflow-uri http://your-server:5000` when using a remote server.

## Feedback Path Finder

After training, generate improvement guidance for losing player samples:

```bash
# Linear interpolation (fast):
uv run python feedback_path.py --strategy linear --method centroid

# Gradient ascent + KDE density (3 feedback signals):
uv run python feedback_path.py --strategy gradient_kde --top-k 15
```

### Strategies

| Strategy | Path method | Feedback | Speed |
|---|---|---|---|
| `linear` | Straight-line interpolation toward centroid or k-NN target | Raw delta only | Fast |
| `gradient_kde` | Gradient ascent on P(win) + KDE density regularisation | Raw delta, minimum-viable delta, P(win)-gain-weighted delta | Slower |

### Gradient+KDE hyperparameters

| Flag | Default | Description |
|---|---|---|
| `--ga-steps` | 500 | Maximum gradient ascent iterations |
| `--ga-lr` | 0.02 | Learning rate |
| `--ga-momentum` | 0.9 | Momentum coefficient |
| `--density-weight` | 0.3 | KDE density gradient weight |
| `--kde-bandwidth` | 0.5 | Gaussian KDE bandwidth |

See all options:

```bash
uv run python feedback_path.py --help
```
