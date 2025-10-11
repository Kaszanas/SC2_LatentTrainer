# SC2 Latent Trainer

An approach to try and create a traversal through the latent space of a model that has been optimized for classifying player skill in StarCraft 2.

## Dependencies

Please make sure to install PyTorch with CUDA support before running `uv sync`.

## Quick Start Guide

### Step 1: Install dependencies

```bash
uv sync
```

### Step 2: Start the training

```bash
python src\latent_trainer\models\train_model.py
```

### Step 3: Monitor training with TensorBoard (in a separate terminal)

```bash
tensorboard --logdir=output/tensorboard_logs
```

## Advanced Usage

You can customize your training run with various options (see command line interface below).

Example with custom parameters:

```bash
python src\latent_trainer\models\train_model.py --epochs 20 --batch-size 256 --transform mmr_vs_result
```

## Usage

### CLICK command line interface

```python
@click.command()
@click.option('--batch-size', '-b', default=128, help='input batch size for training (default: 128)', type=int)
@click.option('--output', default='output', help='output directory for results')
@click.option('--epochs', default=10, help='number of epochs to train (default: 10)', type=int)
@click.option('--nz', default=16, help='bottleneck size', type=int)
@click.option('--cls', default=200.0, help='classification error weight for supervised Guided-VAE', type=float)
@click.option('--num_workers', default=0, help='number of workers for dataloader', type=int)
@click.option('--test_interval', default=1, help='interval for testing', type=int)
@click.option('--lr', default=1e-4, help='learning rate', type=float)
@click.option('--weight_decay', default=1e-5, help='weight decay', type=float)
@click.option('--lr_c', default=1e-4, help='classifier learning rate(in supervised version)', type=float)
@click.option('--weight_decay_c', default=1e-4, help='classifier weight decay(in supervised version)', type=float)
@click.option('--transform', default='mmr_vs_result', type=click.Choice(['mmr_vs_result', 'economy_average_vs_outcome', 'average_player_stats', 'select_outcome_1v1']), help='which transform to use')
@click.option('--optuna', 'use_optuna', is_flag=True, help='use Optuna for hyperparameter optimization')
@click.option('--n_trials', default=20, help='number of Optuna trials (only used with --optuna)', type=int)
@click.option('--optuna_epochs', default=3, help='number of epochs per trial for Optuna (only used with --optuna)', type=int)
@click.option('--optuna_db', default='sqlite:///optuna_study.db', help='Optuna database URL for dashboard (only used with --optuna)', type=str)
@click.option('--study_name', default='vae_optimization', help='Optuna study name (only used with --optuna)', type=str)
```
