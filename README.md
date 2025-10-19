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
python src\latent_trainer\models\train_model.py --epochs 20 --batch-size 256 --transform economy_average_vs_outcome
```
only economy_average_vs_outcome works 

