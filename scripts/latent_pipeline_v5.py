"""
Latent Space Counterfactual Pipeline  —  N-Dimensional
========================================================
Stack:
  · PyTorch Lightning  — training loop, checkpointing
  · MLflow             — experiment tracking, metric/artifact logging
  · Ray Tune + Optuna  — hyperparameter search (TPE sampler + Median pruner)

Hyperparameters searched:
  lr, beta (KL weight), cls_w (classifier weight),
  latent_dim, hidden_multiplier, batch_size

After the search the best config is used to train the final model,
then the full counterfactual pipeline (OT / GA / Geodesic) runs as before.

Install:
    pip install torch pytorch-lightning mlflow ray[tune] optuna
                scikit-learn POT matplotlib numpy scipy umap-learn
"""

import os
import warnings
from functools import partial
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import ot
import pytorch_lightning as pl
import ray
import scipy.sparse.csgraph as csgraph
import torch
import torch.nn as nn
import torch.optim as optim
import umap
from matplotlib.lines import Line2D
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity, kneighbors_graph
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
HAS_UMAP = True

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

INPUT_DIM = 203
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Real dataset settings
DATA_CACHE_PATH = PROJECT_ROOT / "data" / "cached_dataset_rich.pt"
# "diff"   -> player0 - player1  (203 dims for rich transform)
# "concat" -> [player0, player1]  (406 dims for rich transform)
PLAYER_REPRESENTATION = "diff"

# Provide real names here — must be length INPUT_DIM
FEATURE_NAMES = [f"feature_{i:02d}" for i in range(INPUT_DIM)]

FINAL_EPOCHS = 5

TUNE_NUM_SAMPLES = 2
TUNE_MAX_EPOCHS = 5
TUNE_GRACE_PERIOD = 2

OT_REG = 0.0
GRAD_STEPS = 500
GRAD_LR = 0.02
GRAD_MOMENTUM = 0.9
GRAD_DENSITY_WEIGHT = 0.3
GRAD_KDE_BW = 0.5
GEODESIC_K = 12
N_WAYPOINTS = 10

MLFLOW_EXPERIMENT = "latent_vae_search"
TUNE_LOG_DIR = str(PROJECT_ROOT / "ray_results")  # trial logs
RAY_TEMP_DIR = str(PROJECT_ROOT / "ray_tmp")  # session/actor temp files
os.makedirs(PROJECT_ROOT / "plots", exist_ok=True)
os.makedirs(PROJECT_ROOT / "mlruns", exist_ok=True)
os.makedirs(TUNE_LOG_DIR, exist_ok=True)
os.makedirs(RAY_TEMP_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)


def _build_feature_names(input_dim: int, mode: str) -> list[str]:
    if mode == "diff":
        prefix = "p0_minus_p1"
    elif mode == "concat":
        prefix = "p0_p1_concat"
    else:
        prefix = "feature"
    return [f"{prefix}_{i:03d}" for i in range(input_dim)]


def _project_player_features(X: np.ndarray, mode: str) -> np.ndarray:
    """
    Convert [N, 2, F] player features into [N, D] model inputs.
    """
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(
            f"Expected cached features with shape [N, 2, F], got {tuple(X.shape)}"
        )

    if mode == "diff":
        return (X[:, 0, :] - X[:, 1, :]).astype(np.float32)
    if mode == "concat":
        return X.reshape(X.shape[0], -1).astype(np.float32)

    raise ValueError(
        f"Unsupported PLAYER_REPRESENTATION='{mode}'. Use 'diff' or 'concat'."
    )


def prepare_data(cache_path=DATA_CACHE_PATH, representation=PLAYER_REPRESENTATION):
    set_seed()

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cached dataset not found at '{cache_path}'. "
            "Generate it first (src/latent_trainer/features/main.py)."
        )

    cached = torch.load(cache_path, weights_only=True)
    X_train_raw = cached["train_features"].float().cpu().numpy()
    y_train = cached["train_labels"].float().cpu().numpy()
    X_val_raw = cached["val_features"].float().cpu().numpy()
    y_val = cached["val_labels"].float().cpu().numpy()
    X_test_raw = cached["test_features"].float().cpu().numpy()
    y_test = cached["test_labels"].float().cpu().numpy()

    X_train_flat = _project_player_features(X_train_raw, representation)
    X_val_flat = _project_player_features(X_val_raw, representation)
    X_test_flat = _project_player_features(X_test_raw, representation)

    scaler = StandardScaler().fit(X_train_flat)
    X_train = scaler.transform(X_train_flat).astype(np.float32)
    X_val = scaler.transform(X_val_flat).astype(np.float32)
    X_test = scaler.transform(X_test_flat).astype(np.float32)

    # Keep labels in float32 for BCE loss.
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)

    global FEATURE_NAMES
    FEATURE_NAMES = _build_feature_names(X_train.shape[1], representation)

    print(
        f"Loaded cached dataset from '{cache_path}' | rep='{representation}' | "
        f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


class LatentDataModule(pl.LightningDataModule):
    """
    Improvements over the naive version:
      - TensorDatasets built once at construction — not rebuilt each epoch
        or each Ray Tune trial.  This avoids repeated torch.tensor() copies.
      - pin_memory only enabled when CUDA + num_workers > 0.
        pin_memory with num_workers=0 is a no-op and triggers a Lightning warning.
      - persistent_workers keeps worker processes alive between epochs
        (measurable speedup when num_workers > 0; harmless when 0).
      - prefetch_factor lets workers queue batches ahead of the training loop.
      - drop_last=True on train avoids a tiny tail batch that can destabilise
        LayerNorm / BatchNorm at small effective batch sizes.
    """

    def __init__(
        self, X_train, y_train, X_val, y_val, batch_size: int = 64, num_workers: int = 0
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Pre-build once; Lightning's dataloader hooks just call these
        self._train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        self._val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

    def _make_loader(self, ds, *, shuffle: bool, drop_last: bool) -> DataLoader:
        use_pin = torch.cuda.is_available() and self.num_workers > 0
        use_pref = self.num_workers > 0
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=use_pin,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if use_pref else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train_ds, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False, drop_last=False)


# ══════════════════════════════════════════════════════════════
# ARCHITECTURE BUILDERS
# ══════════════════════════════════════════════════════════════


def _norm_layer(norm: str, dim: int) -> nn.Module | None:
    if norm == "layernorm":
        return nn.LayerNorm(dim)
    if norm == "batchnorm":
        return nn.BatchNorm1d(dim)
    return None  # "none"


def _mlp_block(in_dim: int, out_dim: int, norm: str, dropout: float) -> list:
    """Linear → [Norm] → GELU → [Dropout]."""
    layers: list = [nn.Linear(in_dim, out_dim)]
    n = _norm_layer(norm, out_dim)
    if n is not None:
        layers.append(n)
    layers.append(nn.GELU())
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return layers


def _build_encoder(
    input_dim: int, latent_dim: int, hidden_dims: list[int], norm: str, dropout: float
):
    """
    Returns (enc_shared, enc_mu, enc_logvar).
    hidden_dims: widths of intermediate layers, e.g. [256, 128].
    The last element feeds enc_mu / enc_logvar.
    """
    dims = [input_dim] + hidden_dims
    layers: list = []
    for i in range(len(dims) - 1):
        layers += _mlp_block(dims[i], dims[i + 1], norm, dropout)
    enc_shared = nn.Sequential(*layers)
    enc_mu = nn.Linear(dims[-1], latent_dim)
    enc_logvar = nn.Linear(dims[-1], latent_dim)
    return enc_shared, enc_mu, enc_logvar


def _build_decoder(
    latent_dim: int, output_dim: int, hidden_dims: list[int], norm: str, dropout: float
) -> nn.Sequential:
    """
    hidden_dims: widths starting from the latent side, e.g. [128, 256].
    No activation on the final output layer.
    """
    dims = [latent_dim] + hidden_dims
    layers: list = []
    for i in range(len(dims) - 1):
        layers += _mlp_block(dims[i], dims[i + 1], norm, dropout)
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


def _build_classifier(
    latent_dim: int, hidden_dims: list[int], norm: str, dropout: float
) -> nn.Sequential:
    """Classifier head — returns pre-sigmoid logit."""
    dims = [latent_dim] + hidden_dims
    layers: list = []
    for i in range(len(dims) - 1):
        layers += _mlp_block(dims[i], dims[i + 1], norm, dropout)
    layers.append(nn.Linear(dims[-1], 1))
    return nn.Sequential(*layers)


def config_to_arch(config: dict) -> tuple[list, list, list]:
    """
    Translates flat config keys → (enc_hidden, dec_hidden, clf_hidden).

    Encoder (input → latent, shrinking):
      enc_depth    int  1–4   number of hidden layers
      enc_width_0  int        width of the widest (first) hidden layer
      enc_taper    float      multiplicative width reduction per layer
                              0.5 = halve each time, 1.0 = keep flat

    Decoder (latent → output, expanding):
      dec_depth, dec_width_0, dec_taper  — same scheme
      The widths are built from the latent end outward, so dec_width_0
      is the *narrowest* layer (closest to latent space).

    Classifier head (latent → 1 logit, flat):
      clf_depth  int   1–3   number of hidden layers
      clf_width  int         flat width of all hidden layers
    """
    latent_dim = config["latent_dim"]

    def _taper_dims(depth, width0, taper):
        dims, w = [], float(width0)
        for _ in range(depth):
            dims.append(max(latent_dim, int(round(w))))
            w *= taper
        return dims

    enc_hidden = _taper_dims(
        config.get("enc_depth", 2),
        config.get("enc_width_0", 256),
        config.get("enc_taper", 0.5),
    )
    # Decoder expands: build same way but reverse so first dim is near latent
    dec_hidden = list(
        reversed(
            _taper_dims(
                config.get("dec_depth", 2),
                config.get("dec_width_0", 256),
                config.get("dec_taper", 0.5),
            )
        )
    )
    clf_hidden = [config.get("clf_width", 32)] * config.get("clf_depth", 2)

    return enc_hidden, dec_hidden, clf_hidden


# ══════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════


class VAEClassifierLightning(pl.LightningModule):
    """
    VAE + classifier head with fully searchable depth and width.

    Architecture config keys:
      enc_depth    int   1–4
      enc_width_0  int         widest encoder hidden layer
      enc_taper    float 0.5|1 width decay per layer
      dec_depth    int   1–4
      dec_width_0  int
      dec_taper    float 0.5|1
      clf_depth    int   1–3
      clf_width    int
      norm         str   "layernorm" | "batchnorm" | "none"
      dropout      float 0–0.4
      latent_dim   int

    Training config keys (unchanged from before):
      lr, beta, cls_w, batch_size, epochs
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        norm = config.get("norm", "layernorm")
        dropout = config.get("dropout", 0.0)

        enc_hidden, dec_hidden, clf_hidden = config_to_arch(config)

        (self.enc_shared, self.enc_mu, self.enc_logvar) = _build_encoder(
            config["input_dim"], config["latent_dim"], enc_hidden, norm, dropout
        )

        self.decoder = _build_decoder(
            config["latent_dim"], config["input_dim"], dec_hidden, norm, dropout
        )

        self.clf_body = _build_classifier(
            config["latent_dim"], clf_hidden, norm, dropout
        )
        self.clf_sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.enc_shared(x)
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decoder(z)
        p_win = self.clf_sigmoid(self.clf_body(z))
        return x_hat, p_win, mu, logvar

    @torch.no_grad()
    def embed(self, x):
        mu, _ = self.encode(x)
        return mu

    @torch.no_grad()
    def decode(self, z):
        return self.decoder(z)

    def p_win(self, z):
        return self.clf_sigmoid(self.clf_body(z))

    def logit_win(self, z):
        return self.clf_body(z)

    def _vae_loss(self, x, x_hat, mu, logvar, p_win, y):
        beta = self.config.get("beta", 0.4)
        cls_w = self.config.get("cls_w", 2.5)
        recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        cls = nn.functional.binary_cross_entropy(p_win.squeeze(), y)
        return recon + beta * kl + cls_w * cls, recon, kl, cls

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat, pw, mu, lv = self(x)
        loss, recon, kl, cls = self._vae_loss(x, x_hat, mu, lv, pw, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_recon", recon, on_epoch=True)
        self.log("train_kl", kl, on_epoch=True)
        self.log("train_cls", cls, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat, pw, mu, lv = self(x)
        loss, recon, kl, cls = self._vae_loss(x, x_hat, mu, lv, pw, y)
        acc = ((pw.squeeze() > 0.5).float() == y).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_recon", recon)
        self.log("val_kl", kl)
        self.log("val_cls", cls)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = self.config.get("lr", 8e-4)
        opt = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.config.get("epochs", FINAL_EPOCHS)
        )
        return {"optimizer": opt, "lr_scheduler": sched}


# ══════════════════════════════════════════════════════════════
# RAY TUNE
# ══════════════════════════════════════════════════════════════

SEARCH_SPACE = {
    # ── Training hyperparameters ──────────────────────────────
    "lr": tune.loguniform(1e-4, 1e-2),
    "beta": tune.uniform(0.1, 1.0),  # KL weight
    "cls_w": tune.uniform(1.0, 5.0),  # classifier loss weight
    "batch_size": tune.choice([64, 128, 256]),
    # ── Latent space ──────────────────────────────────────────
    "latent_dim": tune.choice([8, 16, 32, 64]),
    # ── Encoder depth / width ─────────────────────────────────
    "enc_depth": tune.choice([1, 2, 3, 4]),
    "enc_width_0": tune.choice([64, 128, 256, 512]),
    "enc_taper": tune.choice([0.5, 1.0]),  # 0.5=shrink each layer, 1.0=flat
    # ── Decoder depth / width ─────────────────────────────────
    "dec_depth": tune.choice([1, 2, 3, 4]),
    "dec_width_0": tune.choice([64, 128, 256, 512]),
    "dec_taper": tune.choice([0.5, 1.0]),
    # ── Classifier head ───────────────────────────────────────
    "clf_depth": tune.choice([1, 2, 3]),
    "clf_width": tune.choice([16, 32, 64]),
    # ── Regularisation ────────────────────────────────────────
    "norm": tune.choice(["layernorm", "batchnorm", "none"]),
    "dropout": tune.uniform(0.0, 0.4),
    # ── Fixed (not searched) ──────────────────────────────────
    "input_dim": INPUT_DIM,
    "epochs": TUNE_MAX_EPOCHS,
}


def _train_tune(
    config,
    X_train,
    y_train,
    X_val,
    y_val,
    parent_run_id=None,
    tracking_uri=None,
):
    """
    Single Ray Tune trial, runs in a separate Ray worker process.

    parent_run_id is passed explicitly from the main process because
    Ray workers have their own MLflow context — they cannot see any
    run that was started in the driver process.  We re-open the parent
    run here (nested=True) so the trial appears as a child in the UI.
    """
    dm = LatentDataModule(
        X_train, y_train, X_val, y_val, batch_size=config["batch_size"], num_workers=0
    )
    model = VAEClassifierLightning(config)
    tune_cb = TuneReportCallback(
        {"val_loss": "val_loss", "val_acc": "val_acc"}, on="validation_end"
    )

    # Re-establish MLflow context in this worker process
    # We must set the URI explicitly because Ray changes the working directory
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Open a child run.  If we have the parent id we attach to it so the
    # trial shows as nested; otherwise fall back to a flat top-level run.
    run_kwargs = dict(run_name="tune_trial")

    with mlflow.start_run(**run_kwargs) as trial_run:
        mlflow.log_params(
            {k: v for k, v in config.items() if k not in ("input_dim", "epochs")}
        )
        # Tag the run so we know it belongs to the sweep, even without nesting
        if parent_run_id:
            mlflow.set_tag("parent_run_id", parent_run_id)

        # PASS tracking_uri EXPLICITLY to the logger
        mlf_logger = MLFlowLogger(
            experiment_name=MLFLOW_EXPERIMENT,
            run_id=trial_run.info.run_id,
            tracking_uri=tracking_uri,
        )

        trainer = pl.Trainer(
            max_epochs=TUNE_MAX_EPOCHS,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=mlf_logger,
            callbacks=[tune_cb],
        )
        trainer.fit(model, dm)


def run_hyperparameter_search(X_train, y_train, X_val, y_val, tracking_uri=None):
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    print("=" * 60)
    print("HPARAM SEARCH  —  Ray Tune + Optuna")
    print(f"  Trials : {TUNE_NUM_SAMPLES}   Max epochs/trial : {TUNE_MAX_EPOCHS}")
    print("=" * 60)

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=False,
            _temp_dir=RAY_TEMP_DIR,  # keeps session files in the project, off the system drive
        )

    optuna_search = OptunaSearch(metric="val_loss", mode="min")
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=TUNE_MAX_EPOCHS,
        grace_period=TUNE_GRACE_PERIOD,
        reduction_factor=2,
    )
    # All 30 trials are nested under this parent run in the MLflow UI —
    # expand it to compare every trial side by side
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="hparam_search") as search_run:
        # Capture the run_id NOW, before spawning any Ray workers.
        # Workers run in separate processes and cannot see this run
        # unless we pass the id to them explicitly via partial().
        parent_run_id = search_run.info.run_id
        train_fn = partial(
            _train_tune,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            parent_run_id=parent_run_id,
            tracking_uri=tracking_uri,
        )
        mlflow.log_params(
            {
                "tune_num_samples": TUNE_NUM_SAMPLES,
                "tune_max_epochs": TUNE_MAX_EPOCHS,
                "tune_grace_period": TUNE_GRACE_PERIOD,
                "search_algorithm": "OptunaSearch (TPE)",
                "scheduler": "ASHA",
            }
        )

        # Windows has a 260-char path limit.  The default trial dirname
        # includes every param value and easily exceeds it.
        # This creator produces short names like "trial_0042" instead.
        _trial_counter = [0]

        def _short_dirname(trial):
            name = f"trial_{_trial_counter[0]:04d}"
            _trial_counter[0] += 1
            return name

        analysis = tune.run(
            train_fn,
            config=SEARCH_SPACE,
            num_samples=TUNE_NUM_SAMPLES,
            search_alg=optuna_search,
            scheduler=scheduler,
            resources_per_trial={
                "cpu": 2,
                "gpu": 0.5 if torch.cuda.is_available() else 0,
            },
            verbose=1,
            name="vae_optuna_search",
            storage_path=TUNE_LOG_DIR,  # Ray 2.x
            # local_dir=TUNE_LOG_DIR,       # Ray 1.x
            trial_dirname_creator=_short_dirname,
        )

        best_config = analysis.get_best_config(metric="val_loss", mode="min")
        best_trial = analysis.get_best_trial(metric="val_loss", mode="min")

        # Log the winner back onto the parent run for easy access
        mlflow.log_metric("best_val_loss", best_trial.last_result["val_loss"])
        mlflow.log_metric("best_val_acc", best_trial.last_result["val_acc"])
        mlflow.log_params(
            {
                f"best_{k}": v
                for k, v in best_config.items()
                if k not in ("input_dim", "epochs")
            }
        )

    print(f"\n  Best val_loss : {best_trial.last_result['val_loss']:.4f}")
    print(f"  Best val_acc  : {best_trial.last_result['val_acc']:.4f}")
    for k, v in best_config.items():
        if k not in ("input_dim", "epochs"):
            print(f"    {k:<20} = {v}")
    return best_config, analysis


# ══════════════════════════════════════════════════════════════
# FINAL TRAINING
# ══════════════════════════════════════════════════════════════


def train_final_model(config, X_train, y_train, X_val, y_val, tracking_uri=None):
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    config = {**config, "epochs": FINAL_EPOCHS}
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="final_model") as run:
        mlflow.log_params(
            {k: v for k, v in config.items() if k not in ("input_dim", "epochs")}
        )
        mlf_logger = MLFlowLogger(
            experiment_name=MLFLOW_EXPERIMENT,
            run_id=run.info.run_id,
            tracking_uri=tracking_uri,
        )
        dm = LatentDataModule(
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=config["batch_size"],
            num_workers=4,
        )
        model = VAEClassifierLightning(config)
        ckpt_cb = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:03d}-{val_loss:.4f}",
            dirpath="checkpoints/",
        )
        early_stop_cb = EarlyStopping(
            monitor="val_loss", patience=40, mode="min", verbose=False
        )
        trainer = pl.Trainer(
            max_epochs=FINAL_EPOCHS,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=mlf_logger,
            callbacks=[ckpt_cb, early_stop_cb],
            enable_progress_bar=True,
            log_every_n_steps=5,
        )
        print("=" * 60)
        print(
            f"FINAL TRAINING  —  up to {FINAL_EPOCHS} epochs (early stop patience=40)"
        )
        print(f"  Device : {DEVICE}   latent_dim : {config['latent_dim']}")
        print("=" * 60)
        trainer.fit(model, dm)

        best_val = ckpt_cb.best_model_score
        print(f"\n  Best val_loss : {best_val:.4f}")
        mlflow.log_metric("best_val_loss", float(best_val))
        mlflow.log_artifact("checkpoints/")

    model = VAEClassifierLightning.load_from_checkpoint(
        ckpt_cb.best_model_path, config=config
    )
    model.eval()
    model.to(DEVICE)
    return model


# ══════════════════════════════════════════════════════════════
# EMBED
# ══════════════════════════════════════════════════════════════


def embed_training_data(model, X_train, y_train):
    with torch.no_grad():
        Z_train = model.embed(torch.tensor(X_train).to(DEVICE)).cpu().numpy()
    return Z_train, Z_train[y_train == 1], Z_train[y_train == 0]


def pick_subject(model, X_test, y_test):
    print("=" * 60)
    print("STEP  —  Embedding new (unseen) point")
    print("=" * 60)
    loss_idx = np.where(y_test == 0)[0]
    x_raw = X_test[loss_idx[0]]
    xt = torch.tensor(x_raw).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        z_new = model.embed(xt).cpu().numpy().squeeze()
        p = model.p_win(model.embed(xt)).item()
    print(f"  Latent dim : {z_new.shape[0]}")
    print(f"  P(win)     : {p:.3f}  ← predicted loss\n")
    return z_new, x_raw


# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════


def path_optimal_transport(z_new, Z_win, reg=OT_REG, n_waypoints=N_WAYPOINTS):
    a = np.array([1.0])
    M = ot.dist(z_new.reshape(1, -1), Z_win, metric="sqeuclidean")
    T = (
        ot.sinkhorn(a, ot.unif(len(Z_win)), M, reg=reg)
        if reg > 0
        else ot.emd(a, ot.unif(len(Z_win)), M)
    )
    weights = T[0]
    z_target = (weights[:, None] * Z_win).sum(0) / weights.sum()
    print(f"  [OT]  Wasserstein distance : {(M * T).sum():.4f}")
    alphas = np.linspace(0, 1, n_waypoints)
    return np.array([(1 - a) * z_new + a * z_target for a in alphas])


def path_gradient_ascent(
    model,
    z_new,
    Z_all,
    steps=GRAD_STEPS,
    lr=GRAD_LR,
    momentum=GRAD_MOMENTUM,
    density_w=GRAD_DENSITY_WEIGHT,
    kde_bw=GRAD_KDE_BW,
    n_waypoints=N_WAYPOINTS,
):
    kde = KernelDensity(kernel="gaussian", bandwidth=kde_bw).fit(Z_all)
    z = torch.tensor(
        z_new.copy(), dtype=torch.float32, requires_grad=True, device=DEVICE
    )
    velocity = torch.zeros_like(z)
    trajectory = [z_new.copy()]

    model.eval()
    for step in range(steps):
        if z.grad is not None:
            z.grad.zero_()
        model.logit_win(z.unsqueeze(0)).backward()

        with torch.no_grad():
            grad_cls = z.grad.clone()
            z_np = z.detach().cpu().numpy()
            grads_kde = np.zeros_like(z_np)
            for i in range(len(z_np)):
                zp, zm = z_np.copy(), z_np.copy()
                zp[i] += 1e-3
                zm[i] -= 1e-3
                grads_kde[i] = (
                    float(
                        kde.score_samples(zp.reshape(1, -1))[0]
                        - kde.score_samples(zm.reshape(1, -1))[0]
                    )
                    / 2e-3
                )
            total_grad = grad_cls + density_w * torch.tensor(
                grads_kde, dtype=torch.float32, device=DEVICE
            )
            velocity = momentum * velocity + lr * total_grad
            z = z.detach().to(DEVICE) + velocity
            z.requires_grad_(True)
            trajectory.append(z.detach().cpu().numpy().copy())
            current_p = model.p_win(z.unsqueeze(0)).item()
            if step % 50 == 0:
                print(
                    f"    step {step:4d}  P(win)={current_p:.4f}  |grad|={grad_cls.norm().item():.5f}"
                )
            if current_p > 0.95:
                print(f"  [GA]  Converged at step {step}  P(win)={current_p:.4f}")
                break

    trajectory = np.array(trajectory)
    path = trajectory[np.linspace(0, len(trajectory) - 1, n_waypoints, dtype=int)]
    final_p = model.p_win(
        torch.tensor(path[-1], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    ).item()
    print(f"  [GA]  Final P(win) : {final_p:.4f}  |  steps : {len(trajectory) - 1}")
    return path


def path_geodesic(z_new, Z_win, Z_all, k=GEODESIC_K, n_waypoints=N_WAYPOINTS):
    kde_win = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(Z_win)
    best_win = Z_win[np.argmax(kde_win.score_samples(Z_win))]
    Z_graph = np.vstack([Z_all, z_new.reshape(1, -1), best_win.reshape(1, -1)])
    src_idx, tgt_idx = len(Z_graph) - 2, len(Z_graph) - 1

    A = kneighbors_graph(Z_graph, n_neighbors=k, mode="distance", include_self=False)
    A = (A + A.T) / 2
    dist_matrix, predecessors = csgraph.shortest_path(
        A, method="D", directed=False, indices=src_idx, return_predecessors=True
    )

    path_indices, node = [], tgt_idx
    while node != src_idx and node >= 0:
        path_indices.append(node)
        node = predecessors[node]
    path_indices.append(src_idx)
    path_indices = path_indices[::-1]

    if len(path_indices) < 2:
        print("  [GEO] Warning: no path found — straight line fallback.")
        return np.array(
            [(1 - a) * z_new + a * best_win for a in np.linspace(0, 1, n_waypoints)]
        )

    full_path = Z_graph[path_indices]
    path = full_path[np.linspace(0, len(full_path) - 1, n_waypoints, dtype=int)]
    print(
        f"  [GEO] Nodes in path : {len(path_indices)}  |  graph dist : {dist_matrix[tgt_idx]:.4f}"
    )
    return path


# ══════════════════════════════════════════════════════════════
# DECODE
# ══════════════════════════════════════════════════════════════


def decode_path(model, path_z):
    zt = torch.tensor(path_z, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        x_hat = model.decode(zt).cpu().numpy()
        p_vals = model.p_win(zt).cpu().numpy().squeeze()
    return x_hat, p_vals


# ══════════════════════════════════════════════════════════════
# FEEDBACK  —  three signal types
# ══════════════════════════════════════════════════════════════


def generate_feedback(model, scaler, path_z, feature_names, top_k=5, method_name="OT"):
    """
    Three complementary feedback signals:

      1. Raw delta (start → end)         — overall direction of change
      2. Minimum-viable delta             — change only up to the P(win)=0.5 crossover
      3. P(win)-gain-weighted delta       — features that moved *while P(win) rose*

    Returns a dict with all three ranked lists plus raw arrays.
    """
    x_decoded, p_vals = decode_path(model, path_z)

    # ── Inverse-transform every waypoint back to original feature scale
    x_orig = scaler.inverse_transform(x_decoded)  # (N_WAYPOINTS, input_dim)
    x_start = x_orig[0]
    x_end = x_orig[-1]

    # ── 1. Raw start → end delta
    raw_delta = x_end - x_start

    # ── 2. Minimum-viable delta  (stop at first P(win) ≥ 0.5 crossover)
    cross_idx = np.where(np.diff(np.sign(p_vals - 0.5)))[0]
    if len(cross_idx) > 0:
        threshold_wp = cross_idx[0] + 1  # first waypoint already past the boundary
        mv_delta = x_orig[threshold_wp] - x_start
        mv_p = p_vals[threshold_wp]
        mv_label = f"waypoint {threshold_wp}/{len(p_vals) - 1}  (P(win)={mv_p:.3f})"
    else:
        # Path never crosses — use midpoint as best available
        mid = len(p_vals) // 2
        mv_delta = x_orig[mid] - x_start
        mv_p = p_vals[mid]
        mv_label = f"midpoint (P(win)={mv_p:.3f}, no crossing found)"

    # ── 3. P(win)-gain-weighted delta
    #    For each consecutive waypoint pair, weight the feature change
    #    by how much P(win) improved in that step (ignore steps that hurt)
    p_gains = np.diff(p_vals).clip(min=0)  # (N_WAYPOINTS-1,)
    x_steps = np.diff(x_orig, axis=0)  # (N_WAYPOINTS-1, input_dim)
    if p_gains.sum() > 0:
        weighted_delta = (x_steps * p_gains[:, None]).sum(0) / p_gains.sum()
    else:
        weighted_delta = raw_delta.copy()

    # ── Build ranked feedback tables for each signal
    def _rank_table(delta, label):
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

    # ── Print all three tables
    header = f"  {'Feature':<22}  {'Current':>9}  {'Target':>9}  {'Δ':>9}"
    sep = "  " + "─" * 54

    print(f"\n{'═' * 60}")
    print(f"  [{method_name}] FEEDBACK REPORT  —  top {top_k} features")
    print(f"{'═' * 60}")

    for rows, lbl in [(raw_rows, raw_lbl), (mv_rows, mv_lbl), (wgt_rows, wgt_lbl)]:
        print(f"\n  ── {lbl}")
        print(header)
        print(sep)
        for r in rows:
            print(
                f"  {r['feature']:<22}  {r['current']:>9.3f}"
                f"  {r['target']:>9.3f}  {r['direction']} {abs(r['delta']):>7.3f}"
            )

    return {
        "method": method_name,
        "raw": raw_rows,
        "minimum_viable": mv_rows,
        "gain_weighted": wgt_rows,
        # raw arrays for downstream use / plotting
        "_raw_delta": raw_delta,
        "_mv_delta": mv_delta,
        "_weighted_delta": weighted_delta,
        "_x_start": x_start,
        "_p_vals": p_vals,
        "_mv_crossover_wp": int(cross_idx[0]) if len(cross_idx) > 0 else None,
    }


# ══════════════════════════════════════════════════════════════
# FEEDBACK VISUALISATION
# ══════════════════════════════════════════════════════════════


def visualise_feedback(all_feedback: dict, out_prefix="latent_paths"):
    """
    One figure per path method.
    Each figure has 4 panels:
      Row 0: P(win) curve with minimum-viable crossover marker
      Row 1: Raw delta bars  (start → end)
      Row 2: Minimum-viable delta bars  (up to P(win)=0.5)
      Row 3: P(win)-gain-weighted delta bars
    Top-k features are annotated with their name on all bar charts.
    """
    TOP_K_ANNOTATE = 5

    for method_name, fb in all_feedback.items():
        c_light, c_dark = COLORS.get(method_name, ("#fff", "#aaa"))

        raw_delta = fb["_raw_delta"]
        mv_delta = fb["_mv_delta"]
        weighted_delta = fb["_weighted_delta"]
        p_vals = fb["_p_vals"]
        mv_wp = fb["_mv_crossover_wp"]
        alphas_wp = np.linspace(0, 1, len(p_vals))
        feat_names = [
            r["feature"] for r in sorted(fb["raw"], key=lambda x: x["priority"])
        ]
        n_feat = len(raw_delta)

        fig = plt.figure(figsize=(14, 14))
        fig.patch.set_facecolor("#0D1117")
        gs = gridspec.GridSpec(
            4, 1, figure=fig, hspace=0.45, left=0.08, right=0.97, top=0.93, bottom=0.05
        )

        # ── Row 0: P(win) curve ──────────────────────────────
        ax_p = fig.add_subplot(gs[0])
        ax_p.plot(alphas_wp, p_vals, "o-", color=c_light, lw=2, markersize=6)
        ax_p.axhline(0.5, color="#555", linestyle="--", lw=1, label="Decision boundary")
        ax_p.fill_between(
            alphas_wp,
            p_vals,
            0.5,
            where=(p_vals >= 0.5),
            color=COLORS["win"],
            alpha=0.2,
            label="Win zone",
        )
        ax_p.fill_between(
            alphas_wp,
            p_vals,
            0.5,
            where=(p_vals < 0.5),
            color=COLORS["loss"],
            alpha=0.2,
            label="Loss zone",
        )

        # Mark crossover
        cross = np.where(np.diff(np.sign(p_vals - 0.5)))[0]
        for c in cross:
            ax_p.axvline(alphas_wp[c], color="white", lw=1.2, alpha=0.5, linestyle=":")
            ax_p.text(
                alphas_wp[c] + 0.01,
                0.08,
                "min viable →",
                color="white",
                fontsize=7,
                alpha=0.7,
            )

        # Annotate each waypoint with its P(win)
        for i, (a, p) in enumerate(zip(alphas_wp, p_vals)):
            ax_p.text(
                a,
                p + 0.03,
                f"{p:.2f}",
                color=c_light,
                fontsize=5.5,
                ha="center",
                va="bottom",
            )

        ax_p.set_ylim(0, 1.12)
        ax_p.set_xlim(-0.02, 1.02)
        ax_p.legend(
            fontsize=7,
            facecolor="#161B22",
            edgecolor="#30363D",
            labelcolor="white",
            loc="upper left",
        )
        _style_ax(
            ax_p,
            f"{method_name.upper()} — P(win) along path",
            xlabel="Waypoint α  (0 = current player, 1 = target)",
            ylabel="P(win)",
        )

        # ── Shared bar-chart helper ───────────────────────────
        def _bar_panel(ax, delta, title, show_names=True):
            feat_idx = np.arange(n_feat)
            bar_cols = [COLORS["win"] if v > 0 else COLORS["loss"] for v in delta]
            ax.bar(feat_idx, delta, color=bar_cols, alpha=0.8, width=0.8)
            ax.axhline(0, color="#555", lw=0.8)

            # Annotate top-k by absolute magnitude
            top_k_idx = np.argsort(np.abs(delta))[::-1][:TOP_K_ANNOTATE]
            y_max = np.abs(delta).max() if np.abs(delta).max() > 0 else 1
            for rank, i in enumerate(top_k_idx):
                offset = np.sign(delta[i]) * 0.04 * y_max
                ax.text(
                    i,
                    delta[i] + offset,
                    f"#{rank + 1} {FEATURE_NAMES[i]}",
                    color="white",
                    fontsize=5,
                    ha="center",
                    va="bottom" if delta[i] >= 0 else "top",
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="#0D1117")],
                )
            _style_ax(ax, title, xlabel="Feature index", ylabel="Δ (original scale)")

        _bar_panel(
            ax_p2 := fig.add_subplot(gs[1]),
            raw_delta,
            "Signal 1 — Raw Δ  (full path start → end)",
        )
        _bar_panel(
            ax_mv := fig.add_subplot(gs[2]),
            mv_delta,
            "Signal 2 — Minimum-viable Δ  (up to P(win) ≥ 0.5 crossover)",
        )
        _bar_panel(
            ax_wgt := fig.add_subplot(gs[3]),
            weighted_delta,
            "Signal 3 — P(win)-gain-weighted Δ  (causally attributed)",
        )

        fig.suptitle(
            f"Player Feedback Report  ·  Method: {method_name.upper()}",
            color="white",
            fontsize=13,
            fontweight="bold",
            y=0.97,
        )

        out = f"plots/{out_prefix}_feedback_{method_name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved → {out}")
        mlflow.log_artifact(out)


# ══════════════════════════════════════════════════════════════
# PROJECTIONS
# ══════════════════════════════════════════════════════════════


def fit_projections(Z_train):
    projections = {}
    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(Z_train)
    projections["PCA"] = pca.transform
    projections["_pca_obj"] = pca

    print("  Fitting t-SNE …", flush=True)
    tsne_full = TSNE(
        n_components=2,
        perplexity=min(30, len(Z_train) // 4),
        random_state=SEED,
        max_iter=600,
        init="pca",
        learning_rate="auto",
    ).fit_transform(Z_train)
    projections["_tsne_train"] = tsne_full

    def tsne_project(Z_query):
        dists = np.linalg.norm(Z_train[:, None] - Z_query[None], axis=2)
        return tsne_full[dists.argmin(axis=0)]

    projections["tSNE"] = tsne_project

    if HAS_UMAP:
        print("  Fitting UMAP …", flush=True)
        try:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                random_state=SEED,
                verbose=False,
            )
            reducer.fit(Z_train)
            projections["_umap_train"] = reducer.transform(Z_train)
            projections["UMAP"] = reducer.transform
        except Exception as e:
            warnings.warn(
                "UMAP fit failed; continuing with PCA and t-SNE only. "
                f"Reason: {type(e).__name__}: {e}"
            )
            projections["UMAP"] = None
    else:
        projections["UMAP"] = None

    return projections


# ══════════════════════════════════════════════════════════════
# VISUALISATION  (projection + analytics)
# ══════════════════════════════════════════════════════════════

COLORS = {
    "ot": ("#F4A261", "#E76F51"),
    "ga": ("#57CC99", "#22577A"),
    "geo": ("#C77DFF", "#7B2FBE"),
    "win": "#457B9D",
    "loss": "#E63946",
    "new_pt": "#FFBA08",
}


def _style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor("#161B22")
    ax.set_title(title, color="white", fontsize=9, pad=5)
    ax.set_xlabel(xlabel, color="#666", fontsize=7)
    ax.set_ylabel(ylabel, color="#666", fontsize=7)
    ax.tick_params(colors="#555", labelsize=6)
    for sp in ax.spines.values():
        sp.set_color("#30363D")


def _draw_projection_panel(
    ax, proj_name, proj_fn, Z_win_train, Z_loss_train, z_new, paths, **_
):
    ax.scatter(
        *proj_fn(Z_loss_train).T,
        c=COLORS["loss"],
        alpha=0.18,
        s=8,
        lw=0,
        rasterized=True,
    )
    ax.scatter(
        *proj_fn(Z_win_train).T, c=COLORS["win"], alpha=0.18, s=8, lw=0, rasterized=True
    )
    ax.scatter(
        *proj_fn(z_new.reshape(1, -1))[0],
        c=COLORS["new_pt"],
        s=200,
        zorder=12,
        marker="*",
        edgecolors="white",
        linewidths=0.8,
    )

    for name, path_z in paths.items():
        c_light, c_dark = COLORS.get(name, ("#fff", "#aaa"))
        path_2d = proj_fn(path_z)
        n = len(path_2d)
        for i in range(n - 1):
            ax.plot(
                path_2d[i : i + 2, 0],
                path_2d[i : i + 2, 1],
                "-",
                color=c_light,
                lw=1.5 + 1.0 * (i / (n - 1)),
                alpha=0.4 + 0.6 * (i / (n - 1)),
                solid_capstyle="round",
            )
        ax.scatter(
            path_2d[:, 0],
            path_2d[:, 1],
            c=[c_light] * n,
            s=np.linspace(10, 55, n),
            zorder=10,
            alpha=0.9,
            edgecolors="none",
        )
        mid = int(n * 0.6)
        if mid < n - 1:
            ax.annotate(
                "",
                xy=(path_2d[mid + 1, 0], path_2d[mid + 1, 1]),
                xytext=(path_2d[mid, 0], path_2d[mid, 1]),
                arrowprops=dict(
                    arrowstyle="-|>", color=c_light, lw=1.6, mutation_scale=10
                ),
                zorder=11,
            )
        ax.scatter(
            *path_2d[-1], c=c_dark, s=80, zorder=11, edgecolors="white", linewidths=0.6
        )
        ax.text(
            path_2d[-1, 0],
            path_2d[-1, 1] - 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            name.upper(),
            color=c_light,
            fontsize=6,
            ha="center",
            va="top",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="#0D1117")],
        )

    _style_ax(
        ax, f"{proj_name} projection", xlabel=f"{proj_name}₁", ylabel=f"{proj_name}₂"
    )


def visualise_all(
    Z_win,
    Z_loss,
    z_new,
    paths,
    projections,
    model,
    latent_dim,
    out_prefix="latent_paths",
):
    proj_items = [("PCA", projections["PCA"]), ("tSNE", projections["tSNE"])]
    if projections.get("UMAP") is not None:
        proj_items.append(("UMAP", projections["UMAP"]))

    n_proj = len(proj_items)
    n_methods = len(paths)
    alphas_wp = np.linspace(0, 1, N_WAYPOINTS)

    fig1, axes1 = plt.subplots(1, n_proj, figsize=(6.5 * n_proj, 6.5))
    fig1.patch.set_facecolor("#0D1117")
    if n_proj == 1:
        axes1 = [axes1]
    for ax, (pname, pfn) in zip(axes1, proj_items):
        _draw_projection_panel(ax, pname, pfn, Z_win, Z_loss, z_new, paths)

    legend_elems = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=COLORS["new_pt"],
            markersize=10,
            label="New point (loss)",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["loss"],
            markersize=6,
            alpha=0.6,
            label="Loss cloud",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["win"],
            markersize=6,
            alpha=0.6,
            label="Win cloud",
            linestyle="None",
        ),
    ] + [
        Line2D([0], [0], color=COLORS[n][0], lw=2, label=f"Path: {n.upper()}")
        for n in paths
    ]
    fig1.legend(
        handles=legend_elems,
        loc="lower center",
        ncol=len(legend_elems),
        fontsize=8,
        facecolor="#161B22",
        edgecolor="#30363D",
        labelcolor="white",
        bbox_to_anchor=(0.5, -0.02),
    )
    fig1.suptitle(
        f"Latent Space Projections  ·  latent_dim={latent_dim}",
        color="white",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig1.tight_layout()
    out1 = f"plots/{out_prefix}_projections.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
    plt.close(fig1)
    print(f"  Saved → {out1}")
    mlflow.log_artifact(out1)

    fig2 = plt.figure(figsize=(6 * n_methods, 10))
    fig2.patch.set_facecolor("#0D1117")
    gs2 = gridspec.GridSpec(
        3,
        n_methods,
        figure=fig2,
        hspace=0.5,
        wspace=0.35,
        left=0.06,
        right=0.97,
        top=0.91,
        bottom=0.07,
    )

    for col, (name, path_z) in enumerate(paths.items()):
        c_light, c_dark = COLORS.get(name, ("#fff", "#aaa"))
        x_decoded, p_vals = decode_path(model, path_z)

        ax_p = fig2.add_subplot(gs2[0, col])
        ax_p.plot(alphas_wp, p_vals, "o-", color=c_light, lw=2, markersize=5)
        ax_p.axhline(0.5, color="#555", linestyle="--", lw=1)
        ax_p.fill_between(
            alphas_wp,
            p_vals,
            0.5,
            where=(p_vals >= 0.5),
            color=COLORS["win"],
            alpha=0.2,
        )
        ax_p.fill_between(
            alphas_wp,
            p_vals,
            0.5,
            where=(p_vals < 0.5),
            color=COLORS["loss"],
            alpha=0.2,
        )
        for c in np.where(np.diff(np.sign(p_vals - 0.5)))[0]:
            ax_p.axvline(alphas_wp[c], color="white", lw=0.8, alpha=0.4, linestyle=":")
        ax_p.set_ylim(0, 1)
        ax_p.set_xlim(0, 1)
        _style_ax(
            ax_p,
            f"{name.upper()} — P(win) along path",
            xlabel="α  (0=start → 1=target)",
            ylabel="P(win)",
        )

        ax_d = fig2.add_subplot(gs2[1, col])
        delta = x_decoded[-1] - x_decoded[0]
        ax_d.bar(
            np.arange(len(delta)),
            delta,
            color=[COLORS["win"] if v > 0 else COLORS["loss"] for v in delta],
            alpha=0.85,
            width=0.7,
        )
        ax_d.axhline(0, color="#555", lw=0.8)
        for i in np.argsort(np.abs(delta))[-3:]:
            ax_d.text(
                i,
                delta[i] + np.sign(delta[i]) * 0.02 * np.abs(delta).max(),
                f"f{i}",
                color="white",
                fontsize=6,
                ha="center",
            )
        _style_ax(
            ax_d,
            f"{name.upper()} — Feature Δ (start→end)",
            xlabel="Feature index",
            ylabel="Δ (decoded space)",
        )

        ax_h = fig2.add_subplot(gs2[2, col])
        hmap = ax_h.imshow(
            x_decoded.T,
            aspect="auto",
            cmap="RdBu_r",
            interpolation="nearest",
            extent=[0, 1, -0.5, x_decoded.shape[1] - 0.5],
        )
        ax_h.set_yticks(range(x_decoded.shape[1]))
        ax_h.set_yticklabels(
            [f"f{i}" for i in range(x_decoded.shape[1])], color="#888", fontsize=5
        )
        ax_h.set_xticks(np.linspace(0, 1, N_WAYPOINTS))
        ax_h.set_xticklabels(
            [f"{a:.1f}" for a in alphas_wp], color="#888", fontsize=5, rotation=45
        )
        plt.colorbar(hmap, ax=ax_h, pad=0.02, fraction=0.04).ax.tick_params(
            colors="#888", labelsize=5
        )
        _style_ax(
            ax_h,
            f"{name.upper()} — Feature values along path",
            xlabel="α",
            ylabel="Feature",
        )

    fig2.suptitle(
        f"Path Analytics  ·  latent_dim={latent_dim}",
        color="white",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )
    out2 = f"plots/{out_prefix}_analytics.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"  Saved → {out2}")
    mlflow.log_artifact(out2)


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════


def run_pipeline():
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data()
    input_dim = X_train.shape[1]

    # Keep Ray search-space input_dim aligned with the selected dataset representation.
    SEARCH_SPACE["input_dim"] = input_dim

    print(f"\n{'═' * 60}")
    print(f"  Latent OT Pipeline  ·  input={input_dim}d  ·  device={DEVICE}")
    print(f"{'═' * 60}\n")

    # 1. Define SQLite DB path (for metrics/params)
    db_path = PROJECT_ROOT / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"

    # Keep fluent MLflow API and Lightning MLFlowLogger on the same backend.
    mlflow.set_tracking_uri(tracking_uri)

    # Defensive cleanup in case a previous failed run left an active context.
    if mlflow.active_run() is not None:
        mlflow.end_run()

    print(f"  Tracking URI : {tracking_uri}")

    best_config, analysis = run_hyperparameter_search(
        X_train,
        y_train,
        X_val,
        y_val,
        tracking_uri=tracking_uri,
    )
    model = train_final_model(
        best_config,
        X_train,
        y_train,
        X_val,
        y_val,
        tracking_uri=tracking_uri,
    )
    latent_dim = best_config["latent_dim"]

    Z_train, Z_win, Z_loss = embed_training_data(model, X_train, y_train)
    z_new, x_new_raw = pick_subject(model, X_test, y_test)

    print("=" * 60)
    print("Fitting 2D projections")
    print("=" * 60)
    projections = fit_projections(Z_train)
    print()

    print("=" * 60)
    print("Computing paths")
    print("=" * 60)
    path_ot = path_optimal_transport(z_new, Z_win)
    print()
    path_ga = path_gradient_ascent(model, z_new, Z_train)
    print()
    path_geo = path_geodesic(z_new, Z_win, Z_train)
    print()
    paths = {"ot": path_ot, "ga": path_ga, "geo": path_geo}

    with mlflow.start_run(run_name="path_analysis"):
        # ── Path comparison table
        print("=" * 60)
        print("Path comparison")
        print("=" * 60)
        print(f"  {'Method':<6}  {'P(win) start':<14}  {'P(win) end':<12}  {'Length'}")
        print("  " + "-" * 50)
        for name, path_z in paths.items():
            _, p_vals = decode_path(model, path_z)
            length = np.linalg.norm(np.diff(path_z, axis=0), axis=1).sum()
            print(
                f"  {name.upper():<6}  {p_vals[0]:<14.3f}  {p_vals[-1]:<12.3f}  {length:.4f}"
            )
            mlflow.log_metrics(
                {
                    f"{name}_p_start": float(p_vals[0]),
                    f"{name}_p_end": float(p_vals[-1]),
                    f"{name}_length": float(length),
                }
            )

        # ── Projection + analytics plots
        print()
        visualise_all(
            Z_win,
            Z_loss,
            z_new,
            paths,
            projections,
            model,
            latent_dim=latent_dim,
            out_prefix="latent_paths",
        )

        # ── Feedback: all three signals for each path method
        print("\n" + "=" * 60)
        print("FEEDBACK  —  actionable feature recommendations")
        print("=" * 60)
        all_feedback = {}
        for name, path_z in paths.items():
            fb = generate_feedback(
                model,
                scaler,
                path_z,
                feature_names=FEATURE_NAMES,
                top_k=5,
                method_name=name.upper(),
            )
            all_feedback[name] = fb
            mlflow.log_dict(
                {k: v for k, v in fb.items() if not k.startswith("_")},
                f"feedback_{name}.json",
            )

        # ── Feedback visualisation (one figure per method)
        print()
        visualise_feedback(all_feedback, out_prefix="latent_paths")

    print("\nDone.  Run `mlflow ui` to explore results.\n")


if __name__ == "__main__":
    run_pipeline()
