"""Smoke test: verify LitGuidedVAE works with TensorDict batches end-to-end."""

from pathlib import Path

from torch.utils.data import DataLoader

from latent_trainer.features.data_utils import _collate_sc2, load_and_normalize
from latent_trainer.features.type import CachedSC2Dataset
from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE

DATA_PATH = Path("data/cached_dataset_rich")

nd = load_and_normalize(DATA_PATH)

sample_player = nd.train_X[0, 0]
input_dim = sum(t.numel() for t in sample_player.values(True, True))
print(f"input_dim: {input_dim}")

train_loader = DataLoader(
    CachedSC2Dataset(features=nd.train_X, labels=nd.train_y),
    batch_size=8,
    shuffle=False,
    collate_fn=_collate_sc2,
)
batch = next(iter(train_loader))
print(f"batch[0] type: {type(batch[0])}")
print(f"batch[1] shape: {batch[1].shape}")

model = LitGuidedVAE(
    input_dim=input_dim,
    encoder_hidden_dims=[64, 128],
    supervised_dim=4,
    vae_latent_dim=16,
    mean=nd.mean,
    std=nd.std,
)
print(f"in_keys: {model.in_keys}")
print(f"mean buffer shape: {model.mean.shape}")

data, label = model._unpack_batch(batch)
print(f"data shape: {data.shape}")  # expect [8, 2, 202]
print(f"label shape: {label.shape}")  # expect [8]

recon, mu, logvar, cls_pred = model.model(data)
print(f"recon shape:  {recon.shape}")  # [8, 2, 202]
print(f"mu shape:     {mu.shape}")  # [8, 2, 16]
print(f"cls_pred shape: {cls_pred.shape}")  # [8, 1]

print("OK — all shapes correct")

# ── Feature inspection ──────────────────────────────────────────────────────
def print_td_values(td, title: str) -> None:
    print(f"\n=== {title} ===")
    flat = td.flatten_keys(".")
    for key, val in sorted(flat.items()):
        print(f"  {key:<60s}: {val.item():>12.4f}")


print_td_values(nd.mean, "Normalization mean (per feature)")
print_td_values(nd.std,  "Normalization std  (per feature)")
print_td_values(nd.train_X[0, 0], "Normalized values: replay 0, player 0")
print_td_values(nd.train_X[0, 1], "Normalized values: replay 0, player 1")
