"""Quick test to check MMR vs result data shape."""

import sys
sys.path.append("./src/latent_trainer/models")

from sc2_datasets.lightning.sc2_egset_datamodule import SC2EGSetDataModule
from sc2_datasets.available_replaypacks import SC2EGSET_DATASET_REPLAYPACKS
from sc2_datasets.transforms.mmr_vs_result import mmr_vs_result

# Set up data module with MMR transform
datamodule = SC2EGSetDataModule(
    unpack_dir="./data/unpack",
    download_dir="./data/download", 
    download=True,
    replaypacks=SC2EGSET_DATASET_REPLAYPACKS,
    transform=mmr_vs_result,
)

print("Preparing data...")
datamodule.prepare_data()
datamodule.setup()

dataloader = datamodule.train_dataloader()

print("Checking first batch...")
for batch_idx, (data, label) in enumerate(dataloader):
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Label shape: {label.shape}")
    print(f"Label type: {label.dtype}")
    print(f"Data sample: {data}")
    print(f"Label sample: {label}")
    break

print("Test completed!")
