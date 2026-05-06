from pathlib import Path

from sc2_datasets.lightning.sc2_egset_datamodule import (
    SC2EGSetDataModuleSingleJSON,
)

from latent_trainer.features.data_utils import load_and_normalize
from latent_trainer.settings import DATA_DIR

if __name__ == "__main__":
    normalized_data = load_and_normalize(DATA_DIR / "cached_dataset_rich_sc2egset.pt")

    print("Train features shape:", normalized_data.train_X.shape)

    print("Validation features shape:", normalized_data.val_X.shape)

    print("Test features shape:", normalized_data.test_X.shape)

    summed = (
        normalized_data.train_X.shape[0]
        + normalized_data.val_X.shape[0]
        + normalized_data.test_X.shape[0]
    )
    print("Total samples:", summed)

    single_json_dataset_path = Path("H:/sc2egset_merged/sc2egset_merged.json")

    sc2egset = SC2EGSetDataModuleSingleJSON(
        json_path=single_json_dataset_path,
        download=False,
    )

    sc2egset.prepare_data()

    print(f"sc2egset dataset length: {len(sc2egset.dataset)}")
