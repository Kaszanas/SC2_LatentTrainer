from latent_trainer.features.data_utils import load_and_normalize
from latent_trainer.settings import DATA_DIR

if __name__ == "__main__":
    normalized_data = load_and_normalize(
        DATA_DIR / "cached_dataset_rich_sc2egset_sc2reset.pt"
    )

    print("Train features shape:", normalized_data.train_X.shape)

    print("Validation features shape:", normalized_data.val_X.shape)

    print("Test features shape:", normalized_data.test_X.shape)
