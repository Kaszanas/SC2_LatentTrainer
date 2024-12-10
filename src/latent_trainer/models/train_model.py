"""Train models module."""

import logging
from pathlib import Path

from latent_trainer.config import LOGGING_FORMAT

from latent_trainer.models.guided_vae import SupervisedGuidedVAE

from sc2_datasets.torch.sc2_egset_dataset import SC2EGSetDataset
from sc2_datasets.available_replaypacks import EXAMPLE_SYNTHETIC_REPLAYPACKS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)

    download_path = Path("./data/download").resolve().as_posix()
    unpack_path = Path("./data/unpack").resolve().as_posix()

    logging.info(f"{download_path=}")
    logging.info(f"{unpack_path=}")

    dataset = SC2EGSetDataset(
        download_dir=download_path,
        unpack_dir=unpack_path,
        names_urls=EXAMPLE_SYNTHETIC_REPLAYPACKS,
    )
    # Iterate over instances:
    for i in range(len(dataset)):
        replay = dataset[i]

    # TODO: Train the model on aggregated data:

    # TODO: Train the model on sequential data with metadata:
