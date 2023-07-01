from sc2_datasets.lightning.sc2_egset_datamodule import SC2EGSetDataModule

from latent_trainer.config import LOGGING_FORMAT

import logging

from pathlib import Path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)

    download_path = Path("./data/download").resolve().as_posix()
    unpack_path = Path("./data/unpack").resolve().as_posix()

    logging.info(f"{download_path=}")
    logging.info(f"{unpack_path=}")
