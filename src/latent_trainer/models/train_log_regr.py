import logging
from guided_vae import Classifier, suGuidedVAE
import torch
from pathlib import Path

from latent_trainer.config import LOGGING_FORMAT


# TODO regresja dla wyfrany/przegrany na srednich wartosciach

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)

    download_path = Path("./data/download").resolve().as_posix()
    unpack_path = Path("./data/unpack").resolve().as_posix()

    device = torch.device("cpu")

    torch.manual_seed(1024)
