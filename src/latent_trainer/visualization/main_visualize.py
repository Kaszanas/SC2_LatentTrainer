"""Module for visualizing the dataset, and all of the intermediate steps."""

from sc2_datasets.lightning.sc2_egset_datamodule import SC2EGSetDataModule

from pathlib import Path

if __name__ == "__main__":
    # Paths:
    root_project_path = Path("../../../").resolve()
    data_dir_path = Path(root_project_path, "data").resolve()
    data_download_path = Path(data_dir_path, "download").resolve().as_posix()
    data_unpack_path = Path(data_dir_path, "unpack").resolve().as_posix()

    # DataModule:
    datamodule = SC2EGSetDataModule(
        download_dir=data_download_path,
        unpack_dir=data_unpack_path,
    )
