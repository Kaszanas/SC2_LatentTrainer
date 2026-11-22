"""
Write a sample of SC2EGSet / SC2GGSet replays to a JSONL file.

Replays are picked evenly spaced across the whole index range of the merged
single-JSON dataset (default), or uniformly at random without replacement
(``--strategy random``). Each output line is one replay
object, copied byte-for-byte from the source file (no re-serialization), so the
output is exactly what ``SC2DatasetSingleJSON`` would load for that index.

Usage:
    uv run --extra cuda python scripts/sample_sc2egset_jsonl.py
    uv run --extra cuda python scripts/sample_sc2egset_jsonl.py --n-samples 10000 --seed 42 --output data/sc2egset_sample.jsonl
"""

import argparse
import json
import logging
import random
from pathlib import Path

from sc2_datasets.utils.json_utils import get_json_offsets

from latent_trainer.settings import DATA_DIR

DEFAULT_JSON_PATH = Path("H:/sc2egset_merged/sc2egset_merged.json")

logger = logging.getLogger(__name__)


def read_raw_object(file_handle, offset: int) -> str:
    """
    Read the replay object starting at ``offset`` as a JSON string.

    Parameters
    ----------
    file_handle : BinaryIO
        Single-JSON file opened in binary mode.
    offset : int
        Byte offset of the object's line.

    Returns
    -------
    str
        The object's JSON text without the trailing comma.
    """
    file_handle.seek(offset)
    line = file_handle.readline().decode("utf-8").strip()
    if line.endswith(","):
        line = line[:-1].rstrip()
    return line


def sample_to_jsonl(
    json_path: Path,
    output_path: Path,
    n_samples: int,
    seed: int,
    strategy: str,
) -> None:
    """
    Write a sample of replays from the single-JSON dataset as JSON Lines.

    The chosen source indices are also written to ``<output>.indices.json``.

    Parameters
    ----------
    json_path : Path
        Single-JSON SC2EGSet file.
    output_path : Path
        Destination ``.jsonl`` file.
    n_samples : int
        Number of replays to write.
    seed : int
        Seed for random sampling.
    strategy : str
        ``"linspace"`` for evenly spaced indices, otherwise a seeded random sample.

    Raises
    ------
    ValueError
        If ``n_samples`` exceeds the dataset size or a line is not a JSON object.
    """
    offsets_path = json_path.with_name(json_path.stem + "_offsets.json")
    offsets = get_json_offsets(json_filepath=json_path, offsets_filepath=offsets_path)
    total = len(offsets)

    if n_samples > total:
        raise ValueError(f"Requested {n_samples} samples but dataset has only {total}.")

    if strategy == "linspace":
        # Evenly spaced across the whole index range; distinct since n_samples <= total:
        indices = [i * total // n_samples for i in range(n_samples)]
    else:
        # Sorted so the source file is read sequentially rather than seeking randomly:
        indices = sorted(random.Random(seed).sample(range(total), n_samples))
    logger.info(
        f"Sampling {n_samples} of {total} replays (strategy={strategy}, seed={seed})."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("rb") as src, output_path.open("w", encoding="utf-8") as dst:
        for written, index in enumerate(indices, start=1):
            line = read_raw_object(src, offsets[index])
            if not line.startswith("{"):
                raise ValueError(f"Index {index} did not yield a JSON object.")
            json.loads(line)  # fail loudly on a malformed object
            dst.write(line + "\n")
            if written % 500 == 0:
                logger.info(f"  {written}/{n_samples} written")

    output_path.with_suffix(".indices.json").write_text(json.dumps(indices))
    logger.info(f"Wrote {output_path} ({output_path.stat().st_size / 1e9:.2f} GB).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument(
        "--output", type=Path, default=DATA_DIR / "sc2egset_sample.jsonl"
    )
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42, help="Only for 'random'.")
    parser.add_argument(
        "--strategy",
        choices=["linspace", "random"],
        default="linspace",
        help="'linspace': evenly spaced indexes; 'random': uniform random draw.",
    )
    args = parser.parse_args()

    sample_to_jsonl(
        args.json_path, args.output, args.n_samples, args.seed, args.strategy
    )
