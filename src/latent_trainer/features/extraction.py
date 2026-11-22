"""
Ordered, manifest-driven replay extraction shared by every cache builder.

Replays are extracted in index order (``executor.map`` keeps submission order),
identified by stable id, and placed strictly by ``splits.json``: a listed replay
that the transform skips or fails on is an error, never a silent split change.
"""

from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from sc2_datasets.lightning.sc2_egset_datamodule import SC2EGSetDataModuleSingleJSON
from tqdm import tqdm

from latent_trainer.features.splits import (
    SPLIT_NAMES,
    ReplayRef,
    SplitManifest,
    SplitManifestError,
    assign_splits,
    replay_id_from_filepath,
    sc2_datasets_version,
)

logger = logging.getLogger(__name__)

# Transform contract: any value for a kept replay, None to skip it.
TransformFn = Callable[[Any], Any]


def load_source_dataset(single_json_dataset_path: Path | str) -> Any:
    """
    Load the full single-JSON dataset, indexed as recorded in ``splits.json``.

    Parameters
    ----------
    single_json_dataset_path : Path | str
        Path to the single-JSON SC2EGSet file.

    Returns
    -------
    SC2DatasetSingleJSON
        Dataset returning ``SC2ReplayData`` by source index.
    """
    datamodule = SC2EGSetDataModuleSingleJSON(
        json_path=single_json_dataset_path,
        download=False,
    )
    datamodule.prepare_data()
    return datamodule.dataset


@dataclass
class ExtractionRecord:
    """
    Outcome of one replay: an output, a skip, or an error.

    Parameters
    ----------
    index : int
        Index of the replay in the source dataset.
    replay_id : str | None
        Stable replay id, or None if the replay could not be read.
    output : Any
        Transform output for a kept replay, by default None.
    skipped : bool
        Whether the transform returned None, by default False.
    error : str | None
        Exception type and message if extraction failed, by default None.
    """

    index: int
    replay_id: str | None
    output: Any = None
    skipped: bool = False
    error: str | None = None

    @property
    def kept(self) -> bool:
        """
        Whether the replay produced an output.

        Returns
        -------
        bool
            True if the replay neither errored nor was skipped.
        """
        return self.error is None and not self.skipped


@dataclass
class ExtractionResult:
    """
    Per-split transform outputs, in manifest (index) order.

    Parameters
    ----------
    outputs : dict[str, list[Any]]
        Transform outputs per split name.
    manifest : SplitManifest
        Manifest used to place the outputs (subsampled if ``n_samples`` was set).
    n_skipped : int
        Number of extracted replays the transform skipped.
    n_errors : int
        Number of extracted replays that raised.
    """

    outputs: dict[str, list[Any]]
    manifest: SplitManifest
    n_skipped: int
    n_errors: int


def _extract_chunk(
    dataset_object: Any,
    indices: list[int],
    transform_fn: TransformFn,
) -> list[ExtractionRecord]:
    """
    Extract one chunk of replays inside a single worker call.

    Parameters
    ----------
    dataset_object : Any
        Indexable dataset returning ``SC2ReplayData``.
    indices : list[int]
        Source indices of the chunk.
    transform_fn : TransformFn
        Transform returning an output, or None to skip the replay.

    Returns
    -------
    list[ExtractionRecord]
        One record per index, in input order.
    """
    records: list[ExtractionRecord] = []
    for index in indices:
        replay_id: str | None = None
        try:
            replay = dataset_object[index]
            replay_id = replay_id_from_filepath(replay.filepath)
            output = transform_fn(replay)
        except Exception as exc:  # noqa: BLE001 - recorded; fatal for manifest replays
            records.append(
                ExtractionRecord(
                    index=index,
                    replay_id=replay_id,
                    error=f"{type(exc).__name__}: {exc}"[:200],
                )
            )
            continue

        if output is None:
            records.append(
                ExtractionRecord(index=index, replay_id=replay_id, skipped=True)
            )
        else:
            records.append(
                ExtractionRecord(index=index, replay_id=replay_id, output=output)
            )
    return records


def extract_ordered(
    dataset_object: Any,
    indices: list[int],
    transform_fn: TransformFn,
    n_workers: int = 8,
    chunk_size: int = 64,
    desc: str = "extract",
) -> list[ExtractionRecord]:
    """
    Run a transform over replays in parallel, keeping input order.

    ``executor.map`` returns chunk results in submission order whatever order
    the workers finish in.

    Parameters
    ----------
    dataset_object : Any
        Indexable dataset returning ``SC2ReplayData``.
    indices : list[int]
        Source indices to extract.
    transform_fn : TransformFn
        Transform returning an output, or None to skip the replay.
    n_workers : int
        Worker processes; 1 uses a single thread, by default 8.
    chunk_size : int
        Replays per worker task, by default 64.
    desc : str
        Progress bar label, by default "extract".

    Returns
    -------
    list[ExtractionRecord]
        One record per index, in the order of ``indices``.
    """
    chunks = [
        indices[start : start + chunk_size]
        for start in range(0, len(indices), chunk_size)
    ]
    executor = ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor

    records: list[ExtractionRecord] = []
    with executor(max_workers=n_workers) as pool:
        results = pool.map(
            _extract_chunk,
            [dataset_object] * len(chunks),
            chunks,
            [transform_fn] * len(chunks),
        )
        with tqdm(total=len(indices), desc=desc) as progress:
            for chunk, chunk_records in zip(chunks, results):
                records.extend(chunk_records)
                progress.update(len(chunk))
    return records


def _log_errors(records: list[ExtractionRecord]) -> None:
    """
    Log the number of failed replays and their most common causes.

    Parameters
    ----------
    records : list[ExtractionRecord]
        Extraction records to summarise.
    """
    errors = [record.error for record in records if record.error is not None]
    if errors:
        logger.warning(f"  {len(errors)} replays failed; top causes:")
        for cause, count in Counter(errors).most_common(5):
            logger.warning(f"    {count:>5}x  {cause}")


def place_by_manifest(
    records: list[ExtractionRecord],
    manifest: SplitManifest,
) -> dict[str, list[Any]]:
    """
    Put each manifest replay's output into its split, strictly.

    Parameters
    ----------
    records : list[ExtractionRecord]
        Extraction records covering every manifest replay.
    manifest : SplitManifest
        Manifest defining split membership.

    Returns
    -------
    dict[str, list[Any]]
        Outputs per split name, in manifest (index) order.

    Raises
    ------
    SplitManifestError
        If a listed replay is missing, skipped, errored, or its id differs from
        the manifest (the source JSON changed under it).
    """
    by_index = {record.index: record for record in records}

    problems: list[str] = []
    outputs: dict[str, list[Any]] = {split: [] for split in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        for ref in manifest.refs(split):
            record = by_index.get(ref.index)
            if record is None:
                problems.append(f"{split}: index {ref.index} was not extracted")
            elif record.replay_id != ref.replay_id:
                problems.append(
                    f"{split}: index {ref.index} is {record.replay_id!r}, "
                    f"manifest says {ref.replay_id!r}"
                )
            elif record.error is not None:
                problems.append(f"{split}: {ref.replay_id} failed: {record.error}")
            elif record.skipped:
                problems.append(
                    f"{split}: {ref.replay_id} was skipped by the transform"
                )
            else:
                outputs[split].append(record.output)

    if problems:
        shown = "\n  ".join(problems[:10])
        raise SplitManifestError(
            f"{len(problems)} manifest replays could not be placed "
            f"(first {min(10, len(problems))}):\n  {shown}\n"
            "This transform keeps fewer replays than the one that created the manifest."
        )
    return outputs


def extract_with_manifest(
    dataset_object: Any,
    transform_fn: TransformFn,
    *,
    transform_name: str,
    dataset_file: str,
    splits_path: Path,
    create_splits: bool,
    n_samples: int = 0,
    seed: int = 42,
    n_workers: int = 8,
    chunk_size: int = 64,
) -> ExtractionResult:
    """
    Extract replays and place them into train/val/test via ``splits.json``.

    Parameters
    ----------
    dataset_object : Any
        Indexable dataset returning ``SC2ReplayData``.
    transform_fn : TransformFn
        Transform returning an output, or None to skip the replay.
    transform_name : str
        Transform name, recorded in a newly created manifest.
    dataset_file : str
        File name of the source dataset, checked against the manifest.
    splits_path : Path
        Location of ``splits.json``.
    create_splits : bool
        Create a new manifest from the replays this run keeps (never
        overwriting); otherwise the existing manifest is loaded and enforced.
    n_samples : int
        If > 0, restrict the run to a seeded subset: of all replays when
        creating, proportionally per split when reusing, by default 0.
    seed : int
        Seed for the split assignment and subsampling, by default 42.
    n_workers : int
        Worker processes, by default 8.
    chunk_size : int
        Replays per worker task, by default 64.

    Returns
    -------
    ExtractionResult
        Outputs per split, the manifest used, and skip/error counts.

    Raises
    ------
    FileExistsError
        If ``create_splits`` is set and ``splits_path`` already exists.
    FileNotFoundError
        If ``create_splits`` is not set and there is no manifest.
    SplitManifestError
        If the manifest does not match the dataset or a listed replay cannot
        be placed.
    """
    n_source = len(dataset_object)

    if create_splits:
        if splits_path.exists():
            raise FileExistsError(
                f"--create_splits given but {splits_path} already exists. Reuse it "
                "or delete it explicitly to re-split."
            )
        indices = list(range(n_source))
        if n_samples > 0:
            generator = torch.Generator().manual_seed(seed)
            indices = sorted(
                torch.randperm(n_source, generator=generator)[:n_samples].tolist()
            )

        records = extract_ordered(
            dataset_object,
            indices,
            transform_fn,
            n_workers,
            chunk_size,
            desc=f"{transform_name} (all replays)",
        )
        _log_errors(records)
        kept = [
            ReplayRef(index=record.index, replay_id=record.replay_id)
            for record in records
            if record.kept
        ]
        manifest = SplitManifest(
            seed=seed,
            dataset_file=dataset_file,
            n_source_replays=n_source,
            sc2_datasets_version=sc2_datasets_version(),
            created_by_transform=transform_name,
            subsampled=n_samples > 0,
            splits=assign_splits(kept, seed=seed),
        )
        manifest.save(splits_path)
        logger.info(
            f"  Created split manifest {splits_path} from {len(kept)} kept replays: "
            f"{manifest.sizes()} (sha256 {manifest.sha256()[:12]})"
        )
    else:
        manifest = SplitManifest.load(splits_path)
        manifest.check_matches_dataset(
            dataset_file=dataset_file, n_source_replays=n_source
        )
        if n_samples > 0:
            manifest = manifest.subsample(n_samples=n_samples, seed=seed)
        logger.info(
            f"  Using split manifest {splits_path}: {manifest.sizes()} "
            f"(created by '{manifest.created_by_transform}', "
            f"sha256 {manifest.sha256()[:12]})"
        )
        records = extract_ordered(
            dataset_object,
            manifest.all_indices(),
            transform_fn,
            n_workers,
            chunk_size,
            desc=f"{transform_name} (manifest replays)",
        )
        _log_errors(records)

    outputs = place_by_manifest(records, manifest)
    return ExtractionResult(
        outputs=outputs,
        manifest=manifest,
        n_skipped=sum(record.skipped for record in records),
        n_errors=sum(record.error is not None for record in records),
    )
