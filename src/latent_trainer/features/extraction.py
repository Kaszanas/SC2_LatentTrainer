"""Ordered, manifest-driven replay extraction shared by every cache builder.

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
    """The full single-JSON dataset, indexed exactly as recorded in ``splits.json``."""
    datamodule = SC2EGSetDataModuleSingleJSON(
        json_path=single_json_dataset_path,
        download=False,
    )
    datamodule.prepare_data()
    return datamodule.dataset


@dataclass
class ExtractionRecord:
    """One replay's outcome: an output, a skip, or an error."""

    index: int
    replay_id: str | None
    output: Any = None
    skipped: bool = False
    error: str | None = None

    @property
    def kept(self) -> bool:
        return self.error is None and not self.skipped


@dataclass
class ExtractionResult:
    """Per-split outputs in manifest (index) order."""

    outputs: dict[str, list[Any]]
    manifest: SplitManifest
    n_skipped: int
    n_errors: int


def _extract_chunk(
    dataset_object: Any,
    indices: list[int],
    transform_fn: TransformFn,
) -> list[ExtractionRecord]:
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
    """Run ``transform_fn`` over ``indices``; records come back in input order."""
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
    errors = [record.error for record in records if record.error is not None]
    if errors:
        logger.warning(f"  {len(errors)} replays failed; top causes:")
        for cause, count in Counter(errors).most_common(5):
            logger.warning(f"    {count:>5}x  {cause}")


def place_by_manifest(
    records: list[ExtractionRecord],
    manifest: SplitManifest,
) -> dict[str, list[Any]]:
    """Put each manifest replay's output into its split, or raise.

    Fails if a listed replay is missing, skipped, errored, or its id differs
    from the manifest (i.e. the source JSON changed under it).
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
    """Extract replays and place them into train/val/test via ``splits.json``.

    ``create_splits`` builds a new manifest from the replays this run keeps
    (never overwriting). Otherwise the manifest must exist and is enforced.
    ``n_samples > 0`` limits the run to a seeded subset.
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
