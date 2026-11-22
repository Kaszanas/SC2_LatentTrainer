"""
Report why the events transform skipped each source replay.

The skipped replays are the source replays missing from a split manifest that
the events transform created. Each one is re-parsed and passed through
``event_skip_reason``, which runs the transform's own code path, so the
reported reason is the exact branch that rejected it. Writes a JSON report with
the count per reason and one record per replay.

Usage:
    uv run --extra cuda python scripts/report_skipped_replays.py
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from latent_trainer.features.event_transform import SKIP_REASONS, event_skip_reason
from latent_trainer.features.extraction import load_source_dataset
from latent_trainer.features.splits import SplitManifest, replay_id_from_filepath
from latent_trainer.settings import DATA_DIR, OUTPUT_DIR

DEFAULT_JSON_PATH = Path("H:/sc2egset_merged/sc2egset_merged.json")

logger = logging.getLogger(__name__)


def skipped_replay_report(manifest: SplitManifest, dataset) -> dict:
    """
    Explain every source replay that is not in the manifest.

    Parameters
    ----------
    manifest : SplitManifest
        Manifest created by the events transform over the whole dataset.
    dataset : Any
        Indexable source dataset the manifest was created from.

    Returns
    -------
    dict
        ``manifest_sha256``, ``n_source_replays``, ``n_kept``, ``n_skipped``,
        ``counts`` (code -> replays) and ``replays`` (``index``, ``replay_id``,
        ``code``, ``detail`` per skipped replay). ``code`` is ``"error"`` if
        loading or transforming raised, and ``"kept_now"`` if the replay passes
        today (the manifest is then stale for this code).

    Raises
    ------
    ValueError
        If the manifest was not created by the events transform over the full
        dataset, or does not match the dataset's size.
    """
    if manifest.created_by_transform != "events" or manifest.subsampled:
        raise ValueError(
            "skip reasons are only defined for a manifest created by the events "
            f"transform over the full dataset (got {manifest.created_by_transform!r}, "
            f"subsampled={manifest.subsampled})"
        )
    if manifest.n_source_replays != len(dataset):
        raise ValueError(
            f"manifest covers {manifest.n_source_replays} replays, "
            f"dataset has {len(dataset)}"
        )

    kept = set(manifest.all_indices())
    skipped = [i for i in range(len(dataset)) if i not in kept]
    logger.info(
        f"{len(skipped)} of {len(dataset)} source replays are not in the manifest"
    )

    records = []
    for position, index in enumerate(skipped, start=1):
        replay_id, code, detail = "", "", ""
        try:
            replay = dataset[index]
            replay_id = replay_id_from_filepath(replay.filepath)
            reason = event_skip_reason(replay)
            code, detail = reason if reason else ("kept_now", "")
        except Exception as error:  # noqa: BLE001 -- report, do not stop
            code, detail = "error", repr(error)
        records.append(
            {"index": index, "replay_id": replay_id, "code": code, "detail": detail}
        )
        if position % 25 == 0:
            logger.info(f"  {position}/{len(skipped)}")

    counts = Counter(record["code"] for record in records)
    return {
        "manifest_sha256": manifest.sha256(),
        "n_source_replays": len(dataset),
        "n_kept": len(kept),
        "n_skipped": len(skipped),
        "counts": dict(counts.most_common()),
        "replays": records,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--splits-path", type=Path, default=DATA_DIR / "splits.json")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR / "skipped_replays_events.json"
    )
    args = parser.parse_args()

    report = skipped_replay_report(
        SplitManifest.load(args.splits_path), load_source_dataset(args.json_path)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSkipped {report['n_skipped']} of {report['n_source_replays']} replays:")
    for code, count in report["counts"].items():
        print(f"  {count:5d}  {code:18s} {SKIP_REASONS.get(code, '')}")
    print(f"Report: {args.output}")
