"""
Per-replay metadata the events cache does not hold: patch, map, date, players.

Built once through ``extract_with_manifest`` (``create_splits=False``), so it
covers exactly the manifest's replays in manifest order and fails, like every
cache, if a listed replay cannot be read. Saved as Parquet next to a JSON
sidecar holding the manifest hash; loading refuses a different manifest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from latent_trainer.features.extraction import (
    extract_with_manifest,
    load_source_dataset,
)
from latent_trainer.features.splits import SPLIT_NAMES, SplitManifest

logger = logging.getLogger(__name__)

METADATA_FILE = "replay_metadata.parquet"
SIDECAR_FILE = "replay_metadata.json"
PLAYER_FIELDS = ("toon", "nickname", "race", "selected_race", "apm", "result")


def replay_metadata(sc2_replay: Any) -> dict:
    """
    Read one replay's metadata; never skips, so manifest placement is strict.

    Parameters
    ----------
    sc2_replay : Any
        Parsed ``SC2ReplayData``.

    Returns
    -------
    dict
        ``elapsed_loops``, ``game_version``, ``base_build``, ``map_name``,
        ``time_utc`` and ``p<1|2>_<field>`` for each of :data:`PLAYER_FIELDS`;
        missing values are None.
    """
    metadata = getattr(sc2_replay, "metadata", None)
    details = getattr(sc2_replay, "details", None)
    header = getattr(sc2_replay, "header", None)
    row: dict = {
        "elapsed_loops": getattr(header, "elapsedGameLoops", None),
        "game_version": getattr(metadata, "gameVersion", None),
        "base_build": getattr(metadata, "baseBuild", None),
        "map_name": getattr(metadata, "mapName", None),
        "time_utc": getattr(details, "timeUTC", None),
    }
    by_player = {}
    for toon_desc in getattr(sc2_replay, "toonPlayerDescMap", []):
        info = toon_desc.toon_player_info
        by_player[int(info.playerID)] = (toon_desc.toon, info)
    for player_id in (1, 2):
        toon, info = by_player.get(player_id, (None, None))
        row[f"p{player_id}_toon"] = toon
        row[f"p{player_id}_nickname"] = getattr(info, "nickname", None)
        row[f"p{player_id}_race"] = getattr(info, "race", None)
        row[f"p{player_id}_selected_race"] = getattr(info, "selectedRace", None)
        row[f"p{player_id}_apm"] = getattr(info, "APM", None)
        row[f"p{player_id}_result"] = getattr(info, "result", None)
    return row


def build_metadata(
    single_json_dataset_path: Path,
    splits_path: Path,
    out_dir: Path,
    n_workers: int = 24,
) -> pd.DataFrame:
    """
    Extract metadata for every manifest replay and save it.

    Parameters
    ----------
    single_json_dataset_path : Path
        The single-JSON source dataset.
    splits_path : Path
        The split manifest.
    out_dir : Path
        Directory for the Parquet file and its sidecar.
    n_workers : int
        Worker processes, by default 24.

    Returns
    -------
    pd.DataFrame
        One row per manifest replay with ``split`` and ``replay_id``.
    """
    result = extract_with_manifest(
        load_source_dataset(single_json_dataset_path),
        replay_metadata,
        transform_name="eda_metadata",
        dataset_file=Path(single_json_dataset_path).name,
        splits_path=splits_path,
        create_splits=False,
        n_workers=n_workers,
    )
    manifest = result.manifest
    rows = []
    for split in SPLIT_NAMES:
        for replay_id, row in zip(
            manifest.ids(split), result.outputs[split], strict=True
        ):
            rows.append({"split": split, "replay_id": replay_id, **row})
    table = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / METADATA_FILE, index=False)
    sidecar = {"splits_sha256": manifest.sha256(), "n_rows": len(table)}
    (out_dir / SIDECAR_FILE).write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    logger.info(f"Wrote metadata for {len(table)} replays to {out_dir / METADATA_FILE}")
    return table


def load_metadata(out_dir: Path, manifest: SplitManifest) -> pd.DataFrame:
    """
    Load saved metadata, refusing one built from another manifest.

    Parameters
    ----------
    out_dir : Path
        Directory holding the Parquet file and its sidecar.
    manifest : SplitManifest
        The manifest the analysis uses.

    Returns
    -------
    pd.DataFrame
        The metadata table.

    Raises
    ------
    FileNotFoundError
        If the metadata has not been built.
    ValueError
        If it was built from a different manifest.
    """
    sidecar_path = out_dir / SIDECAR_FILE
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"{sidecar_path} missing; build it with --build-metadata"
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar["splits_sha256"] != manifest.sha256():
        raise ValueError(
            f"metadata was built from manifest {sidecar['splits_sha256'][:12]}, "
            f"the analysis uses {manifest.sha256()[:12]}; rebuild it"
        )
    return pd.read_parquet(out_dir / METADATA_FILE)
