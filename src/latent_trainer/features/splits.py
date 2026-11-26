"""
Persisted train/val/test split manifest (``splits.json``) shared by every cache.

The manifest is created once from the replays an extraction run keeps (seeded
80/10/10 cut of whole series, see :func:`series_groups`), then every cache places
replays by stable id against it. Caches store their row ids and the manifest
hash, so shared test sets are checkable.
"""

from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePath
from typing import Any

import torch

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
MANIFEST_VERSION = 2
SUPPORTED_VERSIONS = (1, 2)
GROUPINGS = ("replay", "series")

# Identity of one replay's players: (p1 nickname, p2 nickname, p1 toon, p2 toon).
ReplayIdentity = tuple[str | None, str | None, str | None, str | None]


class SplitManifestError(RuntimeError):
    """The manifest does not match the dataset or the extraction results."""


@dataclass(frozen=True)
class ReplayRef:
    """
    A replay's position in the source JSON and its stable identity.

    Parameters
    ----------
    index : int
        Index of the replay in the single-JSON source dataset.
    replay_id : str
        Stable id ``<replaypack>/<file>``, see :func:`replay_id_from_filepath`.
    series : str | None
        Series group id (the smallest replay id of the group) when the manifest
        is grouped by series, else None.
    """

    index: int
    replay_id: str
    series: str | None = None


def replay_id_from_filepath(filepath: str | PurePath) -> str:
    """
    Normalise ``SC2ReplayData.filepath`` to a platform-independent replay id.

    Parameters
    ----------
    filepath : str | PurePath
        Filepath as built by ``sc2_datasets`` (backslashes on Windows).

    Returns
    -------
    str
        The id ``<replaypack>/<file>`` with forward slashes on every OS.
    """
    return str(filepath).replace("\\", "/")


def subsample_manifest_path(splits_path: Path, n_samples: int) -> Path:
    """
    Return where a ``--create_splits --n_samples`` run writes its manifest.

    Parameters
    ----------
    splits_path : Path
        Path of the full manifest, e.g. ``data/splits.json``.
    n_samples : int
        Number of sampled replays.

    Returns
    -------
    Path
        A sibling path such as ``data/splits_n500.json``.
    """
    splits_path = Path(splits_path)
    return splits_path.with_name(f"{splits_path.stem}_n{n_samples}{splits_path.suffix}")


_CLAN_TAG = re.compile(r"^<[^>]*>")


def normalise_nickname(raw: str | None) -> str | None:
    """
    Reduce a lobby nickname to the player's name, for matching across games.

    The data stores clan tags HTML-escaped (``&lt;mouz&gt;HeroMarine``) and some
    clients write them as ``clan<sp/>name``; both are stripped, then the name is
    case-folded.

    Parameters
    ----------
    raw : str | None
        Nickname as recorded in ``toonPlayerDescMap``.

    Returns
    -------
    str | None
        The normalised name, or None if nothing is left.
    """
    if raw is None:
        return None
    name = html.unescape(str(raw)).split("<sp/>")[-1]
    name = _CLAN_TAG.sub("", name).strip().casefold()
    return name or None


def replay_identity(sc2_replay: Any) -> ReplayIdentity:
    """
    Read both players' normalised nicknames and toons (account ids).

    Parameters
    ----------
    sc2_replay : Any
        Parsed ``SC2ReplayData``.

    Returns
    -------
    ReplayIdentity
        ``(p1 nickname, p2 nickname, p1 toon, p2 toon)``; missing values are None.
    """
    by_player: dict[int, tuple[str | None, str | None]] = {}
    for toon_desc in getattr(sc2_replay, "toonPlayerDescMap", []) or []:
        info = toon_desc.toon_player_info
        by_player[int(info.playerID)] = (
            normalise_nickname(getattr(info, "nickname", None)),
            getattr(toon_desc, "toon", None) or None,
        )
    (n1, t1), (n2, t2) = (by_player.get(pid, (None, None)) for pid in (1, 2))
    return n1, n2, t1, t2


def series_groups(
    kept: list[ReplayRef], identities: dict[int, ReplayIdentity]
) -> dict[int, str]:
    """
    Group replays into series: the same two players within one replaypack.

    Two replays share a series if they come from the same replaypack and have
    the same unordered nickname pair *or* the same unordered toon pair; the
    links are joined transitively. Matching on either catches both ways one
    player looks different within a tournament: a changed clan tag or display
    name (same toon) and a second account or server (same nickname). A pair
    with a missing name or toon is not linked on that key.

    Parameters
    ----------
    kept : list[ReplayRef]
        Replays to group.
    identities : dict[int, ReplayIdentity]
        Identity of each replay by source index, see :func:`replay_identity`.

    Returns
    -------
    dict[int, str]
        Series id of each replay by source index: the smallest replay id in its
        series, so ids are stable and readable.
    """
    parent: dict[Any, Any] = {}

    def find(node: Any) -> Any:
        root = node
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(a: Any, b: Any) -> None:
        parent[find(a)] = find(b)

    for ref in kept:
        pack = ref.replay_id.split("/", 1)[0]
        n1, n2, t1, t2 = identities[ref.index]
        node = ("replay", ref.index)
        find(node)
        if n1 is not None and n2 is not None:
            union(node, ("nick", pack, tuple(sorted((n1, n2)))))
        if t1 is not None and t2 is not None:
            union(node, ("toon", pack, tuple(sorted((t1, t2)))))

    members: dict[Any, list[ReplayRef]] = {}
    for ref in kept:
        members.setdefault(find(("replay", ref.index)), []).append(ref)
    return {
        ref.index: min(r.replay_id for r in group)
        for group in members.values()
        for ref in group
    }


def sc2_datasets_version() -> str:
    """
    Return the installed ``sc2_datasets`` version.

    Returns
    -------
    str
        Package version, or ``"unknown"`` if it is not installed as a distribution.
    """
    try:
        return importlib.metadata.version("sc2_datasets")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True)
class SplitManifest:
    """
    Train/val/test membership of every replay, persisted as ``splits.json``.

    Validated on construction, see :meth:`validate`.

    Parameters
    ----------
    seed : int
        Seed of the random 80/10/10 assignment.
    dataset_file : str
        File name of the single-JSON source dataset the indices refer to.
    n_source_replays : int
        Number of replays in the source dataset.
    sc2_datasets_version : str
        ``sc2_datasets`` version used to read the source dataset.
    created_by_transform : str
        Transform whose kept replays defined the manifest.
    subsampled : bool
        Whether this manifest is a ``--n_samples`` subset rather than the full split.
    splits : dict[str, list[ReplayRef]]
        Replays per split name (``train``, ``val``, ``test``), each sorted by index.
    grouping : str
        Unit of assignment: ``"series"`` (whole series per split, what
        ``--create_splits`` writes) or ``"replay"`` (each replay on its own, as
        in version 1 manifests), by default ``"replay"``.
    version : int
        Manifest file format version.
    """

    seed: int
    dataset_file: str
    n_source_replays: int
    sc2_datasets_version: str
    created_by_transform: str
    subsampled: bool
    splits: dict[str, list[ReplayRef]] = field(default_factory=dict)
    grouping: str = "replay"
    version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """
        Check split names, disjointness, index range and ordering.

        Raises
        ------
        SplitManifestError
            If a split is missing or extra, a replay appears twice, an index is out
            of range, a split is not sorted by index, or (series grouping) a
            replay has no series or a series spans two splits.
        """
        if set(self.splits) != set(SPLIT_NAMES):
            raise SplitManifestError(
                f"Manifest must contain exactly {SPLIT_NAMES}, got {sorted(self.splits)}"
            )
        if self.grouping not in GROUPINGS:
            raise SplitManifestError(
                f"Unknown grouping {self.grouping!r}; expected one of {GROUPINGS}."
            )
        if self.grouping == "series":
            series_split: dict[str, str] = {}
            for split in SPLIT_NAMES:
                for ref in self.splits[split]:
                    if ref.series is None:
                        raise SplitManifestError(
                            f"{ref.replay_id} has no series in a series-grouped "
                            "manifest."
                        )
                    if series_split.setdefault(ref.series, split) != split:
                        raise SplitManifestError(
                            f"Series {ref.series!r} spans '{series_split[ref.series]}'"
                            f" and '{split}'."
                        )

        seen_index: dict[int, str] = {}
        seen_id: dict[str, str] = {}
        for split in SPLIT_NAMES:
            for ref in self.splits[split]:
                for key, seen in ((ref.index, seen_index), (ref.replay_id, seen_id)):
                    if key in seen:
                        raise SplitManifestError(
                            f"Replay {key!r} appears in both '{seen[key]}' and "
                            f"'{split}' (or twice in one split)."
                        )
                    seen[key] = split

                if not 0 <= ref.index < self.n_source_replays:
                    raise SplitManifestError(
                        f"Index {ref.index} is outside the source dataset "
                        f"(n_source_replays={self.n_source_replays})."
                    )

            indices = [ref.index for ref in self.splits[split]]
            if indices != sorted(indices):
                raise SplitManifestError(f"Split '{split}' is not sorted by index.")

    # Accessors:
    def refs(self, split: str) -> list[ReplayRef]:
        """
        Return the replays of one split.

        Parameters
        ----------
        split : str
            Split name: ``train``, ``val`` or ``test``.

        Returns
        -------
        list[ReplayRef]
            Replays of the split, sorted by index.
        """
        return self.splits[split]

    def ids(self, split: str) -> list[str]:
        """
        Return the replay ids of one split.

        Parameters
        ----------
        split : str
            Split name: ``train``, ``val`` or ``test``.

        Returns
        -------
        list[str]
            Replay ids of the split, in index order.
        """
        return [ref.replay_id for ref in self.splits[split]]

    def all_indices(self) -> list[int]:
        """
        Return the source indices of every replay in the manifest.

        Returns
        -------
        list[int]
            Indices across all splits, sorted.
        """
        return sorted(ref.index for split in SPLIT_NAMES for ref in self.splits[split])

    def sizes(self) -> dict[str, int]:
        """
        Return the number of replays per split.

        Returns
        -------
        dict[str, int]
            Split name to replay count.
        """
        return {split: len(self.splits[split]) for split in SPLIT_NAMES}

    # Persistence:
    def to_dict(self) -> dict:
        """
        Serialise the manifest to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Manifest metadata and splits as lists of ``{"index", "replay_id"}``
            (plus ``"series"`` when grouped by series).
        """
        data: dict[str, Any] = {
            "version": self.version,
            "seed": self.seed,
            "dataset_file": self.dataset_file,
            "n_source_replays": self.n_source_replays,
            "sc2_datasets_version": self.sc2_datasets_version,
            "created_by_transform": self.created_by_transform,
            "subsampled": self.subsampled,
            "splits": {
                split: [
                    {k: v for k, v in asdict(ref).items() if v is not None}
                    for ref in self.splits[split]
                ]
                for split in SPLIT_NAMES
            },
        }
        # Version 1 had no grouping field; leaving it out keeps its hash unchanged.
        if self.version >= 2:
            data["grouping"] = self.grouping
        return data

    @classmethod
    def from_dict(cls, data: dict) -> SplitManifest:
        """
        Build a manifest from its dictionary form.

        Parameters
        ----------
        data : dict
            Dictionary as produced by :meth:`to_dict`.

        Returns
        -------
        SplitManifest
            The validated manifest.

        Raises
        ------
        SplitManifestError
            If the format version is unsupported or the manifest is invalid.
        """
        if data.get("version") not in SUPPORTED_VERSIONS:
            raise SplitManifestError(
                f"Unsupported manifest version {data.get('version')!r}; "
                f"expected one of {SUPPORTED_VERSIONS}."
            )
        return cls(
            # Version 1 manifests predate series grouping and cut per replay.
            grouping=data.get("grouping", "replay"),
            seed=data["seed"],
            dataset_file=data["dataset_file"],
            n_source_replays=data["n_source_replays"],
            sc2_datasets_version=data["sc2_datasets_version"],
            created_by_transform=data["created_by_transform"],
            subsampled=data["subsampled"],
            splits={
                split: [ReplayRef(**ref) for ref in data["splits"][split]]
                for split in SPLIT_NAMES
            },
            version=data["version"],
        )

    def sha256(self) -> str:
        """
        Return a content hash of the manifest.

        Returns
        -------
        str
            SHA-256 hex digest of the canonical JSON; equal membership and metadata
            give an equal hash.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, path: Path | str, overwrite: bool = False) -> None:
        """
        Write the manifest as JSON.

        Parameters
        ----------
        path : Path | str
            Destination file.
        overwrite : bool
            Allow replacing an existing file, by default False.

        Raises
        ------
        FileExistsError
            If ``path`` exists and ``overwrite`` is False.
        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite the split manifest at {path}; caches built "
                "from it would stop matching. Delete it explicitly to re-split."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> SplitManifest:
        """
        Read a manifest from JSON.

        Parameters
        ----------
        path : Path | str
            Manifest file.

        Returns
        -------
        SplitManifest
            The validated manifest.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No split manifest at {path}. Create it once with --create_splits."
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # Operations:
    def check_matches_dataset(self, dataset_file: str, n_source_replays: int) -> None:
        """
        Check that the manifest was built for the given source dataset.

        Parameters
        ----------
        dataset_file : str
            File name of the source dataset in use.
        n_source_replays : int
            Number of replays in the source dataset in use.

        Raises
        ------
        SplitManifestError
            If the file name or replay count differs from the manifest.
        """
        if (
            dataset_file != self.dataset_file
            or n_source_replays != self.n_source_replays
        ):
            raise SplitManifestError(
                "Split manifest was built for a different source dataset: "
                f"manifest=({self.dataset_file}, {self.n_source_replays} replays), "
                f"current=({dataset_file}, {n_source_replays} replays)."
            )

    def subsample(self, n_samples: int, seed: int) -> SplitManifest:
        """
        Return a seeded subset drawn proportionally from each split.

        Split assignment is inherited unchanged, so a smoke-test cache uses the same
        membership as the full one.

        Parameters
        ----------
        n_samples : int
            Total number of replays to keep. Values ``<= 0`` or ``>=`` the manifest
            size return the manifest unchanged.
        seed : int
            Seed for the per-split random selection.

        Returns
        -------
        SplitManifest
            The subset, marked ``subsampled=True``.
        """
        total = sum(self.sizes().values())
        if n_samples <= 0 or n_samples >= total:
            return self

        generator = torch.Generator().manual_seed(seed)
        n_val = round(n_samples * len(self.splits["val"]) / total)
        n_test = round(n_samples * len(self.splits["test"]) / total)
        targets = {"train": n_samples - n_val - n_test, "val": n_val, "test": n_test}

        subset: dict[str, list[ReplayRef]] = {}
        for split in SPLIT_NAMES:
            refs = self.splits[split]
            keep = torch.randperm(len(refs), generator=generator)[: targets[split]]
            subset[split] = [refs[i] for i in sorted(keep.tolist())]

        return SplitManifest(
            seed=self.seed,
            dataset_file=self.dataset_file,
            n_source_replays=self.n_source_replays,
            sc2_datasets_version=self.sc2_datasets_version,
            created_by_transform=self.created_by_transform,
            subsampled=True,
            splits=subset,
            grouping=self.grouping,
        )


def assign_splits(
    kept: list[ReplayRef], seed: int, series: dict[int, str] | None = None
) -> dict[str, list[ReplayRef]]:
    """
    Randomly cut the kept replays into 80/10/10 splits.

    The input is sorted by index before permuting, so the result depends only on
    which replays were kept and on ``seed``, never on extraction order.

    Without ``series`` each replay is assigned on its own and the sizes are
    exact. With ``series`` whole series are assigned: series are shuffled, then
    each goes to test while test stays within its target, else to val likewise,
    else to train. Val and test land within one series of their targets.

    Parameters
    ----------
    kept : list[ReplayRef]
        Replays kept by the transform that creates the manifest.
    seed : int
        Seed of the permutation.
    series : dict[int, str] | None
        Series id per source index (see :func:`series_groups`), by default None.

    Returns
    -------
    dict[str, list[ReplayRef]]
        Replays per split name, each sorted by index; refs carry their series.
    """
    ordered = sorted(kept, key=lambda ref: ref.index)
    n_total = len(ordered)
    n_val = round(n_total * VAL_FRACTION)
    n_test = round(n_total * TEST_FRACTION)

    generator = torch.Generator().manual_seed(seed)
    if series is not None:
        return _assign_series(ordered, series, n_val, n_test, generator)
    permutation = torch.randperm(n_total, generator=generator).tolist()

    cut = {
        "test": permutation[:n_test],
        "val": permutation[n_test : n_test + n_val],
        "train": permutation[n_test + n_val :],
    }
    return {
        split: sorted((ordered[i] for i in cut[split]), key=lambda ref: ref.index)
        for split in SPLIT_NAMES
    }


def _assign_series(
    ordered: list[ReplayRef],
    series: dict[int, str],
    n_val: int,
    n_test: int,
    generator: torch.Generator,
) -> dict[str, list[ReplayRef]]:
    """
    Assign whole series to splits, filling test, then val, then train.

    Parameters
    ----------
    ordered : list[ReplayRef]
        Kept replays sorted by index.
    series : dict[int, str]
        Series id per source index.
    n_val : int
        Target number of val replays.
    n_test : int
        Target number of test replays.
    generator : torch.Generator
        Seeded generator for the series permutation.

    Returns
    -------
    dict[str, list[ReplayRef]]
        Replays per split, sorted by index, each ref carrying its series.

    Raises
    ------
    KeyError
        If a kept replay has no series.
    """
    members: dict[str, list[ReplayRef]] = {}
    for ref in ordered:
        group = series[ref.index]
        members.setdefault(group, []).append(
            ReplayRef(index=ref.index, replay_id=ref.replay_id, series=group)
        )
    names = sorted(members)
    permutation = torch.randperm(len(names), generator=generator).tolist()

    targets = {"test": n_test, "val": n_val}
    splits: dict[str, list[ReplayRef]] = {split: [] for split in SPLIT_NAMES}
    for i in permutation:
        group = members[names[i]]
        split = next(
            (
                name
                for name in ("test", "val")
                if len(splits[name]) + len(group) <= targets[name]
            ),
            "train",
        )
        splits[split].extend(group)
    return {
        split: sorted(splits[split], key=lambda ref: ref.index) for split in SPLIT_NAMES
    }
