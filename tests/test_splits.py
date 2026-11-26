"""Tests for the split manifest and manifest-driven extraction."""

import json
import random
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from latent_trainer.features import preprocess_dataset as dense_module
from latent_trainer.features.extraction import extract_with_manifest
from latent_trainer.features.splits import (
    SPLIT_NAMES,
    ReplayRef,
    SplitManifest,
    SplitManifestError,
    assign_splits,
    normalise_nickname,
    replay_id_from_filepath,
    replay_identity,
    series_groups,
)

N_REPLAYS = 200


@dataclass
class FakeReplay:
    """Minimal replay with a ``filepath``, its source index and its players."""

    filepath: str
    index: int
    toonPlayerDescMap: list = field(default_factory=list)


def _players(*players: tuple[str, str]) -> list:
    """Return a ``toonPlayerDescMap`` for ``(nickname, toon)`` of players 1, 2."""
    return [
        SimpleNamespace(
            toon=toon,
            toon_player_info=SimpleNamespace(playerID=pid, nickname=nickname),
        )
        for pid, (nickname, toon) in enumerate(players, start=1)
    ]


class FakeDataset:
    """Indexable fake source dataset; ``rename`` overrides file names by index."""

    def __init__(self, n: int = N_REPLAYS, rename: dict[int, str] | None = None):
        self.n = n
        self.rename = rename or {}

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> FakeReplay:
        name = self.rename.get(index, f"replay_{index:04d}.json")
        return FakeReplay(filepath=f"Tournament\\{name}", index=index)


def keep_unless_multiple_of(k: int):
    """Return a transform that skips replays whose index is a multiple of ``k``."""
    def transform(replay: FakeReplay):
        return None if replay.index % k == 0 else replay.index

    return transform


def _refs(n: int) -> list[ReplayRef]:
    return [ReplayRef(index=i, replay_id=f"t/r{i}") for i in range(n)]


def _manifest(n: int = 100, seed: int = 0) -> SplitManifest:
    return SplitManifest(
        seed=seed,
        dataset_file="data.json",
        n_source_replays=n,
        sc2_datasets_version="x",
        created_by_transform="t",
        subsampled=False,
        splits=assign_splits(_refs(n), seed=seed),
    )


def _extract(tmp_path, transform, create: bool, dataset=None, **kwargs):
    return extract_with_manifest(
        dataset or FakeDataset(),
        transform,
        transform_name="fake",
        dataset_file="data.json",
        splits_path=tmp_path / "splits.json",
        create_splits=create,
        n_workers=1,
        chunk_size=7,
        **kwargs,
    )


# Assignment:
def test_assign_splits_exact_disjoint_and_complete():
    """Assignment is exactly 80/10/10, disjoint and covers every replay."""
    splits = assign_splits(_refs(1000), seed=1)
    assert {s: len(v) for s, v in splits.items()} == {"train": 800, "val": 100, "test": 100}
    all_ids = [ref.replay_id for s in SPLIT_NAMES for ref in splits[s]]
    assert len(all_ids) == len(set(all_ids)) == 1000


def test_assign_splits_ignores_input_order_and_depends_on_seed():
    """Assignment ignores input order and changes with the seed."""
    refs = _refs(300)
    shuffled = refs[:]
    random.Random(5).shuffle(shuffled)
    assert assign_splits(refs, seed=3) == assign_splits(shuffled, seed=3)
    assert assign_splits(refs, seed=3) != assign_splits(refs, seed=4)


# Series grouping:
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("&lt;mouz&gt;HeroMarine", "heromarine"),
        ("<Rival>Cuddlebear", "cuddlebear"),
        ("Liquid<sp/>Clem", "clem"),
        ("  Serral ", "serral"),
        ("&lt;TAG&gt;", None),
        (None, None),
    ],
)
def test_nicknames_lose_clan_tags_and_case(raw, expected):
    """Escaped and raw clan tags, ``<sp/>`` prefixes and case are removed."""
    assert normalise_nickname(raw) == expected


def test_replay_identity_reads_both_players():
    """Identity is (p1 nick, p2 nick, p1 toon, p2 toon), normalised."""
    replay = FakeReplay("P\\r", 0, _players(("<A>Alpha", "1-S2-1-1"), ("Beta", "2")))
    assert replay_identity(replay) == ("alpha", "beta", "1-S2-1-1", "2")
    assert replay_identity(FakeReplay("P\\r", 0)) == (None, None, None, None)


def test_series_link_by_nickname_or_toon_within_a_pack_only():
    """Same pair by name or by account joins a series; another pack never does."""
    kept = [ReplayRef(i, f"{pack}/r{i}") for i, pack in enumerate("AAAAB")]
    identities = {
        0: ("x", "y", "t1", "t2"),
        1: ("y", "x", "t9", "t8"),  # same names, other accounts (other server)
        2: ("x2", "y", "t1", "t2"),  # renamed, same accounts
        3: ("x", "z", "t1", "t3"),  # different opponent
        4: ("x", "y", "t1", "t2"),  # same pair, different tournament
    }
    groups = series_groups(kept, identities)
    assert groups[0] == groups[1] == groups[2] == "A/r0"
    assert groups[3] == "A/r3"
    assert groups[4] == "B/r4"


def test_missing_identity_never_links_replays():
    """Replays without names or toons stay in their own series."""
    kept = [ReplayRef(i, f"A/r{i}") for i in range(3)]
    groups = series_groups(kept, {i: (None, None, None, None) for i in range(3)})
    assert len(set(groups.values())) == 3


def _series_refs(n_series: int, seed: int) -> tuple[list[ReplayRef], dict[int, str]]:
    """Return refs in series of 1-5 replays, plus their series ids."""
    rng = random.Random(seed)
    refs, series = [], {}
    for s in range(n_series):
        for _ in range(rng.randint(1, 5)):
            index = len(refs)
            refs.append(ReplayRef(index, f"t/r{index}"))
            series[index] = f"s{s}"
    return refs, series


def test_series_assignment_keeps_series_whole_and_sizes_close():
    """No series spans two splits; val/test sit within one series of 10%."""
    refs, series = _series_refs(400, seed=0)
    splits = assign_splits(refs, seed=1, series=series)
    owner: dict[str, str] = {}
    for split in SPLIT_NAMES:
        for ref in splits[split]:
            assert ref.series == series[ref.index]
            assert owner.setdefault(ref.series, split) == split
    n = len(refs)
    for split in ("val", "test"):
        assert round(0.1 * n) - 5 <= len(splits[split]) <= round(0.1 * n)
    assert sum(len(v) for v in splits.values()) == n
    assert splits == assign_splits(list(reversed(refs)), seed=1, series=series)
    assert splits != assign_splits(refs, seed=2, series=series)


def test_series_manifest_rejects_a_series_in_two_splits():
    """Validation catches a series split across train and test."""
    refs = [ReplayRef(i, f"t/r{i}", series="s") for i in range(3)]
    with pytest.raises(SplitManifestError, match="spans"):
        SplitManifest(
            seed=0,
            dataset_file="d",
            n_source_replays=3,
            sc2_datasets_version="x",
            created_by_transform="t",
            subsampled=False,
            splits={"train": refs[:2], "val": [], "test": refs[2:]},
            grouping="series",
        )


def test_version_1_manifest_loads_with_its_hash_unchanged(tmp_path):
    """Old per-replay manifests still load and hash as they always did."""
    manifest = _manifest()
    old = manifest.to_dict()
    old["version"] = 1
    old.pop("grouping")
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    loaded = SplitManifest.load(path)
    assert loaded.grouping == "replay" and loaded.version == 1
    assert loaded.to_dict() == old


# Manifest:
def test_manifest_round_trip_and_hash(tmp_path):
    """Save/load preserves the manifest and its hash; another seed changes the hash."""
    manifest = _manifest()
    path = tmp_path / "splits.json"
    manifest.save(path)
    loaded = SplitManifest.load(path)
    assert loaded == manifest
    assert loaded.sha256() == manifest.sha256()
    assert _manifest(seed=1).sha256() != manifest.sha256()


def test_manifest_refuses_overwrite(tmp_path):
    """Saving over an existing manifest raises."""
    path = tmp_path / "splits.json"
    _manifest().save(path)
    with pytest.raises(FileExistsError):
        _manifest().save(path)


def test_manifest_rejects_replay_in_two_splits():
    """A replay listed in two splits is rejected."""
    refs = _refs(3)
    with pytest.raises(SplitManifestError):
        SplitManifest(
            seed=0,
            dataset_file="d",
            n_source_replays=3,
            sc2_datasets_version="x",
            created_by_transform="t",
            subsampled=False,
            splits={"train": refs[:2], "val": refs[1:2], "test": refs[2:]},
        )


def test_manifest_rejects_other_dataset():
    """A manifest for another source dataset is rejected."""
    with pytest.raises(SplitManifestError):
        _manifest().check_matches_dataset("other.json", 100)


def test_subsample_keeps_assignment():
    """Subsampling is seeded and keeps each replay in its original split."""
    manifest = _manifest(n=1000)
    small = manifest.subsample(n_samples=100, seed=7)
    assert sum(small.sizes().values()) == 100
    assert small.subsampled
    for split in SPLIT_NAMES:
        assert set(small.ids(split)) <= set(manifest.ids(split))
    assert small == manifest.subsample(n_samples=100, seed=7)


def test_replay_id_is_platform_independent():
    """Replay ids use forward slashes on every OS."""
    assert replay_id_from_filepath("Pack\\sub\\r.json") == "Pack/sub/r.json"


# Extraction:
def test_create_then_reuse_gives_identical_index_ordered_splits(tmp_path):
    """Reusing a manifest reproduces the created splits, in index order."""
    created = _extract(tmp_path, keep_unless_multiple_of(10), create=True)
    kept = N_REPLAYS - N_REPLAYS // 10
    assert sum(created.manifest.sizes().values()) == kept
    assert created.n_skipped == N_REPLAYS // 10

    # A different (more permissive) transform placed by the same manifest.
    reused = _extract(tmp_path, lambda replay: replay.index, create=False)
    for split in SPLIT_NAMES:
        assert created.outputs[split] == reused.outputs[split]
        assert created.outputs[split] == sorted(created.outputs[split])
        assert [f"Tournament/replay_{i:04d}.json" for i in created.outputs[split]] == (
            created.manifest.ids(split)
        )


def test_create_refuses_existing_manifest(tmp_path):
    """Creating splits over an existing manifest raises."""
    _extract(tmp_path, keep_unless_multiple_of(10), create=True)
    with pytest.raises(FileExistsError):
        _extract(tmp_path, keep_unless_multiple_of(10), create=True)


def test_reuse_fails_when_transform_skips_a_manifest_replay(tmp_path):
    """Reuse fails if the transform skips a listed replay."""
    _extract(tmp_path, keep_unless_multiple_of(10), create=True)
    with pytest.raises(SplitManifestError, match="skipped"):
        _extract(tmp_path, keep_unless_multiple_of(7), create=False)


def test_reuse_fails_when_transform_errors(tmp_path):
    """Reuse fails if the transform raises on a listed replay."""
    _extract(tmp_path, keep_unless_multiple_of(10), create=True)

    def broken(replay):
        if replay.index == 13:
            raise RuntimeError("boom")
        return replay.index

    with pytest.raises(SplitManifestError, match="boom"):
        _extract(tmp_path, broken, create=False)


def test_reuse_fails_when_source_replay_changed(tmp_path):
    """Reuse fails if a source replay's id no longer matches."""
    _extract(tmp_path, keep_unless_multiple_of(10), create=True)
    changed = FakeDataset(rename={11: "different.json"})
    with pytest.raises(SplitManifestError, match="manifest says"):
        _extract(tmp_path, lambda r: r.index, create=False, dataset=changed)


def test_created_manifest_keeps_each_series_in_one_split(tmp_path):
    """--create_splits groups replays of the same pair in one tournament."""

    class PairedDataset(FakeDataset):
        def __getitem__(self, index: int) -> FakeReplay:
            replay = super().__getitem__(index)
            pair = index // 4  # series of four games
            replay.toonPlayerDescMap = _players((f"a{pair}", ""), (f"b{pair}", ""))
            return replay

    result = _extract(tmp_path, lambda r: r.index, create=True, dataset=PairedDataset())
    assert result.manifest.grouping == "series"
    for split in SPLIT_NAMES:
        pairs = {i // 4 for i in result.outputs[split]}
        for other in SPLIT_NAMES:
            if other != split:
                assert not pairs & {i // 4 for i in result.outputs[other]}
    assert SplitManifest.load(tmp_path / "splits.json") == result.manifest


def test_create_with_n_samples_writes_a_separate_manifest(tmp_path):
    """A smoke-test manifest goes to splits_n<N>.json and a full run refuses it."""
    _extract(tmp_path, lambda r: r.index, create=True, n_samples=40)
    assert not (tmp_path / "splits.json").exists()
    small_path = tmp_path / "splits_n40.json"
    assert SplitManifest.load(small_path).subsampled

    with pytest.raises(SplitManifestError, match="subset"):
        extract_with_manifest(
            FakeDataset(),
            lambda r: r.index,
            transform_name="fake",
            dataset_file="data.json",
            splits_path=small_path,
            create_splits=False,
            n_workers=1,
        )


def test_reuse_with_n_samples_draws_from_manifest(tmp_path):
    """``n_samples`` draws a subset of each manifest split."""
    full = _extract(tmp_path, keep_unless_multiple_of(10), create=True)
    small = _extract(tmp_path, lambda r: r.index, create=False, n_samples=40, seed=3)
    assert sum(small.manifest.sizes().values()) == 40
    for split in SPLIT_NAMES:
        assert set(small.outputs[split]) <= set(full.outputs[split])


# Dense cache:
def test_dense_cache_records_ids_and_manifest_hash(tmp_path, monkeypatch):
    """The dense cache stores row-aligned replay ids, labels and the manifest hash."""
    monkeypatch.setattr(dense_module, "load_source_dataset", lambda path: FakeDataset())

    def transform(replay: FakeReplay):
        if replay.index % 10 == 0:
            return None
        return torch.full((2, 3), float(replay.index)), replay.index % 2

    kwargs = dict(
        transform_name="fake",
        transform_fn=transform,
        single_json_dataset_path=tmp_path / "data.json",
        splits_path=tmp_path / "splits.json",
        output_directory=tmp_path,
        n_workers=1,
    )
    dense_module.preprocess_dataset(create_splits=True, **kwargs)
    cache = torch.load(tmp_path / "cached_dataset_fake.pt", weights_only=False)
    manifest = SplitManifest.load(tmp_path / "splits.json")

    assert cache["splits_sha256"] == manifest.sha256()
    for split in SPLIT_NAMES:
        ids = cache[f"{split}_ids"]
        assert ids == manifest.ids(split)
        indices = [int(i.split("_")[-1].split(".")[0]) for i in ids]
        assert cache[f"{split}_features"][:, 0, 0].tolist() == [float(i) for i in indices]
        assert cache[f"{split}_labels"].tolist() == [i % 2 for i in indices]
