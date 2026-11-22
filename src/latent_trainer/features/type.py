"""Dataclasses, datasets and cache layouts shared by the feature and training code."""

import math
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class NormalizedDataloaders:
    """
    Train/val(/test) dataloaders over normalised dense features.

    Parameters
    ----------
    train_loader : DataLoader
        Training batches of ``(features, label)``.
    val_loader : DataLoader
        Validation batches of ``(features, label)``.
    input_dim : int
        Number of features per player.
    mean : torch.Tensor
        Per-feature training mean used for normalisation.
    std : torch.Tensor
        Per-feature training std (floored at 1.0) used for normalisation.
    test_loader : DataLoader | None
        Test batches, if requested, by default None.
    """

    train_loader: DataLoader
    val_loader: DataLoader

    input_dim: int
    mean: torch.Tensor
    std: torch.Tensor

    test_loader: DataLoader | None = None


@dataclass
class NormalizedData:
    """
    Return type of :func:`normalize`: normalised splits plus their statistics.

    Parameters
    ----------
    train_X : torch.Tensor
        Normalised training features.
    val_X : torch.Tensor
        Normalised validation features.
    test_X : torch.Tensor
        Normalised test features.
    mean : torch.Tensor
        Per-feature training mean, shape ``[1, F]``.
    std : torch.Tensor
        Per-feature training std floored at 1.0, shape ``[1, F]``.
    """

    train_X: torch.Tensor
    val_X: torch.Tensor
    test_X: torch.Tensor

    # Normalisation parameters (for test-time normalization)
    mean: torch.Tensor
    std: torch.Tensor


@dataclass
class NormalizedDataWithLabels:
    """
    Return type of :func:`load_and_normalize`: normalised splits, labels and statistics.

    Parameters
    ----------
    train_X : torch.Tensor
        Normalised training features, ``[N, 2, F]``.
    train_y : torch.Tensor
        Training labels as float (1 = player 1 won).
    val_X : torch.Tensor
        Normalised validation features.
    val_y : torch.Tensor
        Validation labels as float.
    test_X : torch.Tensor
        Normalised test features.
    test_y : torch.Tensor
        Test labels as float.
    mean : torch.Tensor
        Per-feature training mean, shape ``[1, F]``.
    std : torch.Tensor
        Per-feature training std floored at 1.0, shape ``[1, F]``.
    """

    train_X: torch.Tensor
    train_y: torch.Tensor
    val_X: torch.Tensor
    val_y: torch.Tensor
    test_X: torch.Tensor
    test_y: torch.Tensor

    # Normalisation parameters (for test-time normalization)
    mean: torch.Tensor
    std: torch.Tensor


# ------------------------------------------------------------------
# Protocol for any VAE-like encoder (LitVAE, SimpleVAE, suGuidedVAE…)
# ------------------------------------------------------------------
class Encoder(Protocol):
    """
    Structural type for any model that exposes an ``encode`` method.

    The ``encode`` method must return a tuple ``(mu, logvar)`` where:

    - **mu** — latent mean, shape ``[batch, latent_dim]``
    - **logvar** — latent log-variance, shape ``[batch, latent_dim]``
    """

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode features into the latent posterior.

        Parameters
        ----------
        x : torch.Tensor
            Features, ``[N, F]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Posterior ``(mu, logvar)``, each ``[N, latent_dim]``.
        """
        ...


# ------------------------------------------------------------------
# Dataset wrapper (used by train_model.py / GuidedVAE path)
# ------------------------------------------------------------------
class CachedSC2Dataset(Dataset):
    """
    Wrap pre-processed feature and label tensors as a PyTorch Dataset.

    Parameters
    ----------
    features : torch.Tensor
        Features, indexed along the first dimension.
    labels : torch.Tensor
        Labels aligned with ``features``.
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


@dataclass
class CachedDatasetFileSpec:
    """
    On-disk layout of a dense feature cache (``cached_dataset_<transform>.pt``).

    Parameters
    ----------
    train_features, val_features, test_features : torch.Tensor
        Features per split, ``[N, 2, F]`` (player 1, player 2).
    train_labels, val_labels, test_labels : torch.Tensor
        Labels per split, ``[N]`` int64; 1 = player 1 won.
    transform : str
        Name of the transform that produced the cache.
    train_ids, val_ids, test_ids : list[str]
        Replay ids aligned with the rows of each split; empty in caches built
        before the split manifest.
    splits_sha256 : str
        Hash of the ``splits.json`` the cache was built from; empty in old caches.
    """

    train_features: torch.Tensor
    train_labels: torch.Tensor
    val_features: torch.Tensor
    val_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor
    transform: str
    # Provenance: row-aligned replay ids and the splits.json hash (empty in old caches).
    train_ids: list[str] = field(default_factory=list)
    val_ids: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    splits_sha256: str = ""


# ------------------------------------------------------------------
# Event-sequence pipeline (Transformer encoder -> guided VAE)
# ------------------------------------------------------------------
@dataclass
class EventCacheFileSpec:
    """
    On-disk layout of the ``events`` cache.

    Streams are ragged: a replay contributes 72-266 stat ticks (p25-p99) and up
    to ~13.5k events per player, so there is no dense ``[N, 2, T, C]`` tensor.
    Each split keeps its streams concatenated with CSR-style offsets: stream
    ``s = replay_index * 2 + player_index`` occupies
    ``values[offsets[s] : offsets[s + 1]]``, so ``offsets`` has ``2N + 1``
    entries. Batches are padded per batch by ``collate_event_sequences``.

    Parameters
    ----------
    train_stat_values, val_stat_values, test_stat_values : torch.Tensor
        PlayerStats channels per tick, ``[sum_t, C]`` float32, unnormalised.
    train_stat_loops, val_stat_loops, test_stat_loops : torch.Tensor
        Game loop of each tick, ``[sum_t]`` int32.
    train_stat_tick, val_stat_tick, test_stat_tick : torch.Tensor
        Tick index on the 160-loop grid, ``[sum_t]`` int32.
    train_stat_offsets, val_stat_offsets, test_stat_offsets : torch.Tensor
        CSR offsets into the stat arrays, ``[2N + 1]`` int64.
    train_evt_type, val_evt_type, test_evt_type : torch.Tensor
        Event type id into ``event_type_names``, ``[sum_e]`` int8.
    train_evt_unit, val_evt_unit, test_evt_unit : torch.Tensor
        Unit/upgrade id into ``unit_vocab``, ``[sum_e]`` int16.
    train_evt_loops, val_evt_loops, test_evt_loops : torch.Tensor
        Game loop of each event, ``[sum_e]`` int32.
    train_evt_tick, val_evt_tick, test_evt_tick : torch.Tensor
        Tick index each event belongs to, ``[sum_e]`` int32.
    train_evt_offsets, val_evt_offsets, test_evt_offsets : torch.Tensor
        CSR offsets into the event arrays, ``[2N + 1]`` int64.
    train_labels, val_labels, test_labels : torch.Tensor
        Labels per replay, ``[N]`` int64; 1 = player 1 won.
    stat_channel_names : list[str]
        Names of the ``C`` stat channels.
    unit_vocab : list[str]
        Unit/upgrade vocabulary; ids 0 and 1 are ``<PAD>`` and ``<UNK>``.
    event_type_names : list[str]
        Event type vocabulary; id 0 is ``<PAD>``.
    tick_interval_loops : int
        Game loops between PlayerStats ticks (160).
    transform : str
        Transform name (``events``).
    train_ids, val_ids, test_ids : list[str]
        Replay ids aligned with each split's replays; empty in old caches.
    splits_sha256 : str
        Hash of the ``splits.json`` the cache was built from; empty in old caches.
    """

    train_stat_values: torch.Tensor  # [sum_t, 39] float32
    train_stat_loops: torch.Tensor  # [sum_t]     int32
    train_stat_tick: torch.Tensor  # [sum_t]     int32
    train_stat_offsets: torch.Tensor  # [2N + 1]    int64
    train_evt_type: torch.Tensor  # [sum_e]     int8
    train_evt_unit: torch.Tensor  # [sum_e]     int16
    train_evt_loops: torch.Tensor  # [sum_e]     int32
    train_evt_tick: torch.Tensor  # [sum_e]     int32
    train_evt_offsets: torch.Tensor  # [2N + 1]    int64
    train_labels: torch.Tensor  # [N]         int64

    val_stat_values: torch.Tensor
    val_stat_loops: torch.Tensor
    val_stat_tick: torch.Tensor
    val_stat_offsets: torch.Tensor
    val_evt_type: torch.Tensor
    val_evt_unit: torch.Tensor
    val_evt_loops: torch.Tensor
    val_evt_tick: torch.Tensor
    val_evt_offsets: torch.Tensor
    val_labels: torch.Tensor

    test_stat_values: torch.Tensor
    test_stat_loops: torch.Tensor
    test_stat_tick: torch.Tensor
    test_stat_offsets: torch.Tensor
    test_evt_type: torch.Tensor
    test_evt_unit: torch.Tensor
    test_evt_loops: torch.Tensor
    test_evt_tick: torch.Tensor
    test_evt_offsets: torch.Tensor
    test_labels: torch.Tensor

    stat_channel_names: list[str]
    unit_vocab: list[str]
    event_type_names: list[str]
    tick_interval_loops: int
    transform: str
    # Provenance: row-aligned replay ids and the splits.json hash (empty in old caches).
    train_ids: list[str] = field(default_factory=list)
    val_ids: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    splits_sha256: str = ""


@dataclass
class EventSplit:
    """
    One split's ragged streams, as handed to :class:`EventSequenceDataset`.

    Fields mirror the per-split fields of :class:`EventCacheFileSpec`.

    Parameters
    ----------
    stat_values : torch.Tensor
        PlayerStats channels per tick, ``[sum_t, C]`` float32 (normalised once
        loaded by ``load_event_cache``).
    stat_loops : torch.Tensor
        Game loop of each tick, ``[sum_t]``.
    stat_tick : torch.Tensor
        Tick index of each tick, ``[sum_t]``.
    stat_offsets : torch.Tensor
        CSR offsets into the stat arrays, ``[2N + 1]``.
    evt_type : torch.Tensor
        Event type ids, ``[sum_e]``.
    evt_unit : torch.Tensor
        Unit/upgrade ids, ``[sum_e]``.
    evt_loops : torch.Tensor
        Game loop of each event, ``[sum_e]``.
    evt_tick : torch.Tensor
        Tick index of each event, ``[sum_e]``.
    evt_offsets : torch.Tensor
        CSR offsets into the event arrays, ``[2N + 1]``.
    labels : torch.Tensor
        Labels per replay, ``[N]``; 1 = player 1 won.
    """

    stat_values: torch.Tensor
    stat_loops: torch.Tensor
    stat_tick: torch.Tensor
    stat_offsets: torch.Tensor
    evt_type: torch.Tensor
    evt_unit: torch.Tensor
    evt_loops: torch.Tensor
    evt_tick: torch.Tensor
    evt_offsets: torch.Tensor
    labels: torch.Tensor


@dataclass
class EventDataBundle:
    """
    Return type of ``load_event_cache``: three splits plus their metadata.

    ``mean``/``std`` are per stat channel, fitted on the training split only,
    and must be persisted with the checkpoint so inference normalises identically.

    Parameters
    ----------
    train : EventSplit
        Training split, stat channels normalised.
    val : EventSplit
        Validation split, stat channels normalised.
    test : EventSplit
        Test split, stat channels normalised.
    mean : torch.Tensor
        Per-channel training mean, ``[1, C]``.
    std : torch.Tensor
        Per-channel training std floored at 1.0, ``[1, C]``.
    stat_channel_names : list[str]
        Names of the stat channels.
    unit_vocab : list[str]
        Unit/upgrade vocabulary.
    event_type_names : list[str]
        Event type vocabulary.
    tick_interval_loops : int
        Game loops between PlayerStats ticks.
    split_ids : dict[str, list[str]]
        Replay ids per split name; empty lists for old caches.
    splits_sha256 : str
        Hash of the source ``splits.json``; empty for old caches.
    """

    train: "EventSplit"
    val: "EventSplit"
    test: "EventSplit"

    mean: torch.Tensor  # [1, 39]
    std: torch.Tensor  # [1, 39]

    stat_channel_names: list[str]
    unit_vocab: list[str]
    event_type_names: list[str]
    tick_interval_loops: int

    split_ids: dict[str, list[str]] = field(default_factory=dict)
    splits_sha256: str = ""


class EventSequenceDataset(Dataset):
    """
    Serve one replay (both players' ragged streams) per index.

    Two load-time policies are applied here rather than baked into the cache,
    so both stay sweepable without re-extracting the dataset.

    Parameters
    ----------
    split : EventSplit
        Split to serve.
    slice_ticks : int
        PlayerStats ticks (7.14 s each) per level-1 slice; the slice index is
        ``tick // slice_ticks``. With the default 1, slice index and tick index
        coincide, by default 1.
    drop_tail_frac : float
        Fraction of each game's slices to drop from the end. Endgame slices encode
        the outcome through collapsed counters rather than actionable behaviour;
        0.0 keeps everything, by default 0.1.

    Raises
    ------
    ValueError
        If ``slice_ticks < 1`` or ``drop_tail_frac`` is outside ``[0, 1)``.
    """

    def __init__(
        self,
        split: EventSplit,
        slice_ticks: int = 1,
        drop_tail_frac: float = 0.1,
    ) -> None:
        if slice_ticks < 1:
            raise ValueError(f"slice_ticks must be >= 1, got {slice_ticks}")
        if not 0.0 <= drop_tail_frac < 1.0:
            raise ValueError(
                f"drop_tail_frac must be in [0.0, 1.0), got {drop_tail_frac}"
            )

        self.split = split
        self.slice_ticks = slice_ticks
        self.drop_tail_frac = drop_tail_frac

    def __len__(self) -> int:
        return len(self.split.labels)

    def _stream(self, stream_index: int) -> dict[str, torch.Tensor]:
        """
        Slice one player stream out of the split and apply the tail drop.

        The stat stream alone defines the horizon: it always reaches the final
        slice, so a player who is event-silent late does not shorten it.

        Parameters
        ----------
        stream_index : int
            Stream number ``replay_index * 2 + player_index``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``stat_values``, ``stat_loops``, ``stat_slice``, ``evt_type``,
            ``evt_unit``, ``evt_loops`` and ``evt_slice`` for the kept slices.
        """
        split = self.split

        stat_start = int(split.stat_offsets[stream_index])
        stat_end = int(split.stat_offsets[stream_index + 1])
        evt_start = int(split.evt_offsets[stream_index])
        evt_end = int(split.evt_offsets[stream_index + 1])

        stat_slice = split.stat_tick[stat_start:stat_end] // self.slice_ticks
        evt_slice = split.evt_tick[evt_start:evt_end] // self.slice_ticks

        # Slices are numbered from 0, so the count is the last index + 1. The
        # stat stream always reaches the final slice, so it alone defines the
        # horizon -- a player can be event-silent late without shortening it.
        n_slices = int(stat_slice[-1]) + 1 if stat_slice.numel() else 0
        keep = math.ceil(n_slices * (1.0 - self.drop_tail_frac))

        stat_keep = stat_slice < keep
        evt_keep = evt_slice < keep

        return {
            "stat_values": split.stat_values[stat_start:stat_end][stat_keep],
            "stat_loops": split.stat_loops[stat_start:stat_end][stat_keep],
            "stat_slice": stat_slice[stat_keep],
            "evt_type": split.evt_type[evt_start:evt_end][evt_keep],
            "evt_unit": split.evt_unit[evt_start:evt_end][evt_keep],
            "evt_loops": split.evt_loops[evt_start:evt_end][evt_keep],
            "evt_slice": evt_slice[evt_keep],
        }

    def __getitem__(self, idx: int) -> dict:
        return {
            "players": [self._stream(idx * 2), self._stream(idx * 2 + 1)],
            "label": self.split.labels[idx],
        }
