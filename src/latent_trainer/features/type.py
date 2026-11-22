"""Dataclasses, datasets and cache layouts shared by the feature and training code."""

import math
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch.utils.data import DataLoader, Dataset

from latent_trainer.features.perception_action_cycle import GAME_LOOPS_PER_SECOND


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


# Encoder protocol:
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


# Dense dataset wrapper:
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


# Event-sequence pipeline:
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
    train_race, val_race, test_race : torch.Tensor
        Race actually played per stream, ``[2N]`` int8, ids into ``race_names``.
    stat_channel_names : list[str]
        Names of the ``C`` stat channels (PlayerStats counters, then PAC).
    race_names : list[str]
        Race vocabulary (``Terr``, ``Prot``, ``Zerg``).
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

    train_stat_values: torch.Tensor  # [sum_t, C] float32
    train_stat_loops: torch.Tensor  # [sum_t]     int32
    train_stat_tick: torch.Tensor  # [sum_t]     int32
    train_stat_offsets: torch.Tensor  # [2N + 1]    int64
    train_evt_type: torch.Tensor  # [sum_e]     int8
    train_evt_unit: torch.Tensor  # [sum_e]     int16
    train_evt_loops: torch.Tensor  # [sum_e]     int32
    train_evt_tick: torch.Tensor  # [sum_e]     int32
    train_evt_offsets: torch.Tensor  # [2N + 1]    int64
    train_labels: torch.Tensor  # [N]         int64
    train_race: torch.Tensor  # [2N]        int8

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
    val_race: torch.Tensor

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
    test_race: torch.Tensor

    stat_channel_names: list[str]
    race_names: list[str]
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
        Stat channels per tick in raw units, ``[sum_t, C]`` float32 (the model
        scales them with its fitted ``ChannelTransform``).
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
    race : torch.Tensor
        Race actually played per stream, ``[2N]``.
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
    race: torch.Tensor


@dataclass
class EventDataBundle:
    """
    Return type of ``load_event_cache``: three splits plus their metadata.

    Stat values are raw; the event model fits its ``ChannelTransform`` on
    ``train.stat_values`` and stores it in the checkpoint.

    Parameters
    ----------
    train : EventSplit
        Training split.
    val : EventSplit
        Validation split.
    test : EventSplit
        Test split.
    stat_channel_names : list[str]
        Names of the stat channels.
    race_names : list[str]
        Race vocabulary.
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

    stat_channel_names: list[str]
    race_names: list[str]
    unit_vocab: list[str]
    event_type_names: list[str]
    tick_interval_loops: int

    split_ids: dict[str, list[str]] = field(default_factory=dict)
    splits_sha256: str = ""


class EventSequenceDataset(Dataset):
    """
    Serve one replay (both players' ragged streams) per index.

    Three load-time policies are applied here rather than baked into the cache,
    so they stay sweepable without re-extracting the dataset. Each one is
    applied to the *pair*, so both players always cover the same game time:

    1. Slices of ``slice_ticks`` ticks.
    2. Tail drop: the last ``drop_tail_frac`` of the pair's slices (counted on
       the shorter of the two streams) is removed. This sets the pair's
       *horizon*, the loop of the last tick both players keep.
    3. Prefix crop at a cutoff loop ``<= horizon``: a stat tick is kept if its
       loop is ``<= cutoff``, an event if its loop is ``<= cutoff`` and its tick
       is kept. Nothing after the cutoff reaches the model.

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
    crop : None | str | float
        ``None`` keeps everything up to the horizon. ``"random"`` draws a cutoff
        uniformly in ``[min_crop_s, horizon]`` seconds on every access (games
        shorter than ``min_crop_s`` are served uncropped); the draw uses torch's
        global RNG, which the DataLoader seeds per worker, so a seeded run is
        reproducible. A number is a fixed cutoff in game seconds for
        evaluation; replays whose horizon ends before it are excluded (see
        ``n_excluded``), by default None.
    min_crop_s : float
        Shortest random cutoff in game seconds, by default 180.0.

    Raises
    ------
    ValueError
        If ``slice_ticks < 1``, ``drop_tail_frac`` is outside ``[0, 1)``,
        ``crop`` is neither None, ``"random"`` nor a positive number, or
        ``min_crop_s < 0``.
    """

    def __init__(
        self,
        split: EventSplit,
        slice_ticks: int = 1,
        drop_tail_frac: float = 0.1,
        crop: None | str | float = None,
        min_crop_s: float = 180.0,
    ) -> None:
        if slice_ticks < 1:
            raise ValueError(f"slice_ticks must be >= 1, got {slice_ticks}")
        if not 0.0 <= drop_tail_frac < 1.0:
            raise ValueError(
                f"drop_tail_frac must be in [0.0, 1.0), got {drop_tail_frac}"
            )
        fixed_crop = isinstance(crop, (int, float)) and not isinstance(crop, bool)
        if not (crop is None or crop == "random" or (fixed_crop and crop > 0)):
            raise ValueError(
                f"crop must be None, 'random' or a positive number, got {crop!r}"
            )
        if min_crop_s < 0:
            raise ValueError(f"min_crop_s must be >= 0, got {min_crop_s}")

        self.split = split
        self.slice_ticks = slice_ticks
        self.drop_tail_frac = drop_tail_frac
        self.crop = crop
        self.min_crop_s = min_crop_s

        n_replays = len(split.labels)
        self.keep_slices = torch.empty(n_replays, dtype=torch.long)
        self.horizon_loops = torch.empty(n_replays, dtype=torch.long)
        for replay in range(n_replays):
            keep, horizon = self._pair_tail(replay)
            self.keep_slices[replay] = keep
            self.horizon_loops[replay] = horizon

        replays = torch.arange(n_replays)
        if fixed_crop:
            cutoff = round(float(crop) * GAME_LOOPS_PER_SECOND)
            replays = replays[self.horizon_loops >= cutoff]
        self.replay_indices = replays
        self.n_excluded = n_replays - len(replays)

    def _stat_rows(self, stream_index: int) -> tuple[int, int]:
        """
        Return the ``[start, end)`` rows of one stream in the stat arrays.

        Parameters
        ----------
        stream_index : int
            Stream number ``replay_index * 2 + player_index``.

        Returns
        -------
        tuple[int, int]
            Start and end row.
        """
        offsets = self.split.stat_offsets
        return int(offsets[stream_index]), int(offsets[stream_index + 1])

    def _pair_tail(self, replay: int) -> tuple[int, int]:
        """
        Compute a replay's shared tail cut and the horizon it leaves.

        Parameters
        ----------
        replay : int
            Replay index within the split.

        Returns
        -------
        tuple[int, int]
            Slices kept (slices ``< keep``) and the horizon loop: the last kept
            tick's loop, the earlier of the two players' (0 if nothing is kept).
        """
        split = self.split
        n_slices = []
        for stream_index in (2 * replay, 2 * replay + 1):
            start, end = self._stat_rows(stream_index)
            # The stat stream reaches the final slice, so it defines the count;
            # a player can be event-silent late without the game being over.
            if end > start:
                n_slices.append(int(split.stat_tick[end - 1]) // self.slice_ticks + 1)
            else:
                n_slices.append(0)
        keep = math.ceil(min(n_slices) * (1.0 - self.drop_tail_frac))

        horizons = []
        for stream_index in (2 * replay, 2 * replay + 1):
            start, end = self._stat_rows(stream_index)
            kept = split.stat_tick[start:end] // self.slice_ticks < keep
            loops = split.stat_loops[start:end][kept]
            horizons.append(int(loops.max()) if loops.numel() else 0)
        return keep, min(horizons)

    def _cutoff(self, replay: int) -> int:
        """
        Return the cutoff loop for one access to a replay.

        Parameters
        ----------
        replay : int
            Replay index within the split.

        Returns
        -------
        int
            The horizon for ``crop=None``, a fresh uniform draw for
            ``"random"``, or the fixed cutoff.
        """
        horizon = int(self.horizon_loops[replay])
        if self.crop is None:
            return horizon
        if self.crop == "random":
            low = self.min_crop_s * GAME_LOOPS_PER_SECOND
            if horizon <= low:
                return horizon
            return int(low + float(torch.rand(())) * (horizon - low))
        return round(float(self.crop) * GAME_LOOPS_PER_SECOND)

    def __len__(self) -> int:
        return len(self.replay_indices)

    def _stream(
        self, stream_index: int, keep_slices: int, cutoff_loop: int
    ) -> dict[str, torch.Tensor]:
        """
        Slice one player stream out of the split, cut to the pair's tail and crop.

        Parameters
        ----------
        stream_index : int
            Stream number ``replay_index * 2 + player_index``.
        keep_slices : int
            Slices ``< keep_slices`` survive the pair's tail drop.
        cutoff_loop : int
            Last game loop that may reach the model.

        Returns
        -------
        dict[str, torch.Tensor]
            ``stat_values``, ``stat_loops``, ``stat_slice``, ``evt_type``,
            ``evt_unit``, ``evt_loops`` and ``evt_slice`` for the kept part.
        """
        split = self.split

        stat_start, stat_end = self._stat_rows(stream_index)
        evt_start = int(split.evt_offsets[stream_index])
        evt_end = int(split.evt_offsets[stream_index + 1])

        stat_tick = split.stat_tick[stat_start:stat_end]
        stat_loops = split.stat_loops[stat_start:stat_end]
        evt_tick = split.evt_tick[evt_start:evt_end]
        evt_loops = split.evt_loops[evt_start:evt_end]

        stat_slice = stat_tick // self.slice_ticks
        evt_slice = evt_tick // self.slice_ticks

        stat_keep = (stat_slice < keep_slices) & (stat_loops <= cutoff_loop)
        # An event belongs to the tick at or after it, so it is kept only with
        # its tick; the loop test is the literal "nothing after the cutoff".
        last_tick = int(stat_tick[stat_keep].max()) if bool(stat_keep.any()) else -1
        evt_keep = (
            (evt_slice < keep_slices)
            & (evt_tick <= last_tick)
            & (evt_loops <= cutoff_loop)
        )

        return {
            "stat_values": split.stat_values[stat_start:stat_end][stat_keep],
            "stat_loops": stat_loops[stat_keep],
            "stat_slice": stat_slice[stat_keep],
            "evt_type": split.evt_type[evt_start:evt_end][evt_keep],
            "evt_unit": split.evt_unit[evt_start:evt_end][evt_keep],
            "evt_loops": evt_loops[evt_keep],
            "evt_slice": evt_slice[evt_keep],
        }

    def __getitem__(self, idx: int) -> dict:
        """
        Return both players' streams, races and label of one replay.

        ``idx`` indexes this dataset; it maps to replay ``r =
        replay_indices[idx]`` of the split, which is ``idx`` itself unless a
        fixed crop excluded short replays.

        Why the layout: every player stream has its own length (72-266 stat
        ticks, up to ~13.5k events), so a dense ``[N, 2, T, ...]`` tensor would be
        mostly padding, and a list of per-replay tensors would make the cache
        hundreds of thousands of small objects that are slow to save, load and
        share with DataLoader workers. Instead each split stores every stream
        concatenated into a few large arrays, plus an ``offsets`` array (CSR, as in
        sparse matrices): stream ``s`` is ``values[offsets[s] : offsets[s + 1]]``.
        The unit is the *player* stream, not the replay, because each player's
        stream has its own boundaries.

        Why ``2 * r`` and ``2 * r + 1``: ``_flatten_split`` in
        ``preprocess_events.py`` loops over replays in order and, inside, over
        that replay's players, appending one offset per stream and one label per
        replay. ``event_transform`` emits a replay only when both players
        (``PLAYER_IDS = (1, 2)``, in that order) have streams, otherwise the
        replay is skipped and never cached. So every replay contributes exactly
        two streams, replays ``0 .. r-1`` occupy streams ``0 .. 2*r-1``, and
        replay ``r`` owns stream ``2 * r`` (player 1) and ``2 * r + 1``
        (player 2). ``offsets`` starts at 0 and gains one entry per stream, so it
        has ``2N + 1`` entries and ``offsets[2 * r + 2]`` always exists. Labels
        get one entry per replay, so they are indexed by ``r`` directly; races
        one per stream, like the offsets. ``load_event_cache`` checks all of this.

        Example with two replays of 3 and 5 ticks per player::

            stream s      0      1      2       3
            replay, p     0, 1   0, 2   1, 1    1, 2
            offsets       [0,    3,     6,      11,     16]
            rows          0:3    3:6    6:11    11:16

        Replay 1 reads streams 2 and 3, rows ``6:11`` and ``11:16``.

        The player order must not change: the label means "player 1 won".

        Parameters
        ----------
        idx : int
            Index into this dataset.

        Returns
        -------
        dict
            ``players``: ``[player 1, player 2]`` stream dicts from
            :meth:`_stream`; ``label``: 1 if player 1 won, else 0; ``race``:
            ``[2]`` race ids in player order; ``crop_loop``: the cutoff loop
            applied to both players.
        """
        replay = int(self.replay_indices[idx])
        keep = int(self.keep_slices[replay])
        cutoff = self._cutoff(replay)
        return {
            "players": [
                self._stream(2 * replay, keep, cutoff),
                self._stream(2 * replay + 1, keep, cutoff),
            ],
            "label": self.split.labels[replay],
            "race": self.split.race[2 * replay : 2 * replay + 2].long(),
            "crop_loop": torch.tensor(cutoff, dtype=torch.long),
        }
