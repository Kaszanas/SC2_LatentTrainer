import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class NormalizedDataloaders:
    train_loader: DataLoader
    val_loader: DataLoader

    input_dim: int
    mean: torch.Tensor
    std: torch.Tensor

    test_loader: DataLoader | None = None


@dataclass
class NormalizedData:
    """Return type of :func:`load_and_normalize`."""

    train_X: torch.Tensor
    val_X: torch.Tensor
    test_X: torch.Tensor

    # Normalisation parameters (for test-time normalization)
    mean: torch.Tensor
    std: torch.Tensor


@dataclass
class NormalizedDataWithLabels:
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
    """Structural type for any model that exposes an ``encode`` method.

    The ``encode`` method must return a tuple ``(mu, logvar)`` where:

    - **mu** — latent mean, shape ``[batch, latent_dim]``
    - **logvar** — latent log-variance, shape ``[batch, latent_dim]``
    """

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


# ------------------------------------------------------------------
# Dataset wrapper (used by train_model.py / GuidedVAE path)
# ------------------------------------------------------------------
class CachedSC2Dataset(Dataset):
    """Wraps pre-processed feature and label tensors as a PyTorch Dataset."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


@dataclass
class CachedDatasetFileSpec:
    train_features: torch.Tensor
    train_labels: torch.Tensor
    val_features: torch.Tensor
    val_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor
    transform: str


# ------------------------------------------------------------------
# Event-sequence pipeline (Transformer encoder -> guided VAE)
# ------------------------------------------------------------------
@dataclass
class EventCacheFileSpec:
    """Cache layout for the ``events`` transform.

    Unlike :class:`CachedDatasetFileSpec`, the streams here are **ragged**: a
    replay contributes 72-266 stat ticks (p25-p99) and anywhere from a few
    hundred to ~13.5k events per player, so there is no dense ``[N, 2, T, C]``
    tensor to store.  Each split keeps its streams concatenated end to end with
    an offsets array, exactly like a CSR row pointer: stream ``s`` occupies
    ``values[offsets[s] : offsets[s + 1]]``.

    Streams are ordered ``s = replay_index * 2 + player_index``, so both of a
    replay's players sit adjacent and ``offsets`` has ``2 * N + 1`` entries.

    Storing ragged rather than padded is what makes "keep the whole game" cheap:
    padding every stream to the dataset maximum would waste roughly 4x the space
    on the event stream alone, and would force a percentile cap that discards
    real events from long games.  Batches are padded by
    ``collate_event_sequences`` to that batch's own maximum instead.

    The name arrays are the point of this spec: feature identity lives in the
    cache, never inferred from tensor width again.
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


@dataclass
class EventSplit:
    """One split's ragged streams, as handed to :class:`EventSequenceDataset`."""

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
    """Return type of ``load_event_cache``: three splits plus their metadata.

    ``mean``/``std`` are per-stat-channel, fitted on the training split only,
    and are persisted with the checkpoint so inference normalises identically.
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


class EventSequenceDataset(Dataset):
    """Serves one replay (both players' ragged streams) per index.

    Two load-time policies are applied here rather than baked into the cache,
    so both stay sweepable without re-extracting the dataset:

    ``slice_ticks``
        How many 7.1 s ``PlayerStats`` ticks form one level-1 slice.  The
        cache stores a per-token tick index; the slice index is just
        ``tick // slice_ticks``.  Default 1, so slice index, tick index and
        decoder output row are the same index and therefore name the same
        7.1 seconds.

    ``drop_tail_frac``
        Fraction of each game's slices to drop from the end.  Endgame slices
        encode the outcome through collapsed counters rather than through
        actionable behaviour; 0.1 mirrors the published ``k18``-of-20-bins
        decision for the same reason.  Set to 0.0 for the keep-everything
        ablation arm.
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
