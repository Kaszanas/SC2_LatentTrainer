"""
Per-channel input scaling of the stat channels, owned by the model.

Each channel gets one treatment (agreed from the EDA, training split):

- ``"z"``: z-score only (bounded, evenly spread channels);
- ``"clip"``: cap at the training 99.9th percentile, then z-score (PAC
  durations and action counts, whose extremes are idle stretches or
  auto-generated bursts);
- ``"symlog"``: ``sign(x) * log1p(|x|)``, then z-score (resource accounting
  channels, where ratios matter and refunds make some values negative);
- ``"drop"``: removed from the model input (near-constant channels).

The z-score divides by each channel's true standard deviation, floored only at
a tiny epsilon. All parameters are fitted on the training split and stored as
buffers, so a checkpoint carries its own scaling and serving applies exactly
the same transform.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

Z_ONLY: frozenset[str] = frozenset(
    {
        "foodMade",
        "foodUsed",
        "workersActiveCount",
        "mineralsCollectionRate",
        "vespeneCollectionRate",
        "pac_cycles",
        "pac_has_latency",
    }
)
CLIPPED: frozenset[str] = frozenset({"pac_actions", "pac_latency_s", "pac_gap_s"})
DROPPED: frozenset[str] = frozenset({"vespeneFriendlyFireEconomy"})
SYMLOG_PREFIXES: tuple[str, ...] = ("minerals", "vespene")
KINDS: tuple[str, ...] = ("z", "clip", "symlog")


def channel_kind(name: str) -> str:
    """
    Return the treatment of one stat channel.

    Parameters
    ----------
    name : str
        Channel name.

    Returns
    -------
    str
        ``"z"``, ``"clip"``, ``"symlog"`` or ``"drop"``.

    Raises
    ------
    ValueError
        If the channel has no agreed treatment; new channels must be added to
        the plan explicitly.
    """
    if name in DROPPED:
        return "drop"
    if name in Z_ONLY:
        return "z"
    if name in CLIPPED:
        return "clip"
    if name.startswith(SYMLOG_PREFIXES):
        return "symlog"
    raise ValueError(f"no scaling plan for channel {name!r}")


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the signed log ``sign(x) * log1p(|x|)``.

    Parameters
    ----------
    x : torch.Tensor
        Input.

    Returns
    -------
    torch.Tensor
        Transformed input.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    Invert :func:`symlog`.

    Parameters
    ----------
    x : torch.Tensor
        Signed-log values.

    Returns
    -------
    torch.Tensor
        Original-scale values.
    """
    return torch.sign(x) * torch.expm1(torch.abs(x))


class ChannelTransform(nn.Module):
    """
    Fit-once, per-channel scaling from raw stat channels to model inputs.

    Parameters
    ----------
    channel_names : list[str]
        Names of the raw input channels, in cache order.
    clip_quantile : float
        Quantile used as the upper clip of ``"clip"`` channels, by default 0.999.
    eps : float
        Floor on each channel's standard deviation, by default 1e-6.

    Raises
    ------
    ValueError
        If a channel has no treatment or ``clip_quantile`` is not in ``(0, 1]``.
    """

    def __init__(
        self,
        channel_names: list[str],
        clip_quantile: float = 0.999,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 < clip_quantile <= 1.0:
            raise ValueError(f"clip_quantile must be in (0, 1], got {clip_quantile}")
        self.channel_names = list(channel_names)
        self.clip_quantile = clip_quantile
        self.eps = eps

        kinds = [channel_kind(name) for name in self.channel_names]
        keep = [i for i, kind in enumerate(kinds) if kind != "drop"]
        self.output_names = [self.channel_names[i] for i in keep]
        kept_kinds = [kinds[i] for i in keep]
        n_out = len(keep)

        self.register_buffer("keep_index", torch.tensor(keep, dtype=torch.long))
        self.register_buffer(
            "is_clip", torch.tensor([k == "clip" for k in kept_kinds], dtype=torch.bool)
        )
        self.register_buffer(
            "is_symlog",
            torch.tensor([k == "symlog" for k in kept_kinds], dtype=torch.bool),
        )
        self.register_buffer("clip_max", torch.full((n_out,), float("inf")))
        self.register_buffer("mean", torch.zeros(n_out))
        self.register_buffer("std", torch.ones(n_out))
        self.register_buffer("fitted", torch.tensor(False))

    @property
    def n_out(self) -> int:
        """
        Return the number of model input channels.

        Returns
        -------
        int
            Channels kept after dropping.
        """
        return len(self.output_names)

    def kinds(self) -> list[str]:
        """
        Return the treatment of each output channel.

        Returns
        -------
        list[str]
            ``"z"``, ``"clip"`` or ``"symlog"`` per output channel.
        """
        return [channel_kind(name) for name in self.output_names]

    def _scale(self, kept: torch.Tensor) -> torch.Tensor:
        """
        Apply the clip and the signed log, before the z-score.

        Parameters
        ----------
        kept : torch.Tensor
            Raw kept channels ``[..., n_out]``.

        Returns
        -------
        torch.Tensor
            Clipped and log-scaled channels.
        """
        clipped = torch.where(self.is_clip, torch.minimum(kept, self.clip_max), kept)
        return torch.where(self.is_symlog, symlog(clipped), clipped)

    @torch.no_grad()
    def fit(self, raw: torch.Tensor) -> ChannelTransform:
        """
        Fit the clip limits, means and stds on raw training ticks.

        Parameters
        ----------
        raw : torch.Tensor
            Raw training stat values ``[N, C]`` (no padding rows).

        Returns
        -------
        ChannelTransform
            ``self``, fitted.

        Raises
        ------
        ValueError
            If ``raw`` does not have one column per input channel.
        """
        if raw.dim() != 2 or raw.shape[1] != len(self.channel_names):
            raise ValueError(
                f"expected [N, {len(self.channel_names)}] raw values, "
                f"got {tuple(raw.shape)}"
            )
        kept = raw.double()[:, self.keep_index.cpu()]
        limits = np.quantile(kept.numpy(), self.clip_quantile, axis=0)
        clip_max = torch.where(
            self.is_clip.cpu(),
            torch.from_numpy(limits),
            torch.full_like(torch.from_numpy(limits), float("inf")),
        )
        self.clip_max.copy_(clip_max.to(self.clip_max))
        # Statistics in float64: millions of rows lose precision in float32.
        is_clip, is_symlog = self.is_clip.cpu(), self.is_symlog.cpu()
        clipped = torch.where(is_clip, torch.minimum(kept, clip_max), kept)
        scaled = torch.where(is_symlog, symlog(clipped), clipped)
        self.mean.copy_(scaled.mean(dim=0).to(self.mean))
        self.std.copy_(scaled.std(dim=0).clamp(min=self.eps).to(self.std))
        self.fitted.fill_(True)
        return self

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Turn raw stat channels into model inputs.

        Parameters
        ----------
        raw : torch.Tensor
            Raw values ``[..., C]``.

        Returns
        -------
        torch.Tensor
            Scaled values ``[..., n_out]``.

        Raises
        ------
        RuntimeError
            If the transform has not been fitted.
        """
        if not bool(self.fitted):
            raise RuntimeError("ChannelTransform used before fit()")
        kept = raw.index_select(-1, self.keep_index).to(self.mean.dtype)
        return (self._scale(kept) - self.mean) / self.std

    def inverse(self, scaled: torch.Tensor) -> torch.Tensor:
        """
        Map model-space values back to raw units of the kept channels.

        Clipped channels come back at most at their clip limit.

        Parameters
        ----------
        scaled : torch.Tensor
            Model-space values ``[..., n_out]``.

        Returns
        -------
        torch.Tensor
            Raw-unit values ``[..., n_out]``, in :attr:`output_names` order.
        """
        x = scaled * self.std + self.mean
        return torch.where(self.is_symlog, symexp(x), x)
