"""
Guided VAE over per-player event streams (hierarchical transformer).

Each player is encoded on their own (shared weights), so ``z`` describes that
player only; the opponent enters through the race tokens and the outcome
head. Shapes below use ``B`` replays, 2 players, ``S`` slices, ``K`` events
per slice, ``C`` scaled stat channels and ``d`` the model width.

Encoder:

1. Stat token per slice: ``Linear(C -> d)`` of the slice's last stat tick,
   scaled by the model's own :class:`ChannelTransform`.
2. Event tokens: type + unit embedding + the event's offset before the end of
   its slice.
3. Level 1 (within a slice): the stat token attends over the slice's events
   and a learned "no events" token; plus ``log1p(event count)``.
4. Level 2 (across slices): ``[CLS, own race, opponent race, slices...]``
   through a pre-norm transformer, slices carrying a sinusoidal encoding of
   their real game time; ``CLS`` gives ``mu`` and ``logvar``.

Decoder: ``MLP(z, fourier(t seconds), race)`` -> the ``C`` channels at time
``t``, so the model can be queried at any game time.

Outcome: an antisymmetric head on both players' supervised dims,
``g(a, b) - g(b, a)``: swapping the players negates the logit exactly, so the
model cannot use the player slot.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from latent_trainer.features.channel_transform import ChannelTransform
from latent_trainer.features.event_transform import TICK_INTERVAL_LOOPS
from latent_trainer.features.perception_action_cycle import GAME_LOOPS_PER_SECOND

N_PREFIX_TOKENS = 3  # CLS, own race, opponent race


def sinusoidal_time(
    seconds: torch.Tensor,
    dim: int,
    min_period: float = 7.0,
    max_period: float = 4000.0,
) -> torch.Tensor:
    """
    Encode game times with sines and cosines of geometric periods.

    Parameters
    ----------
    seconds : torch.Tensor
        Game times in seconds, any shape.
    dim : int
        Output width (even).
    min_period : float
        Shortest period in seconds (one PlayerStats tick is 7.14 s), by
        default 7.0.
    max_period : float
        Longest period in seconds, by default 4000.0.

    Returns
    -------
    torch.Tensor
        ``[..., dim]`` encoding.
    """
    half = dim // 2
    exponent = torch.arange(half, device=seconds.device, dtype=torch.float32)
    exponent = exponent / max(half - 1, 1)
    periods = min_period * (max_period / min_period) ** exponent
    angles = 2 * math.pi * seconds.float().unsqueeze(-1) / periods
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class PairHead(nn.Module):
    """
    Antisymmetric pair classifier: ``logit = g(a, b) - g(b, a)``.

    Swapping the two players negates the logit, so
    ``P(player 1 wins | swapped) = 1 - P(player 1 wins)`` exactly.

    Parameters
    ----------
    in_dim : int
        Features per player.
    hidden : int
        Hidden width of ``g``, by default 32.
    """

    def __init__(self, in_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the logit that player 1 wins.

        Parameters
        ----------
        x : torch.Tensor
            Per-player features ``[B, 2, in_dim]``.

        Returns
        -------
        torch.Tensor
            Logits ``[B, 1]``.
        """
        a, b = x[:, 0], x[:, 1]
        return self.net(torch.cat([a, b], dim=-1)) - self.net(torch.cat([b, a], dim=-1))


class suEventGuidedVAE(nn.Module):
    """
    Hierarchical transformer guided VAE over per-player event streams.

    Parameters
    ----------
    channel_names : list[str]
        Raw stat channel names (cache order); scaled by :class:`ChannelTransform`.
    n_units : int
        Unit vocabulary size (id 0 is padding).
    n_event_types : int
        Event type vocabulary size (id 0 is padding).
    n_races : int
        Number of races, by default 3.
    d_model : int
        Model width, by default 128.
    n_heads : int
        Attention heads (must divide ``d_model``), by default 4.
    n_layers : int
        Level-2 transformer layers, by default 4.
    ffn_mult : int
        Feed-forward width as a multiple of ``d_model``, by default 4.
    dropout : float
        Dropout in the attention and feed-forward blocks, by default 0.1.
    latent_dim : int
        Latent size per player, by default 32.
    supervised_dim : int
        Latent dims per player read by the outcome head, by default 8.
    decoder_width : int
        Hidden width of the decoder MLP, by default 256.
    decoder_layers : int
        Hidden layers of the decoder MLP, by default 3.
    n_time_features : int
        Sine/cosine pairs encoding the decoder's query time, by default 16.

    Raises
    ------
    ValueError
        If ``supervised_dim`` is not in ``(0, latent_dim)`` or ``n_heads`` does
        not divide ``d_model``.
    """

    def __init__(
        self,
        channel_names: list[str],
        n_units: int,
        n_event_types: int,
        n_races: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        latent_dim: int = 32,
        supervised_dim: int = 8,
        decoder_width: int = 256,
        decoder_layers: int = 3,
        n_time_features: int = 16,
    ) -> None:
        super().__init__()
        if not 0 < supervised_dim < latent_dim:
            raise ValueError(
                f"supervised_dim must be in (0, latent_dim), got {supervised_dim} "
                f"and {latent_dim}"
            )
        if d_model % n_heads:
            raise ValueError(f"n_heads={n_heads} must divide d_model={d_model}")

        self.latent_dim = latent_dim
        self.supervised_dim = supervised_dim
        self.d_model = d_model
        self.n_time_features = n_time_features

        self.transform = ChannelTransform(channel_names)
        n_channels = self.transform.n_out

        # Level 1: one token per slice, attending over that slice's events.
        self.stat_proj = nn.Linear(n_channels, d_model)
        self.type_emb = nn.Embedding(n_event_types, d_model, padding_idx=0)
        self.unit_emb = nn.Embedding(n_units, d_model, padding_idx=0)
        self.offset_proj = nn.Linear(1, d_model)
        self.no_event = nn.Parameter(torch.zeros(1, 1, d_model))
        self.slice_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.count_proj = nn.Linear(1, d_model)
        self.slice_norm = nn.LayerNorm(d_model)

        # Level 2: prefix tokens, then slices with their game time.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.race_emb = nn.Embedding(n_races, d_model)
        self.role_emb = nn.Embedding(2, d_model)  # 0 = own race, 1 = opponent
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, n_layers, enable_nested_tensor=False
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Linear(d_model, 2 * latent_dim)

        # Decoder queried by time.
        self.dec_race = nn.Embedding(n_races, 16)
        widths = [latent_dim + 2 * n_time_features + 16] + [
            decoder_width
        ] * decoder_layers
        blocks: list[nn.Module] = []
        for w_in, w_out in zip(widths[:-1], widths[1:], strict=True):
            blocks += [nn.Linear(w_in, w_out), nn.GELU()]
        blocks.append(nn.Linear(widths[-1], n_channels))
        self.decoder = nn.Sequential(*blocks)

        self.outcome_head = PairHead(supervised_dim)

    # Encoder:
    def _slice_tokens(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build one token per slice from its stats and its events.

        Parameters
        ----------
        batch : dict
            Output of ``collate_event_sequences``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Slice tokens ``[B, 2, S, d]`` (zeros at padded slices) and the
            scaled stats of each slice ``[B, 2, S, C]``.
        """
        scaled = self.transform(batch["stat_values"])  # [B, 2, Ls, C]
        index = batch["slice_stat_index"]  # [B, 2, S]
        slice_stats = torch.gather(
            scaled, 2, index.unsqueeze(-1).expand(-1, -1, -1, scaled.shape[-1])
        )
        stat_tok = self.stat_proj(slice_stats)  # [B, 2, S, d]
        batch_size, n_players, n_slices, width = stat_tok.shape

        # Event tokens; one zero row is appended so empty streams can be gathered.
        evt_tok = self.type_emb(batch["evt_type"]) + self.unit_emb(batch["evt_unit"])
        end_loop = torch.gather(
            batch["slice_loops"], 2, batch["evt_slice"].clamp(max=max(n_slices - 1, 0))
        )
        ticks_before_end = (end_loop - batch["evt_loops"]).float() / TICK_INTERVAL_LOOPS
        evt_tok = evt_tok + self.offset_proj(
            ticks_before_end.clamp(min=0).unsqueeze(-1)
        )
        evt_tok = torch.cat(
            [evt_tok, evt_tok.new_zeros(batch_size, n_players, 1, width)], 2
        )

        group = batch["evt_group"]  # [B, 2, S, K]
        k = group.shape[-1]
        flat = evt_tok.reshape(batch_size * n_players, -1, width)
        gathered = torch.gather(
            flat,
            1,
            group.reshape(batch_size * n_players, n_slices * k, 1).expand(
                -1, -1, width
            ),
        ).reshape(batch_size, n_players, n_slices, k, width)

        # Level 1, on real slices only.
        valid = batch["slice_mask"]
        query = stat_tok[valid].unsqueeze(1)  # [Nv, 1, d]
        keys = gathered[valid]  # [Nv, K, d]
        n_valid = keys.shape[0]
        keys = torch.cat([self.no_event.expand(n_valid, 1, width), keys], dim=1)
        padding = torch.cat(
            [
                torch.zeros(n_valid, 1, dtype=torch.bool, device=keys.device),
                ~batch["evt_group_mask"][valid],
            ],
            dim=1,
        )
        attended, _ = self.slice_attn(
            query, keys, keys, key_padding_mask=padding, need_weights=False
        )
        count = torch.log1p(batch["evt_count"][valid].float()).unsqueeze(-1)
        slice_vec = self.slice_norm(
            query.squeeze(1) + attended.squeeze(1) + self.count_proj(count)
        )
        tokens = stat_tok.new_zeros(batch_size, n_players, n_slices, width)
        tokens[valid] = slice_vec.to(tokens.dtype)
        return tokens, slice_stats

    def _encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode both players and keep the scaled slice stats for the decoder.

        Parameters
        ----------
        batch : dict
            Output of ``collate_event_sequences``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``mu``, ``logvar`` ``[B, 2, latent_dim]`` and slice stats
            ``[B, 2, S, C]``.
        """
        tokens, slice_stats = self._slice_tokens(batch)
        batch_size, n_players, n_slices, width = tokens.shape
        seconds = batch["slice_loops"].float() / GAME_LOOPS_PER_SECOND
        tokens = tokens + sinusoidal_time(seconds, width).to(tokens.dtype)

        own = self.race_emb(batch["race"]) + self.role_emb.weight[0]
        opponent = self.race_emb(batch["race"].flip(1)) + self.role_emb.weight[1]
        cls = self.cls_token.expand(batch_size, n_players, width)
        prefix = torch.stack([cls, own, opponent], dim=2)  # [B, 2, 3, d]
        sequence = torch.cat([prefix, tokens], dim=2).reshape(
            batch_size * n_players, N_PREFIX_TOKENS + n_slices, width
        )
        padding = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    n_players,
                    N_PREFIX_TOKENS,
                    dtype=torch.bool,
                    device=tokens.device,
                ),
                ~batch["slice_mask"],
            ],
            dim=2,
        ).reshape(batch_size * n_players, N_PREFIX_TOKENS + n_slices)

        hidden = self.encoder(sequence, src_key_padding_mask=padding)
        stats = self.to_latent(self.out_norm(hidden[:, 0]))
        stats = stats.reshape(batch_size, n_players, 2 * self.latent_dim)
        mu, logvar = stats.split(self.latent_dim, dim=-1)
        return mu, torch.clamp(logvar, min=-20, max=2), slice_stats

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both players into their latent posteriors.

        Parameters
        ----------
        batch : dict
            Output of ``collate_event_sequences``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``mu`` and ``logvar`` ``[B, 2, latent_dim]``; ``logvar`` clamped to
            ``[-20, 2]``.
        """
        mu, logvar, _ = self._encode(batch)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample ``z = mu + eps * std``.

        Parameters
        ----------
        mu : torch.Tensor
            Posterior mean.
        logvar : torch.Tensor
            Posterior log-variance.

        Returns
        -------
        torch.Tensor
            Sampled latent, same shape as ``mu``.
        """
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    # Decoder and heads:
    def decode(
        self, z: torch.Tensor, seconds: torch.Tensor, race: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict the scaled stat channels at the given game times.

        Parameters
        ----------
        z : torch.Tensor
            Latents ``[N, latent_dim]``.
        seconds : torch.Tensor
            Query times in game seconds ``[N, T]``.
        race : torch.Tensor
            Race id per latent ``[N]``.

        Returns
        -------
        torch.Tensor
            Scaled channels ``[N, T, C]``; ``transform.inverse`` maps them to
            raw units.
        """
        n_times = seconds.shape[1]
        features = torch.cat(
            [
                z.unsqueeze(1).expand(-1, n_times, -1),
                sinusoidal_time(seconds, 2 * self.n_time_features).to(z.dtype),
                self.dec_race(race).unsqueeze(1).expand(-1, n_times, -1).to(z.dtype),
            ],
            dim=-1,
        )
        return self.decoder(features)

    def cls(self, z: torch.Tensor) -> torch.Tensor:
        """
        Return the logit that player 1 wins from both players' supervised dims.

        Parameters
        ----------
        z : torch.Tensor
            Latents ``[B, 2, latent_dim]``.

        Returns
        -------
        torch.Tensor
            Logits ``[B, 1]``.
        """
        return self.outcome_head(z[:, :, : self.supervised_dim])

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """
        Encode, sample, reconstruct every observed slice and predict the outcome.

        Parameters
        ----------
        batch : dict
            Output of ``collate_event_sequences``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``mu``, ``logvar``, ``z`` ``[B, 2, L]``; ``recon`` and ``target``
            ``[B, 2, S, C]``; ``slice_mask`` ``[B, 2, S]``; ``logit`` ``[B, 1]``.
        """
        mu, logvar, slice_stats = self._encode(batch)
        z = self.reparameterize(mu, logvar)
        batch_size, n_players, latent = z.shape
        n_slices = slice_stats.shape[2]
        seconds = batch["slice_loops"].float() / GAME_LOOPS_PER_SECOND
        recon = self.decode(
            z.reshape(batch_size * n_players, latent),
            seconds.reshape(batch_size * n_players, n_slices),
            batch["race"].reshape(batch_size * n_players),
        ).reshape(batch_size, n_players, n_slices, -1)
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "recon": recon,
            "target": slice_stats,
            "slice_mask": batch["slice_mask"],
            "logit": self.cls(z),
        }
