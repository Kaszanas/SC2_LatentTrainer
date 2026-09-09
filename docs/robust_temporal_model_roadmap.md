# Roadmap: a duration-robust, leakage-free temporal model

Follow-up research notes from the leakage-audit work on `rich_transform.py`. Not scheduled against the 3-day publication deadline — this is what to tackle afterward. See also `local.md` (repo root) for the original raw-sequence-model sketch this roadmap supersedes/extends.

## What we found (context)

Two ablation passes on the `rich` transform's temporal windows:

1. **Coarse block ablation** (`early`/`mid`/`late`/`final`/`econDelta`/`meta`): `early+mid+meta` alone (79 dims) sits at chance (52.49% val_acc); adding `final` alone (118 dims) hits 99.11%; `late` alone (118 dims) hits 94.63%.
2. **Fine-grained 5%-of-game sweep** (20 bins, cumulative prefix `--max_input_dim` slicing of one `granular` cache): accuracy is flat at chance through the first 65% of each replay's own event-index timeline (bins 1-13, 49.5-53.1%), then jumps to 72.26% at bin 14 (70%) and climbs smoothly to 97.74% at bin 20 (100%).

Both point to the same thing: **the current windowing scheme (event-index fraction of each replay's own length) puts the "outcome becomes readable" transition inside the observation window for a large fraction of games**, because it defines "early/mid/late" relative to a game's own total duration rather than to any fixed, externally-meaningful point in time.

## The bias the user flagged: percentage bins != equal information

Percentage-of-own-duration bins have two separate problems, not one:

1. **Placement leaks total duration.** `start_idx = int(start_frac * n)` (`rich_transform.py`) uses `n = len(events)` — the replay's own total event count — to decide where a window falls. A model conditioned on features built this way implicitly "knows" how long the game was, even though duration itself is a strong outcome-correlated signal (blowouts are short).
2. **Sample density scales with game length, not bin fraction.** A 5% bin in a 40-minute game covers roughly 2 minutes of real time and many more `PlayerStats` ticks than the same 5% bin in a 12-minute game covering ~36 seconds. Averaging over more ticks is a lower-variance estimate. So two replays' "bin 5" values aren't estimating the same underlying real-time quantity with the same precision — longer games get cleaner (lower-noise) features "for free," independent of whether their in-game state is actually more informative. This could artificially inflate accuracy for longer games and deflate it for short ones, confounded with duration in a second, distinct way from point 1.

Both problems point the same direction: stop defining windows as fractions of each replay's own length.

## Proposed direction 1 — fixed wall-clock (game-loop) windows

Replace `_temporal_snapshot`'s `start_frac * n` indexing with absolute game-loop cutoffs (`event.loop`, already available and already used for the existing ≥4032-loop / 3-minute minimum-duration filter):

- Bin boundaries become e.g. `[0, 30s), [30s, 60s), ...` in game-loop terms, identical across every replay, regardless of total length.
- Replays shorter than the cutoff span analyzed are excluded (extending the existing minimum-duration filter, not a new mechanism).
- This removes leak #1 outright (no replay-specific total length used to place a boundary) and substantially reduces leak #2 (if `PlayerStats` ticks fire at a roughly constant game-loop interval — worth verifying empirically across the dataset/patch versions before relying on it — then a fixed-duration bin has a roughly constant tick count regardless of total game length).
- Re-run the same expanding-window leak-localization sweep under this scheme once implemented; expect the transition point to move and, more importantly, to mean the same thing ("N minutes into the game") across every replay, which the current 65-70%-of-own-length transition does not.

## Proposed direction 2 — architecture: convolution (or recurrent) over fixed-duration bins

Once bins are fixed-duration, a real time-series architecture becomes worthwhile instead of flattening bins into one vector for an MLP:

- **1D conv encoder**: reshape features as `[batch, 39 channels, T timesteps]`. Stack `Conv1d → BatchNorm → ReLU → MaxPool1d(2)`, increasing channels (39→64→128) while halving `T` each pool — a learned version of the hierarchical bin-merging ("5+5→10→20") discussed in conversation, instead of hand-picked averaging.
- **`AdaptiveAvgPool1d(1)` at the end** collapses whatever's left of `T` to a fixed-size vector regardless of how many bins were fed in. This is the practical win over the current MLP approach: **one trained model handles "5 minutes of game" or "20 minutes of game" input without retraining**, unlike today's setup where every sweep point (`--max_input_dim`) needs its own independent retrain because the first `Linear` layer's shape is hard-baked to `input_dim`.
- Feeds into the existing `suGuidedVAE` latent head (mu/logvar) + classifier unchanged — only the encoder/decoder front-end changes.
- **Generative/decoder side**: mirror with `ConvTranspose1d` layers upsampling `T` back up as channels shrink (128→64→39), reconstructing `[39, T]` from `z` — standard 1D-conv-VAE pattern (same idea as image VAEs' `Conv2d`/`ConvTranspose2d`, just in 1D over time).
- **Alternative**: GRU/LSTM with `pack_padded_sequence` (the original sketch in `local.md`) or a small Transformer encoder with positional encoding over bin index. Conv is likely the best cost/benefit for this data — SC2 games at 1-2 minute bin granularity are short sequences (10-40 timesteps), where conv/RNN capacity is plenty and a Transformer's extra flexibility probably isn't needed.
- None of the current best-hyperparameters (`encoder_hidden_dims=[32,16]`, etc.) transfer to this architecture — needs its own search space and sweep from scratch (`configs/search_space.py`).

## Known hard limitation — duration-conditional leakage isn't fully fixable by rebinning

Switching to fixed wall-clock cutoffs removes the *windowing* artifacts above, but does **not** remove a deeper, real property of SC2: **short games are often short because they were decided early** (cheese/all-in aggression ending a game at minute 6 was, by construction, "decided" well before minute 6 would be "late game" in a 25-minute macro game). A fixed real-time cutoff (say, minute 10) is "clearly early game, nothing decided yet" for a long macro game but "already near the end" for a short blowout of the same wall-clock length. This is not an artifact of feature engineering — it's intrinsic to the sport — and no amount of rebinning fully separates "time elapsed" from "how decided the game already is" when game length itself is an outcome-correlated variable.

Mitigations worth considering, not mutually exclusive:
- Report/evaluate accuracy stratified by total game duration bucket, rather than one pooled number — makes the "how much time do you need to watch before you can call it" question honestly duration-conditional instead of averaging over a mix of game lengths.
- Consider event-count-based cutoffs (first N `PlayerStats` ticks) as a third windowing scheme alongside fraction-based and time-based, and compare which best decorrelates from total game duration in practice — worth an empirical check rather than assuming wall-clock time alone fully solves it.

## Suggested phased order

1. Verify the `PlayerStats` tick-interval-is-roughly-constant assumption empirically (quick data check, no model training needed) — this determines how much of leak #2 fixed-duration bins actually solve.
2. Implement fixed wall-clock windowing in `rich_transform.py` (or a new `rich_transform_timewindowed.py`, reusing `_get_player_stats_timeseries`/`_get_stats_values`), re-run the leak-localization sweep with the existing MLP + `--max_input_dim` slicing to get a first, cleaner read cheaply before investing in new architecture.
3. Stratify results by game-duration bucket to characterize the hard limitation above.
4. If the cleaner windowing still shows a sharp, useful "enough game observed" point, build the conv-VAE encoder/decoder and its own hyperparameter search space, so a single model can be queried at any input duration instead of needing one retrain per cutoff.
