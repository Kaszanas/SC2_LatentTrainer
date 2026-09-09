# Research note (deferred): raw PlayerStats sequence model

Deferred out of the active 3-day leakage-fix/publication plan (see the plan doc) because it requires new architecture (sequence encoder, padded/packed batching) rather than a config-level change to the existing MLP `suGuidedVAE`. Revisit after publication.

## Idea

Instead of the hand-crafted early/mid/late/final/econDelta aggregation in `src/latent_trainer/features/rich_transform.py`, feed the model the raw per-event `Stats` time series directly, to test whether hand-engineered windowing throws away signal (or hides leakage inside the aggregation choices).

## Why it's bigger than the leakage-ablation change

`suGuidedVAE`/`LitGuidedVAE`'s encoder is an MLP over one fixed-size vector. A raw event sequence is variable-length per replay (different games have different numbers of PlayerStats ticks), so it needs a sequence encoder front-end, not just a different vector size.

## Sketch

1. **New transform** `src/latent_trainer/features/sequence_transform.py`: for each player, emit the raw ordered list of 39-dim `Stats` vectors (reuse `_get_player_stats_timeseries`/`_get_stats_values` from `rich_transform.py`), truncated to exclude the final events/fraction (same leakage concern as the `final`/`late` blocks, just at the raw-sequence level).
2. **New cache format**: `CachedDatasetFileSpec` (`features/type.py`) stores fixed-shape `[N, 2, 196]` tensors — can't hold ragged sequences. Add a padded variant: `[N, 2, max_len, 39]` zero-padded tensor + `lengths[N, 2]`, built with `torch.nn.utils.rnn.pad_sequence`.
3. **Batching**: `collate_fn` in `features/data_utils.py` pads each batch to that batch's max length, returns `(padded_batch, lengths)`; feed `lengths` into `torch.nn.utils.rnn.pack_padded_sequence` before a recurrent layer (or use an attention mask for a Transformer encoder instead).
4. **New encoder front-end**: GRU/LSTM (packed-sequence) or masked 1D-CNN/Transformer consuming `(padded_sequence, lengths)`, producing one fixed-size embedding per player that feeds into the *existing* `suGuidedVAE` latent/classifier head unchanged — only the front-end is new.
5. Compare against the ablated hand-feature model (early+mid+meta, 79 dims) on the same held-out test split (accuracy/AUC/F1); confirm the sequence model needs the same "no final-game events" truncation to avoid the same leakage problem.
6. **Hyperparameters**: new knobs (recurrent hidden size, num layers, truncation length) not covered by `get_guided_vae_search_space` (`configs/search_space.py`) — would need its own small search space, one sweep, then reuse the existing `train.py --mode best` retrain-from-saved-params flow for subsequent runs.

**Files that would be touched/created:** `src/latent_trainer/features/sequence_transform.py` (new), `src/latent_trainer/features/type.py` (new padded cache spec), `src/latent_trainer/features/data_utils.py` (collate/padding), `src/latent_trainer/models/` (new sequence encoder module + wiring into `suGuidedVAE`), `src/latent_trainer/configs/search_space.py` (new search space).
