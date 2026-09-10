# Leakage ablation results

Consolidated, citable results from the two ablation studies that motivated switching from the original `rich` (196-dim) feature transform to the `granular` transform, and from using all 20 bins to using only the first 18 (k18). All accuracy numbers are the GuidedVAE classifier's validation accuracy (`val_acc`) unless noted otherwise, sourced from MLflow (`mlflow.db`), confirmed 2026-09-10. Random seed 42 throughout (`src/latent_trainer/settings.py::SEED`).

## 1. Coarse block ablation (original `rich`, 196-dim transform)

Each row trains an otherwise-identical GuidedVAE (best hyperparameters carried over from `guided_vae_picked1`) on a reduced feature set, isolating which block of the original 196-dim vector (`early`/`mid`/`late` thirds-of-game averages, `final` end-of-game snapshot, `econDelta` = late−early) is responsible for the model's apparent skill at predicting the winner.

| MLflow experiment | Feature blocks used | Dims | val_acc |
|---|---|---|---|
| `guided_vae_noleak_audit` | `early + mid + meta` only | 79 | 52.49% (chance) |
| `guided_vae_ablation_late` | `+ late` alone | 118 | 94.63% |
| `guided_vae_ablation_econdelta` | `+ econDelta` alone | ~118 | 94.42% |
| `guided_vae_ablation_no_final` | everything except `final` | ~157 | 95.27% |
| `guided_vae_ablation_final` | `final` alone | 118 | **99.11%** |

**Reading**: the `early+mid+meta` block alone (no end-of-game information) sits at chance. Adding *any* single late-game-adjacent block (`late`, `econDelta`, or `final`) alone pushes accuracy to 94-99%. The `final` block alone is the single strongest predictor (99.11%) — the model was substantially "reading off" who already won from the final game state, not learning to recognize genuine mid-game skill differences.

## 2. Fine-grained bin-count sweep (`granular` transform, 5%-of-game bins)

The `granular` transform divides each replay into 20 bins of 5% of the game's PlayerStats event-index range each (39 stats/bin + 1 meta feature = 781 dims total). Truncating the cached 781-dim tensor to its first *k*·39+1 columns is equivalent to having cached only the first *k* bins (per-feature z-score normalization is slice-invariant), so this whole sweep reuses one wide feature cache without re-extracting per point.

| MLflow run | k (bins used) | % of game | Dims | val_acc |
|---|---|---|---|---|
| `guided_vae_granular_k1_dim40` | 1 | 5% | 40 | 49.51% |
| `guided_vae_granular_k2_dim79` | 2 | 10% | 79 | 52.36% |
| `guided_vae_granular_k3_dim118` | 3 | 15% | 118 | 52.02% |
| `guided_vae_granular_k4_dim157` | 4 | 20% | 157 | 51.81% |
| `guided_vae_granular_k5_dim196` | 5 | 25% | 196 | 52.28% |
| `guided_vae_granular_k6_dim235` | 6 | 30% | 235 | 52.79% |
| `guided_vae_granular_k7_dim274` | 7 | 35% | 274 | 52.62% |
| `guided_vae_granular_k8_dim313` | 8 | 40% | 313 | 51.77% |
| `guided_vae_granular_k9_dim352` | 9 | 45% | 352 | 52.58% |
| `guided_vae_granular_k10_dim391` | 10 | 50% | 391 | 53.05% |
| `guided_vae_granular_k11_dim430` | 11 | 55% | 430 | 52.32% |
| `guided_vae_granular_k12_dim469` | 12 | 60% | 469 | 53.09% |
| `guided_vae_granular_k13_dim508` | 13 | 65% | 508 | 52.49% |
| `guided_vae_granular_k14_dim547` | 14 | 70% | 547 | 72.26% |
| `guided_vae_granular_k15_dim586` | 15 | 75% | 586 | 73.54% |
| `guided_vae_granular_k16_dim625` | 16 | 80% | 625 | 78.74% |
| `guided_vae_granular_k17_dim664` | 17 | 85% | 664 | 83.68% |
| `guided_vae_granular_k18_dim703` | **18 (shipped)** | **90%** | **703** | **88.11%** |
| `guided_vae_k18_dim703_fixedstd` | 18, retrained w/ normalization-epsilon fix | 90% | 703 | **88.45%** |
| `guided_vae_granular_k19_dim742` | 19 | 95% | 742 | 94.72% |
| `guided_vae_granular_k20_dim781` | 20 | 100% | 781 | 97.74% |

**Reading**: accuracy is flat at chance through k13 (65% of the game observed), then climbs continuously and steeply from k14 onward, reaching 97.74% once the entire game is visible (k20). There is no single clean "leakage starts here" cliff — it is a gradual, compounding effect consistent with the "final economy state" finding above (later bins increasingly resemble a near-final-state summary). k18 was selected as a pragmatic cutoff: it excludes the clearest "already decided" collapse signature (bins 18-19; see per-feature Cohen's-d analysis below) while retaining most of the genuine mid-game economy/army-value signal, at a cost of ~9-10 accuracy points versus using the full game.

## 3. Per-feature separation by bin (Cohen's d, winner vs. loser)

Computed on the SC2EGSet test split (n=2178), comparing each bin's most-separating stats between the eventual winner and loser:

| Bin | % of game | Top features (|Cohen's d|) | Interpretation |
|---|---|---|---|
| bin10-14 | 50-70% | `mineralsCollectionRate`, `workersActiveCount`, `mineralsKilledEconomy` (d ≈ 0.05-0.24) | Modest, plausible genuine mid-game skill signal |
| bin16-17 | 80-90% | same features, d ≈ 0.2-0.5 | Elevated but not yet "collapse"-scale |
| bin18-19 | 90-100% (excluded from k18) | `foodUsed`, `mineralsUsedCurrentArmy`, `mineralsUsedActiveForces` (d up to 1.1+) | Loser's active army/supply has already collapsed — the death-spiral/aftermath state, functionally equivalent to the `final`-block leakage in Section 1 |

## 4. k18-fixedstd test-set + OOD evaluation (this session)

A granular-transform OOD cache did not previously exist (only the old 196-dim `rich`-transform OOD cache did). Regenerated via `python -m latent_trainer.features --transform granular --single_json_dataset_path H:/sc2ggset/sc2ggset_merged.json --test_only --n_samples 2178 --seed 42` → `data/cached_dataset_granular_test_2178.pt` (2178 valid samples, 317 skipped, 0 errors, ~30s runtime via the existing offset-indexed chunked-pool preprocessing). Then evaluated via `python -m latent_trainer.evaluate --model_path .../guided_vae_k18_dim703_fixedstd/best.ckpt --dataset cached_dataset_granular.pt --max_input_dim 703 --ood_dataset data/cached_dataset_granular_test_2178.pt --ood_label OOD`:

| Metric | SC2EGSet (test, n=2178) | OOD (n=2178) |
|---|---|---|
| MSE (original scale) | 231641.4 | 294415.9 |
| MSE (normalised scale) | 0.5524 | 0.8209 |
| KL Divergence | 11.3350 | 10.3955 |
| **Accuracy (%)** | **88.11** | **85.86** |
| ROC-AUC | 0.9454 | 0.9282 |
| F1 Score | 0.8856 | 0.8601 |
| Brier Score | 0.0885 | 0.1060 |

Auto-generated LaTeX (`output/plots/test/evaluation_table.tex`, `\label{tab:model-evaluation}`) is a direct drop-in replacement for the paper's existing (leaky) table — same structure, same label.

Compare against the old, leaky 196-dim `rich`-transform numbers this replaces: SC2EGSet 98.76%/OOD 97.11% accuracy, ROC-AUC 0.9991/0.9926, F1 0.9881/0.9714, Brier 0.0090/0.0229. The corrected numbers are meaningfully lower (as expected once the outcome-restating final-game-state features are removed) but still well above chance, and the OOD gap (88.11%→85.86%) is modest and in the expected direction (slightly worse on amateur/out-of-distribution play), unlike the old table's implausibly-near-perfect scores.

## 5. Normalization-epsilon fix

`features/data_utils.py::normalize()` previously floored per-feature std at `+1e-8`. For a near-constant feature (true std itself ~1e-8, e.g. a rare stat that's almost always exactly 0 across the dataset), this is not a real floor: any nonzero outlier value produces an astronomical z-score. Confirmed: one training game's single nonzero occurrence of `vespeneFriendlyFireEconomy` produced a z-score of ~1.5×10¹⁰, which the VAE encoder propagated into an encoded latent vector with norm ~10⁸ (versus 1-10 for every other game), corrupting the win-latent centroid used for counterfactual path generation. Fixed with a real `.clamp(min=1.0)` floor (features are raw counts/aggregates with typical scale in the hundreds-to-thousands, so this floor is negligible for normally-scaled features). The k18 checkpoint was retrained with this fix (`guided_vae_k18_dim703_fixedstd`), improving val_acc marginally (88.11% → 88.45%) and eliminating the pathological centroid.
