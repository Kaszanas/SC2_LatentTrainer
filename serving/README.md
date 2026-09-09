# latent-trainer-serve

Single-replay hosting pipeline for the SC2 Latent Trainer counterfactual model. Wraps `latent_trainer.inference` (extraction -> features -> encode -> path -> feedback) behind a CLI and a FastAPI upload API + minimal web page.

**Model used**: the served model is always `guided_vae_k18_dim703_fixedstd/best.ckpt` -- the granular transform's first 18 of 20 5%-of-game bins (90% of the game, 703 dims), trained with a fixed normalization epsilon (see below). This is a deliberate accuracy/leakage tradeoff, not just "use everything": a full k=1..20 bin-count sweep (`docs/robust_temporal_model_roadmap.md`) showed validation accuracy flat at chance through k13 (65% of the game), then climbing continuously to 98% by k20 -- and per-feature analysis showed bins 18-19 (the last 10%) carry a clear "already decided" collapse signature (e.g. the loser's active army/supply has already crashed), the same kind of leakage Part 1 found in the old `final`/`late` blocks. k18 (88% val_acc) excludes that clearest collapse signal while keeping most of the real economy/army-value signal; k19/k20 and the older 196-dim `best.ckpt`/`best_flow.ckpt` must never be pointed at by this package.

**Normalization fix**: `features/data_utils.py::normalize()` used to floor per-feature std at `+1e-8`, which is not a real floor for a near-constant feature (true std can itself be ~1e-8) -- a single training game with one nonzero occurrence of an otherwise-always-zero stat produced a z-score of ~1.5e10, which the VAE encoder propagated into an astronomically large latent vector that poisoned the win-latent centroid (visible as absurd counterfactual targets, e.g. a feature currently at ~2000 getting a "target" of 140+ million). Fixed with a real `.clamp(min=1.0)` floor and this checkpoint was retrained with it. `inference/pipeline.py::_load_model` also floors any already-loaded checkpoint's `std` defensively, and the `centroid` method filters norm-outliers (MAD-based) before averaging as a second line of defense.

## Setup

```bash
docker pull kaszanas/sc2infoextractorgo:dev
cd serving
uv sync
```

Known limitation: `latent-trainer-serve` depends on the full `latent-trainer` package, which (via its own `pyproject.toml`) pulls in the complete training dependency set (ray, optuna, mkdocs, umap, ...), not just the inference path. Splitting `latent_trainer` into a lean "core" install and a "training-only" extra is a legitimate follow-up, not done here.

## Usage

**1. Build a reference pack** (one-time, per checkpoint) -- precomputes the win/loss latent pool the path-charting strategies compare a new replay against, so serving doesn't need the full training cache:

```bash
python -m latent_trainer_serve build-reference-pack \
    --model ../output/checkpoints/.../best.ckpt \
    --dataset ../data/cached_dataset_granular.pt \
    --output reference_pack.pt
```

**2. Predict** on a replay:

```bash
python -m latent_trainer_serve predict \
    --replay game.SC2Replay \
    --model ../output/checkpoints/.../best.ckpt \
    --reference-pack reference_pack.pt
```

`--strategy` selects one of `linear` (centroid/nearest-k-NN target), `gradient_ascent` (P(win) ascent + KDE density regularization), or `optimal_transport` (Wasserstein-barycentric path). `neural_flow` is not available here -- it needs a separately trained OT-flow-matching checkpoint (`latent_trainer/paths/cli.py`'s `cmd_neural_flow`) that doesn't exist for this granular feature set.

Extraction results are cached by replay content hash under `~/.cache/latent_trainer_serve/` (override with `--cache-dir`), so re-running `predict` on the same replay (e.g. to try a different `--strategy`) skips both the extraction and the feature computation.

## API + web UI

`docker-compose.yml` defines an `api` service that runs the FastAPI app (`latent_trainer_serve.api:app`) instead of the CLI, exposing an upload endpoint and a small web page:

```bash
docker compose -f serving/docker-compose.yml build api
docker compose -f serving/docker-compose.yml up api
```

Then either:

```bash
curl -F file=@game.SC2Replay http://localhost:8000/analyze
```

or open <http://localhost:8000/> in a browser, upload a replay, and view the ranked feedback tables + plots inline (each plot links to its PDF for the publication-quality version). `/analyze` always computes all three strategies (`linear`, `gradient_ascent`, `optimal_transport`) for every upload -- one report comparing all three, rather than picking one -- so a single upload takes a few minutes. If one strategy fails (e.g. `optimal_transport`'s non-finite-result guard), its section shows the error while the other two still render normally.

The `api` service's `MODEL_PATH`/`REFERENCE_PACK_PATH` env vars are hardcoded in `docker-compose.yml` to the k18 checkpoint + `reference_pack_k18_fixedstd.pt` -- these are the only model files mounted into the container, so there's no way to point a request at a different (or leaky) model. Every request's plots are written to a fresh, timestamped directory under `RESULTS_DIR` (`./api_results` on the host, via the compose volume) so concurrent uploads never overwrite each other's output, and results stay traceable by creation time.

Running the CLI (`predict`/`build-reference-pack`) inside a container instead uses the `frontend` service:

```bash
docker compose -f serving/docker-compose.yml run --rm frontend predict \
    --replay /replays/game.SC2Replay \
    --model /models/guided_vae_granular_sweep/guided_vae_k18_dim703_fixedstd/best.ckpt \
    --reference-pack /models/... # mount your reference pack similarly, or copy it into ./replays \
    --extractor-binary /usr/local/bin/SC2InfoExtractorGo
```

## Docker

SC2InfoExtractorGo's binary is baked directly into this image via a multi-stage build (`FROM kaszanas/sc2infoextractorgo:dev AS extractor` in `serving/Dockerfile`, copied to `/usr/local/bin/SC2InfoExtractorGo`), and `extract.py` invokes it directly as a local subprocess when `extractor_binary` is set (both `frontend` and `api` set `EXTRACTOR_BINARY` to that path). This avoids Docker-out-of-Docker entirely -- no docker CLI, no host socket mount, no sibling container spun up per replay. `extract_replay()` still supports the Docker-run fallback (`docker_image=...`) for local/dev use without building this image.
