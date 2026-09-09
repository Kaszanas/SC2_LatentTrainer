"""FastAPI app: upload a .SC2Replay, get a counterfactual feedback report back.

Configured entirely via environment variables (never per-request), so a
client can never point the server at a different checkpoint -- in
particular, never at the old leaky 196-dim `best.ckpt`/`best_flow.ckpt`
(full-game-state features), nor at k19/k20 of the granular sweep (bins
covering the last 10-15% of the game show a clear "already decided"
collapse signature -- see docs/robust_temporal_model_roadmap.md). The
default here is k18 (first 18 of 20 5%-of-game bins, 90% of the game,
703 dims, ~88% val_acc) -- the container/compose file should set these to
match:

    MODEL_PATH            -- guided-VAE checkpoint (.ckpt)
    REFERENCE_PACK_PATH   -- matching reference pack (.pt)
    DOCKER_IMAGE          -- SC2InfoExtractorGo image (default: dev tag)
    CACHE_DIR             -- per-replay feature/extraction cache
    RESULTS_DIR           -- where per-request plots/reports are written
                             and served back from

Run locally (from serving/, with the `api` dependency group installed):

    uvicorn latent_trainer_serve.api:app --host 0.0.0.0 --port 8000

Then either POST a replay directly:

    curl -F file=@game.SC2Replay http://localhost:8000/analyze

or open http://localhost:8000/ for the upload page.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from latent_trainer.inference.extract import DEFAULT_DOCKER_IMAGE, ReplayExtractionError
from latent_trainer.inference.pipeline import DEFAULT_CACHE_DIR, predict_replay

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config -- environment variables only, set once at process startup. This is
# the enforcement point for "never serve the leaky model": there is no
# request parameter that can change MODEL_PATH/REFERENCE_PACK_PATH.
# ---------------------------------------------------------------------------
_SERVING_ROOT = Path(__file__).resolve().parents[3]  # serving/

MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        _SERVING_ROOT.parent
        / "output"
        / "checkpoints"
        / "guided_vae_granular_sweep"
        / "guided_vae_k18_dim703_fixedstd"
        / "best.ckpt",
    )
).resolve()
REFERENCE_PACK_PATH = Path(
    os.environ.get(
        "REFERENCE_PACK_PATH", _SERVING_ROOT / "reference_pack_k18_fixedstd.pt"
    )
).resolve()
DOCKER_IMAGE = os.environ.get("DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)
# Set by default to the path the Dockerfile bakes the extractor binary into
# (multi-stage COPY from kaszanas/sc2infoextractorgo:dev) -- no Docker CLI or
# host socket needed inside this container. Empty string / unset env var
# falls back to Docker-out-of-Docker for non-containerized/local use.
_extractor_binary_env = os.environ.get(
    "EXTRACTOR_BINARY", "/usr/local/bin/SC2InfoExtractorGo"
)
EXTRACTOR_BINARY = (
    Path(_extractor_binary_env).resolve()
    if _extractor_binary_env and Path(_extractor_binary_env).is_file()
    else None
)
CACHE_DIR = Path(os.environ.get("CACHE_DIR", DEFAULT_CACHE_DIR)).resolve()
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", CACHE_DIR / "api_results")).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SC2 Latent Trainer -- counterfactual feedback API")


@app.on_event("startup")
def _check_config() -> None:
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"MODEL_PATH does not exist: {MODEL_PATH}. Set the MODEL_PATH env "
            "var to a trained guided-VAE checkpoint (the non-leaking granular "
            "model)."
        )
    if not REFERENCE_PACK_PATH.is_file():
        raise RuntimeError(
            f"REFERENCE_PACK_PATH does not exist: {REFERENCE_PACK_PATH}. Set "
            "the REFERENCE_PACK_PATH env var to a reference pack built for "
            "MODEL_PATH (see 'build-reference-pack')."
        )
    logger.info(
        "Serving model=%s reference_pack=%s extractor_binary=%s (docker_image=%s)",
        MODEL_PATH,
        REFERENCE_PACK_PATH,
        EXTRACTOR_BINARY,
        DOCKER_IMAGE,
    )


def _request_dir() -> Path:
    """A fresh, timestamped, collision-free directory for one request's plots.

    Named by wall-clock time (to UTC seconds) plus a short random suffix, so
    concurrent uploads never write into the same directory and the creation
    time of any report is visible directly from its path/URL.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    request_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
    request_dir = RESULTS_DIR / request_id
    request_dir.mkdir(parents=True, exist_ok=False)
    return request_dir


def _list_history(query: str | None = None) -> list[dict]:
    """List past analyses (newest first), each described by its meta.json.

    ``query`` filters by case-insensitive substring match against the
    original replay filename, so a replay can be found again by name.
    """
    entries = []
    for meta_path in RESULTS_DIR.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if query and query.lower() not in meta.get("replay_name", "").lower():
            continue
        entries.append(meta)
    entries.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return entries


def _feedback_to_json(feedback: dict) -> dict:
    """Strip the raw numpy-array (`_`-prefixed) keys -- only the already-
    plain-Python ranked tables and labels are meant for a client."""
    return {k: v for k, v in feedback.items() if not k.startswith("_")}


def _plot_urls(saved_files: dict, request_id: str) -> dict:
    urls = {}
    for plot_key, paths in saved_files.items():
        urls[plot_key] = {
            fmt: f"/results/{request_id}/{Path(path).name}"
            for fmt, path in paths.items()
        }
    return urls


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    player: Optional[int] = Form(None),
    method: str = Form("centroid"),
    k_neighbours: int = Form(5),
    n_steps: int = Form(50),
    top_k: int = Form(15),
) -> JSONResponse:
    if player is not None and player not in (0, 1):
        raise HTTPException(status_code=400, detail="player must be 0 or 1")
    if method not in ("centroid", "nearest"):
        raise HTTPException(status_code=400, detail="method must be 'centroid' or 'nearest'")

    request_dir = _request_dir()
    request_id = request_dir.name
    created_at = datetime.now(timezone.utc).isoformat()

    upload_name = Path(file.filename or "replay.SC2Replay").name
    replay_path = request_dir / upload_name
    with open(replay_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Written before extraction/prediction so the entry (and replay name) is
    # already searchable via /history even if analysis itself fails later.
    (request_dir / "meta.json").write_text(
        json.dumps(
            {
                "request_id": request_id,
                "replay_name": upload_name,
                "created_at": created_at,
            }
        ),
        encoding="utf-8",
    )

    try:
        result = predict_replay(
            replay_path=replay_path,
            model_path=MODEL_PATH,
            reference_pack_path=REFERENCE_PACK_PATH,
            player=player,
            strategy="linear",
            method=method,
            k_neighbours=k_neighbours,
            n_steps=n_steps,
            top_k=top_k,
            docker_image=DOCKER_IMAGE,
            extractor_binary=EXTRACTOR_BINARY,
            cache_dir=CACHE_DIR,
            output_dir=request_dir,
        )
    except ReplayExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    response = {
        "request_id": request_id,
        "replay_name": upload_name,
        "created_at": created_at,
        "feedback": _feedback_to_json(result["feedback"]),
        "plots": _plot_urls(result["saved_files"], request_id),
    }
    (request_dir / "report.json").write_text(json.dumps(response), encoding="utf-8")
    return JSONResponse(response)


@app.get("/history")
async def history(q: Optional[str] = None) -> JSONResponse:
    """List past analyses, newest first. ?q= filters by replay filename."""
    return JSONResponse(_list_history(query=q))


_REQUEST_ID_RE = re.compile(r"^\d{8}T\d{6}Z_[0-9a-f]{8}$")


@app.get("/history/{request_id}")
async def history_detail(request_id: str) -> JSONResponse:
    """Return one past analysis's full report, by its request_id."""
    if not _REQUEST_ID_RE.match(request_id):
        # request_id is used to build a filesystem path -- reject anything
        # that doesn't match the format _request_dir() actually generates,
        # rather than trusting client input (path traversal guard).
        raise HTTPException(status_code=400, detail="Invalid request_id")
    report_path = RESULTS_DIR / request_id / "report.json"
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail=f"No report found for {request_id!r}")
    return JSONResponse(json.loads(report_path.read_text(encoding="utf-8")))


# Static mounts -- registered after the routes above so /analyze/etc always
# match first; StaticFiles("/") would otherwise shadow them.
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

_WEB_DIR = Path(__file__).resolve().parent / "web"
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
