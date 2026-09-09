# latent-trainer-serve: single-replay hosting pipeline.
#
# Build context must be the repo root (not serving/), since this needs both
# the root `latent_trainer` project (a path dependency of this sub-project)
# and `serving/` itself, keeping the same "../" relative layout that
# serving/pyproject.toml's `[tool.uv.sources]` expects:
#
#   docker build -f serving/Dockerfile -t latent-trainer-serve .
#
# Extraction (SC2InfoExtractorGo) is baked directly into this image -- its
# statically-linked binary is copied from the official image at build time,
# so `extract.py` can invoke it directly (extractor_binary=...) with no
# Docker CLI, no Docker-out-of-Docker, and no host socket mount needed.
FROM kaszanas/sc2infoextractorgo:dev AS extractor

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

COPY --from=extractor /app/SC2InfoExtractorGo /usr/local/bin/SC2InfoExtractorGo

# `latent-trainer` depends on `sc2-datasets` sourced from a git branch (see
# root pyproject.toml's [tool.uv.sources]), which `uv sync` resolves by
# shelling out to `git` -- not present in this slim base image by default.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Root project: latent_trainer_serve depends on it via a relative path
# source ("../" from serving/), so the copied layout must match. README.md
# is required too -- the root pyproject.toml's build backend (uv_build)
# reads it directly (`readme = "README.md"`) and fails without it.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# The servable sub-project.
COPY serving/pyproject.toml serving/uv.lock serving/README.md ./serving/
COPY serving/src ./serving/src

WORKDIR /app/serving
# --group api installs fastapi/uvicorn/python-multipart alongside the base
# CLI deps, so the same image serves both the `predict` CLI (frontend
# service) and the FastAPI app (api service) -- see docker-compose.yml.
RUN uv sync --frozen --no-dev --group api

ENV PATH="/app/serving/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "latent_trainer_serve"]
CMD ["--help"]
