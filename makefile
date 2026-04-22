sources = src

DOCKER_DIRECTORY = ./docker

DOCKERFILE = $(DOCKER_DIRECTORY)/Dockerfile
DOCKER_COMPOSE_FILE = $(DOCKER_DIRECTORY)/docker-compose.yml

# Training configuration (override on the CLI):
DATASET       ?= cached_dataset_rich.pt
N_TRIALS      ?= 100

TS_STUDY      ?= SC2_TwoStage
TS_EXPERIMENT ?= SC2_TwoStage_ArchSearch

GV_EXPERIMENT ?= SC2_GuidedVAE_ArchSearch

.PHONY: test
test: format lint unittest

.PHONY: format
format:
	ruff format $(sources)
	ruff format tests

.PHONY: lint
lint:
	ruff check $(sources)
	ruff check tests
	mypy $(sources) tests

.PHONY: unittest
test:
	pytest tests/

.PHONY: coverage
coverage:
	pytest --cov=$(sources) --cov-branch --cov-report=term-missing tests

.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: clean
clean:
	rm -rf .mypy_cache .pytest_cache
	rm -rf *.egg-info
	rm -rf .tox dist site
	rm -rf coverage.xml .coverage

.PHONY: clean_runs
clean_runs:
	rm -f mlflow.db optuna_study.db
	rm -rf mlruns ray_results ray_tmp
	rm -rf output checkpoints

# Usage: make uv_update EXTRA=cpu (default: cuda)
EXTRA ?= cuda

.PHONy: uv_update
uv_update:
	uv lock --upgrade-package sc2_datasets
	uv sync --extra $(EXTRA)

.PHONY: process_features_rich
process_features_rich:
	uv run python src/latent_trainer/features/main.py

.PHONY: process_features_averaged
process_features_averaged:
	uv run python src/latent_trainer/features/main.py --transform averaged_economy

# Guided-VAE pipeline
.PHONY: guided_vae_sweep
guided_vae_sweep:
	python src/latent_trainer/train.py \
		--pipeline guided_vae \
		--dataset_filename $(DATASET) \
		--mode sweep \
		--n_trials $(N_TRIALS) \
		--experiment_name $(GV_EXPERIMENT)

.PHONY: guided_vae_train
guided_vae_train:
	python src/latent_trainer/train.py \
		--pipeline guided_vae \
		--dataset_filename $(DATASET) \
		--mode best \
		--experiment_name $(GV_EXPERIMENT)

# Dashboards
.PHONY: mlflow
mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

.PHONY: optuna
optuna:
	optuna-dashboard sqlite:///optuna_study.db

# Docker commands
.PHONY: docker-build
docker-build:
	docker-compose build

.PHONY: docker-build-cpu
docker-build-cpu:
	docker-compose -f docker-compose.cpu.yml build

.PHONY: docker-up
docker-up:
	docker-compose up -d

.PHONY: docker-up-cpu
docker-up-cpu:
	docker-compose -f docker-compose.cpu.yml up -d

.PHONY: docker-down
docker-down:
	docker-compose down

.PHONY: docker-logs
docker-logs:
	docker-compose logs -f

.PHONY: docker-shell
docker-shell:
	docker-compose exec trainer bash

.PHONY: docker-shell-cpu
docker-shell-cpu:
	docker-compose -f docker-compose.cpu.yml exec trainer-cpu bash

.PHONY: docker-train
docker-train:
	docker-compose exec trainer python src/latent_trainer/models/train_model.py

.PHONY: docker-tensorboard
docker-tensorboard:
	@echo "TensorBoard is available at http://localhost:6006"
	@echo "Optuna Dashboard is available at http://localhost:8080"

.PHONY: docker-clean
docker-clean:
	docker-compose down -v --remove-orphans
	docker-compose -f docker-compose.cpu.yml down -v --remove-orphans

.PHONY: docker-rebuild
docker-rebuild:
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
