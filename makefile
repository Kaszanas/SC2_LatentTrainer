sources = src

DOCKER_DIRECTORY = ./docker

DOCKERFILE = $(DOCKER_DIRECTORY)/Dockerfile
DOCKER_COMPOSE_FILE = $(DOCKER_DIRECTORY)/docker-compose.yml


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

# Usage: make uv_update EXTRA=cpu (default: cuda)
EXTRA ?= cuda

.PHONy: uv_update
uv_update:
	uv lock --upgrade-package sc2_datasets
	uv sync --extra $(EXTRA)


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
