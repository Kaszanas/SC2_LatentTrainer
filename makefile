sources = src

.PHONY: test format lint unittest coverage pre-commit clean
test: format lint unittest

format:
	ruff format $(sources)
	ruff format tests

lint:
	ruff check $(sources)
	ruff check tests
	mypy $(sources) tests

unittest:
	pytest

coverage:
	pytest --cov=$(sources) --cov-branch --cov-report=term-missing tests

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf .mypy_cache .pytest_cache
	rm -rf *.egg-info
	rm -rf .tox dist site
	rm -rf coverage.xml .coverage

# Docker commands
.PHONY: docker-build docker-build-cpu docker-up docker-up-cpu docker-down docker-logs docker-shell docker-train docker-tensorboard docker-clean

docker-build:
	docker-compose build

docker-build-cpu:
	docker-compose -f docker-compose.cpu.yml build

docker-up:
	docker-compose up -d

docker-up-cpu:
	docker-compose -f docker-compose.cpu.yml up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-shell:
	docker-compose exec trainer bash

docker-shell-cpu:
	docker-compose -f docker-compose.cpu.yml exec trainer-cpu bash

docker-train:
	docker-compose exec trainer python src/latent_trainer/models/train_model.py

docker-tensorboard:
	@echo "TensorBoard is available at http://localhost:6006"
	@echo "Optuna Dashboard is available at http://localhost:8080"

docker-clean:
	docker-compose down -v --remove-orphans
	docker-compose -f docker-compose.cpu.yml down -v --remove-orphans

docker-rebuild:
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
