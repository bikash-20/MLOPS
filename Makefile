# Makefile for the neural-network project.
# Run `make help` for a list of targets.

.PHONY: help install test test-unit test-integration test-api \
        train-iris train-wine train-mnist api mlflow-ui \
        docker-build docker-run docker-up docker-down \
        deploy deploy-serve \
        lint format clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Setup ----------------------------------------------------------------

install: ## Install Python dependencies.
	pip install -r requirements.txt

# --- Testing --------------------------------------------------------------

test: ## Run the full test suite.
	pytest

test-unit: ## Run unit tests only.
	pytest tests/unit -v

test-integration: ## Run integration tests only.
	pytest tests/integration -v

test-api: ## Run API tests only.
	pytest tests/api -v

# --- Training -------------------------------------------------------------

train-iris: ## Train the Iris NumPy model.
	python -m src.training.train_iris

train-wine: ## Train the Wine PyTorch model.
	python -m src.training.train_wine

train-wine-fast: ## Train the Wine model for just 5 epochs (smoke test).
	python -m src.training.train_wine train.epochs=5

train-mnist: ## Train the MNIST CNN (downloads MNIST on first run).
	python -m src.training.train_mnist

train-mnist-fast: ## Train the MNIST CNN for just 1 epoch (smoke test).
	python -m src.training.train_mnist train.epochs=1

# --- API ------------------------------------------------------------------

api: ## Run the FastAPI service locally.
	uvicorn src.api.main:app --reload --port 8000

mlflow-ui: ## Launch MLflow tracking UI on http://localhost:5000.
	mlflow ui --port 5000

# --- Docker ---------------------------------------------------------------

docker-build: ## Build the API Docker image.
	docker build -t neural-network:latest .

docker-run: ## Run the API container on port 8000.
	docker run --rm -p 8000:8000 --name wine-api neural-network:latest

docker-up: ## Start api + mlflow via docker-compose.
	docker compose up --build

docker-down: ## Stop docker-compose services.
	docker compose down -v

# --- Deployment (Modal) --------------------------------------------------

deploy: ## Deploy the API to Modal (one-time; needs `modal setup` first).
	modal deploy deploy/modal_app.py

deploy-serve: ## Serve the API on Modal with live-reload (dev mode).
	modal serve deploy/modal_app.py

# --- Quality --------------------------------------------------------------

lint: ## Run ruff linter.
	ruff check src tests

format: ## Auto-format with ruff.
	ruff check --fix src tests && ruff format src tests

# --- Cleanup --------------------------------------------------------------

clean: ## Remove build artefacts and caches.
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache htmlcov .coverage
