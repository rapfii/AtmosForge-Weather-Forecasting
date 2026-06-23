# ═══════════════════════════════════════════════════════════════
# AtmosForge — Makefile
# Production-grade weather forecasting benchmarking framework
# ═══════════════════════════════════════════════════════════════

.PHONY: install install-dev fetch-data train tune benchmark evaluate \
        serve docker-build docker-push test lint typecheck quality \
        reproduce clean help

# ─── Default ─────────────────────────────────────────────────
.DEFAULT_GOAL := help

# ─── Variables ───────────────────────────────────────────────
PYTHON     := python
PIP        := pip
PYTEST     := pytest
RUFF       := ruff
MYPY       := mypy
SEED       ?= 42
MODEL      ?= lstm
DATASET    ?= jena
HORIZON    ?= 24
N_TRIALS   ?= 50
LAT        ?= -6.21
LON        ?= 106.85
START      ?= 2020-01-01
END        ?= 2024-12-31
VARIABLE   ?= 2m_temperature
YEAR       ?= 2022
ALL        ?= false

# ─── Setup ───────────────────────────────────────────────────
install:  ## Install package in production mode
	$(PIP) install -e .

install-dev:  ## Install package with dev dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install || true

# ─── Data Ingestion ──────────────────────────────────────────
fetch-data:  ## Fetch dataset (DATASET=jena|openmeteo|era5)
ifeq ($(DATASET),jena)
	$(PYTHON) -m src.data.ingestion.jena
else ifeq ($(DATASET),openmeteo)
	$(PYTHON) -m src.data.ingestion.openmeteo --lat=$(LAT) --lon=$(LON) --start=$(START) --end=$(END)
else ifeq ($(DATASET),era5)
	$(PYTHON) -m src.data.ingestion.era5 --variable=$(VARIABLE) --year=$(YEAR)
else
	@echo "Unknown DATASET=$(DATASET). Use: jena, openmeteo, era5"
	@exit 1
endif

# ─── Training ────────────────────────────────────────────────
train:  ## Train a model (MODEL=lstm DATASET=jena HORIZON=24)
	$(PYTHON) -m src.training.trainer \
		model=$(MODEL) \
		dataset=$(DATASET) \
		training.forecast_horizon=$(HORIZON) \
		seed=$(SEED)

# ─── HPO ─────────────────────────────────────────────────────
tune:  ## Run Optuna HPO (MODEL=tft DATASET=jena N_TRIALS=50)
	$(PYTHON) -m src.training.tuner \
		model=$(MODEL) \
		dataset=$(DATASET) \
		optuna.n_trials=$(N_TRIALS) \
		seed=$(SEED)

# ─── Evaluation ──────────────────────────────────────────────
evaluate:  ## Run evaluation: DM test + benchmark table
	$(PYTHON) -m scripts.run_benchmark --evaluate-only

benchmark:  ## Full benchmark: all models × datasets × horizons
ifeq ($(ALL),true)
	$(PYTHON) -m scripts.run_benchmark --all --seed=$(SEED)
else
	$(PYTHON) -m scripts.run_benchmark \
		--model=$(MODEL) \
		--dataset=$(DATASET) \
		--horizon=$(HORIZON) \
		--seed=$(SEED)
endif

# ─── Serving ─────────────────────────────────────────────────
serve:  ## Start FastAPI inference server (localhost:8000/docs)
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

# ─── Docker ──────────────────────────────────────────────────
docker-build:  ## Build Docker images (train + serve)
	docker build -f docker/Dockerfile.train -t atmosforge-train .
	docker build -f docker/Dockerfile.serve -t atmosforge-serve .

docker-push:  ## Push Docker images to registry
	docker push atmosforge-train
	docker push atmosforge-serve

docker-up:  ## Start all services via Docker Compose
	docker-compose up -d

docker-down:  ## Stop all Docker Compose services
	docker-compose down

# ─── Quality ─────────────────────────────────────────────────
test:  ## Run all tests with coverage
	$(PYTEST) tests/ --cov=src --cov-report=term-missing -v

test-unit:  ## Run unit tests only
	$(PYTEST) tests/unit/ -v

test-integration:  ## Run integration tests only
	$(PYTEST) tests/integration/ -v --timeout=120

lint:  ## Run ruff linter
	$(RUFF) check src/ tests/ scripts/

lint-fix:  ## Auto-fix linting issues
	$(RUFF) check src/ tests/ scripts/ --fix

format:  ## Format code with ruff
	$(RUFF) format src/ tests/ scripts/

typecheck:  ## Run mypy type checker (strict mode)
	$(MYPY) src/

quality:  ## Run all quality checks (lint + typecheck + test)
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test-unit

# ─── Reproducibility ────────────────────────────────────────
reproduce:  ## Full pipeline with fixed seed (SEED=42)
	@echo "═══ AtmosForge — Full Reproducible Run (seed=$(SEED)) ═══"
	$(MAKE) fetch-data DATASET=jena
	$(MAKE) benchmark ALL=true SEED=$(SEED)
	$(MAKE) evaluate
	@echo "═══ Results saved to results/benchmark.csv ═══"

# ─── Utilities ───────────────────────────────────────────────
mlflow-ui:  ## Launch MLflow tracking UI (localhost:5000)
	mlflow ui --host 0.0.0.0 --port 5000

clean:  ## Remove generated artifacts
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned build artifacts."

clean-all: clean  ## Remove ALL generated files (including data and results)
	rm -rf data/processed mlruns checkpoints logs
	@echo "Cleaned all generated data."

# ─── Help ────────────────────────────────────────────────────
help:  ## Show this help message
	@echo "═══════════════════════════════════════════════════════"
	@echo "  AtmosForge — Available Commands"
	@echo "═══════════════════════════════════════════════════════"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
