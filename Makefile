.PHONY: help dev prod down logs test lint fmt typecheck install install-dev clean

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Development ───────────────────────────────────────────────────────────────

install:       ## Install production Python dependencies
	pip install -r requirements.txt

install-dev:   ## Install all Python dependencies (including dev/test)
	pip install -r requirements-dev.txt
	pre-commit install

dev:           ## Start all services in development mode (hot-reload)
	docker compose up --build

down:          ## Stop and remove containers
	docker compose down

logs:          ## Tail API logs
	docker compose logs -f api

# ── Production ────────────────────────────────────────────────────────────────

prod:          ## Start production stack (requires .env with real secrets)
	docker compose -f docker-compose.prod.yml up --build -d

prod-down:     ## Stop production stack
	docker compose -f docker-compose.prod.yml down

# ── Code quality ──────────────────────────────────────────────────────────────

lint:          ## Run ruff linter
	ruff check sentinel tests

fmt:           ## Auto-format code with ruff
	ruff format sentinel tests

typecheck:     ## Run mypy type checker
	mypy sentinel

# ── Testing ───────────────────────────────────────────────────────────────────

test:          ## Run unit tests with coverage
	pytest tests/unit --cov=sentinel --cov-report=term-missing -v

test-all:      ## Run all tests (unit + integration)
	pytest tests --cov=sentinel --cov-report=term-missing -v

clean:         ## Remove Python caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
