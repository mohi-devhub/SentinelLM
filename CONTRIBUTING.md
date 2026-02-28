# Contributing to SentinelLM

Thanks for your interest in contributing. This document covers how to set up your development environment, the project conventions you should follow, and the process for submitting changes.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Conventions](#code-conventions)
- [Running Tests](#running-tests)
- [Adding an Evaluator](#adding-an-evaluator)
- [Adding an LLM Backend](#adding-an-llm-backend)
- [Submitting Changes](#submitting-changes)

---

## Development Setup

**Prerequisites:** Python 3.11+, Docker, Docker Compose

```bash
git clone https://github.com/mohi-devhub/SentinelLM.git
cd SentinelLM

# Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# Install git hooks (ruff lint + secret detection run on every commit)
pre-commit install

# Copy env and add your LLM API key
cp .env.example .env

# Start backing services only (DB + Redis)
docker compose up postgres redis -d

# Run the API locally with hot-reload
uvicorn sentinel.main:app --reload
```

Dashboard:

```bash
cd dashboard
npm install
npm run dev   # → http://localhost:3000
```

---

## Project Structure

```
sentinel/
├── main.py                   # App factory + lifespan (startup/shutdown)
├── settings.py               # Pydantic settings — single config source
├── api/
│   ├── proxy.py              # POST /v1/chat/completions
│   ├── health.py             # GET /health, GET /v1/sentinel/config
│   ├── middleware.py         # API key auth, request ID, Prometheus
│   ├── metrics.py            # GET /v1/sentinel/metrics/*
│   ├── scores.py             # GET /v1/sentinel/scores
│   ├── review.py             # GET /v1/sentinel/review
│   ├── eval.py               # GET /v1/sentinel/eval/*
│   └── websocket.py          # WS /ws/feed
├── evaluators/
│   ├── base.py               # BaseEvaluator ABC, EvalPayload, EvalResult
│   ├── registry.py           # EVALUATOR_REGISTRY, load_evaluators()
│   ├── input/                # pii, prompt_injection, topic_guardrail
│   └── output/               # toxicity, relevance, hallucination, faithfulness
├── chain/
│   ├── runner.py             # run_input_chain(), run_output_chain()
│   └── aggregator.py         # SentinelResult assembly
├── proxy/
│   ├── base.py               # LLMClient ABC
│   ├── factory.py            # get_llm_client()
│   ├── ollama.py             # Ollama client
│   ├── openai.py             # OpenAI client
│   ├── anthropic.py          # Anthropic client
│   └── gemini.py             # Gemini client
├── storage/
│   ├── database.py           # asyncpg pool creation
│   ├── models.py             # RequestRecord dataclass
│   ├── schema.sql            # DDL (auto-applied on first Postgres start)
│   └── queries/              # insert_request, get_aggregate_metrics, etc.
├── cache/
│   └── client.py             # Redis helpers (get/set input evaluator scores)
├── eval_pipeline/
│   ├── cli.py                # Typer CLI: `sentinel eval run ...`
│   ├── runner.py             # Concurrent request runner
│   └── reporter.py           # Scorecard table + regression detection
└── ws/
    └── broadcaster.py        # WebSocket ConnectionManager
```

---

## Code Conventions

### General

- **Never import `settings` at module level.** Always call `get_settings()` inside a function or lifespan. Importing at module level breaks the settings cache in tests.
- **Never block the event loop.** All CPU-bound model inference must use `run_in_executor` from `sentinel.evaluators.base`.
- **DB writes are always background tasks.** Use `BackgroundTasks.add_task()` so the response is not blocked on a database write.
- **Evaluators fail-open.** Every evaluator must catch exceptions and return `EvalResult(score=None, flag=False, error=str(e))`.

### Python

- Formatter and linter: **ruff** (`line-length = 100`). Run `make fmt` before committing.
- Type checker: **mypy** (non-strict mode). New code should be fully annotated.
- `from __future__ import annotations` at the top of every Python file.
- Commit format: `type(scope): description` — e.g. `feat(evaluators): add toxicity scorer`.

### TypeScript (dashboard)

- Strict mode, no `any` types.
- All data fetching via `@tanstack/react-query`.
- `"use client"` directive on every component that uses hooks or browser APIs.

---

## Running Tests

```bash
# Unit tests only (no Docker services needed)
make test

# All tests (unit + integration — requires Postgres + Redis)
docker compose up postgres redis -d
make test-all

# With coverage report
pytest tests/unit --cov=sentinel --cov-report=html
open htmlcov/index.html
```

### Writing tests

- Unit tests go in `tests/unit/`. They should not require Docker services.
- Integration tests go in `tests/integration/`. They use the `client` and `db_pool` fixtures from `tests/conftest.py`.
- Mock the LLM backend with `unittest.mock.patch` on the client's `chat` method.
- Use the `make_eval_result()` helper in `conftest.py` to build `EvalResult` fixtures.

---

## Adding an Evaluator

1. **Create the file** in `sentinel/evaluators/input/` or `sentinel/evaluators/output/`.

2. **Inherit from `BaseEvaluator`** and implement the two required methods:

    ```python
    from sentinel.evaluators.base import BaseEvaluator, EvalPayload, run_in_executor

    class MyEvaluator(BaseEvaluator):
        name = "my_evaluator"
        runs_on = "input"          # or "output"
        flag_direction = "above"   # flag when score exceeds threshold ("above" | "below")

        def _load_model(self) -> None:
            # Runs once at startup (synchronous). Load model weights here.
            self._model = load_my_model()

        async def _run_inference(self, payload: EvalPayload) -> tuple[float, dict | None]:
            # Must be async. Use run_in_executor for CPU-bound work.
            score = await run_in_executor(self._model.predict, payload.input_text)
            return float(score), {"detail": "..."}
    ```

3. **Register the evaluator** in `sentinel/evaluators/registry.py`:

    ```python
    from sentinel.evaluators.input.my_evaluator import MyEvaluator

    EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {
        ...
        "my_evaluator": MyEvaluator,
    }
    ```

4. **Add config** to `config.yaml` under `evaluators:`:

    ```yaml
    evaluators:
      my_evaluator:
        enabled: true
        threshold: 0.75
    ```

5. **Add unit tests** in `tests/unit/evaluators/test_my_evaluator.py`. Test the pass case, the fail case, and the fail-open case (exception in `_run_inference`).

---

## Adding an LLM Backend

1. **Create a client class** in `sentinel/proxy/`:

    ```python
    from sentinel.proxy.base import LLMClient

    class MyProviderClient(LLMClient):
        def __init__(self, model: str, api_key: str, timeout: float = 60.0) -> None:
            ...

        async def chat(self, request: dict) -> dict:
            # Call provider API.
            # Return an OpenAI-format dict:
            # { "choices": [{ "message": { "role": "assistant", "content": "..." } }] }
            ...
    ```

2. **Register it** in `sentinel/proxy/factory.py` inside `get_llm_client()`.

3. **Add config** under `llm_backend:` in `config.yaml` with at least `model` and `timeout_seconds`.

4. **Add the provider's SDK** to `requirements.txt`.

---

## Submitting Changes

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. Make your changes. Pre-commit hooks will run ruff and secret detection automatically on `git commit`.

3. Run the full test suite before pushing:
   ```bash
   make lint
   make test
   ```

4. Open a pull request against `main`. The CI pipeline (ruff, mypy, pytest, Docker build) runs automatically.

5. Keep PRs focused — one feature or fix per PR. If your change is large, open an issue first to discuss the approach.

### Commit message format

```
feat(evaluators): add jailbreak detection evaluator
fix(proxy): enforce timeout on Gemini API calls
docs: update evaluator configuration examples
test(chain): add short-circuit behaviour assertions
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`.
