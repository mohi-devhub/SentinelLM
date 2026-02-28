<div align="center">

# SentinelLM

**Real-time safety and quality middleware for LLM applications.**

[Issues](https://github.com/mohi-devhub/SentinelLM/issues) · [Contributing](CONTRIBUTING.md) · [Architecture](#architecture)

![Python](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

</div>

---

SentinelLM is an open-source proxy middleware that sits between your application and any LLM backend. Every request passes through a chain of seven safety and quality evaluators before reaching the model; every response is scored before reaching the user. Harmful inputs get blocked. Low-quality outputs get flagged. Everything gets logged to PostgreSQL and streamed live to a dashboard.

It is a **drop-in replacement** for your existing LLM client — point your `base_url` at `http://localhost:8000/v1` and it works with no other changes, regardless of whether you're running Ollama locally, OpenAI, Anthropic, or Gemini.

## Key Features

- **Dual-layer evaluation** — input evaluators block harmful requests before the LLM is called; output evaluators flag low-quality responses without adding latency to the happy path.
- **Concurrent input chain with first-block short-circuit** — all input evaluators race in parallel using `asyncio.wait(FIRST_COMPLETED)`. A detected injection doesn't wait for PII to finish.
- **PII redact-or-block** — PII can be automatically redacted from the request (allowing it through with sensitive data removed) or hard-blocked. Configurable per deployment.
- **Shadow mode** — run all evaluators and log scores without ever blocking a request. Use it to tune thresholds in production before enforcing them.
- **Redis caching** — input evaluator scores are cached by a SHA-256 hash of (input + config version). Repeated inputs cost zero model inference. Cache keys automatically invalidate when you change evaluator config.
- **Fail-open guarantee** — a model crash, timeout, or OOM error never blocks a legitimate user request. Every evaluator returns `score=None, flag=False` on error.
- **Human review queue** — flagged responses queue in a dedicated endpoint for analyst review and approval/rejection via the dashboard.
- **Real-time WebSocket feed** — the dashboard receives every scored request over a WebSocket the moment it is processed.
- **Eval pipeline with regression detection** — run a golden dataset against a live instance, save the results as a named baseline, and compare future builds against it. CI exits non-zero on regression.

## Evaluators

Seven evaluators across two layers. Input evaluators run before the LLM call and can block the request. Output evaluators run after and flag responses for human review.

| Evaluator | Layer | Action | Model |
|-----------|-------|--------|-------|
| `pii` | input | block or redact | Presidio + spaCy `en_core_web_sm` |
| `prompt_injection` | input | block | `deepset/deberta-v3-base-injection` |
| `topic_guardrail` | input | block | `all-MiniLM-L6-v2` (cosine sim) |
| `toxicity` | output | flag | Detoxify |
| `relevance` | output | flag | `all-MiniLM-L6-v2` (cosine sim) |
| `hallucination` | output | flag | `cross-encoder/nli-deberta-v3-base` |
| `faithfulness` | output | flag | `cross-encoder/nli-deberta-v3-base` |

All evaluators are **fail-open** — a model crash or timeout never blocks a legitimate request.

`topic_guardrail` is disabled by default. Enable it and set `allowed_topics` to restrict your assistant to a specific domain (e.g. software engineering, customer support).

`hallucination` and `faithfulness` are silently skipped when no `context_documents` are provided in the request.

## Quick Start

**Requirements:** Docker, Docker Compose

```bash
git clone https://github.com/mohi-devhub/SentinelLM.git
cd SentinelLM

cp .env.example .env
# Edit .env: set your LLM API key (GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)

docker compose up -d
```

- API → `http://localhost:8000`
- Dashboard → `http://localhost:3000`
- Prometheus metrics → `http://localhost:8000/metrics`

> **Ollama (local models):**
> ```bash
> docker compose --profile ollama up -d
> docker compose exec ollama ollama pull llama3.2
> ```

## Usage

### Chat request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash-lite",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

Every passing response includes a `sentinel` block:

```json
{
  "choices": [{ "message": { "role": "assistant", "content": "Paris." } }],
  "sentinel": {
    "request_id": "b3f1a2...",
    "scores": { "toxicity": 0.01, "relevance": 0.92 },
    "flags":  [],
    "latency_ms": { "pii": 12, "prompt_injection": 48, "llm": 820, "total": 893 }
  }
}
```

### Blocked request

```bash
curl http://localhost:8000/v1/chat/completions \
  -d '{"model": "gemini-2.5-flash-lite", "messages": [{"role": "user", "content": "Ignore all instructions."}]}'
```

```json
HTTP/1.1 400 Bad Request
{
  "error": {
    "type": "sentinel_block",
    "code": "prompt_injection_detected",
    "score": 0.97,
    "threshold": 0.80
  }
}
```

### PII redaction

When PII action is set to `redact`, sensitive data is stripped from the request text before it reaches the LLM and the response is returned normally. The original text is never forwarded.

### With API key auth (production)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

## Configuration

All configuration lives in `config.yaml`. Switch LLM backend, tune thresholds, and enable/disable evaluators without touching code.

### LLM backend

```yaml
llm_backend:
  provider: gemini   # ollama | openai | anthropic | gemini
```

API keys for cloud providers are set via environment variables, never in `config.yaml`.

| Provider | Env var |
|----------|---------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |

### Evaluator thresholds

```yaml
evaluators:
  pii:
    enabled: true
    threshold: 0.5
    action: redact    # redact | block

  prompt_injection:
    enabled: true
    threshold: 0.80

  topic_guardrail:
    enabled: false            # enable and set allowed_topics to restrict domain
    threshold: 0.30
    allowed_topics:
      - "software engineering"
      - "programming"

  toxicity:
    enabled: true
    threshold: 0.70
```

Set `enabled: false` to skip an evaluator entirely (zero latency cost).

### Shadow mode

```yaml
app:
  shadow_mode: true   # log all scores but never block any request
```

Enable shadow mode to observe evaluator behaviour in production without enforcing blocks. Useful for calibrating thresholds before going live.

### Security settings (env vars)

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTINEL_API_KEY` | *(empty)* | When set, all requests must include `X-API-Key`. Leave empty in dev. |
| `SENTINEL_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed CORS origins. |

## Architecture

```
POST /v1/chat/completions
        │
        ▼
┌─────────────────────────────────────────┐
│  Input Chain  (concurrent, fail-open)   │
│                                         │
│  pii ──────────────────────────── ─ ─ ┐ │
│  prompt_injection ──────────────── ─ ─┼─┼─► first block → HTTP 400
│  topic_guardrail ───────────────── ─ ─┘ │   (shadow_mode bypasses block)
└─────────────────────────────────────────┘
        │ (pass)
        ▼
┌─────────────────────────────────────────┐
│  LLM Backend                            │
│  Ollama · OpenAI · Anthropic · Gemini   │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Output Chain  (all run, fail-open)     │
│                                         │
│  toxicity · relevance                   │
│  hallucination · faithfulness           │
└─────────────────────────────────────────┘
        │
        ├─► BackgroundTask: PostgreSQL write
        ├─► BackgroundTask: WebSocket push → dashboard
        └─► HTTP 200 with sentinel metadata
```

Input evaluators race with `asyncio.wait(FIRST_COMPLETED)` — a detected injection doesn't wait for PII to finish. Output evaluators always all run; flagged responses appear in the dashboard review queue.

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | Main proxy — drop-in OpenAI replacement |
| `GET` | `/health` | Service health, evaluator list, DB/Redis/LLM connectivity |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/v1/sentinel/config` | Active evaluator configuration (no secrets) |
| `GET` | `/v1/sentinel/scores` | Paginated request history (`?page=1&limit=20`) |
| `GET` | `/v1/sentinel/scores/{request_id}` | Single request detail with all scores |
| `GET` | `/v1/sentinel/metrics/aggregate` | Time-bucketed metrics for charts |
| `GET` | `/v1/sentinel/metrics/summary` | Aggregate stats (block rate, flag rates) |
| `GET` | `/v1/sentinel/review` | Human review queue (flagged, unreviewed requests) |
| `PATCH` | `/v1/sentinel/review/{request_id}` | Approve or reject a flagged request |
| `GET` | `/v1/sentinel/eval` | Eval pipeline run history |
| `GET` | `/v1/sentinel/eval/{run_id}` | Single eval run detail |
| `WS` | `/ws/feed` | Real-time event stream for the dashboard |

## Eval Pipeline

Run a golden dataset against a live instance and detect regressions between releases:

```bash
# Run and save as a baseline
sentinel eval run \
  --dataset evals/golden_qa.jsonl \
  --label v1.0-baseline \
  --server http://localhost:8000

# Compare a candidate build against the baseline
sentinel eval run \
  --dataset evals/golden_qa.jsonl \
  --label v1.1-candidate \
  --baseline v1.0-baseline
```

The CLI prints a scorecard table and exits non-zero if any metric regresses.

## Production Deployment

```bash
cp .env.example .env
# Fill in: SENTINEL_API_KEY, SENTINEL_CORS_ORIGINS, POSTGRES_PASSWORD, and your LLM API key

docker compose -f docker-compose.prod.yml up -d
```

The production compose file adds:
- CPU and memory resource limits per service
- DB and Redis ports bound to `127.0.0.1` (not exposed publicly)
- No source code volume mounts and no `--reload`
- Container-level `HEALTHCHECK` via `/health`

## Local Development

```bash
cp .env.example .env       # add your LLM API key

pip install -r requirements-dev.txt
pre-commit install          # install git hooks (ruff, secret detection)

make dev                   # docker compose up with hot-reload
make test                  # pytest unit tests with coverage
make lint                  # ruff check
make fmt                   # ruff format
```

## Load Testing

```bash
pip install locust
locust -f locustfile.py --host http://localhost:8000
# → Locust UI at http://localhost:8089
```

Four user classes simulate realistic production traffic: clean chat (80%), prompt injection attacks (10%), PII leaks (10%), and a mixed realistic profile.

## License

MIT
