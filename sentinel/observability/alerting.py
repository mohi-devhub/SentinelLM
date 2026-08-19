"""Periodic threshold-based alerting — zero extra infrastructure required.

Runs as a background asyncio task (see sentinel/main.py lifespan), checking
the in-process RollingWindow against configured thresholds every
check_interval_seconds and POSTing to a webhook URL when a threshold is
breached. A no-op (the loop still runs, it just skips the POST) when no
webhook URL is configured — this must never require standing up Prometheus/
Alertmanager or any other infra to work on a bare deployment.
"""

from __future__ import annotations

import asyncio
import logging

from sentinel.observability.rolling_window import RollingWindow
from sentinel.observability.webhook import send_alert

logger = logging.getLogger(__name__)

# Single process-wide window — see RollingWindow's docstring for why this is
# intentionally global rather than per-tenant.
window = RollingWindow()

DEFAULT_THRESHOLDS = {
    "error_rate": 0.10,
    "block_rate": 0.50,
    "evaluator_failure_rate": 0.10,
    "p95_latency_ms": 10_000,
}

# Don't alert on noise from a handful of early requests after startup.
_MIN_SAMPLES = 10


def check_thresholds(thresholds: dict) -> list[str]:
    """Return human-readable breach descriptions, or [] if nothing is breached."""
    snap = window.snapshot()
    breaches: list[str] = []

    if snap["llm_call_count"] >= _MIN_SAMPLES:
        limit = thresholds.get("error_rate", DEFAULT_THRESHOLDS["error_rate"])
        if snap["llm_error_rate"] > limit:
            breaches.append(
                f"LLM call error rate {snap['llm_error_rate']:.0%} exceeds {limit:.0%} "
                f"({snap['llm_call_count']} calls in the last window)"
            )

    if snap["request_count"] >= _MIN_SAMPLES:
        block_limit = thresholds.get("block_rate", DEFAULT_THRESHOLDS["block_rate"])
        if snap["block_rate"] > block_limit:
            breaches.append(
                f"Block rate {snap['block_rate']:.0%} exceeds {block_limit:.0%} "
                f"({snap['request_count']} requests in the last window)"
            )

        latency_limit = thresholds.get("p95_latency_ms", DEFAULT_THRESHOLDS["p95_latency_ms"])
        if snap["p95_latency_ms"] > latency_limit:
            breaches.append(f"p95 latency {snap['p95_latency_ms']}ms exceeds {latency_limit}ms")

    if snap["evaluator_call_count"] >= _MIN_SAMPLES:
        limit = thresholds.get(
            "evaluator_failure_rate", DEFAULT_THRESHOLDS["evaluator_failure_rate"]
        )
        if snap["evaluator_failure_rate"] > limit:
            breaches.append(
                f"Evaluator failure rate {snap['evaluator_failure_rate']:.0%} exceeds {limit:.0%} "
                f"({snap['evaluator_call_count']} evaluator calls in the last window)"
            )

    return breaches


async def run_alert_loop(webhook_url: str, thresholds: dict, check_interval_seconds: float) -> None:
    """Background task: check thresholds on an interval, POST breaches to the webhook."""
    while True:
        await asyncio.sleep(check_interval_seconds)
        try:
            breaches = check_thresholds(thresholds)
            if breaches and webhook_url:
                await send_alert(webhook_url, "SentinelLM alert", breaches)
            elif breaches:
                logger.warning("alert threshold(s) breached (no webhook configured): %s", breaches)
        except Exception:
            logger.exception("alert loop check failed")
