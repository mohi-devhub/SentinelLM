"""Unit tests for sentinel.observability.alerting.check_thresholds and
sentinel.observability.webhook.send_alert.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentinel.observability.alerting import check_thresholds, window
from sentinel.observability.rolling_window import RollingWindow
from sentinel.observability.webhook import send_alert

# ── check_thresholds ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolated_window(monkeypatch):
    """Each test gets a fresh RollingWindow instead of the shared global one."""
    fresh = RollingWindow()
    monkeypatch.setattr("sentinel.observability.alerting.window", fresh)
    return fresh


def test_no_breaches_when_window_is_empty():
    assert check_thresholds({}) == []


def test_no_breach_below_minimum_sample_size():
    from sentinel.observability import alerting

    for _ in range(5):  # fewer than _MIN_SAMPLES
        alerting.window.record_llm_call(errored=True)
    assert check_thresholds({"error_rate": 0.1}) == []


def test_llm_error_rate_breach():
    from sentinel.observability import alerting

    for _ in range(10):
        alerting.window.record_llm_call(errored=True)
    breaches = check_thresholds({"error_rate": 0.1})
    assert any("error rate" in b for b in breaches)


def test_llm_error_rate_within_threshold_is_clean():
    from sentinel.observability import alerting

    for i in range(20):
        alerting.window.record_llm_call(errored=(i == 0))  # 5% error rate
    assert check_thresholds({"error_rate": 0.1}) == []


def test_block_rate_breach():
    from sentinel.observability import alerting

    for i in range(10):
        alerting.window.record_request(blocked=(i < 8), latency_ms=10)  # 80% blocked
    breaches = check_thresholds({"block_rate": 0.5})
    assert any("Block rate" in b for b in breaches)


def test_p95_latency_breach():
    from sentinel.observability import alerting

    for _ in range(10):
        alerting.window.record_request(blocked=False, latency_ms=20_000)
    breaches = check_thresholds({"p95_latency_ms": 10_000})
    assert any("latency" in b for b in breaches)


def test_evaluator_failure_rate_breach():
    from sentinel.observability import alerting

    for i in range(10):
        alerting.window.record_evaluator(errored=(i < 5))  # 50% failure
    breaches = check_thresholds({"evaluator_failure_rate": 0.1})
    assert any("Evaluator failure rate" in b for b in breaches)


def test_default_thresholds_used_when_config_omits_them():
    from sentinel.observability import alerting

    for _ in range(10):
        alerting.window.record_llm_call(errored=True)  # 100% error rate
    # No thresholds passed at all -> falls back to DEFAULT_THRESHOLDS
    breaches = check_thresholds({})
    assert any("error rate" in b for b in breaches)


def test_global_window_singleton_exists():
    assert isinstance(window, RollingWindow)


# ── send_alert ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_alert_posts_expected_payload():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False

    with patch("httpx.AsyncClient", return_value=mock_client):
        await send_alert(
            "https://hooks.example.com/webhook", "Test Alert", ["line one", "line two"]
        )

    mock_client.post.assert_awaited_once()
    call_args = mock_client.post.call_args
    assert call_args.args[0] == "https://hooks.example.com/webhook"
    payload = call_args.kwargs["json"]
    assert "Test Alert" in payload["text"]
    assert "line one" in payload["text"]
    assert "line two" in payload["text"]


@pytest.mark.asyncio
async def test_send_alert_swallows_failures():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=ConnectionError("webhook unreachable"))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False

    with patch("httpx.AsyncClient", return_value=mock_client):
        await send_alert("https://hooks.example.com/webhook", "Test", ["boom"])  # must not raise
