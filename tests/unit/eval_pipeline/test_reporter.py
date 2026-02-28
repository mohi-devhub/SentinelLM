"""Unit tests for eval_pipeline.reporter.

All functions are pure Python (no DB, no models, no I/O except the export_json
file write which uses a tmp_path).  Coverage target: 95%+ of reporter.py.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from sentinel.eval_pipeline.reporter import (
    REGRESSION_THRESHOLD,
    _percentile,
    compute_regression,
    compute_summary,
    export_json,
    print_report,
    print_runs_table,
)
from sentinel.eval_pipeline.runner import EVALUATOR_NAMES, DatasetRecord, RunResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_run_result(
    scores: dict[str, float | None] | None = None,
    flags: list[str] | None = None,
    blocked: bool = False,
) -> RunResult:
    """Build a minimal RunResult for testing."""
    record = DatasetRecord(record_index=0, input="test input")
    return RunResult(
        record=record,
        scores=scores or {ev: 0.1 for ev in EVALUATOR_NAMES},
        flags=flags or [],
        blocked=blocked,
    )


def _quiet_console() -> Console:
    """A Rich console that writes to a string buffer (no terminal output)."""
    return Console(file=StringIO(), highlight=False)


# ── _percentile ───────────────────────────────────────────────────────────────


def test_percentile_empty_returns_zero():
    assert _percentile([], 50) == 0.0


def test_percentile_single_element():
    assert _percentile([0.5], 50) == pytest.approx(0.5)


def test_percentile_median():
    data = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert _percentile(data, 50) == pytest.approx(0.3)


def test_percentile_p95():
    data = list(range(1, 101))  # 1..100
    p95 = _percentile([float(x) for x in data], 95)
    assert 94.0 <= p95 <= 96.0


def test_percentile_p0_returns_min():
    data = [0.3, 0.1, 0.2]
    assert _percentile(data, 0) == pytest.approx(0.1)


def test_percentile_p100_returns_max():
    data = [0.3, 0.1, 0.5, 0.2]
    assert _percentile(data, 100) == pytest.approx(0.5)


# ── compute_summary ───────────────────────────────────────────────────────────


def test_compute_summary_empty_results():
    summary = compute_summary([])
    for ev in EVALUATOR_NAMES:
        assert ev in summary
        assert summary[ev]["n"] == 0
        assert summary[ev]["mean"] is None
        assert summary[ev]["p50"] is None
        assert summary[ev]["p95"] is None
        assert summary[ev]["flag_rate"] == 0.0
        assert summary[ev]["flag_count"] == 0


def test_compute_summary_single_result():
    result = _make_run_result(
        scores={ev: 0.5 for ev in EVALUATOR_NAMES},
        flags=["toxicity"],
    )
    summary = compute_summary([result])
    assert summary["toxicity"]["mean"] == pytest.approx(0.5)
    assert summary["toxicity"]["flag_count"] == 1
    assert summary["toxicity"]["flag_rate"] == pytest.approx(1.0)
    assert summary["pii"]["flag_count"] == 0


def test_compute_summary_none_score_excluded_from_stats():
    """A None score is excluded from mean/percentile but n_scored is accurate."""
    result_with_score = _make_run_result(scores={ev: 0.4 for ev in EVALUATOR_NAMES})
    result_without_score = _make_run_result(
        scores={ev: (None if ev == "pii" else 0.4) for ev in EVALUATOR_NAMES}
    )
    summary = compute_summary([result_with_score, result_without_score])
    assert summary["pii"]["n"] == 2
    assert summary["pii"]["n_scored"] == 1
    assert summary["pii"]["mean"] == pytest.approx(0.4)


def test_compute_summary_multiple_results():
    results = [
        _make_run_result(scores={ev: 0.2 for ev in EVALUATOR_NAMES}, flags=["pii"]),
        _make_run_result(scores={ev: 0.4 for ev in EVALUATOR_NAMES}, flags=["pii"]),
        _make_run_result(scores={ev: 0.6 for ev in EVALUATOR_NAMES}),
    ]
    summary = compute_summary(results)
    assert summary["pii"]["n"] == 3
    assert summary["pii"]["flag_count"] == 2
    assert summary["pii"]["flag_rate"] == pytest.approx(2 / 3)
    assert summary["pii"]["mean"] == pytest.approx(0.4)


def test_compute_summary_all_evaluators_present():
    summary = compute_summary([_make_run_result()])
    for ev in EVALUATOR_NAMES:
        assert ev in summary
        assert "mean" in summary[ev]
        assert "p50" in summary[ev]
        assert "p95" in summary[ev]
        assert "flag_rate" in summary[ev]
        assert "flag_count" in summary[ev]
        assert "n" in summary[ev]
        assert "n_scored" in summary[ev]


# ── compute_regression ────────────────────────────────────────────────────────


def test_compute_regression_no_change():
    summary = {ev: {"flag_rate": 0.1} for ev in EVALUATOR_NAMES}
    regression = compute_regression(summary, summary)
    for ev in EVALUATOR_NAMES:
        assert regression[ev]["delta"] == pytest.approx(0.0)
        assert regression[ev]["regressed"] is False


def test_compute_regression_detects_increase():
    current = {ev: {"flag_rate": 0.20} for ev in EVALUATOR_NAMES}
    baseline = {ev: {"flag_rate": 0.10} for ev in EVALUATOR_NAMES}
    regression = compute_regression(current, baseline)
    for ev in EVALUATOR_NAMES:
        assert regression[ev]["delta"] == pytest.approx(0.10)
        assert regression[ev]["regressed"] is True  # delta > REGRESSION_THRESHOLD (0.05)


def test_compute_regression_improvement():
    current = {ev: {"flag_rate": 0.01} for ev in EVALUATOR_NAMES}
    baseline = {ev: {"flag_rate": 0.10} for ev in EVALUATOR_NAMES}
    regression = compute_regression(current, baseline)
    for ev in EVALUATOR_NAMES:
        assert regression[ev]["delta"] < 0
        assert regression[ev]["regressed"] is False


def test_compute_regression_threshold_boundary():
    """Flag rate just below threshold is NOT a regression."""
    current = {"pii": {"flag_rate": 0.10 + REGRESSION_THRESHOLD - 0.001}}
    baseline = {"pii": {"flag_rate": 0.10}}
    regression = compute_regression(current, baseline)
    assert regression["pii"]["regressed"] is False


def test_compute_regression_just_above_threshold():
    """Flag rate strictly above threshold IS a regression."""
    current = {"pii": {"flag_rate": 0.10 + REGRESSION_THRESHOLD + 0.001}}
    baseline = {"pii": {"flag_rate": 0.10}}
    regression = compute_regression(current, baseline)
    assert regression["pii"]["regressed"] is True


def test_compute_regression_missing_evaluator_defaults_to_zero():
    """Missing evaluator in either dict defaults to 0.0 flag_rate."""
    regression = compute_regression({}, {})
    for ev in EVALUATOR_NAMES:
        assert regression[ev]["baseline_flag_rate"] == 0.0
        assert regression[ev]["current_flag_rate"] == 0.0


# ── print_report ──────────────────────────────────────────────────────────────


def test_print_report_renders_without_error():
    """print_report should not raise for valid inputs."""
    results = [
        _make_run_result(scores={ev: 0.3 for ev in EVALUATOR_NAMES}, flags=["toxicity"]),
        _make_run_result(scores={ev: 0.1 for ev in EVALUATOR_NAMES}),
    ]
    summary = compute_summary(results)
    console = _quiet_console()
    print_report(
        console=console,
        label="test_run",
        dataset_path="evals/test.jsonl",
        n_records=2,
        duration_s=1.5,
        summary=summary,
    )


def test_print_report_with_regression_table():
    """print_report renders regression table when regression dict is provided."""
    results = [_make_run_result(scores={ev: 0.5 for ev in EVALUATOR_NAMES})]
    summary = compute_summary(results)
    regression = compute_regression(summary, {ev: {"flag_rate": 0.0} for ev in EVALUATOR_NAMES})
    console = _quiet_console()
    print_report(
        console=console,
        label="run",
        dataset_path="data.jsonl",
        n_records=1,
        duration_s=0.5,
        summary=summary,
        regression=regression,
        baseline_label="baseline_run",
    )


def test_print_report_regression_detected_message():
    """High delta regression produces a regression warning message."""
    current = {ev: {"flag_rate": 0.20} for ev in EVALUATOR_NAMES}
    baseline = {ev: {"flag_rate": 0.05} for ev in EVALUATOR_NAMES}
    regression = compute_regression(current, baseline)
    buf = StringIO()
    console = Console(file=buf, highlight=False)
    print_report(
        console=console,
        label="run",
        dataset_path="data.jsonl",
        n_records=1,
        duration_s=0.1,
        summary=current,
        regression=regression,
        baseline_label="base",
    )
    output = buf.getvalue()
    assert "Regression" in output or "REGRESSION" in output


def test_print_report_no_regression_message():
    """No regression → stable/no-regression message."""
    stable = {ev: {"flag_rate": 0.05} for ev in EVALUATOR_NAMES}
    regression = compute_regression(stable, stable)
    buf = StringIO()
    console = Console(file=buf, highlight=False)
    print_report(
        console=console,
        label="run",
        dataset_path="data.jsonl",
        n_records=1,
        duration_s=0.1,
        summary=stable,
        regression=regression,
        baseline_label="base",
    )


# ── export_json ───────────────────────────────────────────────────────────────


def test_export_json_writes_valid_json(tmp_path):
    results = [_make_run_result(scores={ev: 0.2 for ev in EVALUATOR_NAMES})]
    summary = compute_summary(results)
    out = tmp_path / "report.json"
    export_json(
        path=out,
        label="test_run",
        dataset_path="evals/test.jsonl",
        n_records=1,
        duration_s=1.23,
        summary=summary,
        regression=None,
    )
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["label"] == "test_run"
    assert data["n_records"] == 1
    assert data["duration_s"] == pytest.approx(1.23)
    assert data["regression"] is None
    for ev in EVALUATOR_NAMES:
        assert ev in data["summary"]


def test_export_json_includes_regression(tmp_path):
    summary = {ev: {"flag_rate": 0.1, "mean": 0.1, "p50": 0.1, "p95": 0.1} for ev in EVALUATOR_NAMES}
    regression = compute_regression(summary, {ev: {"flag_rate": 0.0} for ev in EVALUATOR_NAMES})
    out = tmp_path / "report.json"
    export_json(
        path=out,
        label="run",
        dataset_path="data.jsonl",
        n_records=5,
        duration_s=2.5,
        summary=summary,
        regression=regression,
    )
    data = json.loads(out.read_text())
    assert data["regression"] is not None
    for ev in EVALUATOR_NAMES:
        assert ev in data["regression"]


# ── print_runs_table ──────────────────────────────────────────────────────────


def test_print_runs_table_empty():
    """Empty list prints a dim 'No eval runs found.' message."""
    buf = StringIO()
    console = Console(file=buf, highlight=False)
    print_runs_table(console, [])
    assert "No eval runs found" in buf.getvalue()


def test_print_runs_table_renders_runs():
    """Non-empty list renders a table row for each run."""
    from datetime import datetime, timezone
    from unittest.mock import MagicMock

    run = MagicMock()
    run.label = "test_run"
    run.status = "complete"
    run.record_count = 50
    run.dataset_path = "evals/test.jsonl"
    run.created_at = datetime(2026, 2, 28, 12, 0, 0, tzinfo=timezone.utc)
    run.completed_at = datetime(2026, 2, 28, 12, 1, 0, tzinfo=timezone.utc)

    buf = StringIO()
    console = Console(file=buf, highlight=False)
    print_runs_table(console, [run])
    output = buf.getvalue()
    assert "test_run" in output


def test_print_runs_table_running_status():
    from unittest.mock import MagicMock

    run = MagicMock()
    run.label = "active"
    run.status = "running"
    run.record_count = 0
    run.dataset_path = "data.jsonl"
    run.created_at = None
    run.completed_at = None

    buf = StringIO()
    console = Console(file=buf, highlight=False)
    print_runs_table(console, [run])
    output = buf.getvalue()
    assert "active" in output


def test_print_runs_table_failed_status():
    from unittest.mock import MagicMock

    run = MagicMock()
    run.label = "bad_run"
    run.status = "failed"
    run.record_count = 5
    run.dataset_path = "data.jsonl"
    run.created_at = None
    run.completed_at = None

    buf = StringIO()
    console = Console(file=buf, highlight=False)
    print_runs_table(console, [run])
    output = buf.getvalue()
    assert "bad_run" in output
