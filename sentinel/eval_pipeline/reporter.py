"""Scorecard computation and rich terminal output for eval runs."""

from __future__ import annotations

import json
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sentinel.eval_pipeline.runner import EVALUATOR_NAMES, RunResult

# Flag-rate increase above this threshold is reported as a regression.
REGRESSION_THRESHOLD = 0.05


# ── Statistics helpers ────────────────────────────────────────────────────────


def _percentile(data: list[float], p: float) -> float:
    """Linear-interpolation percentile; returns 0.0 for empty lists."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = (p / 100.0) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


# ── Scorecard computation ─────────────────────────────────────────────────────


def compute_summary(results: list[RunResult]) -> dict:
    """Compute per-evaluator stats from a list of RunResults.

    Returns a dict keyed by evaluator name:
        {
          "pii": {
            "mean": 0.021, "p50": 0.000, "p95": 0.140,
            "flag_rate": 0.02, "flag_count": 1, "n": 50, "n_scored": 40
          },
          ...
        }
    """
    n = len(results)
    summary: dict = {}

    for ev in EVALUATOR_NAMES:
        scores: list[float] = [v for r in results if (v := r.scores.get(ev)) is not None]
        flag_count = sum(1 for r in results if ev in r.flags)

        summary[ev] = {
            "mean": sum(scores) / len(scores) if scores else None,
            "p50": _percentile(scores, 50) if scores else None,
            "p95": _percentile(scores, 95) if scores else None,
            "flag_rate": flag_count / n if n > 0 else 0.0,
            "flag_count": flag_count,
            "n": n,
            "n_scored": len(scores),
        }

    return summary


def compute_regression(current: dict, baseline: dict) -> dict:
    """Compare current vs baseline flag rates.

    A regression is flagged when the current flag_rate exceeds the baseline
    by more than REGRESSION_THRESHOLD (default: 5 percentage points).

    Returns a dict keyed by evaluator name:
        {
          "toxicity": {
            "baseline_flag_rate": 0.02,
            "current_flag_rate": 0.07,
            "delta": 0.05,
            "regressed": true
          },
          ...
        }
    """
    regression: dict = {}
    for ev in EVALUATOR_NAMES:
        cur_rate = current.get(ev, {}).get("flag_rate", 0.0)
        base_rate = baseline.get(ev, {}).get("flag_rate", 0.0)
        delta = cur_rate - base_rate
        regression[ev] = {
            "baseline_flag_rate": base_rate,
            "current_flag_rate": cur_rate,
            "delta": delta,
            "regressed": delta > REGRESSION_THRESHOLD,
        }
    return regression


# ── Terminal output ───────────────────────────────────────────────────────────


def print_report(
    console: Console,
    label: str,
    dataset_path: str,
    n_records: int,
    duration_s: float,
    summary: dict,
    regression: dict | None = None,
    baseline_label: str | None = None,
) -> None:
    """Render the scorecard + optional regression table to the terminal."""

    # ── Header ───────────────────────────────────────────────────────────────
    console.print(
        Panel.fit(
            f"[bold cyan]SentinelLM Eval Report[/bold cyan]\n"
            f"Run: [bold]{label}[/bold]   "
            f"Dataset: [dim]{dataset_path}[/dim]   "
            f"Records: [bold]{n_records}[/bold]   "
            f"Duration: [bold]{duration_s:.1f}s[/bold]",
            box=box.ROUNDED,
        )
    )

    # ── Scorecard table ───────────────────────────────────────────────────────
    table = Table(
        title="Scorecard",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Evaluator", style="cyan", no_wrap=True)
    table.add_column("Scored", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("P50", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("Flag Rate", justify="right")
    table.add_column("Flags", justify="right")

    for ev in EVALUATOR_NAMES:
        s = summary.get(ev, {})
        mean = s.get("mean")
        p50 = s.get("p50")
        p95 = s.get("p95")
        flag_rate = s.get("flag_rate", 0.0)
        flag_count = s.get("flag_count", 0)
        n_scored = s.get("n_scored", 0)

        flag_str = f"{flag_rate:.1%}"
        if flag_count > 0:
            flag_str = f"[yellow]{flag_str}[/yellow]"

        table.add_row(
            ev,
            str(n_scored),
            f"{mean:.3f}" if mean is not None else "[dim]—[/dim]",
            f"{p50:.3f}" if p50 is not None else "[dim]—[/dim]",
            f"{p95:.3f}" if p95 is not None else "[dim]—[/dim]",
            flag_str,
            str(flag_count),
        )

    console.print(table)

    # ── Regression table ──────────────────────────────────────────────────────
    if regression and baseline_label:
        reg_table = Table(
            title=f"Regression vs [cyan]{baseline_label}[/cyan]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
        )
        reg_table.add_column("Evaluator", style="cyan", no_wrap=True)
        reg_table.add_column("Baseline", justify="right")
        reg_table.add_column("Current", justify="right")
        reg_table.add_column("Delta", justify="right")
        reg_table.add_column("Status", justify="center")

        any_regression = False
        for ev in EVALUATOR_NAMES:
            r = regression.get(ev, {})
            base = r.get("baseline_flag_rate", 0.0)
            cur = r.get("current_flag_rate", 0.0)
            delta = r.get("delta", 0.0)
            regressed = r.get("regressed", False)

            delta_str = f"{delta:+.1%}"
            if regressed:
                delta_str = f"[red]{delta_str}[/red]"
                status = "[red]REGRESSION ⚠[/red]"
                any_regression = True
            elif delta < -REGRESSION_THRESHOLD:
                delta_str = f"[green]{delta_str}[/green]"
                status = "[green]IMPROVED ✓[/green]"
            else:
                status = "[dim]stable[/dim]"

            reg_table.add_row(ev, f"{base:.1%}", f"{cur:.1%}", delta_str, status)

        console.print(reg_table)

        if any_regression:
            console.print(
                "\n[red bold]⚠  Regressions detected — flag rate increased > 5pp "
                "for one or more evaluators.[/red bold]"
            )
        else:
            console.print("\n[green]✓  No regressions detected.[/green]")


def print_statistical_regression_table(
    console: Console,
    statistical_regression: dict,
    baseline_label: str,
) -> None:
    """Render a statistical regression table with p-value and Cohen's d columns."""
    stat_table = Table(
        title=f"Statistical Regression vs [cyan]{baseline_label}[/cyan]",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
    )
    stat_table.add_column("Evaluator", style="cyan", no_wrap=True)
    stat_table.add_column("n (cur/base)", justify="right")
    stat_table.add_column("p-value", justify="right")
    stat_table.add_column("Cohen's d", justify="right")
    stat_table.add_column("Effect", justify="center")
    stat_table.add_column("Direction", justify="center")

    any_regression = False
    for ev in EVALUATOR_NAMES:
        r = statistical_regression.get(ev)
        if not r:
            stat_table.add_row(ev, "—", "—", "—", "—", "[dim]no data[/dim]")
            continue

        n_str = f"{r.get('n_current', 0)}/{r.get('n_baseline', 0)}"
        p = r.get("p_value", 1.0)
        d = r.get("cohens_d", 0.0)
        effect = r.get("effect_size", "negligible")
        direction = r.get("direction", "stable")
        significant = r.get("significant", False)

        p_str = f"{p:.4f}" if p < 0.001 else f"{p:.3f}"
        d_str = f"{d:+.3f}"
        if significant:
            p_str = f"[bold]{p_str}[/bold]"

        if direction == "regression":
            dir_str = "[red]REGRESSION ⚠[/red]"
            any_regression = True
        elif direction == "improvement":
            dir_str = "[green]IMPROVED ✓[/green]"
        else:
            dir_str = "[dim]stable[/dim]"

        stat_table.add_row(ev, n_str, p_str, d_str, effect, dir_str)

    console.print(stat_table)

    if any_regression:
        console.print(
            "\n[red bold]⚠  Statistical regressions detected "
            "(p < 0.05, effect ≥ small).[/red bold]"
        )
    else:
        console.print("\n[green]✓  No statistical regressions detected.[/green]")


def export_json(
    path: Path,
    label: str,
    dataset_path: str,
    n_records: int,
    duration_s: float,
    summary: dict,
    regression: dict | None,
) -> None:
    """Write the scorecard as a JSON file for CI/CD consumption."""
    report = {
        "label": label,
        "dataset_path": dataset_path,
        "n_records": n_records,
        "duration_s": round(duration_s, 2),
        "summary": summary,
        "regression": regression,
    }
    path.write_text(json.dumps(report, indent=2, default=str))


def print_runs_table(console: Console, runs) -> None:
    """Render a summary table of past eval runs."""
    if not runs:
        console.print("[dim]No eval runs found.[/dim]")
        return

    table = Table(
        title="Eval Runs",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Label", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Records", justify="right")
    table.add_column("Dataset")
    table.add_column("Created", justify="right")
    table.add_column("Completed", justify="right")

    for run in runs:
        status_str = {
            "complete": "[green]complete[/green]",
            "running": "[yellow]running[/yellow]",
            "failed": "[red]failed[/red]",
        }.get(run.status, run.status)

        created = run.created_at.strftime("%Y-%m-%d %H:%M") if run.created_at else "—"
        completed = run.completed_at.strftime("%Y-%m-%d %H:%M") if run.completed_at else "—"

        table.add_row(
            run.label,
            status_str,
            str(run.record_count),
            run.dataset_path,
            created,
            completed,
        )

    console.print(table)
