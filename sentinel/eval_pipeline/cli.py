"""SentinelLM eval pipeline CLI.

Entry point: `sentinel eval run` / `sentinel eval list`

Usage examples:
    sentinel eval run --dataset evals/golden_qa.jsonl --label run_20260227
    sentinel eval run --dataset evals/golden_qa.jsonl --label run_20260228 --baseline run_20260227
    sentinel eval list
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()
app = typer.Typer(
    name="sentinel",
    help="SentinelLM CLI — eval pipeline and tooling.",
    no_args_is_help=True,
)
eval_app = typer.Typer(help="Eval pipeline commands.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")


# ── eval run ──────────────────────────────────────────────────────────────────


@eval_app.command("run")
def eval_run(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to the golden JSONL dataset.",
        exists=True,
        readable=True,
    ),
    label: str = typer.Option(
        ...,
        "--label",
        "-l",
        help="Human-readable name for this run (must be unique).",
    ),
    baseline: str | None = typer.Option(
        None,
        "--baseline",
        "-b",
        help="Label of a previous run to compare against for regression detection.",
    ),
    server: str = typer.Option(
        "http://localhost:8000",
        "--server",
        help="Base URL of the running SentinelLM server.",
    ),
    concurrency: int = typer.Option(
        4,
        "--concurrency",
        "-c",
        help="Max concurrent requests sent to the server.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the scorecard report to this JSON file.",
    ),
) -> None:
    """Run an eval against a golden JSONL dataset and print a scorecard."""
    asyncio.run(
        _async_eval_run(
            dataset=dataset,
            label=label,
            baseline=baseline,
            server=server,
            concurrency=concurrency,
            output=output,
        )
    )


async def _async_eval_run(
    dataset: Path,
    label: str,
    baseline: str | None,
    server: str,
    concurrency: int,
    output: Path | None,
) -> None:
    from sentinel.eval_pipeline.reporter import (  # noqa: PLC0415
        compute_regression,
        compute_summary,
        export_json,
        print_report,
    )
    from sentinel.eval_pipeline.runner import load_dataset, run_eval  # noqa: PLC0415
    from sentinel.settings import get_settings  # noqa: PLC0415
    from sentinel.storage.database import create_pool  # noqa: PLC0415
    from sentinel.storage.queries.eval_runs import (  # noqa: PLC0415
        complete_eval_run,
        fail_eval_run,
        get_eval_run_by_label,
        insert_eval_result,
        insert_eval_run,
    )

    settings = get_settings()

    # ── Load dataset ─────────────────────────────────────────────────────────
    records = load_dataset(dataset)
    if not records:
        console.print(f"[red]No records found in {dataset}[/red]")
        raise typer.Exit(1)

    console.print(
        f"[cyan]Loaded [bold]{len(records)}[/bold] records from [dim]{dataset}[/dim][/cyan]"
    )

    # ── Connect to DB ─────────────────────────────────────────────────────────
    pool = await create_pool(settings.database_url)

    # ── Validate label uniqueness ─────────────────────────────────────────────
    existing = await get_eval_run_by_label(pool, label)
    if existing:
        console.print(
            f"[red]Eval run label '[bold]{label}[/bold]' already exists "
            f"(status: {existing.status}). Choose a different label.[/red]"
        )
        await pool.close()
        raise typer.Exit(1)

    # ── Resolve baseline ──────────────────────────────────────────────────────
    baseline_run = None
    if baseline:
        baseline_run = await get_eval_run_by_label(pool, baseline)
        if not baseline_run:
            console.print(
                f"[yellow]Warning: baseline run '[bold]{baseline}[/bold]' not found. "
                f"Skipping regression comparison.[/yellow]"
            )
            baseline = None
        elif baseline_run.status != "complete":
            console.print(
                f"[yellow]Warning: baseline run '[bold]{baseline}[/bold]' "
                f"has status '{baseline_run.status}' — expected 'complete'. "
                f"Skipping regression comparison.[/yellow]"
            )
            baseline = None
            baseline_run = None

    # ── Create eval_run row ───────────────────────────────────────────────────
    eval_run_record = await insert_eval_run(
        pool,
        label=label,
        dataset_path=str(dataset),
        baseline_run_id=baseline_run.id if baseline_run else None,
    )
    assert eval_run_record.id is not None  # always set by insert_eval_run

    # ── Run the eval ──────────────────────────────────────────────────────────
    start = time.monotonic()
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Running eval...", total=len(records))

        def on_progress(completed: int, total: int, result) -> None:
            results.append(result)
            status = (
                "[red]BLOCKED[/red]"
                if result.blocked
                else ("[yellow]FLAGGED[/yellow]" if result.flags else "[green]PASS[/green]")
            )
            desc = f"[{result.record.record_index + 1}/{total}] {status}"
            if result.error:
                desc += f" [dim](error: {result.error[:60]})[/dim]"
            progress.update(task_id, advance=1, description=desc)

        try:
            results = await run_eval(
                records=records,
                server_url=server,
                concurrency=concurrency,
                on_progress=on_progress,
            )
        except Exception as exc:
            await fail_eval_run(pool, eval_run_record.id)
            console.print(f"[red]Eval run failed: {exc}[/red]")
            await pool.close()
            raise typer.Exit(1)

    duration_s = time.monotonic() - start

    # ── Persist eval_results ──────────────────────────────────────────────────
    console.print("[dim]Persisting eval results to database…[/dim]")
    for result in results:
        try:
            await insert_eval_result(
                pool=pool,
                eval_run_id=eval_run_record.id,
                request_id=result.request_id,
                record_index=result.record.record_index,
                input_text=result.record.input,
                expected_output=result.record.expected_output,
                actual_output=result.actual_output,
                passed=result.passed,
            )
        except Exception as exc:
            console.print(
                f"[yellow]Warning: failed to persist result #{result.record.record_index}: "
                f"{exc}[/yellow]"
            )

    # ── Compute scorecard ─────────────────────────────────────────────────────

    summary = compute_summary(results)
    regression = None
    if baseline_run and baseline_run.summary_json:
        regression = compute_regression(summary, baseline_run.summary_json)

    # ── Complete the eval_run row ─────────────────────────────────────────────
    await complete_eval_run(
        pool=pool,
        run_id=eval_run_record.id,
        record_count=len(results),
        summary=summary,
        regression=regression,
    )
    await pool.close()

    # ── Print report ──────────────────────────────────────────────────────────
    print_report(
        console=console,
        label=label,
        dataset_path=str(dataset),
        n_records=len(results),
        duration_s=duration_s,
        summary=summary,
        regression=regression,
        baseline_label=baseline,
    )

    # ── Optional JSON export ──────────────────────────────────────────────────
    if output:
        export_json(
            path=output,
            label=label,
            dataset_path=str(dataset),
            n_records=len(results),
            duration_s=duration_s,
            summary=summary,
            regression=regression,
        )
        console.print(f"\n[dim]Report exported to [bold]{output}[/bold][/dim]")

    # Exit with non-zero code if any regressions were detected (useful for CI)
    if regression and any(r["regressed"] for r in regression.values()):
        raise typer.Exit(2)


# ── eval offline ──────────────────────────────────────────────────────────────


@eval_app.command("offline")
def eval_offline(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to a v2 JSONL dataset (must include 'output' field per record).",
        exists=True,
        readable=True,
    ),
    label: str = typer.Option(
        ...,
        "--label",
        "-l",
        help="Human-readable name for this run (must be unique).",
    ),
    baseline: Optional[str] = typer.Option(
        None,
        "--baseline",
        "-b",
        help="Label of a previous offline run to compare against for regression detection.",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to config.yaml. Defaults to SENTINEL_CONFIG_PATH env var or 'config.yaml'.",
        exists=True,
        readable=True,
    ),
    concurrency: int = typer.Option(
        8,
        "--concurrency",
        "-c",
        help="Max concurrent evaluations (no network bottleneck — higher than live mode).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the scorecard report to this JSON file.",
    ),
) -> None:
    """Run offline eval on a scored JSONL dataset without an LLM backend."""
    asyncio.run(
        _async_offline_run(
            dataset=dataset,
            label=label,
            baseline=baseline,
            config_path=config_path,
            concurrency=concurrency,
            output=output,
        )
    )


async def _async_offline_run(
    dataset: Path,
    label: str,
    baseline: str | None,
    config_path: Path | None,
    concurrency: int,
    output: Path | None,
) -> None:
    from sentinel.eval_pipeline.reporter import (  # noqa: PLC0415
        export_json,
        print_report,
        print_statistical_regression_table,
    )
    from sentinel.eval_pipeline.runner import EVALUATOR_NAMES  # noqa: PLC0415
    from sentinel.evaluation.dataset import load_offline_dataset  # noqa: PLC0415
    from sentinel.evaluation.engine import OfflineEvaluationEngine  # noqa: PLC0415
    from sentinel.evaluation.stats import compute_statistical_regression  # noqa: PLC0415
    from sentinel.evaluators.base import BaseEvaluator  # noqa: PLC0415
    from sentinel.evaluators.registry import EVALUATOR_REGISTRY  # noqa: PLC0415
    from sentinel.settings import get_settings  # noqa: PLC0415
    from sentinel.storage.database import create_pool  # noqa: PLC0415
    from sentinel.storage.queries.eval_runs import (  # noqa: PLC0415
        complete_offline_eval_run,
        fail_eval_run,
        get_eval_run_by_label,
        get_offline_run_scores,
        insert_eval_result,
        insert_eval_run,
    )

    settings = get_settings()

    # ── Load config ──────────────────────────────────────────────────────────
    effective_config_path = config_path or Path(settings.config_path)
    with open(effective_config_path) as f:
        config = yaml.safe_load(f)

    # ── Load dataset ─────────────────────────────────────────────────────────
    records = load_offline_dataset(dataset)
    if not records:
        console.print(
            f"[red]No records with 'output' field found in {dataset}. "
            f"Offline mode requires pre-scored outputs.[/red]"
        )
        raise typer.Exit(1)

    console.print(
        f"[cyan]Loaded [bold]{len(records)}[/bold] records from [dim]{dataset}[/dim][/cyan]"
    )

    # ── Connect to DB ─────────────────────────────────────────────────────────
    pool = await create_pool(settings.database_url)

    # ── Validate label uniqueness ─────────────────────────────────────────────
    existing = await get_eval_run_by_label(pool, label)
    if existing:
        console.print(
            f"[red]Eval run label '[bold]{label}[/bold]' already exists "
            f"(status: {existing.status}). Choose a different label.[/red]"
        )
        await pool.close()
        raise typer.Exit(1)

    # ── Resolve baseline ──────────────────────────────────────────────────────
    baseline_run = None
    if baseline:
        baseline_run = await get_eval_run_by_label(pool, baseline)
        if not baseline_run:
            console.print(
                f"[yellow]Warning: baseline '[bold]{baseline}[/bold]' not found. "
                f"Skipping regression comparison.[/yellow]"
            )
            baseline = None
        elif baseline_run.status != "complete":
            console.print(
                f"[yellow]Warning: baseline '[bold]{baseline}[/bold]' "
                f"has status '{baseline_run.status}' — expected 'complete'. "
                f"Skipping regression comparison.[/yellow]"
            )
            baseline = None
            baseline_run = None

    # ── Create eval_run row ───────────────────────────────────────────────────
    eval_run_record = await insert_eval_run(
        pool,
        label=label,
        dataset_path=str(dataset),
        baseline_run_id=baseline_run.id if baseline_run else None,
    )
    assert eval_run_record.id is not None

    # ── Instantiate engine ────────────────────────────────────────────────────
    console.print("[dim]Loading evaluators…[/dim]")
    try:
        engine = OfflineEvaluationEngine(config)
    except Exception as exc:
        await fail_eval_run(pool, eval_run_record.id)
        console.print(f"[red]Failed to load evaluators: {exc}[/red]")
        await pool.close()
        raise typer.Exit(1)

    # ── Run the offline eval ──────────────────────────────────────────────────
    start = time.monotonic()
    run_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Evaluating offline…", total=len(records))

        def on_progress(completed: int, total: int, result) -> None:
            run_results.append(result)
            status = (
                "[red]BLOCKED[/red]"
                if result.blocked
                else ("[yellow]FLAGGED[/yellow]" if result.flags else "[green]PASS[/green]")
            )
            desc = f"[{result.record.record_index + 1}/{total}] {status}"
            if result.error:
                desc += f" [dim](error: {result.error[:60]})[/dim]"
            progress.update(task_id, advance=1, description=desc)

        try:
            run_results = await engine.evaluate_dataset(
                records=records,
                concurrency=concurrency,
                on_progress=on_progress,
            )
        except Exception as exc:
            await fail_eval_run(pool, eval_run_record.id)
            console.print(f"[red]Offline eval failed: {exc}[/red]")
            await pool.close()
            raise typer.Exit(1)

    duration_s = time.monotonic() - start

    # ── Persist eval_results with scores_json ────────────────────────────────
    console.print("[dim]Persisting eval results…[/dim]")
    for result in run_results:
        try:
            await insert_eval_result(
                pool=pool,
                eval_run_id=eval_run_record.id,
                request_id=None,
                record_index=result.record.record_index,
                input_text=result.record.input_text,
                expected_output=result.record.expected_output,
                actual_output=result.record.output_text,
                passed=not result.blocked and not result.flags,
                scores_json=result.scores,
            )
        except Exception as exc:
            console.print(
                f"[yellow]Warning: failed to persist result #{result.record.record_index}: "
                f"{exc}[/yellow]"
            )

    # ── Compute summary ───────────────────────────────────────────────────────
    from sentinel.eval_pipeline.reporter import compute_summary  # noqa: PLC0415

    summary = compute_summary(run_results)  # type: ignore[arg-type]  # duck-typed

    # ── Compute statistical regression against baseline ───────────────────────
    statistical_regression: dict | None = None
    legacy_regression: dict | None = None

    if baseline_run and baseline_run.summary_json:
        # Fetch per-record scores for statistical comparison
        baseline_scores_by_ev = await get_offline_run_scores(pool, baseline_run.id)

        # Build current scores dict from run_results
        current_scores_by_ev: dict[str, list[float]] = {}
        current_flags_by_ev: dict[str, int] = {}
        for res in run_results:
            for ev_name, score in res.scores.items():
                if score is not None:
                    current_scores_by_ev.setdefault(ev_name, []).append(score)
            for ev_name in res.flags:
                current_flags_by_ev[ev_name] = current_flags_by_ev.get(ev_name, 0) + 1

        # Per-evaluator flag direction lookup
        ev_direction: dict[str, str] = {}
        for ev_name, cls in EVALUATOR_REGISTRY.items():
            ev_direction[ev_name] = getattr(cls, "flag_direction", "above")

        statistical_regression = {}
        for ev_name in EVALUATOR_NAMES:
            cur_scores = current_scores_by_ev.get(ev_name, [])
            base_scores = baseline_scores_by_ev.get(ev_name, [])
            cur_flags = current_flags_by_ev.get(ev_name, 0)
            base_summary = baseline_run.summary_json.get(ev_name, {})
            base_flags = base_summary.get("flag_count", 0)

            result = compute_statistical_regression(
                current_scores=cur_scores,
                baseline_scores=base_scores,
                flag_direction=ev_direction.get(ev_name, "above"),  # type: ignore[arg-type]
                current_flags=cur_flags,
                baseline_flags=base_flags,
            )
            statistical_regression[ev_name] = result.to_dict()

        # Keep legacy flag-rate regression for backward compat
        from sentinel.eval_pipeline.reporter import compute_regression  # noqa: PLC0415

        legacy_regression = compute_regression(summary, baseline_run.summary_json)

    # ── Complete the eval_run row ─────────────────────────────────────────────
    await complete_offline_eval_run(
        pool=pool,
        run_id=eval_run_record.id,
        record_count=len(run_results),
        summary=summary,
        regression=legacy_regression,
        statistical_regression=statistical_regression,
    )
    await pool.close()

    # ── Print report ──────────────────────────────────────────────────────────
    print_report(
        console=console,
        label=label,
        dataset_path=str(dataset),
        n_records=len(run_results),
        duration_s=duration_s,
        summary=summary,
        regression=legacy_regression,
        baseline_label=baseline,
    )

    if statistical_regression and baseline:
        print_statistical_regression_table(
            console=console,
            statistical_regression=statistical_regression,
            baseline_label=baseline,
        )

    # ── Optional JSON export ──────────────────────────────────────────────────
    if output:
        export_json(
            path=output,
            label=label,
            dataset_path=str(dataset),
            n_records=len(run_results),
            duration_s=duration_s,
            summary=summary,
            regression=legacy_regression,
        )
        console.print(f"\n[dim]Report exported to [bold]{output}[/bold][/dim]")

    # Exit code 2 on statistical regression (for CI gates)
    if statistical_regression and any(
        r.get("direction") == "regression" and r.get("significant")
        for r in statistical_regression.values()
    ):
        raise typer.Exit(2)


# ── eval list ─────────────────────────────────────────────────────────────────


@eval_app.command("list")
def eval_list() -> None:
    """List past eval runs stored in the database."""
    asyncio.run(_async_list())


async def _async_list() -> None:
    from sentinel.eval_pipeline.reporter import print_runs_table  # noqa: PLC0415
    from sentinel.settings import get_settings  # noqa: PLC0415
    from sentinel.storage.database import create_pool  # noqa: PLC0415
    from sentinel.storage.queries.eval_runs import list_eval_runs  # noqa: PLC0415

    settings = get_settings()
    pool = await create_pool(settings.database_url)
    try:
        runs = await list_eval_runs(pool)
    finally:
        await pool.close()

    print_runs_table(console, runs)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
