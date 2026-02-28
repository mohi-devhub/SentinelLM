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

import typer
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
        ..., "--dataset", "-d",
        help="Path to the golden JSONL dataset.",
        exists=True, readable=True,
    ),
    label: str = typer.Option(
        ..., "--label", "-l",
        help="Human-readable name for this run (must be unique).",
    ),
    baseline: str | None = typer.Option(
        None, "--baseline", "-b",
        help="Label of a previous run to compare against for regression detection.",
    ),
    server: str = typer.Option(
        "http://localhost:8000", "--server",
        help="Base URL of the running SentinelLM server.",
    ),
    concurrency: int = typer.Option(
        4, "--concurrency", "-c",
        help="Max concurrent requests sent to the server.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
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
            status = "[red]BLOCKED[/red]" if result.blocked else (
                "[yellow]FLAGGED[/yellow]" if result.flags else "[green]PASS[/green]"
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
