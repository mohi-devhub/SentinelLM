"""Tenant admin CLI.

Usage examples:
    sentinel tenant create-tenant --name "Acme Corp" --slug acme
    sentinel tenant create-key --tenant acme --label "prod key"
    sentinel tenant list-keys
    sentinel tenant revoke-key <id>
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import typer
from rich.console import Console
from rich.table import Table

console = Console()
tenant_app = typer.Typer(help="Tenant and API key management.", no_args_is_help=True)


@tenant_app.command("create-tenant")
def create_tenant_cmd(
    name: str = typer.Option(..., "--name", help="Human-readable tenant name."),
    slug: str = typer.Option(..., "--slug", help="URL-safe unique tenant slug."),
) -> None:
    """Create a new tenant."""
    asyncio.run(_create_tenant(name, slug))


async def _create_tenant(name: str, slug: str) -> None:
    from sentinel.settings import get_settings  # noqa: PLC0415
    from sentinel.storage.database import create_pool  # noqa: PLC0415
    from sentinel.tenancy.queries import create_tenant, get_tenant_by_slug  # noqa: PLC0415

    settings = get_settings()
    pool = await create_pool(settings.database_url)
    try:
        if await get_tenant_by_slug(pool, slug):
            console.print(f"[red]Tenant slug '[bold]{slug}[/bold]' already exists.[/red]")
            raise typer.Exit(1)
        tenant = await create_tenant(pool, name, slug)
        console.print(f"[green]Created tenant[/green] [bold]{name}[/bold] ({slug})")
        console.print(f"[dim]id: {tenant['id']}[/dim]")
    finally:
        await pool.close()


@tenant_app.command("create-key")
def create_key_cmd(
    tenant: str = typer.Option(..., "--tenant", help="Slug of the owning tenant."),
    label: str | None = typer.Option(None, "--label", help="Optional label for this key."),
) -> None:
    """Issue a new API key for a tenant. The plaintext key is printed once — save it now."""
    asyncio.run(_create_key(tenant, label))


async def _create_key(tenant_slug: str, label: str | None) -> None:
    from sentinel.settings import get_settings  # noqa: PLC0415
    from sentinel.storage.database import create_pool  # noqa: PLC0415
    from sentinel.tenancy.keys import generate_api_key, hash_key, key_prefix  # noqa: PLC0415
    from sentinel.tenancy.queries import create_api_key, get_tenant_by_slug  # noqa: PLC0415

    settings = get_settings()
    pool = await create_pool(settings.database_url)
    try:
        tenant = await get_tenant_by_slug(pool, tenant_slug)
        if tenant is None:
            console.print(f"[red]No tenant with slug '[bold]{tenant_slug}[/bold]'.[/red]")
            raise typer.Exit(1)

        plaintext = generate_api_key()
        await create_api_key(pool, tenant["id"], hash_key(plaintext), key_prefix(plaintext), label)
        console.print(f"[green]Created API key for[/green] [bold]{tenant_slug}[/bold]")
        console.print(f"\n[bold yellow]{plaintext}[/bold yellow]\n")
        console.print(
            "[dim]This key is shown once and is not stored in plaintext anywhere. "
            "Save it now.[/dim]"
        )
    finally:
        await pool.close()


@tenant_app.command("list-keys")
def list_keys_cmd(
    tenant: str | None = typer.Option(None, "--tenant", help="Filter to one tenant's slug."),
) -> None:
    """List API keys (prefix only — plaintext keys are never stored)."""
    asyncio.run(_list_keys(tenant))


async def _list_keys(tenant_slug: str | None) -> None:
    from sentinel.settings import get_settings  # noqa: PLC0415
    from sentinel.storage.database import create_pool  # noqa: PLC0415
    from sentinel.tenancy.queries import get_tenant_by_slug, list_api_keys  # noqa: PLC0415

    settings = get_settings()
    pool = await create_pool(settings.database_url)
    try:
        tenant_id = None
        if tenant_slug is not None:
            tenant = await get_tenant_by_slug(pool, tenant_slug)
            if tenant is None:
                console.print(f"[red]No tenant with slug '[bold]{tenant_slug}[/bold]'.[/red]")
                raise typer.Exit(1)
            tenant_id = tenant["id"]

        keys = await list_api_keys(pool, tenant_id)
        if not keys:
            console.print("[dim]No API keys found.[/dim]")
            return

        table = Table(title="API Keys")
        table.add_column("ID", style="dim")
        table.add_column("Tenant")
        table.add_column("Prefix")
        table.add_column("Label")
        table.add_column("Created")
        table.add_column("Last used")
        table.add_column("Status")
        for k in keys:
            status = "[red]revoked[/red]" if k["revoked_at"] else "[green]active[/green]"
            table.add_row(
                str(k["id"]),
                k["tenant_slug"],
                k["key_prefix"] + "…",
                k["label"] or "",
                k["created_at"].isoformat() if k["created_at"] else "",
                k["last_used_at"].isoformat() if k["last_used_at"] else "never",
                status,
            )
        console.print(table)
    finally:
        await pool.close()


@tenant_app.command("revoke-key")
def revoke_key_cmd(
    key_id: str = typer.Argument(..., help="API key UUID (see `sentinel tenant list-keys`)."),
) -> None:
    """Revoke an API key immediately."""
    asyncio.run(_revoke_key(key_id))


async def _revoke_key(key_id_str: str) -> None:
    from sentinel.settings import get_settings  # noqa: PLC0415
    from sentinel.storage.database import create_pool  # noqa: PLC0415
    from sentinel.tenancy.queries import revoke_api_key  # noqa: PLC0415

    try:
        key_id = UUID(key_id_str)
    except ValueError:
        console.print(f"[red]'{key_id_str}' is not a valid UUID.[/red]")
        raise typer.Exit(1) from None

    settings = get_settings()
    pool = await create_pool(settings.database_url)
    try:
        revoked = await revoke_api_key(pool, key_id)
        if not revoked:
            console.print(f"[red]No active key with id {key_id}.[/red]")
            raise typer.Exit(1)
        console.print(f"[green]Revoked key[/green] {key_id}")
    finally:
        await pool.close()


@tenant_app.command("list-tenants")
def list_tenants_cmd() -> None:
    """List all tenants."""
    asyncio.run(_list_tenants())


async def _list_tenants() -> None:
    from sentinel.settings import get_settings  # noqa: PLC0415
    from sentinel.storage.database import create_pool  # noqa: PLC0415
    from sentinel.tenancy.queries import list_tenants  # noqa: PLC0415

    settings = get_settings()
    pool = await create_pool(settings.database_url)
    try:
        tenants = await list_tenants(pool)
        table = Table(title="Tenants")
        table.add_column("ID", style="dim")
        table.add_column("Name")
        table.add_column("Slug")
        table.add_column("Default")
        table.add_column("Created")
        for t in tenants:
            table.add_row(
                str(t["id"]),
                t["name"],
                t["slug"],
                "yes" if t["is_default"] else "",
                t["created_at"].isoformat() if t["created_at"] else "",
            )
        console.print(table)
    finally:
        await pool.close()
