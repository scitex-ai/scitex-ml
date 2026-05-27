#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/_cli/_mcp.py
# ----------------------------------------
from __future__ import annotations

"""`scitex-ml mcp` — MCP server commands (general/03_interface/03_mcp §3).

Exposes all four required subcommands: start, doctor, list-tools, install.
"""

import json as _json

import click

_SERVER_PATH = "scitex_ml._mcp_server:mcp"
_PKG = "scitex-ml"


def _load_server():
    """Import the FastMCP instance, or raise a clear install hint."""
    try:
        from scitex_ml._mcp_server import mcp as server

        return server
    except ImportError as e:  # pragma: no cover - exercised via doctor
        raise click.ClickException(
            f"MCP not available. Install: pip install {_PKG}[mcp]\n{e}"
        ) from e


@click.group(invoke_without_command=True)
@click.pass_context
def mcp(ctx):
    """MCP (Model Context Protocol) server commands for scitex-ml."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@mcp.command("start")
@click.option("--http", is_flag=True, help="Use HTTP transport instead of stdio.")
@click.option("--host", default="127.0.0.1", show_default=True, help="HTTP host.")
@click.option("--port", default=8100, type=int, show_default=True, help="HTTP port.")
@click.option("--dry-run", is_flag=True, help="Print the launch plan; don't start.")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation (no-op).")
def start(http, host, port, dry_run, yes):
    """Start the scitex-ml MCP server (stdio by default).

    \b
    Example:
      $ scitex-ml mcp start
      $ scitex-ml mcp start --http --port 8100
    """
    del yes
    transport = "http" if http else "stdio"
    if dry_run:
        target = f"{host}:{port}" if http else "stdio"
        click.echo(
            f"DRY RUN — would start scitex-ml MCP server ({transport}, {target})"
        )
        return
    server = _load_server()
    if http:
        server.run(transport="http", host=host, port=port)
    else:
        server.run()


@mcp.command("doctor")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def doctor(as_json):
    """Self-diagnose the MCP install (deps, package, tool count).

    \b
    Example:
      $ scitex-ml mcp doctor
      $ scitex-ml mcp doctor --json
    """
    checks = []
    try:
        import fastmcp

        checks.append(("fastmcp", True, getattr(fastmcp, "__version__", "?")))
    except ImportError:
        checks.append(("fastmcp", False, f"missing — pip install {_PKG}[mcp]"))

    n_tools = None
    try:
        import asyncio

        server = _load_server()
        n_tools = len(asyncio.run(server.list_tools()))
        checks.append(("scitex-ml server", True, f"{n_tools} tools"))
    except Exception as e:  # noqa: BLE001
        checks.append(("scitex-ml server", False, str(e)))

    healthy = all(ok for _, ok, _ in checks)
    if as_json:
        click.echo(
            _json.dumps(
                {
                    "healthy": healthy,
                    "tool_count": n_tools,
                    "checks": [
                        {"name": n, "ok": ok, "detail": d} for n, ok, d in checks
                    ],
                },
                indent=2,
            )
        )
        raise SystemExit(0 if healthy else 1)

    click.secho("scitex-ml MCP doctor", fg="cyan", bold=True)
    for name, ok, detail in checks:
        status = click.style("OK", fg="green") if ok else click.style("FAIL", fg="red")
        click.echo(f"  [{status}] {name}: {detail}", err=True)
    raise SystemExit(0 if healthy else 1)


@mcp.command("list-tools")
@click.option(
    "-v", "--verbose", count=True, help="-v signatures, -vv +docstrings, -vvv +schema"
)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def list_tools(verbose, as_json):
    """List the MCP tools registered by scitex-ml.

    \b
    Example:
      $ scitex-ml mcp list-tools
      $ scitex-ml mcp list-tools -vv --json
    """
    import asyncio

    server = _load_server()
    tools = sorted(asyncio.run(server.list_tools()), key=lambda t: t.name)

    if as_json:
        payload = {
            "total": len(tools),
            "tools": [
                {
                    "name": t.name,
                    "description": (getattr(t, "description", "") or ""),
                    "parameters": getattr(t, "parameters", {}) or {},
                }
                for t in tools
            ],
        }
        click.echo(_json.dumps(payload, indent=2))
        return

    for t in tools:
        desc = getattr(t, "description", "") or ""
        params = getattr(t, "parameters", {}) or {}
        props = params.get("properties", {})
        click.echo(click.style(t.name, fg="green", bold=True))
        if verbose >= 1 and props:
            arglist = ", ".join(
                f"{k}: {v.get('type', 'any')}" for k, v in props.items()
            )
            click.echo(f"    ({arglist})")
        if verbose >= 2 and desc:
            click.echo(f"    {click.style(desc.splitlines()[0], dim=True)}")
        if verbose >= 3:
            click.echo(f"    schema: {_json.dumps(params)}")


@mcp.command("install")
@click.option("--json", "as_json", is_flag=True, help="Emit the raw config JSON.")
@click.option("--dry-run", is_flag=True, help="No-op; this verb is informational.")
@click.option("-y", "--yes", is_flag=True, help="No-op; this verb is informational.")
def install(as_json, dry_run, yes):
    """Print the MCP-host config snippet for scitex-ml.

    \b
    Example:
      $ scitex-ml mcp install
      $ scitex-ml mcp install --json
    """
    del dry_run, yes
    config = {
        "mcpServers": {"scitex-ml": {"command": "scitex-ml", "args": ["mcp", "start"]}}
    }
    if as_json:
        click.echo(
            _json.dumps(
                {
                    "install_command": f"pip install {_PKG}[mcp]",
                    "config": config,
                    "server_path": _SERVER_PATH,
                    "verify": ["scitex-ml mcp doctor"],
                },
                indent=2,
            )
        )
        return
    click.secho("scitex-ml MCP installation", fg="cyan", bold=True)
    click.echo(f"\n  pip install {_PKG}[mcp]\n")
    click.echo("Add to your MCP host config (e.g. Claude Code):\n")
    for line in _json.dumps(config, indent=2).splitlines():
        click.echo(f"  {line}")
    click.echo("\nVerify with:\n\n  scitex-ml mcp doctor")


# EOF
