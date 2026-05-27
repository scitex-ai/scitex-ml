#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/_cli/_main.py
# ----------------------------------------
from __future__ import annotations

"""scitex-ml CLI root group.

Noun-verb grammar with the SciTeX universal flags (--version, --help-recursive,
--json) and the required introspection / mcp / skills / shell-completion
surface (general/03_interface/02_cli). Analysis verbs delegate to
:mod:`scitex_ml._analysis` for CLI ↔ MCP parity.
"""

import click

from ._analysis_cmds import (
    compute_metrics_cmd,
    generate_report_cmd,
    reduce_dimensions_cmd,
)
from ._apis import list_python_apis
from ._mcp import mcp
from ._skills import skills_group

try:
    from scitex_ml import __version__
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

COMMAND_CATEGORIES = [
    ("Analysis", ["compute-metrics", "generate-report", "reduce-dimensions"]),
    ("Introspection", ["list-python-apis"]),
    ("Integration", ["mcp", "skills"]),
    ("Utility", ["install-shell-completion", "print-shell-completion"]),
]


class CategorizedGroup(click.Group):
    """Group whose --help lists commands under the COMMAND_CATEGORIES headings."""

    def format_commands(self, ctx, formatter):
        commands = {}
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is not None and not cmd.hidden:
                commands[name] = cmd
        shown = set()
        for category, names in COMMAND_CATEGORIES:
            rows = []
            for name in names:
                if name in commands and name not in shown:
                    rows.append(
                        (name, commands[name].get_short_help_str(limit=formatter.width))
                    )
                    shown.add(name)
            if rows:
                with formatter.section(category):
                    formatter.write_dl(rows)
        # Anything not categorised (deprecated aliases, etc.) → "Other".
        rows = [
            (n, commands[n].get_short_help_str(limit=formatter.width))
            for n in commands
            if n not in shown
        ]
        if rows:
            with formatter.section("Other"):
                formatter.write_dl(rows)


def _print_help_recursive(ctx) -> None:
    """Dump help for the root group and every subcommand."""
    click.echo(ctx.get_help())
    root = ctx.command
    for name in sorted(root.list_commands(ctx)):
        cmd = root.get_command(ctx, name)
        if cmd is None or cmd.hidden:
            continue
        click.echo("\n" + "=" * 70)
        with click.Context(cmd, info_name=name, parent=ctx) as sub:
            click.echo(sub.get_help())
            if isinstance(cmd, click.Group):
                for sname in sorted(cmd.list_commands(sub)):
                    scmd = cmd.get_command(sub, sname)
                    if scmd is None or scmd.hidden:
                        continue
                    click.echo("\n" + "-" * 60)
                    with click.Context(scmd, info_name=sname, parent=sub) as ssub:
                        click.echo(ssub.get_help())


@click.group(
    cls=CategorizedGroup,
    context_settings=CONTEXT_SETTINGS,
    invoke_without_command=True,
)
@click.option(
    "--help-recursive", is_flag=True, help="Show help for every command and subcommand."
)
@click.option(
    "--json", "as_json", is_flag=True, help="Propagate JSON output to subcommands."
)
@click.version_option(__version__, "--version", "-V", message="%(prog)s/%(version)s")
@click.pass_context
def main(ctx, help_recursive, as_json):
    """scitex-ml: classical/deep ML utilities — metrics, reports, reductions.

    \b
    The CLI exposes the stateless analysis surface (compute-metrics,
    generate-report, reduce-dimensions); training/optimizers stay Python-only.

    \b
    Config precedence (when a command reads config):
      --config → $SCITEX_ML_CONFIG → ./config.yaml → ~/.scitex/ml/config.yaml
    """
    ctx.ensure_object(dict)
    ctx.obj["as_json"] = as_json
    if help_recursive:
        _print_help_recursive(ctx)
        ctx.exit(0)
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Analysis verbs (parity with MCP tools)
main.add_command(compute_metrics_cmd)
main.add_command(generate_report_cmd)
main.add_command(reduce_dimensions_cmd)
# Introspection
main.add_command(list_python_apis)
# Integration
main.add_command(mcp)
main.add_command(skills_group)

# Shell completion (adds install-shell-completion / print-shell-completion).
try:
    from scitex_dev._cli._completion import attach_shell_completion

    attach_shell_completion(main, prog_name="scitex-ml")
except Exception:  # pragma: no cover - optional dep
    pass


# EOF
