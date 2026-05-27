#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/_cli/_skills.py
# ----------------------------------------
from __future__ import annotations

"""`scitex-ml skills` — list / get / install the bundled agent-facing skills.

Self-contained (no scitex-dev runtime dep): walks the package's own
`_skills/scitex-ml/` directory directly, per general/03_interface/02_cli §1a.
"""

import os as _os
from pathlib import Path

import click

PKG = "scitex-ml"


def _skills_root() -> Path:
    import scitex_ml

    return Path(scitex_ml.__file__).parent / "_skills" / PKG


def _skill_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.md") if p.is_file() and p.name != "SKILL.md")


@click.group(name="skills", invoke_without_command=True)
@click.pass_context
def skills_group(ctx) -> None:
    """Agent-facing skill pages bundled with scitex-ml.

    \b
    Examples:
      $ scitex-ml skills list
      $ scitex-ml skills get 02_quick-start
      $ scitex-ml skills install                  # → ~/.scitex/dev/skills/scitex-ml/
      $ scitex-ml skills install --claude-symlink # also ~/.claude/skills/scitex/
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@skills_group.command(name="list")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def skills_list(as_json: bool) -> None:
    """List skill pages bundled with this package.

    \b
    Example:
      $ scitex-ml skills list
      $ scitex-ml skills list --json
    """
    root = _skills_root()
    files = _skill_files(root)
    if as_json:
        import json as _json

        click.echo(
            _json.dumps([{"name": p.stem, "path": str(p)} for p in files], indent=2)
        )
        return
    if not files:
        click.echo(f"no skills found at {root}", err=True)
        raise SystemExit(1)
    for p in files:
        click.echo(f"{p.stem:36s}  {p.relative_to(root)}")


@skills_group.command(name="get")
@click.argument("name")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def skills_get(name: str, as_json: bool) -> None:
    """Print the contents of a skill page by NAME (e.g. `02_quick-start`).

    \b
    Example:
      $ scitex-ml skills get 02_quick-start
      $ scitex-ml skills get 04_classification --json
    """
    root = _skills_root()
    stem = name[:-3] if name.endswith(".md") else name
    match = next((p for p in _skill_files(root) if p.stem == stem), None)
    if match is None:
        click.echo(f"skill not found: {name}", err=True)
        avail = ", ".join(p.stem for p in _skill_files(root)[:8])
        click.echo(f"available: {avail}…", err=True)
        raise SystemExit(1)
    if as_json:
        import json as _json

        click.echo(
            _json.dumps(
                {
                    "name": match.stem,
                    "path": str(match),
                    "content": match.read_text(encoding="utf-8"),
                },
                indent=2,
            )
        )
        return
    click.echo(match.read_text(encoding="utf-8"))


@skills_group.command(name="install")
@click.option(
    "--dest",
    type=click.Path(),
    default=None,
    help="Destination dir (default: ~/.scitex/dev/skills/scitex-ml/).",
)
@click.option(
    "--no-link", is_flag=True, help="Copy files instead of symlinking (default: link)."
)
@click.option(
    "--claude-symlink",
    is_flag=True,
    help="Also expose at ~/.claude/skills/scitex/ for Claude Code.",
)
@click.option("--dry-run", is_flag=True, help="Preview without writing.")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation (no-op).")
def skills_install(dest, no_link, claude_symlink, dry_run, yes) -> None:
    """Install this package's skills into a target directory.

    \b
    Example:
      $ scitex-ml skills install
      $ scitex-ml skills install --claude-symlink --dry-run
    """
    del yes
    src = _skills_root().resolve()
    if not src.is_dir():
        click.echo(f"no skills directory at {src}", err=True)
        raise SystemExit(1)
    base = (
        Path(dest).expanduser() if dest else Path.home() / ".scitex" / "dev" / "skills"
    )
    target = base / PKG

    if dry_run:
        action = "copy" if no_link else "symlink"
        click.echo(f"would {action} {src} → {target}")
        if claude_symlink:
            click.echo(
                f"would symlink {Path.home() / '.claude' / 'skills' / 'scitex'} → {base}"
            )
        return

    base.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        import shutil

        shutil.rmtree(target)
    if no_link:
        import shutil

        shutil.copytree(src, target)
        click.echo(f"copied {src} → {target}")
    else:
        _os.symlink(src, target, target_is_directory=True)
        click.echo(f"linked {target} → {src}")

    if claude_symlink:
        link = Path.home() / ".claude" / "skills" / "scitex"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            _os.symlink(base.resolve(), link, target_is_directory=True)
            click.echo(f"linked {link} → {base}")
        else:
            click.echo(
                f"warning: {link} exists and is not a symlink — skipping", err=True
            )


# EOF
