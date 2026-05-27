#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/_cli/_apis.py
# ----------------------------------------
from __future__ import annotations

"""`scitex-ml list-python-apis` — public Python API introspection.

Required by general/03_interface/02_cli §1a, with the `-v/-vv/-vvv` ladder
(names → signatures → docstrings) and `--json`.
"""

import click

TYPE_COLORS = {"M": "blue", "C": "magenta", "F": "green"}


def _format_signature(func) -> str:
    import inspect

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return ""
    params = []
    for name, param in sig.parameters.items():
        if param.annotation is not inspect.Parameter.empty:
            ann = param.annotation
            type_str = ann.__name__ if hasattr(ann, "__name__") else str(ann)
            type_str = type_str.replace("typing.", "")
        else:
            type_str = None
        if param.default is not inspect.Parameter.empty:
            def_str = repr(param.default)
            if len(def_str) > 20:
                def_str = "..."
            piece = (
                f"{name}: {type_str} = {def_str}" if type_str else f"{name}={def_str}"
            )
        else:
            piece = f"{name}: {type_str}" if type_str else name
        params.append(piece)
    return f"({', '.join(params)})"


def _collect(module, prefix="", max_depth=2, _depth=0, _seen=None):
    """Recursively collect public API items (functions/classes/submodules)."""
    import inspect
    import types

    if _seen is None:
        _seen = set()
    mid = id(module)
    if mid in _seen or _depth > max_depth:
        return []
    _seen.add(mid)

    results = []
    for name in sorted(getattr(module, "__all__", None) or dir(module)):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(module, name)
        except Exception:
            continue
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(obj, type):
            doc = (inspect.getdoc(obj) or "").split("\n")[0]
            results.append(("C", full, "", doc))
        elif isinstance(obj, types.ModuleType):
            pkg = getattr(obj, "__package__", "") or ""
            if "scitex_ml" in pkg:
                results.append(("M", full, "", ""))
                results.extend(_collect(obj, full, max_depth, _depth + 1, _seen))
        elif callable(obj):
            doc = (inspect.getdoc(obj) or "").split("\n")[0]
            results.append(("F", full, _format_signature(obj), doc))
    return results


@click.command("list-python-apis")
@click.option("-v", "--verbose", count=True, help="-v signatures, -vv +docstrings")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def list_python_apis(verbose, as_json):
    """List the public Python API of scitex-ml.

    \b
    Examples:
      $ scitex-ml list-python-apis
      $ scitex-ml list-python-apis -vv
      $ scitex-ml list-python-apis --json
    """
    import scitex_ml

    apis = _collect(scitex_ml, "scitex_ml")

    if as_json:
        import json as _json

        payload = {
            "module": "scitex_ml",
            "apis": [
                {"kind": k, "name": n, "signature": s, "doc": d} for k, n, s, d in apis
            ],
        }
        click.echo(_json.dumps(payload, indent=2))
        return

    if not apis:
        click.echo("No public APIs found.")
        return
    for kind, name, sig, doc in apis:
        color = TYPE_COLORS.get(kind, "white")
        label = click.style(f"[{kind}]", fg=color)
        styled = click.style(name, fg=color, bold=True)
        if verbose == 0:
            click.echo(f"  {label} {styled}")
        elif verbose == 1:
            click.echo(f"  {label} {styled}{sig}")
        else:
            line = f"  {label} {styled}{sig}"
            if doc:
                line += f"\n       {click.style(doc, dim=True)}"
            click.echo(line)


# EOF
