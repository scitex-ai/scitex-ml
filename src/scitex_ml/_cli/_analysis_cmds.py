#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/_cli/_analysis_cmds.py
# ----------------------------------------
from __future__ import annotations

"""scitex-ml analysis verbs — the CLI half of the stateless analysis surface.

Each command delegates to :mod:`scitex_ml._analysis` (the same core the MCP
tools call), so ``scitex-ml compute-metrics --json`` and the ``compute_metrics``
MCP tool return identical JSON (CLI ↔ MCP parity). Compound verb-noun leaf
names mirror the MCP tool names: ``compute-metrics`` ↔ ``compute_metrics``,
``generate-report`` ↔ ``generate_report``, ``reduce-dimensions`` ↔
``reduce_dimensions``.
"""

import json as _json
from typing import NoReturn

import click

from scitex_ml import _analysis


def _fail(msg: str) -> NoReturn:
    """Print an error to stderr and exit 2 (usage/precondition error, §3)."""
    click.echo(f"error: {msg}", err=True)
    raise SystemExit(2)


@click.command("compute-metrics")
@click.argument("predictions", type=click.Path())
@click.option(
    "--label",
    "labels",
    multiple=True,
    help="Class display name (repeat, in label order).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit metrics as JSON.")
def compute_metrics_cmd(predictions, labels, as_json):
    """Compute classification metrics from a saved PREDICTIONS table.

    \b
    PREDICTIONS is a table (CSV/parquet/...) with `y_true` and `y_pred`
    columns, plus optional `y_proba` / `y_proba_*` probability columns.

    \b
    Examples:
      $ scitex-ml compute-metrics preds.csv
      $ scitex-ml compute-metrics preds.csv --label neg --label pos --json
    """
    try:
        out = _analysis.run_metrics(predictions, labels=list(labels) or None)
    except _analysis.AnalysisError as e:
        _fail(str(e))

    if as_json:
        click.echo(_json.dumps(out, indent=2))
        return

    m = out["metrics"]
    click.secho(f"Metrics for {out['predictions']}", fg="cyan", bold=True)
    click.echo(f"  samples            : {m['n_samples']}")
    click.echo(f"  balanced_accuracy  : {m['balanced_accuracy']:.4f}")
    click.echo(f"  mcc                : {m['mcc']:.4f}")
    if "roc_auc" in m:
        click.echo(f"  roc_auc            : {m['roc_auc']:.4f}")
        click.echo(f"  pr_auc             : {m['pr_auc']:.4f}")
    click.echo(f"  labels             : {m['labels']}")
    click.echo("  confusion_matrix   :")
    for row in m["confusion_matrix"]:
        click.echo(f"    {row}")


@click.command("generate-report")
@click.argument("predictions", type=click.Path())
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory for the report artifacts (created if absent).",
)
@click.option(
    "--label",
    "labels",
    multiple=True,
    help="Class display name (repeat, in label order).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit result as JSON.")
@click.option(
    "--dry-run", is_flag=True, help="Print what would be written, do nothing."
)
@click.option(
    "-y", "--yes", is_flag=True, help="Skip confirmation (no-op; never prompts)."
)
def generate_report_cmd(predictions, output_dir, labels, as_json, dry_run, yes):
    """Generate a full ClassificationReporter report from PREDICTIONS.

    Writes a metrics summary and plots under --output-dir.

    \b
    Examples:
      $ scitex-ml generate-report preds.csv -o ./report
      $ scitex-ml generate-report preds.csv -o ./report --json
    """
    del yes  # accepted for §2 universal-flag parity; this verb never prompts
    if dry_run:
        click.echo(
            f"DRY RUN — would read {predictions} and write a report to {output_dir}"
        )
        return
    try:
        out = _analysis.run_report(predictions, output_dir, labels=list(labels) or None)
    except _analysis.AnalysisError as e:
        _fail(str(e))

    if as_json:
        click.echo(_json.dumps(out, indent=2))
        return
    click.secho("Report written.", fg="green", bold=True)
    click.echo(f"  output_dir   : {out['output_dir']}")
    click.echo(f"  summary_path : {out['summary_path']}")
    click.echo(f"  balanced_accuracy : {out['metrics']['balanced_accuracy']:.4f}")


@click.command("reduce-dimensions")
@click.argument("data", type=click.Path())
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(),
    help="Path for the saved figure (extension picks the format).",
)
@click.option(
    "--method",
    type=click.Choice(["pca", "umap"]),
    default="pca",
    show_default=True,
    help="Dimensionality-reduction method.",
)
@click.option("--label-col", default=None, help="Column to colour points by.")
@click.option("--supervised", is_flag=True, help="UMAP only — fit using the labels.")
@click.option("--json", "as_json", is_flag=True, help="Emit result as JSON.")
@click.option(
    "--dry-run", is_flag=True, help="Print what would be written, do nothing."
)
@click.option(
    "-y", "--yes", is_flag=True, help="Skip confirmation (no-op; never prompts)."
)
def reduce_dimensions_cmd(
    data, output, method, label_col, supervised, as_json, dry_run, yes
):
    """Project the feature matrix in DATA to 2-D (PCA/UMAP) and save a figure.

    \b
    Examples:
      $ scitex-ml reduce-dimensions features.csv -o pca.png --label-col target
      $ scitex-ml reduce-dimensions features.csv -o umap.png --method umap
    """
    del yes  # accepted for §2 universal-flag parity; this verb never prompts
    if dry_run:
        click.echo(
            f"DRY RUN — would read {data} and write a {method} figure to {output}"
        )
        return
    try:
        out = _analysis.run_reduce(
            data,
            output,
            method=method,
            label_col=label_col,
            supervised=supervised,
        )
    except _analysis.AnalysisError as e:
        _fail(str(e))

    if as_json:
        click.echo(_json.dumps(out, indent=2))
        return
    click.secho(f"{method.upper()} figure written.", fg="green", bold=True)
    click.echo(f"  figure_path : {out['figure_path']}")
    click.echo(
        f"  shape       : {out['n_samples']} samples × {out['n_features']} features"
    )


# EOF
