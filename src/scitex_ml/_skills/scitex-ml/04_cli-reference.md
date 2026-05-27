---
description: |
  [TOPIC] scitex-ml CLI reference
  [DETAILS] Every subcommand with concrete examples — analysis verbs
  (compute-metrics, generate-report, reduce-dimensions), introspection
  (list-python-apis), the mcp and skills groups, and shell completion.
  Plus --help-recursive for the full tree.
tags: [scitex-ml-cli-reference]
---

# CLI Reference

`scitex-ml` exposes the stateless analysis surface from the terminal. Run
`scitex-ml --help` for the live list or `scitex-ml --help-recursive` to dump
help for every subcommand at once.

## Top-level

```bash
scitex-ml --version            # scitex-ml/X.Y.Z
scitex-ml --help               # categorized command list
scitex-ml --help-recursive     # all commands + subcommands
scitex-ml --json compute-metrics preds.csv   # propagate JSON intent
```

## Analysis verbs

Each verb is file-in → JSON/artifact-out and mirrors an MCP tool exactly
(see [05_mcp-tools.md](05_mcp-tools.md)).

```bash
# Metrics from a saved predictions table (y_true,y_pred[,y_proba])
scitex-ml compute-metrics preds.csv
scitex-ml compute-metrics preds.csv --label neg --label pos --json

# Full ClassificationReporter report (summary.json + plots)
scitex-ml generate-report preds.csv -o ./report
scitex-ml generate-report preds.csv -o ./report --dry-run

# PCA / UMAP 2-D projection figure
scitex-ml reduce-dimensions features.csv -o pca.png --label-col target
scitex-ml reduce-dimensions features.csv -o umap.png --method umap --supervised
```

## Introspection

```bash
scitex-ml list-python-apis          # public API names
scitex-ml list-python-apis -v       # + signatures
scitex-ml list-python-apis -vv      # + one-line docstrings
scitex-ml list-python-apis --json
```

## Integration — MCP server

```bash
scitex-ml mcp start                 # stdio MCP server (needs [mcp] extra)
scitex-ml mcp doctor                # health check (deps, tool count)
scitex-ml mcp list-tools -vv        # tools with parameters
scitex-ml mcp install --json        # MCP-host config snippet
```

## Integration — skills

```bash
scitex-ml skills list               # bundled skill pages
scitex-ml skills get 02_quick-start
scitex-ml skills install            # → ~/.scitex/dev/skills/scitex-ml/
scitex-ml skills install --claude-symlink
```

## Shell completion

```bash
scitex-ml install-shell-completion --shell bash   # writes the rc source line
scitex-ml print-shell-completion --shell bash     # print without writing
```

## See also

- [03_python-api.md](03_python-api.md) — same surface in Python
- [05_mcp-tools.md](05_mcp-tools.md) — the agent-callable MCP tools
- Full RTD: <https://scitex-ml.readthedocs.io/en/latest/cli.html>
