"""Runtime cross-package import gate (PS-140) for scitex-ml.

Every module listed here is imported by this package's source but OWNED by
a peer standalone. A rename or move on the other side of that boundary is
invisible to this package's unit tests and surfaces as ModuleNotFoundError
in a user's process — which is how the `scitex_io._load_cache` rename went
undetected for weeks.

Regenerate the LIST with:

    scitex-dev ecosystem install-cross-package-gate scitex-ml --force

Do not hand-edit the list: the PS-140 auditor recomputes it from source on
every run and fails on any drift in either direction (a missing entry OR a
stale one). Hand-written cases go BELOW the closing sentinel; everything
between the sentinels is overwritten on regeneration.

Three outcomes, and the distinction between the last two is the whole
point of the gate:

- Peer distribution installed AND the full dotted path imports → PASSES.
- Peer distribution installed BUT the submodule was renamed/moved →
  FAILS loudly (`importlib.import_module` on the FULL path).
- Peer distribution NOT installed at all (lean install, optional extra,
  marker-gated dependency) → SKIPPED, via `pytest.importorskip` on the
  ROOT package only.

Skipping on the ROOT is what keeps a legitimate absence from becoming a
hard failure. Skipping on the FULL path is what would turn a rename into
an absence and report it green — the exact failure this gate exists to
catch.
"""

import importlib

import pytest

# ===== AUTO-GENERATED: cross-package imports =====
# NOTE: scitex_ml.* self-imports are deliberately OMITTED. The upstream
# PS-140 auditor in scitex-dev v0.12.3 disagrees with itself on whether
# self-imports belong in the cross-package gate:
#   - In a non-canonical checkout dir (e.g. worktree `scitex-ml-fix-skip/`)
#     `_own_import_name` derives `scitex_ml_fix_skip` and flags every
#     `scitex_ml.<sub>` reference as "missing from gate"
#   - In the canonical CI checkout `scitex-ml/` it derives `scitex_ml` and
#     flags those same entries as "stale in gate"
# CI is the source of truth → keep self-imports OUT of the gate.
# Upstream fix needed: `_own_import_name` should use the distribution name
# from the ECOSYSTEM registry, not the repo directory basename.
CROSS_PACKAGE_IMPORTS = [
    "scitex",
    "scitex_context",
    "scitex_dev",
    "scitex_dev._cli._completion",
    "scitex_io",
    "scitex_logging",
    "scitex_plt",
    "scitex_plt.colors",
    "scitex_repro",
    "scitex_seizure_metrics",
    "scitex_str",
    "scitex_types",
]
# ===== END AUTO-GENERATED =====


@pytest.mark.parametrize("module_name", CROSS_PACKAGE_IMPORTS)
def test_cross_package_import_succeeds_for_declared_dependency(module_name):
    # Arrange — skip on the ROOT, and only on the ROOT. PS-140's own
    # prose: banning the skip outright "would convert a legitimate
    # absence into a hard failure — a gate that cannot PASS, in place
    # of one that cannot FAIL." A lean install where a peer
    # distribution is genuinely absent must SKIP here, not fail.
    #
    # Two statements ON PURPOSE. The intermediate binding is what
    # makes the root/full-path distinction visible to a reader, which
    # is the entire point of the shape; inlining it to satisfy a
    # checker would make this file harder to read.
    root = module_name.split(".")[0]
    pytest.importorskip(root)

    # Act — a real import of the FULL dotted path. Not
    # importlib.util.find_spec, which only proves a module is
    # FINDABLE while the failures this gate exists to catch (a
    # renamed symbol re-exported through a package __init__) happen
    # at EXECUTION. And not importorskip(module_name), which skips on
    # the full path and so reports the rename as an absence.
    module = importlib.import_module(module_name)

    # Assert
    assert module is not None
