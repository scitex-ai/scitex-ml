"""Sphinx configuration for scitex-ml documentation."""

import logging as _stdlib_logging
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))


class _SuppressDuplicateObjectDescription(_stdlib_logging.Filter):
    """Silence Sphinx's "duplicate object description of <name>" warnings.

    These fire when scitex_ml's package-level __init__.py re-exports
    classes from submodules (e.g. ``TimeSeriesStrategy``) and autodoc
    documents both the re-exported name and the original definition in
    the same generated stub. Sphinx emits these via
    ``logger.warning(...)`` with no ``type=`` parameter, so they cannot
    be filtered via the standard ``suppress_warnings`` config. We filter
    at the logging layer so ``sphinx-build -W`` doesn't fail PR builds
    on this cosmetic artefact — genuine warnings still surface.
    """

    def filter(self, record):  # type: ignore[override]
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return "duplicate object description of" not in msg


_stdlib_logging.getLogger("sphinx").addFilter(_SuppressDuplicateObjectDescription())

# -- Project information -----------------------------------------------------

project = "SciTeX ML"
copyright = "2024-2026, Yusuke Watanabe"
author = "Yusuke Watanabe"

try:
    from importlib.metadata import version as _get_version

    release = _get_version("scitex-ml")
except Exception:
    release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx_rtd_theme",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Heavy / optional deps live in [heavy] extra — mock them so RTD builds
# without installing torch / catboost / optuna / pytorch_pretrained_vit.
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "torchaudio",
    "torchsummary",
    "pytorch_pretrained_vit",
    "optuna",
    "psutil",
    "catboost",
    "umap",
    "fastmcp",
    # Peer-only optional sibling — scitex_ml.metrics.seizure re-exports
    # from this module. It is published as scitex-seizure-metrics on
    # PyPI but the docs runner installs only [docs]; mock so the
    # recursive autosummary can document the leaf without it.
    "scitex_seizure_metrics",
]

autosummary_generate = True

# Suppress the autosummary import-failure warning (treated as -W error
# on PR builds) for legacy/vendored submodules that aren't valid
# importable modules — currently the bundled Ranger optimizer's
# `setup.py` shows up under autosummary's recursive walk. Also covers
# residual docutils/autodoc cosmetic warnings in legacy docstrings.
suppress_warnings = [
    "autosummary",
    "autodoc",
    "docutils",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "prev_next_buttons_location": "bottom",
}

html_static_path = ["_static"]
html_title = f"{project} v{release}"
html_short_title = project

html_context = {
    "display_github": True,
    "github_user": "ywatanabe1989",
    "github_repo": "scitex-ml",
    "github_version": "main",
    "conf_py_path": "/docs/sphinx/",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "tasklist",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
