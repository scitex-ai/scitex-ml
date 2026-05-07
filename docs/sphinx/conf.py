"""Sphinx configuration for scitex-ml documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

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
]

autosummary_generate = True

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
