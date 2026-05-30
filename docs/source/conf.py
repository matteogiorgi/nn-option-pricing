"""Sphinx configuration for the option pricing project documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "Neural Network Option Pricing"
author = "Project group"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "pydata_sphinx_theme"
html_static_path: list[str] = []
html_title = "Neural Network Option Pricing"

html_theme_options = {
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "collapse_navigation": False,
    "navbar_align": "left",
}
