# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

package_path = os.path.abspath('../..')
infairness_path = os.path.join(package_path, 'inFairness')

sys.path.insert(0, package_path)


# -- Project information -----------------------------------------------------

project = 'inFairness'
copyright = '2022, IBM Research'
author = 'IBM Research'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinxcontrib.apidoc',
    'sphinx_design',
    'myst_parser',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinxcontrib.gtagjs'
]

# API doc configuration
apidoc_module_dir = infairness_path
# apidoc_output_dir = ''
apidoc_excluded_paths = []
apidoc_separate_modules = True

# Google analytics tracking
gtagjs_ids = [
    'G-3QDFV4L7YB',
]


# Napolean docstring configuration
napoleon_numpy_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Markdown and ReST parser extension configuration
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_theme_options = {
    "sidebar_hide_name": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = "_static/imgs/infairness-logo.png"

html_title = project


intersphinx_mapping = {
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'torch': ('https://pytorch.org/docs/1.9.0/', None)
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
