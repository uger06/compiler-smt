# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys

sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'demo-test'
copyright = '2024, uger'
author = 'uger'
release = '0.0.1beta'
version = '0.0.1'
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# These can be extensions bundled with Sphinx (named sphinx.ext.*) or custom first-party or third-party extensions.
extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.autosummary',
  'sphinx.ext.intersphinx',
  'sphinx.ext.mathjax',
  'sphinx.ext.napoleon',
  'sphinx.ext.viewcode',
  'sphinx_autodoc_typehints',
  'myst_nb',
  'matplotlib.sphinxext.plot_directive',
  'sphinx_thebe',
  'sphinx_design'
#   'sphinx-mathjax-offline',
]

language = 'zh_CN'
templates_path = ['_templates']
source_suffix = ['.rst', '.ipynb', '.md']

# This sets the name of the document containing the master toctree directive, and hence the root of the entire tree.
master_doc = 'index'

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.8", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}

suppress_warnings = ["ref.ref"]

# A list of glob-style patterns that should be excluded when looking for source files. 
exclude_patterns = ['_build']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output



html_theme = 'sphinx_rtd_theme'
html_copy_source = True
# html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "BPU-SDK documentation"
# A list of paths that contain custom static files (such as style sheets or script files).
html_static_path = ['_static']
jupyter_execute_notebooks = "off"
thebe_config = {
    "repository_url": "https://github.com/binder-examples/jupyter-stacks-datascience",
    "repository_branch": "master",
}