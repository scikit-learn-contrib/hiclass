# noqa
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
sys.path.insert(0, os.path.abspath('./../..'))
sys.path.insert(0, os.path.abspath('./../../hiclass'))
print(sys.path)

# -- Project information -----------------------------------------------------

project = 'hiclass'
copyright = '2022, Fabio Malcher Miranda, Niklas Köhnecke'
author = 'Fabio Malcher Miranda, Niklas Köhnecke'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinx_code_tabs',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
use_rtd_scheme = False
try:
    import sphinx_rtd_theme
    extensions.extend(["sphinx_rtd_theme"])
    use_rtd_scheme = True
except ImportError:
    print("sphinx_rtd_theme was not installed, using alabaster as fallback!")

html_theme = 'sphinx_rtd_theme' if use_rtd_scheme else 'alabaster'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# options

html_theme_options = {}

if not use_rtd_scheme:
    html_theme_options["sidebar_width"] = "230px"