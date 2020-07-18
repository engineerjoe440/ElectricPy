# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys
print("Build with:", sys.version)
cwd = os.getcwd()
initfile = os.path.join(os.path.dirname(cwd),'electricpy','__init__.py')
# Gather Version Information from Python File
with open(initfile) as fh:
    file_str = fh.read()
    name = re.search('_name_ = \"(.*)\"', file_str).group(1)
    ver = re.search('_version_ = \"(.*)\"', file_str).group(1)
    # Version Breakdown:
    # MAJOR CHANGE . MINOR CHANGE . MICRO CHANGE
    print("Sphinx HTML Build For:",name,"   Version:",ver)


# Verify Import
try:
    import electricpy
except:
    print("Couldn't import `electricpy` module!")
    sys.exit(9)


# -- Project information -----------------------------------------------------

project = 'electricpy'
copyright = '2020, Joe Stanley'
author = 'Joe Stanley'

# The full version, including alpha/beta/rc tags
release = ver


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [  'sphinx.ext.autodoc',
                'sphinx.ext.napoleon',
                'sphinx.ext.mathjax',
                'sphinx.ext.autosummary',
                'numpydoc',
                'sphinx_sitemap',
]
autosummary_generate = True
numpydoc_show_class_members = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['static']
html_logo="static/ElectricpyLogo.svg"
html_favicon="static/ElectricpyLogo.svg"
html_baseurl="https://engineerjoe440.github.io/ElectricPy/html/"

github_repo = "electricpy"
github_user = "engineerjoe440"
github_button = True