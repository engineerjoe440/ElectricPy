################################################################################
"""Configuration file for the Sphinx documentation builder."""
################################################################################

import os
import re
import sys

print("Build with:", sys.version)
parent_dir = os.path.dirname(os.getcwd())
initfile = os.path.join(parent_dir, 'electricpy', 'version.py')
sys.path.insert(0, parent_dir)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
print(parent_dir)

# Generate all Documentation Images
from render_images import main as render_images
render_images()

# Gather Version Information from Python File
with open(initfile) as fh:
    file_str = fh.read()
    name = re.search('NAME = \"(.*)\"', file_str).group(1)
    ver = re.search('VERSION = \"(.*)\"', file_str).group(1)
    # Version Breakdown:
    # MAJOR CHANGE . MINOR CHANGE . MICRO CHANGE
    print("Sphinx HTML Build For:", name,"   Version:", ver)


# Verify Import
try:
    import electricpy
except:
    print("Couldn't import `electricpy` module!")
    sys.exit(9)


# -- Project information -----------------------------------------------------

project = 'electricpy'
copyright = '2022, Joe Stanley'
author = 'Joe Stanley'

# The full version, including alpha/beta/rc tags
release = ver


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'numpydoc',
    'sphinx_git',
    'myst_parser',
    'sphinx_immaterial',
]
autosummary_generate = True
numpydoc_show_class_members = False

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "requirements.txt"]

templates_path = ["_templates"]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_immaterial'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['static']
html_extra_path = ['extra']
html_logo="static/ElectricpyLogo.svg"
html_favicon="static/ElectricpyLogo.svg"
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "searchbox.html"]
}

# github_repo = "electricpy"
# github_user = "engineerjoe440"
# github_button = True

# Material theme options (see theme.conf for more information)
html_theme_options = {

    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    'site_url': 'https://electricpy.readthedocs.io/en/latest/',

    # Set the color and the accent color
    "palette": [
        {
            "primary": "light-blue",
            "accent": "blue",
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "toggle": {
                "icon": "material/toggle-switch-off-outline",
                "name": "Switch to dark mode",
            }
        },
        {
            "primary": "blue",
            "accent": "light-blue",
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "toggle": {
                "icon": "material/toggle-switch",
                "name": "Switch to light mode",
            }
        },
    ],

    # Set the repo location to get a badge with stats
    'repo_url': 'https://github.com/engineerjoe440/ElectricPy/',
    'repo_name': 'ElectricPy',

    "icon": {
        "repo": "fontawesome/brands/github",
        "logo": "material/library",
    },
}
