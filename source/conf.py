import os
import sys
import pandas
import numpy
import typing
import mplfinance
sys.path.insert(0, os.path.abspath('../../code'))




project = 'mplfinance'
html_logo = 'image/grafic.png'

copyright = '2024, Rey'
author = 'Rey'

# The full version, including alpha/beta/rc tags
release = '1.0'



import pydata_sphinx_theme

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autodoc.typehints',
    'sphinx_copybutton',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.githubpages',
]


autosectionlabel_prefix_document = True
source_encoding = 'utf-8-sig'
autodoc_member_order = 'bysource'
napoleon_google_docstring = True
napoleon_numpy_docstring = True


templates_path = ['_templates']
exclude_patterns = []

language = 'ru'


html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "logo": {
        "image_light": "image/grafic.png",
        "image_dark": "image/grafic.png",
    },
    "navbar_align": "left",
    "navigation_depth": 4,
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Reyqq/mplf_line_trend",
            "icon": "fab fa-github-square",
        },
    ],
    "use_edit_page_button": True,
    "show_nav_level": 2,
    "collapse_navigation": False,
}



html_context = {
    "github_user": "Reyqq",
    "github_repo": "mplf_line_trend",
    "github_version": "master",
    "doc_path": "docs/source/",
}

master_doc = 'index'

html_static_path = ['source/_static']