# Sphinx configuration for Pretrain Experiments documentation

project = "Pretrain Experiments"
author = "Sebastian Bordt"
copyright = "2025, Sebastian Bordt"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "Pretrain Experiments"

html_theme_options = {
    "source_repository": "https://github.com/sbordt/pretrain-experiments",
    "source_branch": "main",
    "source_directory": "docs/",
}

myst_heading_anchors = 3
