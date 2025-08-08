# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = "NeMo-RL"
copyright = "2025, NVIDIA Corporation"
author = "NVIDIA Corporation"
release = "latest"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # For our markdown docs
    "autodoc2",  # Generates API docs
    "sphinx.ext.viewcode",  # For adding a link to view source code in docs
    "sphinx.ext.doctest",  # Allows testing in docstrings
    "sphinx.ext.napoleon",  # For google style docstrings
    "sphinx_copybutton",  # For copy button in code blocks
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for MyST Parser (Markdown) --------------------------------------
# MyST Parser settings
myst_enable_extensions = [
    "dollarmath",  # Enables dollar math for inline math
    "amsmath",  # Enables LaTeX math for display mode
    "colon_fence",  # Enables code blocks using ::: delimiters instead of ```
    "deflist",  # Supports definition lists with term: definition format
    "fieldlist",  # Enables field lists for metadata like :author: Name
    "tasklist",  # Adds support for GitHub-style task lists with [ ] and [x]
]
myst_heading_anchors = 5  # Generates anchor links for headings up to level 5

# -- Options for Autodoc2 ---------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

autodoc2_packages = [
    "../nemo_rl",  # Path to your package relative to conf.py
]
autodoc2_render_plugin = "myst"  # Use MyST for rendering docstrings
autodoc2_output_dir = "apidocs"  # Output directory for autodoc2 (relative to docs/)
# This is a workaround that uses the parser located in autodoc2_docstrings_parser.py to allow autodoc2 to
# render google style docstrings.
# Related Issue: https://github.com/sphinx-extensions2/sphinx-autodoc2/issues/33
autodoc2_docstring_parser_regexes = [
    (r".*", "docs.autodoc2_docstrings_parser"),
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nvidia_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA-NeMo/RL",
            "icon": "fa-brands fa-github",
        }
    ],
    "switcher": {
        "json_url": "../versions1.json",
        "version_match": release,
    },
    "extra_head": {
        """
    <script src="https://assets.adobedtm.com/5d4962a43b79/c1061d2c5e7b/launch-191c2462b890.min.js" ></script>
    """
    },
    "extra_footer": {
        """
    <script type="text/javascript">if (typeof _satellite !== "undefined") {_satellite.pageBottom();}</script>
    """
    },
}
html_extra_path = ["project.json", "versions1.json"]

# -- Supporting rendering GitHub alerts correctly ----------------------------
# https://github.com/executablebooks/MyST-Parser/issues/845

_GITHUB_ADMONITIONS = {
    "> [!NOTE]": "note",
    "> [!TIP]": "tip",
    "> [!IMPORTANT]": "important",
    "> [!WARNING]": "warning",
    "> [!CAUTION]": "caution",
}


def convert_gh_admonitions(app, relative_path, parent_docname, contents):
    # loop through content lines, replace github admonitions
    for i, orig_content in enumerate(contents):
        orig_line_splits = orig_content.split("\n")
        replacing = False
        for j, line in enumerate(orig_line_splits):
            # look for admonition key
            line_roi = line.lstrip()
            for admonition_key in _GITHUB_ADMONITIONS:
                if line_roi.startswith(admonition_key):
                    line = line.replace(
                        admonition_key,
                        "```{" + _GITHUB_ADMONITIONS[admonition_key] + "}",
                    )
                    # start replacing quotes in subsequent lines
                    replacing = True
                    break
            else:  # no break
                if not replacing:
                    continue
                # remove GH directive to match MyST directive
                # since we are replacing on the original line, this will preserve the right indent, if any
                if line_roi.startswith("> "):
                    line = line.replace("> ", "", 1)
                elif line_roi.rstrip() == ">":
                    line = line.replace(">", "", 1)
                else:
                    # missing "> ", so stop replacing and terminate directive
                    line = f"```\n{line}"
                    replacing = False
            # swap line back in splits
            orig_line_splits[j] = line
        # swap line back in original
        contents[i] = "\n".join(orig_line_splits)


def setup(app):
    app.connect("include-read", convert_gh_admonitions)
