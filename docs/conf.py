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
import urllib.parse
from pathlib import Path
from typing import Any

import git
from docutils import nodes
from docutils.transforms import Transform
from sphinx import addnodes
from sphinx.application import Sphinx

project = "NeMo-RL"
copyright = "2025, NVIDIA Corporation"
author = "NVIDIA Corporation"
release = "latest"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # For our markdown docs
    # "autodoc2" - Added conditionally below based on package availability
    "sphinx.ext.viewcode",  # For adding a link to view source code in docs
    "sphinx.ext.doctest",  # Allows testing in docstrings
    "sphinx.ext.napoleon",  # For google style docstrings
    "sphinx_copybutton",  # For copy button in code blocks,
    "sphinx_design",  # For grid layout
    "sphinx.ext.ifconfig",  # For conditional content
    "sphinxcontrib.mermaid",  # For Mermaid diagrams
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


# Github links are now getting rate limited from the Github Actions
linkcheck_ignore = [
    ".*github\\.com.*",
    ".*githubusercontent\\.com.*",
]


def _convert_gh_admonitions_inplace(contents: list[str]) -> None:
    """Mutate contents to convert GitHub blockquote admonitions to MyST.

    https://github.com/executablebooks/MyST-Parser/issues/845
    """
    _github_admonitions = {
        "> [!NOTE]": "note",
        "> [!TIP]": "tip",
        "> [!IMPORTANT]": "important",
        "> [!WARNING]": "warning",
        "> [!CAUTION]": "caution",
    }
    # Use 8 backticks for admonition fences to allow code blocks with 3 or 6 backticks inside
    FENCE = "````````"
    for i, orig_content in enumerate(contents):
        orig_line_splits = orig_content.split("\n")
        replacing = False
        for j, line in enumerate(orig_line_splits):
            line_roi = line.lstrip()
            for admonition_key, admonition_name in _github_admonitions.items():
                if line_roi.startswith(admonition_key):
                    replacement = f"{FENCE}{{{admonition_name}}}"
                    if replacing:
                        # Close previous fence before starting new admonition
                        # Add blank line between admonitions for proper MyST parsing
                        line = (
                            f"{FENCE}\n\n{line.replace(admonition_key, replacement, 1)}"
                        )
                    else:
                        line = line.replace(admonition_key, replacement, 1)
                    replacing = True
                    break
            else:
                if not replacing:
                    continue
                if line_roi.startswith("> "):
                    line = line.replace("> ", "", 1)
                elif line_roi.rstrip() == ">":
                    line = line.replace(">", "", 1)
                else:
                    line = f"{FENCE}\n{line}"
                    replacing = False
            orig_line_splits[j] = line
        if replacing:
            orig_line_splits.append(FENCE)
            replacing = False
        contents[i] = "\n".join(orig_line_splits)


def _convert_gh_admonitions(
    _app: Sphinx, _relative_path: Path, _parent_docname: str, contents: list[str]
) -> None:
    _convert_gh_admonitions_inplace(contents)


def _convert_gh_admonitions_source(
    _app: Sphinx, _docname: str, source: list[str]
) -> None:
    # Sphinx "source-read" event
    _convert_gh_admonitions_inplace(source)


class _GitHubLinkTransform(Transform):
    """Converting the relative path to a file in a Markdown to the URL of that file on GitHub."""

    default_priority = 500  # type: ignore[bad-override]

    @staticmethod
    def _get_github_source_url(repo: git.Repo) -> str:
        # Find out which remote GitHub repo should be the source.
        if "origin" in repo.remotes:
            url = repo.remotes.origin.url
        elif len(repo.remotes) == 1:
            url = repo.remotes[0].url
        else:
            raise ValueError(
                "Cannot determine which remote repo on GitHub this local repo is from."
            )
        # Canonicalize the URL.
        if url.startswith("git@github.com:"):
            url = url.replace("git@github.com:", "https://github.com/", 1)
        if url.endswith(".git"):
            url = url[: -len(".git")]
        return url

    def apply(self, **kwargs: Any) -> None:  # type: ignore[bad-override]
        try:
            local_repo = git.Repo(search_parent_directories=True)
            # Hardcode github url for now due to errors when building in a different environment
            # remote_repo_url = self._get_github_source_url(local_repo)
            remote_repo_url = "https://github.com/NVIDIA-NeMo/RL"
        except Exception:
            # Cannot figure out which source url it should be; leave links as-is.
            return
        if local_repo.working_tree_dir is None:
            # If the local repo is a bare repo, the method below won't work.
            return
        wt_dir = local_repo.working_tree_dir

        for node in self.document.traverse(addnodes.download_reference):
            md_dir = Path(node["refdoc"]).parent
            dst_path = md_dir / Path(node["reftarget"])
            try:
                dst_path = dst_path.resolve(strict=True)
            except OSError:
                # If the path doesn't exist or a symlink loop is encountered.
                continue
            if dst_path.is_file():
                kind = "blob"
            elif dst_path.is_dir():
                kind = "tree"
            else:
                # Cannot figure out what type of thing this path is pointing to.
                continue
            refuri = "/".join(
                (
                    remote_repo_url.rstrip("/"),
                    kind,
                    local_repo.head.object.hexsha,
                    urllib.parse.quote(dst_path.relative_to(wt_dir).as_posix()),
                )
            )
            new_node = nodes.reference(rawsource=node.rawsource, refuri=refuri)
            # Preserve styling and title if present.
            if "classes" in node:
                new_node["classes"] = list(node["classes"])
            if "title" in node:
                new_node["title"] = node["title"]
            if node.children:
                new_node += node.children
            node.replace_self(new_node)


def setup(app: Sphinx) -> None:
    app.add_transform(_GitHubLinkTransform)
    # Convert GH admonitions for included files and top-level sources
    app.connect("include-read", _convert_gh_admonitions)
    app.connect("source-read", _convert_gh_admonitions_source)
