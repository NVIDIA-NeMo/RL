#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert RL-specific MyST/Sphinx syntax: {octicon}, {py:class}, {py:meth}."""

import argparse
import re
from pathlib import Path

API_DOCS_BASE = "https://docs.nvidia.com/nemo/rl/latest/apidocs"


def strip_octicon(content: str) -> str:
    """Remove {octicon}`icon` from text, leaving the rest (e.g. 'Overview')."""
    return re.sub(r"\{octicon\}`[^`]+`\s*", "", content)


def convert_py_roles(content: str) -> str:
    """Convert {py:class}`text` and {py:meth}`text` to inline code `text`."""
    # {py:class}`text` or {py:class}`text <module.path>`
    content = re.sub(
        r"\{py:class\}`([^`<]+)(?:\s*<[^>]+>)?`",
        r"`\1`",
        content,
    )
    content = re.sub(
        r"\{py:meth\}`([^`<]+)(?:\s*<[^>]+>)?`",
        r"`\1`",
        content,
    )
    return content


def convert_file(filepath: Path) -> bool:
    """Convert a single file. Returns True if changes were made."""
    content = filepath.read_text()
    original = content

    content = strip_octicon(content)
    content = convert_py_roles(content)

    if content != original:
        filepath.write_text(content)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RL-specific syntax (octicon, py:class, py:meth)"
    )
    parser.add_argument(
        "pages_dir",
        type=Path,
        help="Path to pages directory (e.g. fern/v0.5.0/pages)",
    )
    args = parser.parse_args()

    pages_dir = args.pages_dir.resolve()
    if not pages_dir.exists():
        raise SystemExit(f"Error: pages directory not found at {pages_dir}")

    changed = []
    for mdx_file in sorted(pages_dir.rglob("*.mdx")):
        if convert_file(mdx_file):
            changed.append(mdx_file.relative_to(pages_dir))
            print(f"  Converted: {mdx_file.relative_to(pages_dir)}")

    print(f"\nConverted {len(changed)} files")


if __name__ == "__main__":
    main()
