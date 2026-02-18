#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quote frontmatter titles that contain colons (invalid unquoted YAML)."""

import argparse
import re
from pathlib import Path


def quote_title(filepath: Path) -> bool:
    """Quote title if it contains a colon. Returns True if changed."""
    content = filepath.read_text()

    if not content.strip().startswith("---"):
        return False

    # Match unquoted title with colon
    match = re.search(r"^title:\s+([^\"'\n]+)$", content, re.MULTILINE)
    if not match:
        return False

    title = match.group(1).strip()
    if ":" not in title or title.startswith('"') or title.startswith("'"):
        return False

    title_escaped = title.replace('\\', '\\\\').replace('"', '\\"')
    new_content = re.sub(
        rf"^title:\s+{re.escape(title)}\s*$",
        f'title: "{title_escaped}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )

    if new_content != content:
        filepath.write_text(new_content)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quote frontmatter titles with colons"
    )
    parser.add_argument("pages_dir", type=Path, help="Path to pages directory")
    args = parser.parse_args()

    pages_dir = args.pages_dir.resolve()
    if not pages_dir.exists():
        raise SystemExit(f"Error: pages directory not found at {pages_dir}")

    changed = []
    for mdx_file in sorted(pages_dir.rglob("*.mdx")):
        if quote_title(mdx_file):
            changed.append(mdx_file.relative_to(pages_dir))
            print(f"  Quoted: {mdx_file.relative_to(pages_dir)}")

    print(f"\nQuoted {len(changed)} titles")


if __name__ == "__main__":
    main()
