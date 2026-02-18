#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find mismatched opening/closing tags in MDX files."""

import argparse
import re
from pathlib import Path


def check_file(filepath: Path) -> list[str]:
    """Check a file for tag mismatches. Returns list of issues."""
    content = filepath.read_text()
    lines = content.split("\n")
    issues = []

    tag_stack: list[str] = []
    tag_pattern = re.compile(r"<(/?)(\w+)(?:\s|>|$)")

    for line_num, line in enumerate(lines, 1):
        for match in tag_pattern.finditer(line):
            is_closing = match.group(1) == "/"
            tag_name = match.group(2)

            known_tags = {
                "Tabs", "Tab", "Cards", "Card", "Accordion",
                "Note", "Warning", "Tip", "Info",
            }
            if tag_name not in known_tags:
                continue

            if is_closing:
                if not tag_stack:
                    issues.append(
                        f"Line {line_num}: Closing </{tag_name}> without opening tag"
                    )
                else:
                    expected = tag_stack.pop()
                    if expected != tag_name:
                        issues.append(
                            f"Line {line_num}: Closing </{tag_name}> but expected "
                            f"</{expected}>"
                        )
            else:
                if "/>" not in line[match.start() :]:
                    tag_stack.append(tag_name)

    if tag_stack:
        issues.append(f"Unclosed tags at end of file: {tag_stack}")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find mismatched tags in MDX files"
    )
    parser.add_argument(
        "pages_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Path to pages directory (default: fern/v0.5.0/pages)",
    )
    args = parser.parse_args()

    if args.pages_dir is not None:
        pages_dir = args.pages_dir.resolve()
    else:
        pages_dir = Path(__file__).resolve().parent.parent / "v0.5.0" / "pages"

    if not pages_dir.exists():
        raise SystemExit(f"Error: pages directory not found at {pages_dir}")

    files_with_issues: list[tuple[Path, list[str]]] = []
    for mdx_file in sorted(pages_dir.rglob("*.mdx")):
        issues = check_file(mdx_file)
        if issues:
            rel_path = mdx_file.relative_to(pages_dir)
            files_with_issues.append((rel_path, issues))

    if files_with_issues:
        print(f"Found issues in {len(files_with_issues)} files:\n")
        for rel_path, issues in files_with_issues:
            print(f"  {rel_path}")
            for issue in issues:
                print(f"    - {issue}")
            print()
    else:
        print("No tag mismatches found!")


if __name__ == "__main__":
    main()
