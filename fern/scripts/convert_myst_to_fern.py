#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert MyST Markdown syntax to Fern MDX components.

Handles: admonitions, dropdowns, tab sets, grid cards, toctree removal,
HTML comments. Run convert_rl_specific.py first to strip {octicon} and {py:*} roles.
"""

import argparse
import re
from pathlib import Path


def convert_admonitions(content: str) -> str:
    """Convert MyST admonitions to Fern components."""
    admonition_map = {
        "note": "Note",
        "warning": "Warning",
        "tip": "Tip",
        "important": "Info",
        "seealso": "Note",
        "caution": "Warning",
        "danger": "Warning",
        "attention": "Warning",
        "hint": "Tip",
    }

    for myst_type, fern_component in admonition_map.items():
        pattern = rf"```\{{{myst_type}\}}\s*\n(.*?)```"
        replacement = rf"<{fern_component}>\n\1</{fern_component}>"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)

        pattern = rf":::\{{{myst_type}\}}\s*\n(.*?):::"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)

    return content


def convert_dropdowns(content: str) -> str:
    """Convert MyST dropdowns to Fern Accordion components."""
    pattern = r"```\{dropdown\}\s+([^\n]+)\s*\n(.*?)```"

    def replace_dropdown(match: re.Match[str]) -> str:
        title = match.group(1).strip()
        body = match.group(2).strip()
        if '"' in title:
            title = title.replace('"', "'")
        return f'<Accordion title="{title}">\n{body}\n</Accordion>'

    return re.sub(pattern, replace_dropdown, content, flags=re.DOTALL)


def convert_tab_sets(content: str) -> str:
    """Convert MyST tab sets to Fern Tabs components."""
    content = re.sub(r"::::+\s*\{tab-set\}\s*", "<Tabs>\n", content)
    content = re.sub(r"```\{tab-set\}\s*", "<Tabs>\n", content)

    def replace_tab_item(match: re.Match[str]) -> str:
        title = match.group(1).strip()
        return f'<Tab title="{title}">'

    content = re.sub(r"::::*\s*\{tab-item\}\s+([^\n]+)", replace_tab_item, content)
    content = re.sub(r":::*\s*\{tab-item\}\s+([^\n]+)", replace_tab_item, content)

    lines = content.split("\n")
    result = []
    in_tab = False

    for line in lines:
        if '<Tab title="' in line:
            if in_tab:
                result.append("</Tab>\n")
            in_tab = True
            result.append(line)
        elif line.strip() in [":::::", "::::", ":::", "</Tabs>"]:
            if in_tab and line.strip() != "</Tabs>":
                result.append("</Tab>")
                in_tab = False
            if line.strip() in [":::::", "::::"]:
                result.append("</Tabs>")
            else:
                result.append(line)
        else:
            result.append(line)

    content = "\n".join(result)
    content = re.sub(r"\n::::+\n", "\n", content)
    content = re.sub(r"\n:::+\n", "\n", content)
    return content


def convert_grid_cards(content: str) -> str:
    """Convert MyST grid cards to Fern Cards components."""
    content = re.sub(r"::::+\s*\{grid\}[^\n]*\n", "<Cards>\n", content)
    content = re.sub(r"```\{grid\}[^\n]*\n", "<Cards>\n", content)

    def replace_card(match: re.Match[str]) -> str:
        full_match = match.group(0)
        title_match = re.search(r"\{grid-item-card\}\s+(.+?)(?:\n|$)", full_match)
        title = title_match.group(1).strip() if title_match else "Card"

        link_match = re.search(r":link:\s*(\S+)", full_match)
        href = link_match.group(1) if link_match else ""

        if href and href != "apidocs/index":
            if not href.startswith("http"):
                href = "/" + href.replace(".md", "").replace(".mdx", "")
            return f'<Card title="{title}" href="{href}">'
        if href == "apidocs/index":
            return f'<Card title="{title}" href="https://docs.nvidia.com/nemo/rl/latest/apidocs/">'
        return f'<Card title="{title}">'

    content = re.sub(
        r"::::*\s*\{grid-item-card\}[^\n]*(?:\n:link:[^\n]*)?(?:\n:link-type:[^\n]*)?",
        replace_card,
        content,
    )
    content = re.sub(
        r":::*\s*\{grid-item-card\}[^\n]*(?:\n:link:[^\n]*)?(?:\n:link-type:[^\n]*)?",
        replace_card,
        content,
    )

    lines = content.split("\n")
    result = []
    in_card = False

    for line in lines:
        if '<Card title="' in line:
            if in_card:
                result.append("</Card>\n")
            in_card = True
            result.append(line)
        elif line.strip() in [":::::", "::::", ":::", "</Cards>"]:
            if in_card and line.strip() != "</Cards>":
                result.append("\n</Card>")
                in_card = False
            if line.strip() in [":::::", "::::"]:
                result.append("\n</Cards>")
        else:
            result.append(line)

    return "\n".join(result)


def remove_toctree(content: str) -> str:
    """Remove toctree blocks entirely."""
    content = re.sub(r"```\{toctree\}.*?```", "", content, flags=re.DOTALL)
    content = re.sub(r":::\{toctree\}.*?:::", "", content, flags=re.DOTALL)
    return content


def convert_html_comments(content: str) -> str:
    """Convert HTML comments to JSX comments."""
    return re.sub(r"<!--\s*(.*?)\s*-->", r"{/* \1 */}", content, flags=re.DOTALL)


def remove_directive_options(content: str) -> str:
    """Remove MyST directive options."""
    for opt in [
        ":icon:", ":class:", ":columns:", ":gutter:", ":margin:", ":padding:",
        ":link-type:", ":maxdepth:", ":titlesonly:", ":hidden:", ":link:",
        ":caption:",
    ]:
        content = re.sub(rf"\n{re.escape(opt)}[^\n]*", "", content)
    return content


def fix_malformed_tags(content: str) -> str:
    """Fix common malformed tag issues."""
    content = re.sub(r'title=""', 'title="Details"', content)
    content = re.sub(
        r"<(Note|Warning|Tip|Info)([^>]*)/>\s*\n([^<]+)",
        r"<\1\2>\n\3</\1>",
        content,
    )
    return content


def clean_multiple_newlines(content: str) -> str:
    """Clean up excessive newlines."""
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip() + "\n"


def convert_file(filepath: Path) -> bool:
    """Convert a single file. Returns True if changes were made."""
    content = filepath.read_text()
    original = content

    content = convert_admonitions(content)
    content = convert_dropdowns(content)
    content = convert_grid_cards(content)
    content = convert_tab_sets(content)
    content = remove_toctree(content)
    content = convert_html_comments(content)
    content = remove_directive_options(content)
    content = fix_malformed_tags(content)
    content = clean_multiple_newlines(content)

    if content != original:
        filepath.write_text(content)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MyST syntax to Fern MDX in pages directory"
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
