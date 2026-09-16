# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural checks for the checked-in unified diffs, independent of the Gym pin.

``stage_gym.py`` applies these patches in sorted order with GNU ``patch``. A hunk
whose body disagrees with its header count is rejected by every patch tool, so a
single stripped blank context line breaks staging after earlier patches have
already been applied. Catch that here, without the pinned Gym tree.
"""

from pathlib import Path
import re
import shutil
import subprocess

import pytest

PATCHES = sorted(
    (Path(__file__).resolve().parents[3] / "tools/super_rl/patches").glob("*.patch")
)
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
FILE_HEADER_PREFIXES = ("diff ", "index ", "--- ", "+++ ", "@@")


def hunk_count_errors(text: str) -> list[str]:
    """Return one message per hunk whose body does not match its header counts."""
    errors: list[str] = []
    lines = text.split("\n")
    index = 0
    while index < len(lines):
        header = HUNK_HEADER.match(lines[index])
        if header is None:
            index += 1
            continue
        start = index + 1
        old = int(header.group(1) or 1)
        new = int(header.group(2) or 1)
        index += 1
        while (old > 0 or new > 0) and index < len(lines):
            line = lines[index]
            if line.startswith("\\"):
                pass  # "\ No newline at end of file" marker
            elif line.startswith(" "):
                old -= 1
                new -= 1
            elif line.startswith("-"):
                old -= 1
            elif line.startswith("+"):
                new -= 1
            elif line == "":
                errors.append(
                    f"hunk at line {start}: blank context line {index + 1} lost its "
                    "leading space (trailing-whitespace stripping)"
                )
                break
            else:
                break
            index += 1
        if old != 0 or new != 0:
            errors.append(
                f"hunk at line {start}: body is shorter than its header "
                f"(missing old={old}, new={new})"
            )
            continue
        if index < len(lines):
            following = lines[index]
            if following and not following.startswith(FILE_HEADER_PREFIXES):
                if following[0] in " -+":
                    errors.append(
                        f"hunk at line {start}: body continues past its header count"
                    )
    return errors


def test_patch_assets_are_present():
    assert PATCHES, "no patch assets found"


@pytest.mark.parametrize("patch", PATCHES, ids=lambda path: path.name)
def test_every_hunk_matches_its_header(patch: Path):
    text = patch.read_text()
    assert text.endswith("\n"), "patch must end with a newline"
    assert hunk_count_errors(text) == []


@pytest.mark.parametrize("patch", PATCHES, ids=lambda path: path.name)
def test_git_accepts_patch_structure(patch: Path):
    if shutil.which("git") is None:
        pytest.skip("git is required for the structural oracle")
    # --stat parses the whole file without needing the target tree.
    result = subprocess.run(
        ["git", "apply", "--stat", str(patch)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_detects_stripped_blank_context_and_short_hunks():
    good = "--- a/f\n+++ b/f\n@@ -1,3 +1,4 @@\n a\n \n+b\n c\n"
    assert hunk_count_errors(good) == []
    stripped = good.replace("\n \n", "\n\n")
    assert any("lost its leading space" in e for e in hunk_count_errors(stripped))
    short = "--- a/f\n+++ b/f\n@@ -1,3 +1,4 @@\n a\n+b\n c\n"
    assert any("shorter than its header" in e for e in hunk_count_errors(short))
