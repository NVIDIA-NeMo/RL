#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Sync megatron-bridge dependency-metadata in pyproject.toml.

The root pyproject.toml carries a [[tool.uv.dependency-metadata]] section for
megatron-bridge (needed because it is a non-workspace path dep that cannot
depend on workspace members like megatron-core).  The source of truth is
3rdparty/Megatron-Bridge-workspace/setup.py::CACHED_DEPENDENCIES.

Usage:
    python tools/check_mbridge_deps.py          # auto-update pyproject.toml
    python tools/check_mbridge_deps.py --check   # CI mode: exit 1 if out of sync
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent.parent
SETUP_PY = ROOT / "3rdparty" / "Megatron-Bridge-workspace" / "setup.py"
PYPROJECT = ROOT / "pyproject.toml"

# Deps intentionally omitted from the metadata because they are workspace
# members (uv name-shadowing restriction).
OMITTED_WORKSPACE_DEPS = {"megatron-core"}


def _normalize(dep: str) -> str:
    """Lowercase, strip whitespace, and collapse spaces."""
    return dep.strip().lower().replace(" ", "")


def _strip_extras(dep: str) -> str:
    """Return just the package name, dropping extras/version specifiers."""
    name = (
        dep.split("[")[0]
        .split(">")[0]
        .split("<")[0]
        .split("=")[0]
        .split("!")[0]
        .split(";")[0]
    )
    return name.strip().lower().replace("_", "-")


def extract_cached_dependencies() -> list[str]:
    """Parse CACHED_DEPENDENCIES from setup.py using AST."""
    source = SETUP_PY.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CACHED_DEPENDENCIES":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"CACHED_DEPENDENCIES not found in {SETUP_PY}")


def extract_metadata_requires_dist() -> list[str]:
    """Extract requires-dist for megatron-bridge from pyproject.toml dependency-metadata."""
    data = tomllib.loads(PYPROJECT.read_text())
    for entry in data.get("tool", {}).get("uv", {}).get("dependency-metadata", []):
        if entry.get("name") == "megatron-bridge":
            return entry.get("requires-dist", [])
    raise RuntimeError(
        "No [[tool.uv.dependency-metadata]] entry for megatron-bridge in pyproject.toml"
    )


def build_requires_dist(cached: list[str]) -> list[str]:
    """Build the requires-dist list from CACHED_DEPENDENCIES, omitting workspace deps."""
    result = []
    for dep in cached:
        if _strip_extras(dep) in OMITTED_WORKSPACE_DEPS:
            continue
        result.append(dep)
    return result


def is_in_sync(
    cached: list[str], metadata: list[str]
) -> tuple[bool, set[str], set[str]]:
    """Compare normalized dep sets. Returns (in_sync, missing, extra)."""
    expected = {
        _normalize(d) for d in cached if _strip_extras(d) not in OMITTED_WORKSPACE_DEPS
    }
    actual = {_normalize(d) for d in metadata}
    missing = expected - actual
    extra = actual - expected
    return (not missing and not extra), missing, extra


def format_requires_dist(deps: list[str], indent: str = "  ") -> str:
    """Format a list of deps as a TOML inline array (multi-line)."""
    lines = ["requires-dist = ["]
    for dep in deps:
        lines.append(f'{indent}"{dep}",')
    lines.append("]")
    return "\n".join(lines)


def update_pyproject(new_deps: list[str]) -> None:
    """Replace the requires-dist array in the megatron-bridge dependency-metadata section."""
    text = PYPROJECT.read_text()

    # Find the megatron-bridge dependency-metadata block.
    # Strategy: locate 'name = "megatron-bridge"', then find the requires-dist
    # array that follows it (before the next [[tool.uv.dependency-metadata]] or
    # [tool.*] section).
    pattern = re.compile(
        r"(# Must stay in sync with.*?CACHED_DEPENDENCIES\.\n)"
        r'(version = "[^"]*"\n)'
        r"requires-dist = \[\n(?:.*?\n)*?\]",
        re.DOTALL,
    )

    new_block = format_requires_dist(new_deps)

    replacement = r"\1\2" + new_block

    updated, count = pattern.subn(replacement, text, count=1)
    if count == 0:
        raise RuntimeError(
            "Could not locate the megatron-bridge requires-dist block in pyproject.toml. "
            "Expected a 'requires-dist = [...]' array following the "
            "'# Must stay in sync with...CACHED_DEPENDENCIES' comment."
        )

    PYPROJECT.write_text(updated)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: exit 1 if out of sync (for CI). Does not modify files.",
    )
    args = parser.parse_args()

    cached = extract_cached_dependencies()
    metadata = extract_metadata_requires_dist()
    in_sync, missing, extra = is_in_sync(cached, metadata)

    if in_sync:
        print("megatron-bridge dependency-metadata is in sync.")
        return 0

    if args.check:
        print(
            "ERROR: megatron-bridge dependency-metadata in pyproject.toml is out of sync.",
            file=sys.stderr,
        )
        print(f"  Source of truth: {SETUP_PY.relative_to(ROOT)}", file=sys.stderr)
        if missing:
            print(
                "  Missing from [[tool.uv.dependency-metadata]] (present in CACHED_DEPENDENCIES):",
                file=sys.stderr,
            )
            for dep in sorted(missing):
                print(f"    + {dep}", file=sys.stderr)
        if extra:
            print(
                "  Extra in [[tool.uv.dependency-metadata]] (not in CACHED_DEPENDENCIES):",
                file=sys.stderr,
            )
            for dep in sorted(extra):
                print(f"    - {dep}", file=sys.stderr)
        print(
            "  Run `python tools/check_mbridge_deps.py` to auto-fix.",
            file=sys.stderr,
        )
        return 1

    # Sync mode: update pyproject.toml
    new_deps = build_requires_dist(cached)
    update_pyproject(new_deps)
    print(
        f"Updated megatron-bridge dependency-metadata in {PYPROJECT.relative_to(ROOT)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
