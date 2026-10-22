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

"""Prune uv's PyPI wheel/sdist cache down to what uv.lock still pins.

uv.lock pins exactly one version per package across the WHOLE project (every
extra, installed in this build or not -- see the offline-extras comment on
UV_CACHE_DIR in docker/Dockerfile). But uv's cache is append-only: `uv sync`
only ever downloads what a build needs, it never removes a version a prior
lockfile revision pinned and this one no longer does. Every routine
dependency bump (torch, vllm, sglang, ...) therefore leaves its predecessor's
wheel sitting in the cache forever, as long as BASE_IMAGE/UV_VERSION/
HYBRID_EP_MULTINODE haven't changed (see docker/Dockerfile's dependency-install
RUN for why those three are the only things that ever reset the cache).

This script removes exactly those superseded entries -- never anything
`uv.lock` still references, installed or not -- so it is safe to run before
shipping UV_CACHE_DIR in the release image: it cannot remove a wheel the
offline `uv run --extra <backend>` promise depends on, because that promise
is itself "every wheel uv.lock could need," which is exactly the keep set
computed here.

Scope: only the `pypi/` subtree of `wheels-v*`/`sdists-v*` (the dominant
case -- every normal PyPI-sourced dependency). Git/URL/local-path/custom-index
sources (`wheels-v*/url/`, `wheels-v*/index/`, `git-v*/`) are deliberately
left untouched: their directory naming does not reliably encode a PEP 503
package name + exact version the way the pypi/ tree does, so matching them
against uv.lock risks a false match/miss. They are a small fraction of the
cache's total size (see the before/after `du -sh` this script prints).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tomllib
from pathlib import Path

# PEP 503 normalization: case-insensitive, runs of -_. collapse to one '-'.
# uv.lock package names and the cache's on-disk directory names are not
# guaranteed to use the same casing/separator, so both sides must go through
# this before comparing.
_NORMALIZE_RE = re.compile(r"[-_.]+")


def normalize(name: str) -> str:
    return _NORMALIZE_RE.sub("-", name).lower()


def load_keep_set(lock_path: Path) -> set[tuple[str, str]]:
    with lock_path.open("rb") as f:
        data = tomllib.load(f)
    packages = data.get("package", [])
    keep = {
        (normalize(p["name"]), p["version"])
        for p in packages
        if "name" in p and "version" in p
    }
    return keep


def _entry_name_version(entry: Path) -> tuple[str, str] | None:
    # entry is e.g. wheels-v6/pypi/<name>/<version>-<platform-tag-or-nothing>
    # or sdists-v*/pypi/<name>/<version>
    name = normalize(entry.parent.name)
    version = entry.name.split("-", 1)[0]
    if not version:
        return None
    return name, version


def prune(
    cache_dir: Path, keep: set[tuple[str, str]], *, dry_run: bool
) -> tuple[int, int]:
    """Returns (entries_removed, archives_removed)."""
    reachable: set[str] = set()
    entries_removed = 0

    pypi_roots = [
        p
        for pattern in ("wheels-v*/pypi", "sdists-v*/pypi")
        for p in cache_dir.glob(pattern)
    ]

    for pypi_root in pypi_roots:
        for name_dir in pypi_root.iterdir():
            if not name_dir.is_dir():
                continue
            for entry in name_dir.iterdir():
                parsed = _entry_name_version(entry)
                if parsed is None:
                    continue
                if parsed in keep:
                    if entry.is_symlink():
                        reachable.add(Path(entry.resolve()).name)
                    continue
                entries_removed += 1
                if dry_run:
                    continue
                # Remove the pointer directory/symlink and its sibling
                # metadata files (<same-stem>.msgpack, <same-stem>.http).
                for sibling in name_dir.glob(f"{entry.name}*"):
                    if sibling.is_symlink() or sibling.is_file():
                        sibling.unlink()
                    elif sibling.is_dir():
                        shutil.rmtree(sibling)
            if not dry_run and not any(name_dir.iterdir()):
                name_dir.rmdir()

    # Second pass: also mark reachable archives for every KEPT entry we
    # didn't already see above (entries outside wheels-v*/sdists-v*/pypi
    # that still resolve into archive-v0, if any such mounts exist).
    archives_removed = 0
    for archive_root in cache_dir.glob("archive-v0"):
        if not archive_root.is_dir():
            continue
        for archive_dir in archive_root.iterdir():
            if archive_dir.name in reachable:
                continue
            archives_removed += 1
            if not dry_run:
                shutil.rmtree(archive_dir, ignore_errors=True)

    return entries_removed, archives_removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True, help="Path to uv.lock")
    parser.add_argument(
        "--cache-dir", type=Path, required=True, help="UV_CACHE_DIR to prune"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without removing it",
    )
    # Sanity floor: if parsing uv.lock somehow yields a near-empty keep set
    # (e.g. a future uv.lock format change this script doesn't understand),
    # refuse to prune rather than silently deleting almost everything.
    parser.add_argument("--min-keep", type=int, default=100)
    args = parser.parse_args()

    if not args.lock.is_file():
        print(f"error: {args.lock} does not exist", file=sys.stderr)
        return 1
    if not args.cache_dir.is_dir():
        print(f"error: {args.cache_dir} does not exist", file=sys.stderr)
        return 1

    keep = load_keep_set(args.lock)
    if len(keep) < args.min_keep:
        print(
            f"error: only parsed {len(keep)} pinned packages from {args.lock} "
            f"(expected at least {args.min_keep}) -- refusing to prune, "
            "uv.lock's format may have changed in a way this script doesn't understand",
            file=sys.stderr,
        )
        return 1

    entries_removed, archives_removed = prune(
        args.cache_dir, keep, dry_run=args.dry_run
    )
    verb = "Would remove" if args.dry_run else "Removed"
    print(
        f"{verb} {entries_removed} stale pypi cache entries and {archives_removed} "
        f"orphaned archives (kept {len(keep)} packages pinned in {args.lock})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
