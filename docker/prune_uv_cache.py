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

"""Prune uv's cache down to what uv.lock still pins.

uv.lock pins one version (or one git commit) per package across the WHOLE
project -- every extra, installed in this build or not. uv's cache is
append-only: `uv sync` downloads what a build needs and never removes a version
an older lockfile pinned. The CI seed persists the cache across builds for as
long as BASE_IMAGE/UV_VERSION/HYBRID_EP_MULTINODE are unchanged (see the
dependency-install RUN in docker/Dockerfile), so every dependency bump would
otherwise leave its predecessor in the seed and in every shipped image.
`uv cache prune` does not help: it removes stale cache-format buckets, cached
environments and unlinked archives, never a valid entry for an older version.

Removed, matched against uv.lock (layout per uv's crates/uv-cache/src/wheel.rs):
  wheels-v*/{pypi,index/<digest>}/<name>/<version>-<tags>[.http|.msgpack]
  sdists-v*/{pypi,index/<digest>}/<name>/<version>[...]
      kept iff (<name>, <version>) is pinned by uv.lock.
  sdists-v*/git/<digest>/<first 16 hex of commit>
  git-v*/checkouts/<digest>/<short commit>
      kept iff the commit prefixes a git commit uv.lock pins.
  archive-v0/<id>
      kept iff any link left under wheels-v*/ or sdists-v*/ resolves to it --
      the same reachability scan uv's own `uv cache prune` uses, so archives
      behind entries this script never touches (url/, path/, editable/) stay.

Never removed: url/, path/, editable/ trees (keyed by URL digest, no version to
match), git-v*/db, and every other bucket. That keeps it safe to run before
shipping UV_CACHE_DIR: it cannot remove anything offline
`uv run --extra <backend>` needs, because that is itself "what uv.lock pins".
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tomllib
from pathlib import Path

# PEP 503: uv.lock and the cache's directory names may differ in case/separator.
_NORMALIZE_RE = re.compile(r"[-_.]+")
_GIT_SHA_RE = re.compile(r"#([0-9a-f]{40})$")
_METADATA_SUFFIXES = (".http", ".msgpack", ".lock", ".rev")
# uv's archive-reference scan skips these; they are never links into archive-v0.
_SCAN_SKIP_SUFFIXES = (".lock", ".whl", ".http", ".rev", ".msgpack")


def normalize(name: str) -> str:
    return _NORMALIZE_RE.sub("-", name).lower()


def load_lock(lock_path: Path) -> tuple[set[tuple[str, str]], set[str]]:
    """Returns ({(normalized name, version)}, {full git commit SHA})."""
    with lock_path.open("rb") as f:
        packages = tomllib.load(f).get("package", [])
    versions = set()
    for p in packages:
        if "name" in p and "version" in p:
            name, version = normalize(p["name"]), p["version"]
            versions.add((name, version))
            # uv's WheelFilename::cache_key truncates long versions to 47 chars.
            versions.add((name, version[:47].rstrip(".+")))
    shas = set()
    for p in packages:
        git = p.get("source", {}).get("git")
        if git and (m := _GIT_SHA_RE.search(git)):
            shas.add(m.group(1))
    return versions, shas


def entry_version(entry_name: str) -> str | None:
    """`<version>-<tags>[.http]` or `<version>` -> `<version>`; None if not a version."""
    for suffix in _METADATA_SUFFIXES:
        if entry_name.endswith(suffix):
            entry_name = entry_name[: -len(suffix)]
            break
    version = entry_name.split("-", 1)[0]
    # PEP 440 versions (including epochs) start with a digit. Anything else is
    # not ours to judge, so it is kept.
    return version if version[:1].isdigit() else None


def _registry_roots(cache_dir: Path) -> list[Path]:
    roots = []
    for bucket in ("wheels-v*", "sdists-v*"):
        for bucket_dir in cache_dir.glob(bucket):
            roots.extend(p for p in [bucket_dir / "pypi"] if p.is_dir())
            index = bucket_dir / "index"
            if index.is_dir():
                roots.extend(p for p in index.iterdir() if p.is_dir())
    return roots


def _sha_kept(prefix: str, shas: set[str]) -> bool:
    # Unparseable names are kept: only a hex commit prefix can be matched.
    if len(prefix) < 7 or not all(c in "0123456789abcdef" for c in prefix):
        return True
    return any(sha.startswith(prefix) for sha in shas)


def plan_removals(
    cache_dir: Path, versions: set[tuple[str, str]], shas: set[str]
) -> list[Path]:
    removals = []
    for root in _registry_roots(cache_dir):
        for name_dir in root.iterdir():
            if not name_dir.is_dir() or name_dir.is_symlink():
                continue
            name = normalize(name_dir.name)
            for entry in name_dir.iterdir():
                version = entry_version(entry.name)
                if version is not None and (name, version) not in versions:
                    removals.append(entry)
    commit_dirs = [
        *cache_dir.glob("sdists-v*/git/*/*"),
        *cache_dir.glob("git-v*/checkouts/*/*"),
    ]
    for commit_dir in commit_dirs:
        if commit_dir.is_dir() and not _sha_kept(commit_dir.name, shas):
            removals.append(commit_dir)
    return removals


def _reachable_archives(cache_dir: Path, removed: set[Path]) -> set[str]:
    archive_root = (cache_dir / "archive-v0").resolve()
    reachable = set()
    for bucket in [*cache_dir.glob("wheels-v*"), *cache_dir.glob("sdists-v*")]:
        for dirpath, dirnames, filenames in os.walk(bucket):
            here = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames if d != "src" and here / d not in removed
            ]
            # os.walk lists directory symlinks under dirnames without following them.
            for name in [*filenames, *dirnames]:
                link = here / name
                if link in removed or name.endswith(_SCAN_SKIP_SUFFIXES):
                    continue
                if link.is_symlink():
                    target = link.resolve()
                    if target.parent == archive_root:
                        reachable.add(target.name)
    return reachable


def prune(
    cache_dir: Path,
    versions: set[tuple[str, str]],
    shas: set[str],
    *,
    dry_run: bool,
    verbose: bool = False,
) -> tuple[int, int]:
    """Returns (entries_removed, archives_removed)."""
    removals = plan_removals(cache_dir, versions, shas)
    reachable = _reachable_archives(cache_dir, set(removals))
    archive_root = cache_dir / "archive-v0"
    orphaned = (
        [a for a in archive_root.iterdir() if a.name not in reachable]
        if archive_root.is_dir()
        else []
    )
    for path in [*removals, *orphaned]:
        if verbose:
            print(f"{'would remove' if dry_run else 'remove'}: {path}")
        if dry_run:
            continue
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    return len(removals), len(orphaned)


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
    parser.add_argument(
        "--verbose", action="store_true", help="Print every removed path"
    )
    # Sanity floor: a near-empty keep set (e.g. a uv.lock format change this
    # script does not understand) must not turn into deleting the whole cache.
    parser.add_argument("--min-keep", type=int, default=100)
    args = parser.parse_args()

    if not args.lock.is_file():
        print(f"error: {args.lock} does not exist", file=sys.stderr)
        return 1
    if not args.cache_dir.is_dir():
        print(f"error: {args.cache_dir} does not exist", file=sys.stderr)
        return 1

    versions, shas = load_lock(args.lock)
    if len(versions) < args.min_keep:
        print(
            f"error: only parsed {len(versions)} pinned packages from {args.lock} "
            f"(expected at least {args.min_keep}) -- refusing to prune, "
            "uv.lock's format may have changed in a way this script doesn't understand",
            file=sys.stderr,
        )
        return 1

    entries_removed, archives_removed = prune(
        args.cache_dir, versions, shas, dry_run=args.dry_run, verbose=args.verbose
    )
    verb = "Would remove" if args.dry_run else "Removed"
    print(
        f"{verb} {entries_removed} stale cache entries and {archives_removed} "
        f"orphaned archives (kept {len(versions)} packages and {len(shas)} git "
        f"commits pinned in {args.lock})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
