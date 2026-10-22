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

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]

# docker/ is not a package (no __init__.py, by design -- it mirrors a Docker
# build context, not Python source), so import the script directly by path
# rather than adding docker/ to sys.path.
_SPEC = importlib.util.spec_from_file_location(
    "prune_uv_cache", REPO_ROOT / "docker" / "prune_uv_cache.py"
)
prune_uv_cache = importlib.util.module_from_spec(_SPEC)
sys.modules["prune_uv_cache"] = prune_uv_cache
_SPEC.loader.exec_module(prune_uv_cache)


def test_normalize_is_pep503() -> None:
    assert prune_uv_cache.normalize("Foo_Bar.Baz") == "foo-bar-baz"
    assert prune_uv_cache.normalize("foo--bar") == "foo-bar"
    assert prune_uv_cache.normalize("UPPER") == "upper"


def test_load_keep_set_parses_uv_lock(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(
        """
version = 1

[[package]]
name = "Orjson"
version = "3.12.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "torch"
version = "2.9.0"
source = { registry = "https://pypi.org/simple" }
"""
    )
    keep = prune_uv_cache.load_keep_set(lock)
    assert keep == {("orjson", "3.12.0"), ("torch", "2.9.0")}


def _make_wheel_entry(
    cache_dir: Path, name: str, version: str, tag: str, archive_id: str
) -> None:
    pypi = cache_dir / "wheels-v6" / "pypi" / name
    pypi.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "archive-v0" / archive_id
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "payload.txt").write_text("real wheel content")
    (pypi / f"{version}-{tag}").symlink_to(archive, target_is_directory=True)
    (pypi / f"{version}-{tag}.http").write_text("cache-control: ...")


def test_prune_keeps_locked_version_drops_stale_ones(tmp_path: Path) -> None:
    cache_dir = tmp_path / "uv-cache"
    # Three versions of the same package on disk, as if orjson got bumped
    # 3.11.3 -> 3.11.9 -> 3.12.0 across three separate dependency-bump PRs
    # while BASE_IMAGE/UV_VERSION/HYBRID_EP_MULTINODE stayed the same.
    _make_wheel_entry(
        cache_dir, "orjson", "3.11.3", "cp313-cp313-linux_x86_64", "archive-old-1"
    )
    _make_wheel_entry(
        cache_dir, "orjson", "3.11.9", "cp313-cp313-linux_x86_64", "archive-old-2"
    )
    _make_wheel_entry(
        cache_dir, "orjson", "3.12.0", "cp313-cp313-linux_x86_64", "archive-kept"
    )

    keep = {("orjson", "3.12.0")}
    entries_removed, archives_removed = prune_uv_cache.prune(
        cache_dir, keep, dry_run=False
    )

    assert entries_removed == 4  # 2 stale versions x (symlink + .http sibling)
    assert archives_removed == 2  # the 2 stale archives, not the kept one

    pypi_dir = cache_dir / "wheels-v6" / "pypi" / "orjson"
    remaining = {p.name for p in pypi_dir.iterdir()}
    assert remaining == {
        "3.12.0-cp313-cp313-linux_x86_64",
        "3.12.0-cp313-cp313-linux_x86_64.http",
    }

    archive_dir = cache_dir / "archive-v0"
    assert {p.name for p in archive_dir.iterdir()} == {"archive-kept"}
    assert (
        archive_dir / "archive-kept" / "payload.txt"
    ).read_text() == "real wheel content"


def test_prune_dry_run_reports_without_removing(tmp_path: Path) -> None:
    cache_dir = tmp_path / "uv-cache"
    _make_wheel_entry(
        cache_dir, "orjson", "3.11.3", "cp313-cp313-linux_x86_64", "archive-old"
    )

    entries_removed, archives_removed = prune_uv_cache.prune(
        cache_dir, keep=set(), dry_run=True
    )

    assert entries_removed == 2
    assert archives_removed == 1
    # Nothing actually removed in dry-run mode.
    assert (cache_dir / "wheels-v6" / "pypi" / "orjson").exists()
    assert (cache_dir / "archive-v0" / "archive-old").exists()
