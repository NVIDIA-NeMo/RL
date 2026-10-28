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
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]

# docker/ is not a package, so import the script by path.
_SPEC = importlib.util.spec_from_file_location(
    "prune_uv_cache", REPO_ROOT / "docker" / "prune_uv_cache.py"
)
prune_uv_cache = importlib.util.module_from_spec(_SPEC)
sys.modules["prune_uv_cache"] = prune_uv_cache
_SPEC.loader.exec_module(prune_uv_cache)

SHA_KEPT = "17cfb817bccec3a9c247013360cc550c2bac441e"
SHA_STALE = "0123456789abcdef0123456789abcdef01234567"
TAG = "cp313-cp313-linux_x86_64"


def _archive(cache: Path, archive_id: str) -> Path:
    archive = cache / "archive-v0" / archive_id
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "payload.txt").write_text(archive_id)
    return archive


def _link(link: Path, archive: Path) -> None:
    # uv writes relative links into archive-v0.
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(archive, link.parent), target_is_directory=True)


def _wheel(root: Path, name: str, entry: str, archive_id: str) -> None:
    cache = root
    while not (cache / "archive-v0").exists() and cache.name != "cache":
        cache = cache.parent
    _link(root / name / entry, _archive(cache, archive_id))
    (root / name / f"{entry}.http").write_text("cache-control")


def _names(path: Path) -> set[str]:
    return {p.name for p in path.iterdir()} if path.exists() else set()


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    c = tmp_path / "cache"
    (c / "archive-v0").mkdir(parents=True)
    return c


def test_normalize_is_pep503() -> None:
    assert prune_uv_cache.normalize("Foo_Bar.Baz") == "foo-bar-baz"
    assert prune_uv_cache.normalize("foo--bar") == "foo-bar"
    assert prune_uv_cache.normalize("UPPER") == "upper"


def test_load_lock_reads_versions_and_git_commits(tmp_path: Path) -> None:
    long_version = "1.0.0+" + "a" * 60
    lock = tmp_path / "uv.lock"
    lock.write_text(
        f"""
version = 1

[[package]]
name = "Orjson"
version = "3.12.0"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "deep-ep"
version = "1.2.1"
source = {{ git = "https://github.com/deepseek-ai/DeepEP.git?rev={SHA_KEPT}#{SHA_KEPT}" }}

[[package]]
name = "long"
version = "{long_version}"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "nemo-rl"
source = {{ editable = "." }}
"""
    )
    versions, shas = prune_uv_cache.load_lock(lock)
    assert ("orjson", "3.12.0") in versions
    assert ("deep-ep", "1.2.1") in versions
    assert ("long", long_version) in versions
    # uv truncates long versions in wheel cache keys; the truncated form is kept too.
    assert ("long", long_version[:47]) in versions
    assert shas == {SHA_KEPT}


@pytest.mark.parametrize(
    "entry, expected",
    [
        (f"3.12.0-{TAG}", "3.12.0"),
        (f"3.12.0-{TAG}.http", "3.12.0"),
        (f"3.12.0-{TAG}.msgpack", "3.12.0"),
        (f"2.9.0+cu130-{TAG}", "2.9.0+cu130"),
        ("1!2.0-py3-none-any", "1!2.0"),
        ("1.0.1", "1.0.1"),
        ("metadata.msgpack", None),
        (".lock", None),
    ],
)
def test_entry_version(entry: str, expected: str | None) -> None:
    assert prune_uv_cache.entry_version(entry) == expected


def test_prunes_stale_pypi_wheels_and_their_archives(cache: Path) -> None:
    pypi = cache / "wheels-v6" / "pypi"
    _wheel(pypi, "orjson", f"3.11.3-{TAG}", "old-1")
    _wheel(pypi, "orjson", f"3.11.9-{TAG}", "old-2")
    _wheel(pypi, "orjson", f"3.12.0-{TAG}", "kept")

    removed, archives = prune_uv_cache.prune(
        cache, {("orjson", "3.12.0")}, set(), dry_run=False
    )

    assert (removed, archives) == (4, 2)
    assert _names(pypi / "orjson") == {f"3.12.0-{TAG}", f"3.12.0-{TAG}.http"}
    assert _names(cache / "archive-v0") == {"kept"}
    assert (cache / "archive-v0" / "kept" / "payload.txt").read_text() == "kept"


def test_does_not_confuse_version_prefixes(cache: Path) -> None:
    # Removing 1.0 must not touch 1.0.1's files (a prefix glob would).
    pypi = cache / "wheels-v6" / "pypi"
    _wheel(pypi, "pkg", "1.0-py3-none-any", "stale")
    _wheel(pypi, "pkg", "1.0.1-py3-none-any", "kept")

    prune_uv_cache.prune(cache, {("pkg", "1.0.1")}, set(), dry_run=False)

    assert _names(pypi / "pkg") == {
        "1.0.1-py3-none-any",
        "1.0.1-py3-none-any.http",
    }
    assert _names(cache / "archive-v0") == {"kept"}


def test_prunes_alternative_index_trees(cache: Path) -> None:
    # torch & co. come from download.pytorch.org: wheels-v6/index/<digest>/...
    index = cache / "wheels-v6" / "index" / "d1g3st"
    _wheel(index, "torch", f"2.8.0+cu130-{TAG}", "torch-old")
    _wheel(index, "torch", f"2.9.0+cu130-{TAG}", "torch-kept")

    prune_uv_cache.prune(cache, {("torch", "2.9.0+cu130")}, set(), dry_run=False)

    assert _names(index / "torch") == {
        f"2.9.0+cu130-{TAG}",
        f"2.9.0+cu130-{TAG}.http",
    }
    assert _names(cache / "archive-v0") == {"torch-kept"}


def test_keeps_archives_referenced_by_untouched_trees(cache: Path) -> None:
    # url/ and editable/ entries carry no version to match, so they are left
    # alone -- and so must the archives they point at.
    wheels = cache / "wheels-v6"
    _link(wheels / "url" / "abc" / "vllm" / f"0.29.0-{TAG}", _archive(cache, "url"))
    _link(wheels / "editable" / "def" / "proj" / "0.1.0", _archive(cache, "edit"))

    removed, archives = prune_uv_cache.prune(cache, set(), set(), dry_run=False)

    assert (removed, archives) == (0, 0)
    assert _names(cache / "archive-v0") == {"url", "edit"}


def test_prunes_stale_sdist_versions_keeping_built_wheel_archives(
    cache: Path,
) -> None:
    sdists = cache / "sdists-v9" / "pypi" / "flash-attn"
    _link(sdists / "2.7.0" / "rev1" / "wheel", _archive(cache, "built-old"))
    _link(sdists / "2.8.3" / "rev1" / "wheel", _archive(cache, "built-kept"))
    (sdists / "2.8.3" / "rev1" / "src").mkdir()

    prune_uv_cache.prune(cache, {("flash-attn", "2.8.3")}, set(), dry_run=False)

    assert _names(sdists) == {"2.8.3"}
    assert _names(cache / "archive-v0") == {"built-kept"}


def test_prunes_git_builds_and_checkouts_by_commit(cache: Path) -> None:
    git_builds = cache / "sdists-v9" / "git" / "repo"
    _link(git_builds / SHA_KEPT[:16] / "rev" / "wheel", _archive(cache, "git-kept"))
    _link(git_builds / SHA_STALE[:16] / "rev" / "wheel", _archive(cache, "git-old"))
    checkouts = cache / "git-v0" / "checkouts" / "repo"
    (checkouts / SHA_KEPT[:7]).mkdir(parents=True)
    (checkouts / SHA_STALE[:7]).mkdir()
    (checkouts / "not-a-sha").mkdir()
    (cache / "git-v0" / "db" / "repo").mkdir(parents=True)

    removed, archives = prune_uv_cache.prune(cache, set(), {SHA_KEPT}, dry_run=False)

    assert (removed, archives) == (2, 1)
    assert _names(git_builds) == {SHA_KEPT[:16]}
    assert _names(checkouts) == {SHA_KEPT[:7], "not-a-sha"}
    assert (cache / "git-v0" / "db" / "repo").is_dir()
    assert _names(cache / "archive-v0") == {"git-kept"}


def test_dry_run_reports_same_counts_without_removing(cache: Path) -> None:
    pypi = cache / "wheels-v6" / "pypi"
    _wheel(pypi, "orjson", f"3.11.3-{TAG}", "old")
    _wheel(pypi, "orjson", f"3.12.0-{TAG}", "kept")
    keep = {("orjson", "3.12.0")}

    assert prune_uv_cache.prune(cache, keep, set(), dry_run=True) == (2, 1)
    assert len(_names(pypi / "orjson")) == 4
    assert _names(cache / "archive-v0") == {"old", "kept"}
    assert prune_uv_cache.prune(cache, keep, set(), dry_run=False) == (2, 1)


def test_second_run_is_a_no_op(cache: Path) -> None:
    pypi = cache / "wheels-v6" / "pypi"
    _wheel(pypi, "orjson", f"3.11.3-{TAG}", "old")
    _wheel(pypi, "orjson", f"3.12.0-{TAG}", "kept")
    keep = {("orjson", "3.12.0")}

    prune_uv_cache.prune(cache, keep, set(), dry_run=False)
    assert prune_uv_cache.prune(cache, keep, set(), dry_run=False) == (0, 0)


def test_dockerfile_prunes_the_seed_before_any_sync() -> None:
    # Deleting cache files in a later layer only hides them; the image still ships
    # them. So the prune runs inside the dependency RUN -- right after the cache
    # mount's content is restored and before the first sync, so it only judges
    # restored content and never deletes the build-isolation deps (setuptools, a
    # build-time torch, ...) the syncs resolve outside uv.lock. The RUN's
    # `uv_cache_persist prune` trap then mirrors the pruned cache into the mount.
    lines = (REPO_ROOT / "docker" / "Dockerfile").read_text().splitlines()
    code = [(i, line) for i, line in enumerate(lines) if not line.startswith("#")]

    copy_script = next(
        i for i, l in code if l.startswith("COPY") and "prune_uv_cache.py" in l
    )
    prunes = [
        i for i, l in code if "prune_uv_cache.py" in l and not l.startswith("COPY")
    ]
    assert len(prunes) == 1, "prune must run exactly once, in the dependency RUN"
    prune = prunes[0]
    run_start = max(i for i, l in code if i < prune and l.startswith("RUN"))
    run_end = next(i for i, l in code if i > prune and l == "EOF")
    restore = next(
        i for i, l in code if i > run_start and l.startswith("uv_cache_restore")
    )
    persist = next(i for i, l in code if i > run_start and "uv_cache_persist" in l)
    venv = next(i for i, l in code if i > run_start and l == "uv venv --seed")
    first_sync = next(i for i, l in code if i > run_start and "uv sync" in l)

    # The script comes from the recipe's own context: an older NRL_GIT_REF lacks it.
    assert "--from=" not in lines[copy_script]
    assert copy_script < run_start
    assert lines[restore] == "uv_cache_restore force"
    assert "uv_cache_persist prune" in lines[persist]
    assert run_start < restore < venv < prune < first_sync < run_end
    assert persist < run_end

    # Dynamo installs with a throwaway cache, so its RUN needs no shared cache mount.
    dynamo = next(i for i, l in code if "docker/dynamo/install.sh" in l and "bash" in l)
    dynamo_run = max(i for i, l in code if i < dynamo and l.startswith("RUN"))
    assert "uv-cache" not in "\n".join(lines[dynamo_run:dynamo])
    assert lines[dynamo].startswith("UV_CACHE_DIR=/tmp/")
    assert lines[dynamo + 1].startswith("rm -rf /tmp/")


@pytest.mark.parametrize("suffix", [".http", ".msgpack", ".lock", ".rev"])
def test_compares_sdist_version_directories_verbatim(cache: Path, suffix: str) -> None:
    version = f"1.0+foo{suffix}"
    sdists = cache / "sdists-v9" / "pypi" / "pkg"
    _link(sdists / version / "rev" / "wheel", _archive(cache, "built"))

    prune_uv_cache.prune(cache, {("pkg", version)}, set(), dry_run=False)
    assert _names(sdists) == {version}
    assert _names(cache / "archive-v0") == {"built"}

    # Pinning only the shorter version must not keep the longer one.
    prune_uv_cache.prune(cache, {("pkg", "1.0+foo")}, set(), dry_run=False)
    assert _names(sdists) == set()
    assert _names(cache / "archive-v0") == set()


def test_keeps_archives_of_a_package_named_src(cache: Path) -> None:
    pypi = cache / "wheels-v6" / "pypi"
    _wheel(pypi, "src", "1.0-py3-none-any", "wheel-kept")
    _wheel(pypi, "src", "0.9-py3-none-any", "wheel-old")
    sdists = cache / "sdists-v9" / "pypi"
    _link(sdists / "src" / "1.0" / "rev" / "wheel", _archive(cache, "sdist-kept"))
    _link(sdists / "src" / "0.9" / "rev" / "wheel", _archive(cache, "sdist-old"))
    # A link inside an unpacked source tree does not keep its target alive.
    _link(sdists / "src" / "1.0" / "rev" / "src" / "x", _archive(cache, "in-src"))

    prune_uv_cache.prune(cache, {("src", "1.0")}, set(), dry_run=False)

    assert _names(pypi / "src") == {"1.0-py3-none-any", "1.0-py3-none-any.http"}
    assert _names(sdists / "src") == {"1.0"}
    assert _names(cache / "archive-v0") == {"wheel-kept", "sdist-kept"}


def test_main_refuses_a_suspiciously_small_lock(
    cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wheel(cache / "wheels-v6" / "pypi", "orjson", f"3.11.3-{TAG}", "old")
    lock = tmp_path / "uv.lock"
    lock.write_text('version = 1\n[[package]]\nname = "a"\nversion = "1"\n')
    monkeypatch.setattr(
        sys, "argv", ["prune", "--lock", str(lock), "--cache-dir", str(cache)]
    )

    assert prune_uv_cache.main() == 1
    assert _names(cache / "archive-v0") == {"old"}
