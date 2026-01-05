# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""
Remote-aware test selection helper for pytest-testmon (Python 3.12).

Purpose
-------
When running unit tests with ``--testmon``, pytest-testmon tracks in-process
Python execution and reruns only affected tests. Code executed inside
``@ray.remote`` actors runs out-of-process, so testmon alone cannot see those
dependencies. This lightweight test-only plugin augments selection so that
edits inside remote actors can still retrigger the relevant tests.

How it works
------------
- Builds a static mapping from each unit test (nodeid) to the transitive set
  of ``nemo_rl`` Python files that the test module imports.
- Stores the mapping in ``.nrl_remote_map.json`` and tracks mtimes in
  ``.nrl_remote_state.json`` at repo root.
- When ``--testmon`` is present:
  - On first run, seeds the state file and does not change selection.
  - On subsequent runs, compares mtimes; if tracked files changed, it replaces
    the pytest positional args with the affected nodeids so those tests run.
- Honors ``-k``. If a ``-k`` filter is provided, the plugin does not alter
  selection and lets user intent win.

Limitations
-----------
- Static import analysis only; dynamic imports/loading are not discovered.
- Only Python files are considered (YAML/JSON/shell edits are not tracked).
- The mapping is conservative; if a test exercises code not visible via
  imports, run it once explicitly to seed the map.

Activation
----------
This plugin auto-loads via ``tests/unit/__init__.py`` and only engages when
``--testmon`` is present.

Artifacts
---------
Two JSON files are written to the repository root:

1) ``.nrl_remote_map.json``
   - Maps test nodeids to the transitive set of project files (under ``nemo_rl/``)
     imported by that test module.
   - Example (paths abbreviated for readability):
     {
       "tests/unit/distributed/test_worker_groups.py::test_configure_worker_interaction": [
         "/workspaces/nemo-rl/nemo_rl/distributed/worker_groups.py",
         "/workspaces/nemo-rl/nemo_rl/distributed/virtual_cluster.py"
       ],
       "tests/unit/models/policy/workers/test_dtensor_worker.py::test_lm_policy_init[True]": [
         "/workspaces/nemo-rl/nemo_rl/models/policy/workers/dtensor_policy_worker.py"
       ]
     }

2) ``.nrl_remote_state.json``
   - Stores the last-seen modification time (mtime) per tracked file to detect changes.
   - Example:
     {
       "/workspaces/nemo-rl/nemo_rl/distributed/worker_groups.py": 1725369123.456,
       "/workspaces/nemo-rl/nemo_rl/models/policy/workers/dtensor_policy_worker.py": 1725369187.012
     }
"""

import ast
import json
import os
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
MAP_PATH: Path = REPO_ROOT / ".nrl_remote_map.json"
STATE_PATH: Path = REPO_ROOT / ".nrl_remote_state.json"
PROJECT_PREFIXES: tuple[str, ...] = ("nemo_rl",)


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""


def _parse_imported_modules(py_path: Path) -> set[str]:
    src = _read_text(py_path)
    try:
        tree = ast.parse(src)
    except Exception:
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module)
    return {m for m in modules if m.startswith(PROJECT_PREFIXES)}


def _module_to_file(module_name: str) -> Path | None:
    mod_path = Path(module_name.replace(".", "/") + ".py")
    abs_path = (REPO_ROOT / mod_path).resolve()
    return abs_path if abs_path.exists() else None


def _discover_test_nodeids_and_files() -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    tests_root = REPO_ROOT / "tests" / "unit"
    for test_path in tests_root.rglob("test_*.py"):
        rel = test_path.relative_to(REPO_ROOT)
        mod_node_prefix = str(rel)
        modules = _parse_imported_modules(test_path)
        files: set[str] = set()
        for m in modules:
            f = _module_to_file(m)
            if f:
                files.add(str(f))
        if not files:
            continue
        src = _read_text(test_path)
        try:
            tree = ast.parse(src)
        except Exception:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                nodeid = f"{mod_node_prefix}::{node.name}"
                mapping[nodeid] = set(files)
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name.startswith(
                        "test_"
                    ):
                        nodeid = f"{mod_node_prefix}::{node.name}::{sub.name}"
                        mapping[nodeid] = set(files)
    return mapping


def _load_mapping() -> dict[str, set[str]]:
    if not MAP_PATH.exists():
        return {}
    try:
        data = json.loads(MAP_PATH.read_text())
        return {k: set(v) for k, v in data.items()}
    except Exception:
        return {}


def _save_mapping(mapping: dict[str, set[str]]) -> None:
    MAP_PATH.write_text(
        json.dumps({k: sorted(v) for k, v in mapping.items()}, indent=2)
    )


def _detect_changed(files: Iterable[str]) -> set[str]:
    prev: dict[str, float] = {}
    if STATE_PATH.exists():
        try:
            prev = json.loads(STATE_PATH.read_text())
        except Exception:
            prev = {}
    changed: set[str] = set()
    state: dict[str, float] = {}
    for f in files:
        try:
            mtime = os.path.getmtime(f)
            state[f] = mtime
            if prev.get(f, 0) < mtime:
                changed.add(f)
        except FileNotFoundError:
            changed.add(f)
    if files:
        STATE_PATH.write_text(json.dumps(state, indent=2))
    return changed


def _has_k_filter(args: list[str]) -> bool:
    """Return True if -k/--keyword filter is present in CLI args."""
    if "-k" in args:
        return True
    for i, a in enumerate(args):
        if a.startswith("-k") or a.startswith("--keyword"):
            return True
        if a in {"-k", "--keyword"} and i + 1 < len(args):
            return True
    return False


def pytest_load_initial_conftests(args, early_config, parser):
    # Only augment when user asked for --testmon and no -k filter is provided
    if "--testmon" not in args or _has_k_filter(args):
        return

    affected = _select_affected(None)
    # None = first run (seed only), empty set = no changes; leave args unchanged
    if affected is None or affected == set():
        return

    # Remove --testmon and narrow args to affected nodeids (execute only those tests)
    while "--testmon" in args:
        args.remove("--testmon")
    if not any(not a.startswith("-") for a in args):
        args[:] = sorted(affected)
    else:
        args.extend(sorted(affected))


def _effective_mapping() -> dict[str, set[str]]:
    mapping = _load_mapping()
    if not mapping:
        mapping = _discover_test_nodeids_and_files()
        if mapping:
            _save_mapping(mapping)
    return mapping


def _select_affected(config) -> set[str] | None:
    mapping = _effective_mapping()
    if not mapping:
        return None
    file_set: set[str] = set()
    for files in mapping.values():
        file_set.update(files)
    if not file_set:
        return None
    if not STATE_PATH.exists():
        _ = _detect_changed(file_set)
        return None
    changed = _detect_changed(file_set)
    if not changed:
        return set()
    affected: set[str] = set()
    for nodeid, files in mapping.items():
        if any(f in changed for f in files):
            affected.add(nodeid)
    return affected


def pytest_configure(config) -> None:
    # Late-stage fallback in case initial hook didn't capture
    tm_on = config.pluginmanager.hasplugin("testmon") or "--testmon" in sys.argv
    if not tm_on:
        return
    # Honor -k/--keyword filters
    if _has_k_filter(sys.argv):
        return
    affected = _select_affected(config)
    if affected is None or affected == set():
        return
    try:
        config.args[:] = sorted(affected)
    except Exception:
        pass


def pytest_collection_modifyitems(config, items):
    tm_on = config.pluginmanager.hasplugin("testmon") or "--testmon" in sys.argv
    if not tm_on:
        return
    # Honor -k/--keyword filters
    if _has_k_filter(sys.argv):
        return
    affected = _select_affected(config)
    if affected is None:
        return
    if affected == set():
        # No changes → deselect all for speed
        items[:] = []
        return
    items[:] = [it for it in items if it.nodeid in affected]
