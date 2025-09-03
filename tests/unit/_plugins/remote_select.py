from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Iterable, Set
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
MAP_PATH = REPO_ROOT / ".nrl_remote_map.json"
STATE_PATH = REPO_ROOT / ".nrl_remote_state.json"
PROJECT_PREFIXES = ("nemo_rl",)


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""


def _parse_imported_modules(py_path: Path) -> Set[str]:
    src = _read_text(py_path)
    try:
        tree = ast.parse(src)
    except Exception:
        return set()
    modules: Set[str] = set()
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


def _discover_test_nodeids_and_files() -> dict[str, Set[str]]:
    mapping: dict[str, Set[str]] = {}
    tests_root = REPO_ROOT / "tests" / "unit"
    for test_path in tests_root.rglob("test_*.py"):
        rel = test_path.relative_to(REPO_ROOT)
        mod_node_prefix = str(rel)
        modules = _parse_imported_modules(test_path)
        files: Set[str] = set()
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
                    if isinstance(sub, ast.FunctionDef) and sub.name.startswith("test_"):
                        nodeid = f"{mod_node_prefix}::{node.name}::{sub.name}"
                        mapping[nodeid] = set(files)
    return mapping


def _load_mapping() -> dict[str, Set[str]]:
    if not MAP_PATH.exists():
        return {}
    try:
        data = json.loads(MAP_PATH.read_text())
        return {k: set(v) for k, v in data.items()}
    except Exception:
        return {}


def _save_mapping(mapping: dict[str, Set[str]]) -> None:
    MAP_PATH.write_text(json.dumps({k: sorted(v) for k, v in mapping.items()}, indent=2))


def _detect_changed(files: Iterable[str]) -> Set[str]:
    prev: dict[str, float] = {}
    if STATE_PATH.exists():
        try:
            prev = json.loads(STATE_PATH.read_text())
        except Exception:
            prev = {}
    changed: Set[str] = set()
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


def pytest_load_initial_conftests(args, early_config, parser):
    # Only augment when user asked for --testmon
    if "--testmon" not in args:
        return

    mapping = _load_mapping()
    if not mapping:
        mapping = _discover_test_nodeids_and_files()
        if mapping:
            _save_mapping(mapping)
    if not mapping:
        return

    file_set: Set[str] = set()
    for files in mapping.values():
        file_set.update(files)
    if not file_set:
        return

    if not STATE_PATH.exists():
        _ = _detect_changed(file_set)
        return

    changed = _detect_changed(file_set)
    if not changed:
        return

    affected: Set[str] = set()
    for nodeid, files in mapping.items():
        if any(f in changed for f in files):
            affected.add(nodeid)
    if not affected:
        return

    # Remove --testmon and narrow args to affected nodeids (execute only those tests)
    while "--testmon" in args:
        args.remove("--testmon")
    if not any(not a.startswith("-") for a in args):
        args[:] = sorted(affected)
    else:
        args.extend(sorted(affected))


def _effective_mapping() -> dict[str, Set[str]]:
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
    file_set: Set[str] = set()
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
    affected: Set[str] = set()
    for nodeid, files in mapping.items():
        if any(f in changed for f in files):
            affected.add(nodeid)
    return affected


def pytest_configure(config) -> None:
    # Late-stage fallback in case initial hook didn't capture
    tm_on = config.pluginmanager.hasplugin("testmon") or "--testmon" in sys.argv
    if not tm_on:
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
    affected = _select_affected(config)
    if affected is None:
        return
    if affected == set():
        # No changes → deselect all for speed
        items[:] = []
        return
    items[:] = [it for it in items if it.nodeid in affected]


