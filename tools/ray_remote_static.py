#!/usr/bin/env python3
"""
Static analyzer for detecting tests impacted by changes to @ray.remote functions/actors.

Overview
--------
This script scans Python test files for calls to ``.remote()`` on symbols imported
from your codebase, resolves those to fully-qualified names (FQNs), and then
verifies the referenced function/class is decorated with ``@ray.remote``. For
each such symbol, it computes a stable hash of its body via the AST. The script
tracks a small on-disk DB (JSON) with the mapping:

- tests: {test_nodeid -> [symbol_fqn, ...]}
- symbols: {"/abs/path/module.py::name" -> {file, qualname, hash}}

Usage
-----
1) Selection (default): prints impacted test nodeids, one per line, if the
   current body hash of any referenced ``@ray.remote`` symbol differs from the
   previous DB state, differs from Git HEAD, or if the symbol's file mtime is
   newer than the DB file mtime. The pytest collection hook reads this output
   and unions those tests into the single test run (only when testmon is active).

2) Update ("--update"): refresh the DB to the current state (no output).
   This is invoked at the end of tests/run_unit.sh so subsequent runs compare
   against the latest snapshot.

Notes
-----
- Only tests that actually call ``.remote()`` on an import-resolvable symbol are
  tracked. If you have other ``@ray.remote`` functions that are not referenced
  by any scanned test files (or are aliased in a way we can't resolve statically),
  they won't be included in the DB. This keeps the DB focused and small.
- The analyzer is intentionally conservative: any detectable body change marks
  referencing tests as impacted.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ImportResolver(ast.NodeVisitor):
    """Collect import aliases to resolve attribute chains to FQNs.

    Tracks two maps:
    - alias_to_module: e.g., ``import nemo_rl.foobar as foobar`` -> {"foobar": "nemo_rl.foobar"}
    - name_to_fqn: e.g., ``from nemo_rl import foobar`` -> {"foobar": "nemo_rl.foobar"}
    """
    def __init__(self):
        self.alias_to_module: Dict[str, str] = {}
        self.name_to_fqn: Dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            mod = alias.name
            asname = alias.asname or mod.split(".")[-1]
            self.alias_to_module[asname] = mod

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            return
        base = node.module
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            self.name_to_fqn[asname] = f"{base}.{name}"


def _resolve_attr_chain(expr: ast.AST) -> Optional[List[str]]:
    """Return the dotted name parts from an attribute chain or None.

    Example: ``foobar.hello_world.remote`` -> ["foobar", "hello_world", "remote"]
    """
    names: List[str] = []
    node = expr
    while isinstance(node, ast.Attribute):
        names.insert(0, node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.insert(0, node.id)
        return names
    return None


def _resolve_symbol_fqn(attr_base: ast.AST, resolver: ImportResolver) -> Optional[str]:
    """Resolve the base of ``<base>.remote()`` to a fully qualified name if possible."""
    parts = _resolve_attr_chain(attr_base)
    if not parts:
        return None
    # Name-only, e.g., hello_world
    if len(parts) == 1:
        name = parts[0]
        return resolver.name_to_fqn.get(name)
    # Module alias + attr, e.g., foobar.hello_world
    alias, *rest = parts
    mod = resolver.alias_to_module.get(alias)
    if mod and rest:
        return mod + "." + ".".join(rest)
    return None


def find_remote_calls_in_test(test_path: Path) -> Dict[str, List[str]]:
    """Return {test_nodeid: [module.symbol, ...]} for calls like ``name.remote()``.

    - Scans only functions whose names start with ``test`` at module level.
    - Records FQNs resolvable via local imports in the test file.
    """
    src = test_path.read_text()
    tree = ast.parse(src, filename=str(test_path))
    resolver = ImportResolver()
    resolver.visit(tree)

    tests_to_symbols: Dict[str, List[str]] = {}

    class TestVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_test: Optional[str] = None

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            prev = self.current_test
            is_test = node.name.startswith("test") and isinstance(node.parent, ast.Module)
            if is_test:
                self.current_test = f"{test_path}::{node.name}"
            self.generic_visit(node)
            if is_test:
                self.current_test = prev

        def visit_Call(self, node: ast.Call) -> None:
            # Match: <base>.remote(...)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "remote":
                fqn = _resolve_symbol_fqn(node.func.value, resolver)
                if fqn and self.current_test:
                    tests_to_symbols.setdefault(self.current_test, []).append(fqn)
            self.generic_visit(node)

    # Set parents for module-level detection
    for child in ast.walk(tree):
        for c in ast.iter_child_nodes(child):
            c.parent = child  # type: ignore[attr-defined]

    TestVisitor().visit(tree)
    return tests_to_symbols


def load_previous_db(db_path: Path) -> Dict:
    """Load DB JSON if present, otherwise return an empty structure."""
    if not db_path.exists():
        return {"tests": {}, "symbols": {}}
    try:
        return json.loads(db_path.read_text())
    except Exception:
        return {"tests": {}, "symbols": {}}


def is_remote_decorated(module_path: Path, symbol_name: str) -> Tuple[bool, Optional[str]]:
    """Check if a symbol is decorated with ``@ray.remote`` and return a body hash.

    Returns (True, hash) if decorated, else (False, None).
    """
    try:
        src = module_path.read_text()
        tree = ast.parse(src, filename=str(module_path))
    except Exception:
        return False, None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol_name:
            for dec in node.decorator_list:
                # Match @ray.remote
                if isinstance(dec, ast.Attribute) and dec.attr == "remote" and isinstance(dec.value, ast.Name) and dec.value.id == "ray":
                    # Hash function body
                    start = node.lineno - 1
                    end = getattr(node, "end_lineno", None)
                    src_lines = src.splitlines()
                    body_src = "\n".join(src_lines[start:end]) if end else src
                    return True, _hash_text(body_src)
        if isinstance(node, ast.ClassDef) and node.name == symbol_name:
            for dec in node.decorator_list:
                if isinstance(dec, ast.Attribute) and dec.attr == "remote" and isinstance(dec.value, ast.Name) and dec.value.id == "ray":
                    start = node.lineno - 1
                    end = getattr(node, "end_lineno", None)
                    src_lines = src.splitlines()
                    body_src = "\n".join(src_lines[start:end]) if end else src
                    return True, _hash_text(body_src)
    return False, None


def symbol_fqn_to_path_and_name(fqn: str, repo_root: Path) -> Optional[Tuple[Path, str]]:
    """Map ``nemo_rl.foobar.hello_world`` -> (repo_root/nemo_rl/foobar.py, "hello_world")."""
    # fqn like nemo_rl.foobar.hello_world -> nemo_rl/foobar.py, hello_world
    parts = fqn.split(".")
    if len(parts) < 2:
        return None
    module_parts = parts[:-1]
    symbol_name = parts[-1]
    module_file = repo_root.joinpath(*module_parts).with_suffix(".py")
    return module_file, symbol_name


def _read_file_from_git(repo_root: Path, abs_path: Path) -> str | None:
    try:
        rel = abs_path.relative_to(repo_root)
    except Exception:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"HEAD:{rel.as_posix()}"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Static impacted tests for @ray.remote changes")
    parser.add_argument("--db", default=".ray_remote_static_db.json")
    parser.add_argument("--analyze", default="tests", help="Directory to analyze tests in")
    parser.add_argument("--update", action="store_true", help="Update DB to current hashes and exit")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    tests_dir = (repo_root / args.analyze).resolve()
    db_path = (repo_root / args.db).resolve()

    prev = load_previous_db(db_path)

    # Build current mapping
    current_tests: Dict[str, List[str]] = {}
    for path in tests_dir.rglob("test_*.py"):
        m = find_remote_calls_in_test(path)
        if m:
            current_tests.update(m)

    current_symbols: Dict[str, Dict] = {}
    for tests, fqns in current_tests.items():
        for fqn in fqns:
            res = symbol_fqn_to_path_and_name(fqn, repo_root)
            if not res:
                continue
            mod_path, name = res
            decorated, body_hash = is_remote_decorated(mod_path, name)
            if decorated and body_hash:
                key = f"{str(mod_path)}::{name}"
                current_symbols[key] = {"file": str(mod_path), "qualname": name, "hash": body_hash}

    if args.update:
        db = {"tests": current_tests, "symbols": current_symbols}
        db_path.write_text(json.dumps(db, indent=2))
        return 0

    # Compute impacted: tests that reference any symbol whose hash changed
    prev_symbols: Dict[str, Dict] = prev.get("symbols", {})
    impacted: set[str] = set()
    db_mtime = db_path.stat().st_mtime if db_path.exists() else 0.0
    # Build reverse map: fqn key for current
    for test, fqns in current_tests.items():
        for fqn in fqns:
            res = symbol_fqn_to_path_and_name(fqn, repo_root)
            if not res:
                continue
            mod_path, name = res
            key = f"{str(mod_path)}::{name}"
            cur = current_symbols.get(key)
            if not cur:
                continue
            prev_entry = prev_symbols.get(key)
            changed = False
            # 0) If the source file changed since last DB update, mark impacted
            try:
                if mod_path.stat().st_mtime > db_mtime:
                    impacted.add(test)
                    continue
            except Exception:
                pass

            # 1) Difference from previous DB state
            if prev_entry and cur.get("hash") != prev_entry.get("hash"):
                changed = True
            # 2) Difference from Git HEAD (detect first-change even if DB already updated previously)
            head_src = _read_file_from_git(repo_root, mod_path)
            if head_src is not None:
                try:
                    head_tree = ast.parse(head_src, filename=str(mod_path))
                    for node in head_tree.body:
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                            start = node.lineno - 1
                            end = getattr(node, "end_lineno", None)
                            lines = head_src.splitlines()
                            body_src = "\n".join(lines[start:end]) if end else head_src
                            head_hash = _hash_text(body_src)
                            if head_hash != cur.get("hash"):
                                changed = True
                            break
                except Exception:
                    pass
            if changed:
                impacted.add(test)

    for t in sorted(impacted):
        print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


