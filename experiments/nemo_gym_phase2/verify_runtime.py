#!/usr/bin/env python3
"""Verify one Phase 2 environment against uv.lock and project override policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def intentional_dependency_policy(pyproject: Path) -> dict[str, list[str]]:
    """Return dependency names whose upstream metadata is intentionally overridden."""
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    uv_config = project.get("tool", {}).get("uv", {})
    policy: dict[str, list[str]] = {}
    entries = [
        *(uv_config.get("override-dependencies", [])),
        *(uv_config.get("exclude-dependencies", [])),
    ]
    for raw in entries:
        if not isinstance(raw, str):
            raise TypeError("tool.uv override/exclude entries must be strings")
        try:
            name = Requirement(raw).name
        except InvalidRequirement:
            name = raw
        canonical_name = canonicalize_name(name)
        policy.setdefault(canonical_name, []).append(raw)
    return {name: sorted(values) for name, values in sorted(policy.items())}


def _installed_inventory(
    distributions: Iterable[importlib.metadata.Distribution],
) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    versions: dict[str, set[str]] = {}
    inventory: list[dict[str, str]] = []
    for distribution in distributions:
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            raise RuntimeError(f"installed distribution has no Name: {distribution!r}")
        name = canonicalize_name(raw_name)
        versions.setdefault(name, set()).add(distribution.version)
        inventory.append({"name": name, "version": distribution.version})
    normalized_versions = {
        name: sorted(package_versions) for name, package_versions in versions.items()
    }
    return normalized_versions, sorted(
        inventory, key=lambda item: (item["name"], item["version"])
    )


def find_unsatisfied_requirements(
    distributions: Iterable[importlib.metadata.Distribution] | None = None,
) -> list[dict[str, Any]]:
    """Return unmet installed-distribution requirements, using pip-check semantics."""
    installed_distributions = list(distributions or importlib.metadata.distributions())
    installed, _ = _installed_inventory(installed_distributions)
    marker_environment = default_environment()
    marker_environment["extra"] = ""
    issues: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for distribution in installed_distributions:
        dependent = canonicalize_name(distribution.metadata["Name"])
        for raw_requirement in distribution.requires or []:
            requirement = Requirement(raw_requirement)
            if requirement.marker and not requirement.marker.evaluate(
                marker_environment
            ):
                continue
            required = canonicalize_name(requirement.name)
            installed_versions = installed.get(required, [])
            if not installed_versions:
                reason = "missing"
            elif requirement.specifier and not any(
                requirement.specifier.contains(version, prereleases=True)
                for version in installed_versions
            ):
                reason = "version_mismatch"
            else:
                continue
            key = (dependent, str(requirement), reason)
            if key in seen:
                continue
            seen.add(key)
            issues.append(
                {
                    "dependent": dependent,
                    "dependent_version": distribution.version,
                    "installed_versions": installed_versions,
                    "reason": reason,
                    "required_name": required,
                    "requirement": str(requirement),
                }
            )
    return sorted(
        issues,
        key=lambda item: (
            item["dependent"],
            item["required_name"],
            item["requirement"],
        ),
    )


def classify_requirement_issues(
    issues: list[dict[str, Any]], policy: dict[str, list[str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    intentional: list[dict[str, Any]] = []
    unexpected: list[dict[str, Any]] = []
    for issue in issues:
        required_name = issue.get("required_name")
        if required_name in policy:
            intentional.append({**issue, "project_policy": policy[required_name]})
        else:
            unexpected.append(issue)
    return intentional, unexpected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--python-install-dir", type=Path, required=True)
    parser.add_argument("--uv-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--group", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve(strict=True)
    environment = args.environment.expanduser().resolve(strict=True)
    python_install_dir = args.python_install_dir.expanduser().resolve(strict=True)
    uv_bin = args.uv_bin.expanduser().resolve(strict=True)
    if Path(sys.prefix).resolve() != environment:
        raise RuntimeError(
            f"verifier must run with {environment}/bin/python, got {sys.executable}"
        )
    interpreter = (environment / "bin/python").resolve(strict=True)
    if not interpreter.is_relative_to(python_install_dir):
        raise RuntimeError(
            "formal Phase 2 interpreter is not in the persistent runtime: "
            f"{interpreter} is outside {python_install_dir}"
        )
    if os.environ.get("DG_USE_LOCAL_VERSION") != "0":
        raise RuntimeError("formal Phase 2 runtime requires DG_USE_LOCAL_VERSION=0")
    uv_version = subprocess.run(
        [str(uv_bin), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.match(r"^uv 0\.11\.28(?:\s|$)", uv_version) is None:
        raise RuntimeError(f"formal Phase 2 runs require uv 0.11.28, got {uv_version}")
    pyvenv_config = (environment / "pyvenv.cfg").read_text(encoding="utf-8")
    if re.search(r"^uv\s*=\s*0\.11\.28\s*$", pyvenv_config, re.MULTILINE) is None:
        raise RuntimeError(f"environment was not created by uv 0.11.28: {environment}")

    sync_command = [
        str(uv_bin),
        "sync",
        "--frozen",
        "--check",
        "--directory",
        str(repo),
    ]
    for extra in args.extra:
        sync_command.extend(["--extra", extra])
    for group in args.group:
        sync_command.extend(["--group", group])
    sync_environment = os.environ.copy()
    sync_environment["UV_PROJECT_ENVIRONMENT"] = str(environment)
    sync_result = subprocess.run(
        sync_command,
        env=sync_environment,
        capture_output=True,
        text=True,
    )

    distributions = list(importlib.metadata.distributions())
    _, inventory = _installed_inventory(distributions)
    inventory_payload = json.dumps(
        inventory, separators=(",", ":"), sort_keys=True
    ).encode()
    policy = intentional_dependency_policy(repo / "pyproject.toml")
    issues = find_unsatisfied_requirements(distributions)
    intentional, unexpected = classify_requirement_issues(issues, policy)
    passed = sync_result.returncode == 0 and not unexpected
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "label": args.label,
        "environment": str(environment),
        "interpreter": str(interpreter),
        "python_install_dir": str(python_install_dir),
        "build_environment": {"DG_USE_LOCAL_VERSION": "0"},
        "python": platform.python_version(),
        "uv_version": uv_version,
        "uv_lock_sha256": sha256(repo / "uv.lock"),
        "pyproject_sha256": sha256(repo / "pyproject.toml"),
        "selection": {"extras": sorted(args.extra), "groups": sorted(args.group)},
        "uv_sync_check": {
            "command": sync_command,
            "returncode": sync_result.returncode,
            "stdout": sync_result.stdout.strip(),
            "stderr": sync_result.stderr.strip(),
        },
        "package_inventory": {
            "count": len(inventory),
            "sha256": hashlib.sha256(inventory_payload).hexdigest(),
        },
        "requirement_check": {
            "intentional_project_overrides": intentional,
            "unexpected": unexpected,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(args.output)
    if not passed:
        if sync_result.returncode != 0:
            print("uv sync --frozen --check failed", file=sys.stderr)
        if unexpected:
            print(
                f"found {len(unexpected)} unexpected requirement incompatibilities",
                file=sys.stderr,
            )
        return 1
    print(
        f"{args.label}: lock synchronized; "
        f"{len(intentional)} intentional override differences; 0 unexpected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
