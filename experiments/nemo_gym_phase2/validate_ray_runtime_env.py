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

"""Verify that a pinned Ray control plane can launch runtime-env workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ray


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--py-executable",
        action="append",
        type=Path,
        required=True,
        help="Runtime-env Python to validate; repeat for multiple actor tiers.",
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-ray-version", required=True)
    parser.add_argument(
        "--expected-control-plane-python",
        type=Path,
        required=True,
        help="Python used by Ray for workers without an explicit runtime env.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--address",
        default="auto",
        help="Ray address, or 'local' to start an isolated local control plane.",
    )
    return parser.parse_args(argv)


def _snapshot(role: str) -> dict[str, Any]:
    # Import inside every remote process: this is the module whose absence caused
    # the original worker-registration failure, so a driver-only import is weaker.
    import ray._private.node

    return {
        "role": role,
        "pid": os.getpid(),
        "python_executable": sys.executable,
        "python_executable_resolved": str(Path(sys.executable).resolve()),
        "python_prefix": sys.prefix,
        "python_version": sys.version,
        "ray_version": ray.__version__,
        "ray_module": ray.__file__,
        "ray_node_module": ray._private.node.__file__,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "uv_project_environment": os.environ.get("UV_PROJECT_ENVIRONMENT"),
        "python_path": os.environ.get("PYTHONPATH"),
        "sys_path": sys.path,
    }


@ray.remote
def _task_snapshot(role: str) -> dict[str, Any]:
    return _snapshot(role)


@ray.remote
class _SnapshotActor:
    def snapshot(self, role: str) -> dict[str, Any]:
        return _snapshot(role)


def _absolute_executable(path: Path) -> Path:
    expanded = path.expanduser()
    absolute = Path(os.path.abspath(expanded))
    if not absolute.is_file():
        raise FileNotFoundError(f"Python executable does not exist: {absolute}")
    return absolute


def _same_environment_executable(actual: Path, expected: Path) -> bool:
    if actual == expected:
        return True
    return actual.parent == expected.parent and actual.resolve(
        strict=True
    ) == expected.resolve(strict=True)


def _runtime_env(
    py_executable: Path,
    *,
    inherit_environment: bool,
) -> dict[str, Any]:
    # Keep the venv entry point rather than resolving its interpreter symlink.
    # Ray needs the venv path to discover pyvenv.cfg and its site-packages.
    executable = _absolute_executable(py_executable)
    runtime_env: dict[str, Any] = {"py_executable": str(executable)}
    if not inherit_environment:
        return runtime_env

    venv = executable.parent.parent
    env_vars = dict(os.environ)
    env_vars.update(
        {
            "VIRTUAL_ENV": str(venv),
            "UV_PROJECT_ENVIRONMENT": str(venv),
        }
    )
    runtime_env["env_vars"] = env_vars
    return runtime_env


def _validate_snapshot(
    snapshot: Mapping[str, Any],
    *,
    expected_role: str,
    expected_python: Path,
    expected_ray_version: str,
    expected_environment: Path | None,
    require_environment_variables: bool,
) -> None:
    if snapshot.get("role") != expected_role:
        raise RuntimeError(
            f"{expected_role}: worker returned role {snapshot.get('role')!r}"
        )

    raw_python = snapshot.get("python_executable")
    if not isinstance(raw_python, str):
        raise TypeError(f"{expected_role}: python_executable is not a string")
    actual_python = _absolute_executable(Path(raw_python))
    absolute_expected_python = _absolute_executable(expected_python)
    if not _same_environment_executable(actual_python, absolute_expected_python):
        raise RuntimeError(
            f"{expected_role}: Python mismatch: "
            f"{actual_python} != {absolute_expected_python}"
        )

    actual_ray_version = snapshot.get("ray_version")
    if actual_ray_version != expected_ray_version:
        raise RuntimeError(
            f"{expected_role}: Ray mismatch: "
            f"{actual_ray_version!r} != {expected_ray_version!r}"
        )

    resolved_expected_environment = (
        expected_environment.expanduser().resolve(strict=True)
        if expected_environment is not None
        else None
    )
    for field in ("ray_module", "ray_node_module"):
        raw_module = snapshot.get(field)
        if not isinstance(raw_module, str):
            raise TypeError(f"{expected_role}: {field} is not a string")
        module_path = Path(raw_module).resolve(strict=True)
        if resolved_expected_environment is not None and not module_path.is_relative_to(
            resolved_expected_environment
        ):
            raise RuntimeError(
                f"{expected_role}: {field} is outside the configured environment: "
                f"{module_path} is not under {resolved_expected_environment}"
            )

    if not require_environment_variables:
        return
    if resolved_expected_environment is None:
        raise ValueError(
            f"{expected_role}: environment-variable checks require an environment"
        )
    for field in ("virtual_env", "uv_project_environment"):
        raw_environment = snapshot.get(field)
        if not isinstance(raw_environment, str):
            raise TypeError(f"{expected_role}: {field} is not a string")
        actual_environment = Path(raw_environment).resolve(strict=True)
        if actual_environment != resolved_expected_environment:
            raise RuntimeError(
                f"{expected_role}: {field} mismatch: "
                f"{actual_environment} != {resolved_expected_environment}"
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot(repo: Path) -> dict[str, Any]:
    validator = Path(__file__).resolve(strict=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_status:
        raise RuntimeError(
            "refusing to validate a Ray control plane from tracked source changes: "
            f"{tracked_status}"
        )
    return {
        "repo": str(repo),
        "git_commit": commit,
        "tracked_source_clean": True,
        "validator": str(validator),
        "validator_sha256": _sha256(validator),
    }


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _get_snapshot(
    object_ref: Any,
    *,
    role: str,
    expected_python: Path,
    expected_ray_version: str,
    expected_environment: Path | None,
    require_environment_variables: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    snapshot = ray.get(object_ref, timeout=timeout_seconds)
    if not isinstance(snapshot, dict):
        raise TypeError(
            f"{role}: expected a snapshot object, got {type(snapshot).__name__}"
        )
    _validate_snapshot(
        snapshot,
        expected_role=role,
        expected_python=expected_python,
        expected_ray_version=expected_ray_version,
        expected_environment=expected_environment,
        require_environment_variables=require_environment_variables,
    )
    return snapshot


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    repo = args.repo.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite Ray validation evidence: {output}"
        )
    expected_control_plane_python = _absolute_executable(
        args.expected_control_plane_python
    )
    expected_runtime_pythons = [
        _absolute_executable(path) for path in args.py_executable
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": _utc_now(),
        "expected_ray_version": args.expected_ray_version,
        "expected_control_plane_python": str(expected_control_plane_python),
        "expected_runtime_pythons": [str(path) for path in expected_runtime_pythons],
        "current_probe": "source",
        "runtime_envs": [],
    }
    try:
        result["source"] = _source_snapshot(repo)
        result["current_probe"] = "driver"
        result["driver"] = _snapshot("driver")
        _validate_snapshot(
            result["driver"],
            expected_role="driver",
            expected_python=Path(sys.executable),
            expected_ray_version=args.expected_ray_version,
            expected_environment=None,
            require_environment_variables=False,
        )
        result["current_probe"] = "ray_init"
        context = ray.init(
            address=None if args.address == "local" else args.address,
            log_to_driver=True,
        )
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        if len(alive_nodes) != 1:
            raise RuntimeError(
                f"expected one live validation node, got {len(alive_nodes)}"
            )
        result["cluster"] = {
            "address": args.address,
            "node_id": context.address_info.get("node_id"),
            "alive_node_ids": sorted(str(node.get("NodeID")) for node in alive_nodes),
        }

        result["current_probe"] = "default_task"
        result["default_task"] = _get_snapshot(
            _task_snapshot.remote("default_task"),
            role="default_task",
            expected_python=expected_control_plane_python,
            expected_ray_version=args.expected_ray_version,
            expected_environment=None,
            require_environment_variables=False,
            timeout_seconds=args.timeout_seconds,
        )
        for index, expected_python in enumerate(expected_runtime_pythons):
            expected_environment = expected_python.parent.parent
            bare_runtime_env = _runtime_env(
                expected_python,
                inherit_environment=False,
            )
            runtime_env = _runtime_env(
                expected_python,
                inherit_environment=True,
            )
            entry: dict[str, Any] = {
                "configured_python": str(expected_python),
                "configured_environment": str(expected_environment),
            }
            result["runtime_envs"].append(entry)

            bare_task_role = f"bare_runtime_task_{index}"
            result["current_probe"] = bare_task_role
            bare_task = _task_snapshot.options(runtime_env=bare_runtime_env).remote(
                bare_task_role
            )
            entry["bare_task"] = _get_snapshot(
                bare_task,
                role=bare_task_role,
                expected_python=expected_python,
                expected_ray_version=args.expected_ray_version,
                expected_environment=expected_environment,
                require_environment_variables=False,
                timeout_seconds=args.timeout_seconds,
            )

            bare_actor_role = f"bare_runtime_actor_{index}"
            result["current_probe"] = bare_actor_role
            bare_actor = _SnapshotActor.options(runtime_env=bare_runtime_env).remote()
            entry["bare_actor"] = _get_snapshot(
                bare_actor.snapshot.remote(bare_actor_role),
                role=bare_actor_role,
                expected_python=expected_python,
                expected_ray_version=args.expected_ray_version,
                expected_environment=expected_environment,
                require_environment_variables=False,
                timeout_seconds=args.timeout_seconds,
            )

            task_role = f"runtime_task_{index}"
            result["current_probe"] = task_role
            task = _task_snapshot.options(runtime_env=runtime_env).remote(task_role)
            entry["task"] = _get_snapshot(
                task,
                role=task_role,
                expected_python=expected_python,
                expected_ray_version=args.expected_ray_version,
                expected_environment=expected_environment,
                require_environment_variables=True,
                timeout_seconds=args.timeout_seconds,
            )

            actor_role = f"runtime_actor_{index}"
            result["current_probe"] = actor_role
            actor = _SnapshotActor.options(runtime_env=runtime_env).remote()
            entry["actor"] = _get_snapshot(
                actor.snapshot.remote(actor_role),
                role=actor_role,
                expected_python=expected_python,
                expected_ray_version=args.expected_ray_version,
                expected_environment=expected_environment,
                require_environment_variables=True,
                timeout_seconds=args.timeout_seconds,
            )
        result["status"] = "passed"
        result["current_probe"] = None
    except Exception as exc:
        # This is the top-level evidence boundary: archive every diagnostic
        # failure, then re-raise it without attempting recovery.
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        raise
    finally:
        result["completed_at"] = _utc_now()
        try:
            _write_result(output, result)
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
