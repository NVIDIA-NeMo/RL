import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from experiments.nemo_gym_phase2.validate_ray_runtime_env import (
    _runtime_env,
    _sha256,
    _validate_snapshot,
    _write_result,
)


@pytest.fixture(scope="session", autouse=True)
def init_ray_cluster() -> Iterator[None]:
    """Keep these pure helper tests independent of a local Ray control plane."""
    yield


@pytest.fixture(scope="session", autouse=True)
def ray_gpu_monitor() -> Iterator[None]:
    yield None


@pytest.fixture(scope="session", autouse=True)
def session_data(_unit_test_data: Any) -> Iterator[Any]:
    yield _unit_test_data


def _fake_snapshot(
    root: Path,
    *,
    role: str = "runtime_task_0",
    ray_version: str = "2.56.1",
) -> tuple[dict[str, object], Path, Path]:
    environment = root / "environment"
    executable = environment / "bin" / "python"
    ray_module = environment / "lib" / "python3.13" / "site-packages" / "ray"
    executable.parent.mkdir(parents=True)
    ray_module.mkdir(parents=True)
    executable.touch()
    (ray_module / "__init__.py").touch()
    (ray_module / "node.py").touch()
    return (
        {
            "role": role,
            "python_executable": str(executable),
            "ray_version": ray_version,
            "ray_module": str(ray_module / "__init__.py"),
            "ray_node_module": str(ray_module / "node.py"),
            "virtual_env": str(environment),
            "uv_project_environment": str(environment),
        },
        executable,
        environment,
    )


def test_runtime_env_distinguishes_bare_and_inherited_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "runtime" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setenv("PHASE2_TEST_SENTINEL", "present")

    bare = _runtime_env(executable, inherit_environment=False)
    inherited = _runtime_env(executable, inherit_environment=True)

    assert bare == {"py_executable": str(executable.resolve())}
    assert inherited["py_executable"] == str(executable.resolve())
    assert inherited["env_vars"]["PHASE2_TEST_SENTINEL"] == "present"
    assert inherited["env_vars"]["VIRTUAL_ENV"] == str(executable.parent.parent)
    assert inherited["env_vars"]["UV_PROJECT_ENVIRONMENT"] == str(
        executable.parent.parent
    )


def test_runtime_env_preserves_venv_python_symlink(tmp_path: Path) -> None:
    base_python = tmp_path / "base" / "bin" / "python3.13"
    base_python.parent.mkdir(parents=True)
    base_python.touch()
    venv_python = tmp_path / "runtime" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)

    runtime_env = _runtime_env(venv_python, inherit_environment=False)

    assert runtime_env["py_executable"] == str(venv_python)
    assert runtime_env["py_executable"] != str(base_python)


def test_validate_snapshot_accepts_matching_worker(tmp_path: Path) -> None:
    snapshot, executable, environment = _fake_snapshot(tmp_path)

    _validate_snapshot(
        snapshot,
        expected_role="runtime_task_0",
        expected_python=executable,
        expected_ray_version="2.56.1",
        expected_environment=environment,
        require_environment_variables=True,
    )


def test_validate_snapshot_accepts_alias_in_same_venv(tmp_path: Path) -> None:
    snapshot, expected_python, environment = _fake_snapshot(tmp_path)
    python_alias = expected_python.with_name("python3")
    python_alias.symlink_to(expected_python.name)
    snapshot["python_executable"] = str(python_alias)

    _validate_snapshot(
        snapshot,
        expected_role="runtime_task_0",
        expected_python=expected_python,
        expected_ray_version="2.56.1",
        expected_environment=environment,
        require_environment_variables=True,
    )


def test_validate_snapshot_distinguishes_venv_from_base_interpreter(
    tmp_path: Path,
) -> None:
    snapshot, venv_python, environment = _fake_snapshot(tmp_path)
    base_python = tmp_path / "base" / "bin" / "python3.13"
    base_python.parent.mkdir(parents=True)
    base_python.touch()
    venv_python.unlink()
    venv_python.symlink_to(base_python)
    snapshot["python_executable"] = str(base_python)

    with pytest.raises(RuntimeError, match="Python mismatch"):
        _validate_snapshot(
            snapshot,
            expected_role="runtime_task_0",
            expected_python=venv_python,
            expected_ray_version="2.56.1",
            expected_environment=environment,
            require_environment_variables=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("role", "wrong_role", "worker returned role"),
        ("ray_version", "2.55.1", "Ray mismatch"),
        ("virtual_env", None, "virtual_env is not a string"),
    ],
)
def test_validate_snapshot_rejects_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    snapshot, executable, environment = _fake_snapshot(tmp_path)
    snapshot[field] = value

    with pytest.raises((RuntimeError, TypeError), match=message):
        _validate_snapshot(
            snapshot,
            expected_role="runtime_task_0",
            expected_python=executable,
            expected_ray_version="2.56.1",
            expected_environment=environment,
            require_environment_variables=True,
        )


def test_write_result_is_atomic_and_removes_temporary_file(tmp_path: Path) -> None:
    output = tmp_path / "evidence" / "result.json"

    _write_result(output, {"status": "passed"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "passed"}
    assert list(output.parent.glob(".*.tmp")) == []


def test_sha256_records_exact_validator_bytes(tmp_path: Path) -> None:
    source = tmp_path / "validator.py"
    source.write_bytes(b"first\nsecond\n")

    assert (
        _sha256(source)
        == "dbea9325179efe46ea2add94f7b6b745ca983fabb208dc6d34aa064623d7ee23"
    )
