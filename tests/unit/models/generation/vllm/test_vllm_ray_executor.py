# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from types import ModuleType
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def ray_executor_module() -> Iterator[ModuleType]:
    """Import the NeMo executor without requiring vLLM in the test venv."""

    module_name = "nemo_rl.models.generation.vllm.ray_executor"
    previous_module = sys.modules.pop(module_name, None)

    vllm = ModuleType("vllm")
    vllm.__path__ = []  # type: ignore[attr-defined]
    vllm_v1 = ModuleType("vllm.v1")
    vllm_v1.__path__ = []  # type: ignore[attr-defined]
    vllm_executor = ModuleType("vllm.v1.executor")
    vllm_executor.__path__ = []  # type: ignore[attr-defined]
    vllm_ray_executor = ModuleType(
        "vllm.v1.executor.ray_executor"
    )

    class FakeRayDistributedExecutor:
        def _init_workers_ray(self, placement_group, **kwargs):
            raise NotImplementedError

    vllm_ray_executor.RayDistributedExecutor = (
        FakeRayDistributedExecutor
    )
    fake_vllm_modules = {
        "vllm": vllm,
        "vllm.v1": vllm_v1,
        "vllm.v1.executor": vllm_executor,
        "vllm.v1.executor.ray_executor": vllm_ray_executor,
    }
    try:
        with patch.dict(sys.modules, fake_vllm_modules):
            module = importlib.import_module(module_name)
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module


def test_custom_executor_injects_authenticated_nested_runtime_env(
    ray_executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    desired = {
        "py_executable": "/venv/bin/python",
        "env_vars": {"RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"},
    }
    merged = {
        **desired,
        "nsight": {"cuda-graph-trace": "node"},
    }
    observed: dict[str, object] = {}

    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    monkeypatch.setattr(
        ray_executor_module.ray_constants,
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV",
        False,
    )
    monkeypatch.setattr(
        ray_executor_module,
        "load_nested_runtime_env_contract",
        lambda: (desired, "a" * 64),
    )

    def merge_runtime_env(existing, requested):
        observed["existing"] = existing
        observed["requested"] = requested
        return merged

    monkeypatch.setattr(
        ray_executor_module,
        "merge_nested_runtime_env",
        merge_runtime_env,
    )

    def parent_init(self, placement_group, **kwargs):
        observed["placement_group"] = placement_group
        observed["kwargs"] = kwargs
        return "initialized"

    monkeypatch.setattr(
        ray_executor_module.RayDistributedExecutor,
        "_init_workers_ray",
        parent_init,
    )
    executor = object.__new__(
        ray_executor_module.NemoRayDistributedExecutor
    )

    assert (
        executor._init_workers_ray(
            "placement-group",
            runtime_env={"env_vars": {"EXISTING": "1"}},
            num_cpus=1,
        )
        == "initialized"
    )
    assert observed["existing"] == {
        "env_vars": {"EXISTING": "1"}
    }
    assert observed["requested"] == desired
    assert observed["placement_group"] == "placement-group"
    assert observed["kwargs"] == {
        "runtime_env": merged,
        "num_cpus": 1,
    }


@pytest.mark.parametrize(
    ("environment_value", "import_constant"),
    (("1", False), ("0", True)),
)
def test_custom_executor_rejects_uncontrolled_ray_uv_environment(
    ray_executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    environment_value: str,
    import_constant: bool,
) -> None:
    monkeypatch.setenv(
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV",
        environment_value,
    )
    monkeypatch.setattr(
        ray_executor_module.ray_constants,
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV",
        import_constant,
    )
    executor = object.__new__(
        ray_executor_module.NemoRayDistributedExecutor
    )

    with pytest.raises(
        ray_executor_module.NestedRuntimeEnvContractError,
        match="RAY_ENABLE_UV_RUN_RUNTIME_ENV|Ray imported",
    ):
        executor._init_workers_ray("placement-group")
