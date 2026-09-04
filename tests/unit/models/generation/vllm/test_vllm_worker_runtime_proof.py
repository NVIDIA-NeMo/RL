# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def backend_module() -> Iterator[ModuleType]:
    """Import the worker extension without requiring vLLM in the test venv."""

    module_name = "nemo_rl.models.generation.vllm.vllm_backend"
    previous_module = sys.modules.pop(module_name, None)
    try:
        with patch.dict(sys.modules, {"vllm": ModuleType("vllm")}):
            module = importlib.import_module(module_name)
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module


def test_runtime_proof_normalizes_nsight_and_exposes_only_whitelist(
    backend_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ray
    from ray._private import ray_constants

    proof_env = {
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "NCCL_CUMEM_ENABLE": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NRL_VLLM_RUNTIME_ENV_CONTRACT_SHA256": "a" * 64,
        "PYTHONPATH": "/immutable/replay",
        "NRL_VLLM_PROPAGATE_PYTHONPATH": "1",
        "NRL_FORCED_SEQUENCE_AUDIT": "1",
        "NRL_VLLM_INNER_NSYS_MODE": "cuda_hw",
    }
    for name in backend_module._RUNTIME_ENV_PROOF_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in proof_env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("SHOULD_NOT_APPEAR_IN_RUNTIME_PROOF", "secret")

    nsight = {
        "t": "cuda-hw,nvtx",
        "capture-range": "cudaProfilerApi",
    }
    runtime_context = SimpleNamespace(
        runtime_env={
            "py_executable": "/venv/bin/python",
            "_nsight": nsight,
            "env_vars": {
                **proof_env,
                "SHOULD_NOT_APPEAR_IN_RUNTIME_PROOF": "secret",
            },
            "working_dir": "gcs://not-part-of-the-proof",
        },
        get_actor_id=lambda: "actor-id",
        get_node_id=lambda: "node-id",
    )
    monkeypatch.setattr(ray, "get_runtime_context", lambda: runtime_context)
    monkeypatch.setattr(
        ray_constants,
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV",
        False,
    )

    proof = (
        backend_module.VllmInternalWorkerExtension()
        .report_runtime_environment()
    )

    assert proof["worker_module_file"] == backend_module.__file__
    assert proof["selected_env"] == proof_env
    assert proof["ray_runtime_env"] == {
        "py_executable": "/venv/bin/python",
        "nsight": nsight,
        "env_vars": proof_env,
    }
    assert proof["actor_id"] == "actor-id"
    assert proof["node_id"] == "node-id"
    assert "SHOULD_NOT_APPEAR_IN_RUNTIME_PROOF" not in str(proof)
    assert "_nsight" not in proof["ray_runtime_env"]


def test_runtime_proof_rejects_conflicting_nsight_spellings(
    backend_module: ModuleType,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="conflicting 'nsight' and '_nsight'",
    ):
        backend_module._sanitize_ray_runtime_env(
            {
                "nsight": {"t": "cuda,nvtx"},
                "_nsight": {"t": "cuda-hw,nvtx"},
            }
        )


def test_runtime_proof_accepts_matching_nsight_spellings(
    backend_module: ModuleType,
) -> None:
    nsight = {"t": "cuda,nvtx"}
    assert backend_module._sanitize_ray_runtime_env(
        {
            "nsight": dict(nsight),
            "_nsight": dict(nsight),
        }
    ) == {"nsight": nsight}
