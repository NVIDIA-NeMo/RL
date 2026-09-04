# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm.vllm_worker import (
    _build_vllm_nested_runtime_env,
    _configure_vllm_ray_worker_nsight,
    _get_vllm_inner_nsys_config,
    _get_vllm_inner_nsys_mode,
)


ENV = "NRL_VLLM_INNER_NSYS_MODE"
RAY_UV_ENV = "RAY_ENABLE_UV_RUN_RUNTIME_ENV"


def clear_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        ENV,
        RAY_UV_ENV,
        "NCCL_CUMEM_ENABLE",
        "NCCL_NVLS_ENABLE",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "NRL_VLLM_PROPAGATE_PYTHONPATH",
        "NRL_FORCED_SEQUENCE_AUDIT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_keeps_existing_vllm_managed_nsight_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_runtime_env(monkeypatch)
    assert _get_vllm_inner_nsys_mode() is None
    assert _build_vllm_nested_runtime_env("/venv/python") == {
        "py_executable": "/venv/python",
        "env_vars": {RAY_UV_ENV: "0"},
    }

    kwargs = {"distributed_executor_backend": "ray"}
    mode = _configure_vllm_ray_worker_nsight(
        kwargs, profiling_requested=True
    )
    assert mode is None
    assert kwargs["ray_workers_use_nsight"] is True

    no_profile_kwargs = {"distributed_executor_backend": "ray"}
    _configure_vllm_ray_worker_nsight(
        no_profile_kwargs, profiling_requested=False
    )
    assert "ray_workers_use_nsight" not in no_profile_kwargs


def test_nested_runtime_env_propagates_present_nccl_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_runtime_env(monkeypatch)
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "0")

    runtime_env = _build_vllm_nested_runtime_env("/venv/python")

    assert runtime_env["env_vars"] == {
        RAY_UV_ENV: "0",
        "NCCL_CUMEM_ENABLE": "1",
        "NCCL_NVLS_ENABLE": "0",
    }


def test_cuda_graph_injects_bounded_graph_level_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_runtime_env(monkeypatch)
    monkeypatch.setenv(ENV, "cuda_graph")
    expected = {
        "t": "cuda,nvtx",
        "o": "'worker_process_%p'",
        "stop-on-exit": "true",
        "capture-range": "cudaProfilerApi",
        "capture-range-end": "stop",
        "cuda-graph-trace": "graph",
    }
    assert _get_vllm_inner_nsys_config("cuda_graph") == expected
    assert _build_vllm_nested_runtime_env("/venv/python") == {
        "py_executable": "/venv/python",
        "nsight": expected,
        "env_vars": {
            RAY_UV_ENV: "0",
            ENV: "cuda_graph",
        },
    }

    kwargs = {
        "distributed_executor_backend": "ray",
        "ray_workers_use_nsight": True,
    }
    assert (
        _configure_vllm_ray_worker_nsight(
            kwargs, profiling_requested=True
        )
        == "cuda_graph"
    )
    assert kwargs["ray_workers_use_nsight"] is False


def test_cuda_hw_injects_bounded_hardware_trace_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_runtime_env(monkeypatch)
    monkeypatch.setenv(ENV, "cuda_hw")
    config = _get_vllm_inner_nsys_config("cuda_hw")
    assert config == {
        "t": "cuda-hw,nvtx",
        "o": "'worker_process_%p'",
        "stop-on-exit": "true",
        "capture-range": "cudaProfilerApi",
        "capture-range-end": "stop",
    }
    assert "cuda-graph-trace" not in config

    runtime_env = _build_vllm_nested_runtime_env("/venv/python")
    assert runtime_env["nsight"] == config
    assert runtime_env["env_vars"][ENV] == "cuda_hw"


def test_cuda_node_injects_bounded_graph_node_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_runtime_env(monkeypatch)
    monkeypatch.setenv(ENV, "cuda_node")
    expected = {
        "t": "cuda,nvtx",
        "o": "'worker_process_%p'",
        "stop-on-exit": "true",
        "capture-range": "cudaProfilerApi",
        "capture-range-end": "stop",
        "cuda-graph-trace": "node",
    }
    assert _get_vllm_inner_nsys_config("cuda_node") == expected
    runtime_env = _build_vllm_nested_runtime_env("/venv/python")
    assert runtime_env == {
        "py_executable": "/venv/python",
        "nsight": expected,
        "env_vars": {
            RAY_UV_ENV: "0",
            ENV: "cuda_node",
        },
    }


def test_async_worker_runtime_provenance_reports_exact_applied_nsight_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_runtime_env(monkeypatch)
    monkeypatch.setenv(ENV, "cuda_node")
    expected = _get_vllm_inner_nsys_config("cuda_node")

    from nemo_rl.models.generation.vllm import vllm_worker_async

    worker_class = (
        vllm_worker_async.VllmAsyncGenerationWorker.__ray_metadata__
        .modified_class
    )
    worker = SimpleNamespace(
        model_name="Qwen/Qwen3-30B-A3B",
        llm_async_engine_args=SimpleNamespace(
            logits_processors=[],
            ray_workers_use_nsight=False,
            enable_prefix_caching=False,
            enforce_eager=True,
            max_num_seqs=128,
            max_num_batched_tokens=8192,
            model="Qwen/Qwen3-30B-A3B",
            revision=None,
        ),
        _vllm_nested_runtime_env={
            "py_executable": "/venv/python",
            "nsight": dict(expected),
            "env_vars": {ENV: "cuda_node"},
        },
        _vllm_nested_runtime_env_patch_verified=True,
        _vllm_nested_runtime_env_contract_sha256="a" * 64,
        _vllm_env_copy_layout="custom_executor_class",
        vllm_model_worker_runtime_provenance=[
            {"python_executable": "/venv/python"}
        ],
    )

    provenance = worker_class.report_replay_runtime_provenance(worker)

    assert provenance["inner_nsys_mode"] == "cuda_node"
    assert provenance["inner_nsys_config"] == expected
    assert provenance["vllm_nested_runtime_env_contract_exported"] is True
    assert provenance["inner_nsys_runtime_env_patch_verified"] is True
    assert (
        provenance["vllm_nested_runtime_env_contract_sha256"] == "a" * 64
    )
    assert provenance["vllm_env_copy_layout"] == "custom_executor_class"
    assert provenance["model_worker_runtime_provenance"] == [
        {"python_executable": "/venv/python"}
    ]
    assert provenance["ray_workers_use_nsight"] is False
    assert provenance["async_engine_args"]["enforce_eager"] is True
    provenance["inner_nsys_config"]["cuda-graph-trace"] = "graph"
    assert (
        worker._vllm_nested_runtime_env["nsight"]["cuda-graph-trace"]
        == "node"
    )


def test_invalid_inner_nsys_mode_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_runtime_env(monkeypatch)
    monkeypatch.setenv(ENV, "node")
    with pytest.raises(ValueError, match="cuda_graph, cuda_hw, cuda_node"):
        _get_vllm_inner_nsys_mode()
    with pytest.raises(ValueError, match="cuda_graph, cuda_hw, cuda_node"):
        _build_vllm_nested_runtime_env("/venv/python")

    kwargs = {"distributed_executor_backend": "ray"}
    with pytest.raises(ValueError, match="cuda_graph, cuda_hw, cuda_node"):
        _configure_vllm_ray_worker_nsight(
            kwargs, profiling_requested=True
        )
    assert "ray_workers_use_nsight" not in kwargs


def test_inner_nsys_mode_preserves_pythonpath_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_runtime_env(monkeypatch)
    monkeypatch.setenv(ENV, "cuda_hw")
    monkeypatch.setenv("NRL_VLLM_PROPAGATE_PYTHONPATH", "1")
    monkeypatch.setenv("PYTHONPATH", "/repo")
    monkeypatch.setenv("NRL_FORCED_SEQUENCE_AUDIT", "1")
    runtime_env = _build_vllm_nested_runtime_env("/venv/python")
    assert runtime_env["env_vars"] == {
        RAY_UV_ENV: "0",
        "PYTHONPATH": "/repo",
        "NRL_VLLM_PROPAGATE_PYTHONPATH": "1",
        "NRL_FORCED_SEQUENCE_AUDIT": "1",
        ENV: "cuda_hw",
    }
