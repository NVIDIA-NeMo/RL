# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.trtllm.config import resolve_role_overrides
from nemo_rl.models.generation.trtllm.trtllm_worker_async import (
    TrtllmAsyncGenerationWorkerImpl,
)

pytestmark = pytest.mark.trtllm


def _config():
    return {
        "backend": "trtllm",
        "model_name": "test/model",
        "max_new_tokens": 4,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 20,
        "stop_token_ids": [9],
        "stop_strings": None,
        "_pad_token_id": 0,
        "colocated": {
            "enabled": True,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        },
        "trtllm_cfg": {
            "tensor_parallel_size": 1,
            "max_model_len": 128,
            "precision": "bfloat16",
            "max_batch_size": 8,
            "max_num_tokens": 256,
            "async_engine": True,
        },
    }


def _worker():
    worker = TrtllmAsyncGenerationWorkerImpl.__new__(TrtllmAsyncGenerationWorkerImpl)
    worker.cfg = _config()
    worker.model_name = worker.cfg["model_name"]
    worker.is_model_owner = True
    worker.llm = AsyncMock()
    worker.TrtSamplingParams = MagicMock()
    return worker


def test_configure_worker_reserves_gpus_for_internal_trtllm_actors(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")

    resources, env_vars, init_kwargs, actor_options = (
        TrtllmAsyncGenerationWorkerImpl.configure_worker(
            num_gpus=2,
            bundle_indices=(0, [2, 3]),
        )
    )

    assert resources == {"num_gpus": 0, "num_cpus": 0}
    assert env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"
    assert env_vars["NCCL_CUMEM_ENABLE"] == "1"
    assert env_vars["PATH"].endswith(":/usr/bin")
    assert init_kwargs == {
        "bundle_indices": [2, 3],
        "bundle_pg_idx": 0,
        "fraction_of_gpus": 2,
    }
    assert actor_options == {}


@pytest.mark.asyncio
async def test_generate_async_converts_padded_batch_and_logprobs():
    worker = _worker()
    sampling_params = object()
    worker._build_sampling_params = MagicMock(return_value=sampling_params)

    responses = {
        (11, 12): SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[21, 22], logprobs=[-0.1, -0.2])]
        ),
        (13,): SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[23],
                    logprobs=[{23: SimpleNamespace(logprob=-0.3)}],
                )
            ]
        ),
    }

    async def generate_async(*, inputs, sampling_params):
        assert sampling_params is not None
        return responses[tuple(inputs["prompt_token_ids"])]

    worker.llm.generate_async.side_effect = generate_async
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[11, 12], [13, 0]]),
            "input_lengths": torch.tensor([2, 1]),
        }
    )

    output = await worker.generate_async(data, greedy=True)

    worker._build_sampling_params.assert_called_once_with(greedy=True)
    assert worker.llm.generate_async.await_count == 2
    assert torch.equal(
        output["output_ids"], torch.tensor([[11, 12, 21, 22], [13, 23, 0, 0]])
    )
    assert torch.equal(output["generation_lengths"], torch.tensor([2, 1]))
    assert torch.equal(output["unpadded_sequence_lengths"], torch.tensor([4, 2]))
    assert torch.allclose(
        output["logprobs"],
        torch.tensor([[0.0, 0.0, -0.1, -0.2], [0.0, -0.3, 0.0, 0.0]]),
    )


@pytest.mark.asyncio
async def test_generate_async_rejects_non_right_padded_input():
    worker = _worker()
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[11, 0, 12]]),
            "input_lengths": torch.tensor([1]),
        }
    )

    with pytest.raises(ValueError, match="Non-padding values"):
        await worker.generate_async(data)
    worker.llm.generate_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_async_empty_batch_does_not_call_engine():
    worker = _worker()
    data = BatchedDataDict(
        {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "input_lengths": torch.empty(0, dtype=torch.long),
        }
    )

    output = await worker.generate_async(data)

    assert output["output_ids"].shape == (0, 0)
    assert output["logprobs"].shape == (0, 0)
    worker.llm.generate_async.assert_not_awaited()


def test_build_sampling_params_null_top_k_maps_to_zero():
    """null top_k (default) must reach TRT-LLM as 0, its spelling of 'no restriction'."""
    worker = _worker()
    worker.cfg["top_k"] = None

    worker._build_sampling_params(greedy=False)

    worker.TrtSamplingParams.assert_called_once_with(
        temperature=0.8,
        top_p=0.9,
        top_k=0,
        max_tokens=4,
        stop_token_ids=[9],
        include_stop_str_in_output=True,
        logprobs=True,
        logprobs_simple_format=True,
    )


def test_build_sampling_params_supports_async_greedy_and_sampling_modes():
    worker = _worker()

    worker._build_sampling_params(greedy=True)
    worker.TrtSamplingParams.assert_called_once_with(
        temperature=0.0,
        top_p=0.9,
        top_k=1,
        max_tokens=4,
        stop_token_ids=[9],
        include_stop_str_in_output=True,
        logprobs=True,
        logprobs_simple_format=True,
    )

    worker.TrtSamplingParams.reset_mock()
    worker._build_sampling_params(greedy=False)
    worker.TrtSamplingParams.assert_called_once_with(
        temperature=0.8,
        top_p=0.9,
        top_k=20,
        max_tokens=4,
        stop_token_ids=[9],
        include_stop_str_in_output=True,
        logprobs=True,
        logprobs_simple_format=True,
    )


@pytest.mark.asyncio
async def test_async_lifecycle_resets_cache_before_sleep_and_resumes_selected_tags():
    worker = _worker()
    worker._all_sleep_tags = MagicMock(return_value=["weights", "kv_cache"])
    worker._resolve_wake_tags = MagicMock(return_value=["weights"])

    await worker.sleep_async()
    await worker.wake_up_async(tags=["weights"])

    assert worker.llm.method_calls == [
        call.collective_rpc("reset_prefix_cache"),
        call.release(["weights", "kv_cache"]),
        call.resume(["weights"]),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ([True], True),
        ([True, True], True),
        ([True, 1], False),
        ([], False),
        ([False], False),
        ([True, False], False),
        ([True, None], False),
    ],
)
async def test_async_collective_refit_propagates_worker_result(
    result: list[bool | int | None], expected: bool
) -> None:
    worker = _worker()
    worker.llm.collective_rpc.return_value = result

    succeeded = await worker.update_weights_from_collective_async(
        drain=False,
        recompute_kv=True,
    )

    assert succeeded is expected
    worker.llm.collective_rpc.assert_awaited_once_with(
        "update_weights_from_collective",
        kwargs={"drain": False, "recompute_kv": True},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ([True], True),
        ([True, True], True),
        ([True, 1], False),
        ([], False),
        ([False], False),
        ([True, False], False),
        ([True, None], False),
    ],
)
async def test_async_ipc_refit_propagates_every_worker_result(
    result: list[bool | int | None], expected: bool
) -> None:
    worker = _worker()
    worker.llm.collective_rpc.return_value = result

    succeeded = await worker.update_weights_via_ipc_zmq_async()

    assert succeeded is expected
    worker.llm.collective_rpc.assert_awaited_once_with("update_weights_via_ipc_zmq")


@pytest.mark.asyncio
async def test_async_refit_surfaces_worker_exception() -> None:
    worker = _worker()
    worker.llm.collective_rpc.side_effect = RuntimeError("worker is poisoned")

    with pytest.raises(RuntimeError, match="worker is poisoned"):
        await worker.update_weights_via_ipc_zmq_async()
    with pytest.raises(RuntimeError, match="worker is poisoned"):
        await worker.update_weights_from_collective_async(
            drain=True, recompute_kv=False
        )


class _RecordedConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def recorded_async_llm(monkeypatch):
    """Stub tensorrt_llm so the worker's constructor can run without a GPU.

    Returns the ``AsyncLLM`` mock; its call kwargs are what the engine would
    have been built with. The llm_args config classes just record their kwargs.
    """
    import importlib
    import sys
    import types

    # ray.util re-exports the placement_group *function* under the module's
    # name, so attribute access would patch the wrong object.
    placement_group_module = importlib.import_module("ray.util.placement_group")

    from nemo_rl.models.generation.trtllm import trtllm_worker_async

    async_llm = MagicMock(name="AsyncLLM")
    trt = types.ModuleType("tensorrt_llm")
    trt.AsyncLLM = async_llm
    trt.SamplingParams = object
    llm_args = types.ModuleType("tensorrt_llm.llmapi.llm_args")
    for name in (
        "CacheTransceiverConfig",
        "CudaGraphConfig",
        "KvCacheConfig",
        "SchedulerConfig",
        "SleepConfig",
    ):
        setattr(llm_args, name, type(name, (_RecordedConfig,), {}))
    llm_args.CapacitySchedulerPolicy = SimpleNamespace(MAX_UTILIZATION="max")
    llm_args.ExecutorMemoryType = SimpleNamespace(KV_CACHE="kv")
    for mod in (trt, types.ModuleType("tensorrt_llm.llmapi"), llm_args):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(
        placement_group_module,
        "get_current_placement_group",
        lambda: MagicMock(bundle_specs=[]),
    )
    monkeypatch.setattr(
        trtllm_worker_async, "get_nsight_config_if_pattern_matches", lambda _: {}
    )
    return async_llm


def _engine_kwargs(async_llm, cfg) -> dict:
    TrtllmAsyncGenerationWorkerImpl(cfg, bundle_indices=[0], bundle_pg_idx=0)
    return async_llm.call_args.kwargs


def _disagg_worker_config(trtllm_kwargs=None, *, prefill_trtllm_kwargs=None):
    # The shape the driver ships: the layout under generation.disaggregation,
    # the role blocks resolved (configure_generation_config), and this engine's
    # role stamped into trtllm_cfg._engine_roles.
    cfg = _config()
    cfg["colocated"] = {"enabled": False}
    cfg["disaggregation"] = {"enabled": True}
    if trtllm_kwargs is not None:
        cfg["trtllm_kwargs"] = trtllm_kwargs
    if prefill_trtllm_kwargs is not None:
        cfg["trtllm_cfg"]["prefill_engine"] = {"trtllm_kwargs": prefill_trtllm_kwargs}
    cfg = resolve_role_overrides(cfg)
    cfg["trtllm_cfg"]["_engine_roles"] = {"0:0": {"role": "prefill", "ordinal": 0}}
    return cfg


def test_a_disaggregated_engine_gets_the_default_cache_transceiver(
    recorded_async_llm,
):
    # Without a transceiver config TRT-LLM builds no transceiver at all, and
    # every context leg comes back without disaggregated_params.
    kwargs = _engine_kwargs(recorded_async_llm, _disagg_worker_config())

    assert kwargs["cache_transceiver_config"].kwargs == {"backend": "DEFAULT"}


def test_an_explicit_cache_transceiver_config_is_not_overridden(recorded_async_llm):
    cfg = _disagg_worker_config({"cache_transceiver_config": {"backend": "NIXL"}})

    kwargs = _engine_kwargs(recorded_async_llm, cfg)

    assert kwargs["cache_transceiver_config"] == {"backend": "NIXL"}


def test_a_role_kv_cache_config_adds_to_the_shared_one(recorded_async_llm):
    # A role naming only its pool size must keep the shared reuse settings:
    # replacing the whole block would bring back TRT-LLM's default
    # enable_partial_reuse=True, which the disagg recipe turns off.
    cfg = _disagg_worker_config(
        {
            "kv_cache_config": {
                "enable_block_reuse": True,
                "enable_partial_reuse": False,
            },
            "cache_transceiver_config": {"backend": "NIXL"},
        },
        prefill_trtllm_kwargs={
            "kv_cache_config": {"max_tokens": 4096},
            "cache_transceiver_config": {"kv_transfer_timeout_ms": 600000},
        },
    )

    kwargs = _engine_kwargs(recorded_async_llm, cfg)

    assert kwargs["kv_cache_config"].kwargs == {
        "enable_block_reuse": True,
        "enable_partial_reuse": False,
        "max_tokens": 4096,
    }
    assert kwargs["cache_transceiver_config"] == {
        "backend": "NIXL",
        "kv_transfer_timeout_ms": 600000,
    }


def test_a_role_kv_cache_config_keeps_the_role_memory_fraction(recorded_async_llm):
    # A role's own kv_cache_config must not displace its gpu_memory_utilization.
    cfg = _config()
    cfg["colocated"] = {"enabled": False}
    cfg["disaggregation"] = {"enabled": True}
    cfg["trtllm_cfg"]["gpu_memory_utilization"] = 0.7
    cfg["trtllm_cfg"]["prefill_engine"] = {
        "gpu_memory_utilization": 0.4,
        "trtllm_kwargs": {"kv_cache_config": {"max_tokens": 4096}},
    }
    cfg = resolve_role_overrides(cfg)
    cfg["trtllm_cfg"]["_engine_roles"] = {"0:0": {"role": "prefill", "ordinal": 0}}

    kwargs = _engine_kwargs(recorded_async_llm, cfg)

    assert kwargs["kv_cache_config"].kwargs == {
        "max_tokens": 4096,
        "free_gpu_memory_fraction": 0.4,
    }


def test_a_disaggregated_engine_reads_its_role_block(recorded_async_llm):
    cfg = _disagg_worker_config()
    cfg["trtllm_cfg"]["prefill_engine"]["max_batch_size"] = 2

    kwargs = _engine_kwargs(recorded_async_llm, cfg)

    assert kwargs["max_batch_size"] == 2


@pytest.mark.parametrize("disaggregation", [None, {"enabled": False}])
def test_an_aggregated_engine_gets_no_cache_transceiver(
    recorded_async_llm, disaggregation
):
    cfg = _config()
    cfg["colocated"] = {"enabled": False}
    if disaggregation is not None:
        cfg["disaggregation"] = disaggregation

    kwargs = _engine_kwargs(recorded_async_llm, cfg)

    assert "cache_transceiver_config" not in kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [[], [True, False]])
async def test_async_ipc_refit_checks_every_worker_result(result):
    worker = _worker()
    worker.llm.collective_rpc.return_value = result

    assert await worker.update_weights_via_ipc_zmq_async() is False
