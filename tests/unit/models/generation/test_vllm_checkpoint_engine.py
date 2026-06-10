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

import asyncio
import concurrent.futures
import importlib.machinery
import importlib.util
import sys
import types

import torch

if importlib.util.find_spec("vllm") is None:
    vllm_stub = types.ModuleType("vllm")
    vllm_stub.__spec__ = importlib.machinery.ModuleSpec("vllm", loader=None)
    sys.modules["vllm"] = vllm_stub

from nemo_rl.models.generation.vllm import vllm_backend
from nemo_rl.models.generation.vllm.vllm_backend import (
    VllmInternalWorkerExtension,
    fix_gpt_oss_export_transpose,
    maybe_preinit_nixl_for_vllm_worker,
)
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
    _resolve_collective_rpc_result,
)


def test_maybe_preinit_nixl_for_vllm_worker_is_idempotent(monkeypatch, capsys):
    calls = []

    def fake_preinit_nixl_agent(*, backend_name, backend_init_params):
        calls.append((backend_name, backend_init_params))
        return "agent"

    monkeypatch.setattr(
        "nemo_rl.utils.checkpoint_engines.nixl.preinit_nixl_agent",
        fake_preinit_nixl_agent,
    )
    worker_wrapper = types.SimpleNamespace()
    config = {"backend_name": "UCX", "backend_init_params": {"device": "cuda"}}

    maybe_preinit_nixl_for_vllm_worker(worker_wrapper, config)
    maybe_preinit_nixl_for_vllm_worker(worker_wrapper, config)

    assert vars(worker_wrapper)["_nrl_nixl_preinit_agent"] == "agent"
    assert calls == [("UCX", {"device": "cuda"})]
    assert "NIXL vLLM worker preinit completed: backend=UCX" in capsys.readouterr().out


def test_fix_gpt_oss_export_transpose_only_changes_down_proj():
    weight = torch.arange(6).reshape(2, 3)

    fixed = fix_gpt_oss_export_transpose("model.mlp.experts.down_proj", weight)
    unchanged = fix_gpt_oss_export_transpose("model.mlp.up_proj", weight)

    assert torch.equal(fixed, weight.transpose(-2, -1).contiguous())
    assert torch.equal(unchanged, weight)


def test_vllm_internal_worker_checkpoint_engine_lifecycle(monkeypatch):
    calls = []

    class FakeCheckpointEngine:
        def prepare(self):
            calls.append("prepare")
            return "metadata"

        def init_rollout_process_group(
            self,
            *,
            rollout_rank,
            train_world_size,
            rollout_world_size,
            metadata,
        ):
            calls.append(
                (
                    rollout_rank,
                    train_world_size,
                    rollout_world_size,
                    metadata,
                )
            )

        def finalize(self):
            calls.append("finalize")

    def fake_create_checkpoint_engine(backend, *, bucket_size_bytes, engine_kwargs):
        calls.append((backend, bucket_size_bytes, engine_kwargs))
        return FakeCheckpointEngine()

    monkeypatch.setattr(
        "nemo_rl.utils.checkpoint_engine.create_checkpoint_engine",
        fake_create_checkpoint_engine,
    )
    monkeypatch.setattr(vllm_backend.torch.distributed, "get_rank", lambda: 2)
    worker = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)

    worker.finalize_checkpoint_engine()
    worker.init_checkpoint_engine("nixl", 1024, {"device": "cuda"})
    worker.init_checkpoint_engine("nixl", 2048, {"ignored": True})
    assert worker.prepare_checkpoint_engine() == "metadata"
    worker.init_checkpoint_engine_process_group(
        rank_prefix=10,
        train_world_size=4,
        rollout_world_size=8,
        metadata=["policy", "rollout"],
    )
    worker.finalize_checkpoint_engine()

    assert calls == [
        ("nixl", 1024, {"device": "cuda"}),
        "prepare",
        (12, 4, 8, ["policy", "rollout"]),
        "finalize",
    ]


def test_vllm_internal_worker_loads_checkpoint_engine_batches(monkeypatch, capsys):
    loaded_batches = []
    processed = []
    synchronized = []
    cleaned = []

    class WeightBatch(list):
        def record_cuda_load_complete(self):
            synchronized.append("event")

    class FakeCheckpointEngine:
        cleanup_after_load = True

        async def receive_weight_batches(self):
            yield WeightBatch([("w0", torch.ones(2))])
            yield [("w1", torch.ones(3, dtype=torch.int8))]

    worker = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    worker.checkpoint_engine = FakeCheckpointEngine()
    worker._load_weights = lambda batch: loaded_batches.append(list(batch))
    worker._maybe_process_fp8_kv_cache = lambda: processed.append(True)
    monkeypatch.setattr(
        vllm_backend.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(
            synchronize=lambda: synchronized.append("stream")
        ),
    )
    monkeypatch.setattr(vllm_backend.gc, "collect", lambda: cleaned.append("gc"))
    monkeypatch.setattr(
        vllm_backend.torch.cuda, "empty_cache", lambda: cleaned.append("cache")
    )

    assert asyncio.run(worker._update_weights_from_checkpoint_engine_async())
    assert [batch[0][0] for batch in loaded_batches] == ["w0", "w1"]
    assert processed == [True]
    assert synchronized == ["event", "stream"]
    assert cleaned == ["gc", "cache"]
    assert "Loaded 2 tensors in 2 batches" in capsys.readouterr().out


def test_vllm_internal_worker_checkpoint_engine_wrapper_handles_errors(
    monkeypatch, capsys
):
    async def raise_error(self):
        raise RuntimeError("load failed")

    worker = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    monkeypatch.setattr(
        VllmInternalWorkerExtension,
        "_update_weights_from_checkpoint_engine_async",
        raise_error,
    )

    assert worker.update_weights_from_checkpoint_engine() is False
    assert "load failed" in capsys.readouterr().out


def test_vllm_generation_checkpoint_engine_worker_methods():
    class FakeWorkerGroup:
        workers = [object(), object(), object(), object()]

        def __init__(self):
            self.single_calls = []
            self.multi_calls = []

        def run_all_workers_single_data(self, method_name, **kwargs):
            self.single_calls.append((method_name, kwargs))
            return [method_name]

        def run_all_workers_multiple_data(self, method_name, **kwargs):
            self.multi_calls.append((method_name, kwargs))
            return [method_name]

        def shutdown(self, *args, **kwargs):
            pass

    checkpoint_config = {"enabled": True, "backend": "nixl"}
    generation = VllmGeneration.__new__(VllmGeneration)
    generation.cfg = {
        "checkpoint_engine": checkpoint_config,
        "vllm_cfg": {"async_engine": False},
    }
    generation.worker_group = FakeWorkerGroup()
    generation.dp_size = 2

    assert generation.get_checkpoint_engine_config() is checkpoint_config
    assert generation.init_checkpoint_engine(
        backend="nixl",
        bucket_size_bytes=4096,
        engine_kwargs={"device": "cuda"},
    ) == ["init_checkpoint_engine"]
    assert generation.prepare_checkpoint_engine() == ["prepare_checkpoint_engine"]
    assert generation.init_checkpoint_engine_process_group(
        metadata=["m0"],
        train_world_size=2,
        rollout_world_size=4,
    ) == ["init_checkpoint_engine_process_group"]
    assert generation.update_weights_from_checkpoint_engine() == [
        "update_weights_from_checkpoint_engine"
    ]

    generation.cfg["vllm_cfg"]["async_engine"] = True
    assert generation.finalize_checkpoint_engine() == [
        "finalize_checkpoint_engine_async"
    ]
    assert generation.worker_group.multi_calls[-1] == (
        "init_checkpoint_engine_process_group",
        {
            "rank_prefix": [0, 2],
            "run_rank_0_only_axes": ["tensor_parallel", "pipeline_parallel"],
            "common_kwargs": {
                "metadata": ["m0"],
                "train_world_size": 2,
                "rollout_world_size": 4,
            },
        },
    )


def test_vllm_generation_worker_checkpoint_engine_rpc_methods():
    class FakeLLM:
        def __init__(self):
            self.calls = []

        def collective_rpc(self, method_name, *, args):
            self.calls.append((method_name, args))
            if method_name == "update_weights_from_checkpoint_engine":
                return [True, None]
            return ["metadata"]

    worker = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker.llm = FakeLLM()

    worker.init_checkpoint_engine("nixl", 1024, {"device": "cuda"})
    assert worker.prepare_checkpoint_engine() == ["metadata"]
    worker.init_checkpoint_engine_process_group(0, 1, 2, ["m"])
    worker.finalize_checkpoint_engine()
    assert worker.update_weights_from_checkpoint_engine()
    assert worker.llm.calls == [
        ("init_checkpoint_engine", ("nixl", 1024, {"device": "cuda"})),
        ("prepare_checkpoint_engine", ()),
        ("init_checkpoint_engine_process_group", (0, 1, 2, ["m"])),
        ("finalize_checkpoint_engine", ()),
        ("update_weights_from_checkpoint_engine", ()),
    ]


def test_resolve_collective_rpc_result_handles_nested_async_values():
    async def nested_value(value):
        return value

    future = concurrent.futures.Future()
    future.set_result((nested_value("tuple"), [nested_value("list"), "plain"]))

    result = asyncio.run(_resolve_collective_rpc_result(nested_value(future)))

    assert result == ("tuple", ["list", "plain"])


def test_vllm_async_generation_worker_checkpoint_engine_rpc_methods():
    class FakeAsyncLLM:
        def __init__(self):
            self.calls = []

        async def collective_rpc(self, method_name, *, args):
            self.calls.append((method_name, args))
            if method_name == "update_weights_from_checkpoint_engine":
                return [True, None]
            return ["metadata"]

    async def run_worker_methods():
        worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
        worker.llm = FakeAsyncLLM()

        await worker.init_checkpoint_engine_async("nixl", 1024, {"device": "cuda"})
        assert await worker.prepare_checkpoint_engine_async() == ["metadata"]
        await worker.init_checkpoint_engine_process_group_async(0, 1, 2, ["m"])
        await worker.finalize_checkpoint_engine_async()
        assert await worker.update_weights_from_checkpoint_engine_async()
        return worker.llm.calls

    assert asyncio.run(run_worker_methods()) == [
        ("init_checkpoint_engine", ("nixl", 1024, {"device": "cuda"})),
        ("prepare_checkpoint_engine", ()),
        ("init_checkpoint_engine_process_group", (0, 1, 2, ["m"])),
        ("finalize_checkpoint_engine", ()),
        ("update_weights_from_checkpoint_engine", ()),
    ]
