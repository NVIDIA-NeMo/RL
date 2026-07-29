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
import pickle
import urllib.error
from typing import Any

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.dynamo import DynamoConfig, DynamoGeneration
from nemo_rl.models.generation.dynamo import dynamo_generation as _dynmod


def _runtime_cfg(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "engine_world_size": 1,
        "namespace": None,
        "frontend_port": 0,
        "dynamo_python": "/opt/dynamo_venv/bin/python",
        "startup_timeout_s": 300.0,
        "request_timeout_s": 30.0,
        "etcd_port": 0,
        "etcd_peer_port": 0,
        "nats_port": 0,
        "system_port_base": 29000,
        "metrics_include_prefixes": None,
        "metrics_exclude_prefixes": None,
        "worker_args": {
            "tool_call_parser": None,
            "reasoning_parser": None,
            "exclude_tools_when_tool_choice_none": True,
            "enable_structural_tag": False,
            "structural_tag_scope": "auto",
            "structural_tag_schema": "auto",
            "custom_jinja_template": None,
            "endpoint_types": ["chat", "completions"],
            "extra_cli_args": [],
        },
        "frontend_args": {
            "tokenizer": "default",
            "tokenizer_cache": False,
            "tokenizer_cache_bytes": 52428800,
            "router_mode": "round-robin",
            "router_reset_states": True,
            "extra_cli_args": [],
        },
    }
    config.update(overrides)
    return config


def _base_config(**runtime_overrides: Any) -> DynamoConfig:
    return {
        "backend": "dynamo",
        "model_name": "Qwen/Qwen3-0.6B",
        "max_new_tokens": 16,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "_pad_token_id": 0,
        "dynamo_cfg": _runtime_cfg(**runtime_overrides),
    }


def _generation_data(
    input_ids: list[list[int]],
    input_lengths: list[int],
    stop_strings: list[list[str] | None] | None = None,
) -> BatchedDataDict:
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        }
    )
    if stop_strings is not None:
        data["stop_strings"] = stop_strings
    return data


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    workers: list[dict[str, Any]] | None = None,
    calls: list[Any] | None = None,
) -> None:
    from nemo_rl.models.generation.dynamo import managed_runtime as runtime_module

    fixed_workers = workers or [
        {"instance_id": "worker-0", "system_url": "http://10.0.0.2:29000"}
    ]
    runtime_calls = calls if calls is not None else []

    class FakeManagedRuntime:
        frontend_url = "http://10.0.0.1:8000/v1"

        def __init__(self, *, cluster, config, dynamo_cfg):
            runtime_calls.append(("init", cluster, config["model_name"]))

        def refit_workers(self):
            return [dict(worker) for worker in fixed_workers]

        def validate_workers(self, expected):
            runtime_calls.append(("validate", expected))
            return expected

        def shutdown(self):
            runtime_calls.append(("shutdown",))

    monkeypatch.setattr(runtime_module, "ManagedDynamoRuntime", FakeManagedRuntime)


def _completion_response(token_ids: list[int]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "logprobs": {"token_logprobs": [-0.25] * len(token_ids)},
            }
        ],
        "nvext": {"completion_token_ids": token_ids},
    }


def test_managed_dynamo_requires_inference_cluster() -> None:
    with pytest.raises(RuntimeError, match="non-colocated inference"):
        DynamoGeneration(cluster=None, config=_base_config())


def test_managed_runtime_owns_fixed_workers_and_shutdown(monkeypatch) -> None:
    calls: list[Any] = []
    workers = [
        {"instance_id": "worker-a", "system_url": "http://10.0.0.2:29000"},
        {"instance_id": "worker-b", "system_url": "http://10.0.0.3:29001"},
    ]
    _patch_runtime(monkeypatch, workers=workers, calls=calls)

    generation = DynamoGeneration(
        cluster=object(),
        config=_base_config(engine_world_size=2),
    )

    assert generation.dp_openai_server_base_urls == ["http://10.0.0.1:8000/v1"]
    assert generation.get_inference_world_size() == 4
    assert generation._validate_refit_workers() == workers
    assert generation.shutdown() is True
    assert calls[-1] == ("shutdown",)


def test_token_wrapper_is_used_for_openai_rollouts(monkeypatch) -> None:
    _patch_runtime(monkeypatch)
    created = []

    class FakeTokenWrapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.stopped = False
            created.append(self)

        def start(self):
            return "http://127.0.0.1:9000/v1"

        def shutdown(self):
            self.stopped = True

    monkeypatch.setattr(_dynmod, "DynamoTokenWrapperServer", FakeTokenWrapper)
    config = _base_config()
    config["vllm_cfg"] = {"expose_http_server": True}
    tokenizer = object()

    generation = DynamoGeneration(
        cluster=object(),
        config=config,
        tokenizer=tokenizer,
        tokenizer_config={"chat_template_kwargs": {"enable_thinking": False}},
    )

    assert generation.dp_openai_server_base_urls == ["http://127.0.0.1:9000/v1"]
    assert generation._completion_url() == "http://10.0.0.1:8000/v1/completions"
    assert created[0].kwargs["tokenizer"] is tokenizer
    assert created[0].kwargs["tokenizer_chat_template_kwargs"] == {
        "enable_thinking": False
    }
    generation.shutdown()
    assert created[0].stopped


def test_direct_sync_and_async_generation(monkeypatch) -> None:
    _patch_runtime(monkeypatch)
    requests = []

    def fake_post(url, payload, timeout_s):
        requests.append((url, payload, timeout_s))
        return _completion_response([8, 9])

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post)
    config = _base_config()
    config["vllm_cfg"] = {"max_model_len": 5}
    generation = DynamoGeneration(cluster=object(), config=config)
    data = _generation_data([[1, 2, 3, 0]], [3], [["sample-stop"]])

    sync_output = generation.generate(data)

    async def collect():
        return [item async for item in generation.generate_async(data)]

    async_output = asyncio.run(collect())

    assert sync_output["output_ids"].tolist() == [[1, 2, 3, 8, 9, 0]]
    assert async_output[0][0] == 0
    assert async_output[0][1]["output_ids"].tolist() == [[1, 2, 3, 8, 9]]
    assert all(request[1]["max_tokens"] == 2 for request in requests)
    assert all(request[1]["stop"] == ["sample-stop"] for request in requests)


def test_direct_generation_retries_transient_errors(monkeypatch) -> None:
    _patch_runtime(monkeypatch)
    responses = [
        {"status": "error", "http_status": 503, "raw": "busy"},
        {"status": "error", "transport_error": "reset"},
        _completion_response([7]),
    ]
    monkeypatch.setattr(
        _dynmod,
        "_http_post_json",
        lambda *args, **kwargs: responses.pop(0),
    )
    monkeypatch.setattr(_dynmod.time, "sleep", lambda _: None)
    generation = DynamoGeneration(cluster=object(), config=_base_config())

    output = generation.generate(_generation_data([[1]], [1]))

    assert output["output_ids"].tolist() == [[1, 7]]
    assert responses == []


def test_completion_parser_rejects_misaligned_logprobs() -> None:
    response = _completion_response([8, 9])
    response["choices"][0]["logprobs"]["token_logprobs"] = [-0.1]

    with pytest.raises(RuntimeError, match="1 token logprobs for 2"):
        _dynmod._parse_dynamo_completion_response(
            response,
            request_url="http://dynamo/v1/completions",
        )


def test_refit_rank_offsets_transaction_and_cache_flush(monkeypatch) -> None:
    workers = [
        {"instance_id": "worker-a", "system_url": "http://10.0.0.2:29000"},
        {"instance_id": "worker-b", "system_url": "http://10.0.0.3:29001"},
    ]
    _patch_runtime(monkeypatch, workers=workers)
    init_calls = []
    update_calls = []
    flush_calls = []

    monkeypatch.setattr(
        _dynmod._post_dynamo_worker_route_remote,
        "remote",
        lambda **kwargs: init_calls.append(kwargs) or f"init-{len(init_calls)}",
    )
    monkeypatch.setattr(
        _dynmod._update_dynamo_worker_weights_remote,
        "remote",
        lambda **kwargs: update_calls.append(kwargs) or f"update-{len(update_calls)}",
    )

    generation = DynamoGeneration(
        cluster=object(),
        config=_base_config(engine_world_size=2),
    )
    generation.prepare_refit_info(
        {
            "model.embed.weight": (torch.Size([4, 8]), torch.bfloat16),
            "model.norm.weight": (torch.Size([8]), torch.float32),
        }
    )

    assert generation.init_collective(
        ip="10.1.0.1",
        port=23456,
        world_size=7,
        train_world_size=3,
    ) == ["init-1", "init-2"]
    assert [call["payload"]["init_info"]["rank_offset"] for call in init_calls] == [
        3,
        5,
    ]
    assert generation.update_weights_from_collective() == ["update-1", "update-2"]
    assert update_calls[0]["update_info"] == {
        "names": ["model.embed.weight", "model.norm.weight"],
        "dtype_names": ["bfloat16", "float32"],
        "shapes": [[4, 8], [8]],
        "packed": True,
    }

    monkeypatch.setattr(
        _dynmod._post_dynamo_worker_route_remote,
        "remote",
        lambda **kwargs: flush_calls.append(kwargs) or _dynmod.ray.put(True),
    )
    assert generation.invalidate_kv_cache()
    assert [
        (call["system_url"], call["route"], call["payload"]) for call in flush_calls
    ] == [
        ("http://10.0.0.2:29000", "flush_cache", {}),
        ("http://10.0.0.3:29001", "flush_cache", {}),
    ]


def test_native_weight_update_uses_unpaused_layerwise_transaction(monkeypatch) -> None:
    calls = []

    def fake_post(url, payload, timeout_s):
        calls.append((url, payload, timeout_s))
        return {"status": "ok"}

    monkeypatch.setattr(_dynmod, "_http_post_json", fake_post)

    assert _dynmod._update_dynamo_worker_weights_remote._function(
        system_url="http://worker:29000",
        update_info={"names": ["weight"]},
        timeout_s=30,
    )
    assert [call[1]["engine_rpc"] for call in calls] == [
        "start_weight_update",
        "update_weights",
        "finish_weight_update",
    ]
    assert all(call[1]["allow_unpaused"] is True for call in calls)
    assert all(call[1]["reset_prefix_cache"] is False for call in calls)


def test_prometheus_metrics_are_curated_for_logger(monkeypatch) -> None:
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(DynamoGeneration, "_start_metrics_sampler", lambda self: None)
    config = _base_config(metrics_include_prefixes=[], metrics_exclude_prefixes=[])
    config["vllm_cfg"] = {
        "enable_vllm_metrics_logger": True,
        "vllm_metrics_logger_interval": 0.5,
    }
    generation = DynamoGeneration(cluster=object(), config=config)
    text = (
        'dynamo_component_inflight_requests{worker="0"} 3\n'
        'dynamo_component_inflight_requests{worker="1"} 2\n'
        "dynamo_work_handler_queue_depth 1\n"
        "python_gc_objects_collected_total 10\n"
    )

    parsed = _dynmod._parse_prometheus_metrics(
        text,
        generation._metrics_include_prefixes,
        generation._metrics_exclude_prefixes,
    )
    with generation._metrics_lock:
        generation._dynamo_logger_metrics = {
            name: {0: [value]} for name, value in parsed.items()
        }

    metrics = generation.get_logger_metrics()
    assert metrics["inflight_batch_sizes"] == {0: [5.0]}
    assert metrics["num_pending_samples"] == {0: [1.0]}
    generation.clear_logger_metrics()
    assert generation.get_logger_metrics()["inflight_batch_sizes"] == {}


def test_http_get_text_handles_transport_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("refused")),
    )
    assert _dynmod._http_get_text("http://worker/metrics", 1.0) is None


def test_pickle_roundtrip_drops_driver_owned_runtime(monkeypatch) -> None:
    _patch_runtime(monkeypatch)
    generation = DynamoGeneration(cluster=object(), config=_base_config())

    restored = pickle.loads(pickle.dumps(generation))

    assert restored.dp_openai_server_base_urls == ["http://10.0.0.1:8000/v1"]
    assert restored._managed_runtime is None
    assert restored._owns_managed_runtime is False
    assert restored.shutdown() is True
