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
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.dynamo import DynamoGeneration
from nemo_rl.models.generation.dynamo import dynamo_generation as generation_module
from nemo_rl.models.generation.dynamo import metrics as metrics_module
from nemo_rl.models.generation.dynamo import refit as refit_module
from nemo_rl.models.generation.dynamo.config import (
    VLLM_PACKED_BUFFER_SIZE_BYTES,
    VLLM_PACKED_NUM_BUFFERS,
)
from nemo_rl.models.generation.dynamo.metrics import (
    DynamoMetricsSampler,
    parse_prometheus_metrics,
)
from nemo_rl.models.generation.dynamo.refit import DynamoRefitChannel


def _config(*, tp: int = 1, expose_http_server: bool = False) -> dict[str, Any]:
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
        "colocated": {"enabled": False},
        "dynamo_cfg": {
            "engine": "vllm",
            "startup_timeout_s": 5,
            "request_timeout_s": 30,
            "control_timeout_s": 10,
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
                "tokenizer_cache_bytes": 50 * 1024 * 1024,
                "router_mode": "kv",
                "router_reset_states": True,
                "extra_cli_args": [],
            },
        },
        "vllm_cfg": {
            "async_engine": True,
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": tp,
            "gpu_memory_utilization": 0.8,
            "precision": "bfloat16",
            "kv_cache_dtype": "auto",
            "max_model_len": 5,
            "load_format": "auto",
            "enforce_eager": False,
            "expose_http_server": expose_http_server,
            "enable_vllm_metrics_logger": True,
            "vllm_metrics_logger_interval": 1.0,
            "env_vars": None,
        },
        "vllm_kwargs": {},
    }


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    workers: list[dict[str, Any]] | None = None,
    calls: list[str] | None = None,
) -> None:
    endpoints = workers or [
        {"instance_id": "worker-0", "system_url": "http://10.0.0.2:4000"}
    ]
    events = calls if calls is not None else []

    class FakeRuntime:
        def __init__(self, *, cluster, config):
            events.append("init")

        def start(self):
            events.append("start")

        @property
        def frontend_url(self):
            return "http://10.0.0.1:3000/v1"

        def refit_workers(self):
            return [dict(worker) for worker in endpoints]

        def validate_workers(self, expected):
            return expected

        def shutdown(self):
            events.append("shutdown")

    monkeypatch.setattr(generation_module, "ManagedDynamoRuntime", FakeRuntime)


def _data() -> BatchedDataDict:
    return BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
            "input_lengths": torch.tensor([3], dtype=torch.long),
            "stop_strings": [["stop"]],
        }
    )


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


def test_runtime_start_world_size_sender_geometry_and_shutdown(monkeypatch) -> None:
    calls: list[str] = []
    _patch_runtime(
        monkeypatch,
        workers=[
            {"instance_id": "a", "system_url": "http://10.0.0.2:4000"},
            {"instance_id": "b", "system_url": "http://10.0.0.3:4000"},
        ],
        calls=calls,
    )
    generation = DynamoGeneration(cluster=object(), config=_config(tp=2))

    assert calls[:2] == ["init", "start"]
    assert generation.frontend_url == "http://10.0.0.1:3000/v1"
    assert generation.dp_openai_server_base_urls == [None]
    assert generation.get_inference_world_size() == 4
    sender = generation.get_collective_sender_spec()
    assert sender.nccl_peer == "vllm"
    assert sender.buffer_size_bytes == VLLM_PACKED_BUFFER_SIZE_BYTES
    assert sender.num_buffers == VLLM_PACKED_NUM_BUFFERS
    assert generation.shutdown()
    assert generation.shutdown()
    assert calls.count("shutdown") == 1


def test_blocking_generate_is_rejected_and_async_generation_uses_http(
    monkeypatch,
) -> None:
    _patch_runtime(monkeypatch)
    requests = []

    async def fake_post(url, payload, timeout_s):
        requests.append((url, payload, timeout_s))
        return _completion_response([8, 9])

    monkeypatch.setattr(generation_module, "async_http_post_json", fake_post)
    generation = DynamoGeneration(cluster=object(), config=_config())
    with pytest.raises(NotImplementedError, match="generate_async"):
        generation.generate(_data())

    async def collect():
        return [item async for item in generation.generate_async(_data())]

    outputs = asyncio.run(collect())
    assert outputs[0][0] == 0
    assert outputs[0][1]["output_ids"].tolist() == [[1, 2, 3, 8, 9]]
    assert requests[0][0].endswith("/v1/completions")
    assert requests[0][1]["max_tokens"] == 2
    assert requests[0][1]["stop"] == ["stop"]
    assert "return_tokens_as_token_ids" not in requests[0][1]


def test_prompt_at_context_limit_is_rejected(monkeypatch) -> None:
    _patch_runtime(monkeypatch)
    generation = DynamoGeneration(cluster=object(), config=_config())

    with pytest.raises(ValueError, match="prompt length 5 must be less than"):
        generation._allowed_new_tokens(5)


def test_finish_generation_invalidates_sync_rollout_cache(monkeypatch) -> None:
    _patch_runtime(monkeypatch)
    generation = DynamoGeneration(cluster=object(), config=_config())
    generation.invalidate_kv_cache = MagicMock(return_value=True)

    assert generation.finish_generation()
    generation.invalidate_kv_cache.assert_called_once_with()


def test_merged_stop_strings_enforce_dynamo_limit(monkeypatch) -> None:
    _patch_runtime(monkeypatch)
    config = _config()
    config["stop_strings"] = [f"configured-{index}" for index in range(16)]
    generation = DynamoGeneration(cluster=object(), config=config)

    assert (
        len(
            generation._merge_stop_strings(
                [[f"request-{index}" for index in range(16)]]
            )
        )
        == 32
    )
    with pytest.raises(ValueError, match="at most 32 stop strings"):
        generation._merge_stop_strings([[f"request-{index}" for index in range(17)]])


def test_token_wrapper_is_used_for_nemo_gym(monkeypatch) -> None:
    _patch_runtime(monkeypatch)
    wrappers = []

    class FakeWrapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            wrappers.append(self)

        def start(self):
            return "http://127.0.0.1:3001/v1"

        def shutdown(self):
            pass

    monkeypatch.setattr(generation_module, "DynamoTokenWrapperServer", FakeWrapper)
    tokenizer = object()
    generation = DynamoGeneration(
        cluster=object(),
        config=_config(expose_http_server=True),
        tokenizer=tokenizer,
        tokenizer_config={"chat_template_kwargs": {"enable_thinking": False}},
    )
    assert generation.dp_openai_server_base_urls == ["http://127.0.0.1:3001/v1"]
    assert wrappers[0].kwargs["tokenizer"] is tokenizer


def test_refit_rank_offsets_update_and_pickled_cache_invalidation(monkeypatch) -> None:
    workers = [
        {"instance_id": "a", "system_url": "http://10.0.0.2:4000"},
        {"instance_id": "b", "system_url": "http://10.0.0.3:4000"},
    ]
    _patch_runtime(monkeypatch, workers=workers)
    init_calls = []
    update_calls = []
    cache_calls = []
    monkeypatch.setattr(
        refit_module._post_worker_route,
        "remote",
        lambda **kwargs: init_calls.append(kwargs) or True,
    )
    monkeypatch.setattr(
        refit_module._update_worker_weights,
        "remote",
        lambda **kwargs: update_calls.append(kwargs) or True,
    )
    monkeypatch.setattr(refit_module.ray, "get", lambda refs: refs)

    generation = DynamoGeneration(cluster=object(), config=_config(tp=2))
    generation.prepare_refit_info({"weight": (torch.Size([4, 8]), torch.bfloat16)})
    assert generation.init_collective("10.1.0.1", 1500, 7, train_world_size=3) == [
        True,
        True,
    ]
    assert [call["payload"]["init_info"]["rank_offset"] for call in init_calls] == [
        3,
        5,
    ]
    assert all(call["timeout_s"] == 10 for call in init_calls)
    assert generation.update_weights_from_collective() == [True, True]
    assert update_calls[0]["update_info"]["packed"] is True
    assert all(call["timeout_s"] == 10 for call in update_calls)

    restored = pickle.loads(pickle.dumps(generation))
    assert restored.frontend_url == generation.frontend_url
    monkeypatch.setattr(
        refit_module._post_worker_route,
        "remote",
        lambda **kwargs: cache_calls.append(kwargs) or True,
    )
    assert restored.invalidate_kv_cache()
    assert [call["route"] for call in cache_calls] == [
        "pause_generation",
        "pause_generation",
        "resume_generation",
        "resume_generation",
    ]
    assert all(call["timeout_s"] == 10 for call in cache_calls)
    assert all(
        call["payload"] == {"mode": "wait", "clear_cache": True}
        for call in cache_calls[:2]
    )
    assert restored._managed_runtime is None


def test_cache_invalidation_resumes_workers_that_paused_before_peer_failure(
    monkeypatch,
) -> None:
    calls = []

    def remote(**kwargs):
        calls.append(kwargs)
        return (kwargs["route"], kwargs["system_url"])

    def get(ref):
        if ref == ("pause_generation", "http://worker-b:4000"):
            raise RuntimeError("pause refused")
        return True

    monkeypatch.setattr(refit_module._post_worker_route, "remote", remote)
    monkeypatch.setattr(refit_module.ray, "get", get)
    channel = DynamoRefitChannel(
        [
            {"instance_id": "a", "system_url": "http://worker-a:4000"},
            {"instance_id": "b", "system_url": "http://worker-b:4000"},
        ],
        engine_world_size=1,
        control_timeout_s=10,
    )

    with pytest.raises(RuntimeError, match="pause/clear failed"):
        channel.flush_cache()

    assert [(call["route"], call["system_url"]) for call in calls] == [
        ("pause_generation", "http://worker-a:4000"),
        ("pause_generation", "http://worker-b:4000"),
        ("resume_generation", "http://worker-a:4000"),
    ]


def test_native_refit_transaction_keeps_cache_mode_external(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        refit_module,
        "http_post_json",
        lambda url, payload, timeout_s: calls.append(payload) or {"status": "ok"},
    )
    assert refit_module._update_worker_weights._function(
        system_url="http://worker:4000",
        update_info={"names": ["weight"]},
        timeout_s=30,
    )
    assert [call["engine_rpc"] for call in calls] == [
        "start_weight_update",
        "update_weights",
        "finish_weight_update",
    ]
    assert all(call["reset_prefix_cache"] is False for call in calls)


def test_metrics_parser_and_sampler_aliases() -> None:
    sampler = DynamoMetricsSampler(
        [{"instance_id": "a", "system_url": "http://worker:4000"}],
        interval_s=1,
        include_prefixes=None,
        exclude_prefixes=None,
    )
    parsed = parse_prometheus_metrics(
        'vllm:num_requests_running{model_name="model",engine="0"} 3\n'
        'vllm:num_requests_waiting{model_name="model",engine="0"} 2\n'
        'vllm:kv_cache_usage_perc{model_name="model",engine="0"} 0.5\n'
        'vllm:generation_tokens_total{model_name="model",engine="0"} 7\n'
        'vllm:gpu_cache_usage_perc{model_name="model",engine="0"} 0.9\n'
        "python_gc_objects_collected_total 10\n",
        sampler._include_prefixes,
        sampler._exclude_prefixes,
    )
    sampler._samples = {name: {0: [value]} for name, value in parsed.items()}
    metrics = sampler.snapshot()
    assert metrics["inflight_batch_sizes"] == {0: [3.0]}
    assert metrics["num_pending_samples"] == {0: [2.0]}
    assert metrics["kv_cache_usage_perc"] == {0: [0.5]}
    assert metrics["generation_tokens"] == {0: [7.0]}
    assert "vllm_gpu_cache_usage_perc" not in parsed
    assert "python_gc_objects_collected_total" not in parsed


def test_metrics_http_errors_are_ignored(monkeypatch) -> None:
    monkeypatch.setattr(
        metrics_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            metrics_module.urllib.error.URLError("refused")
        ),
    )
    assert metrics_module._http_get_text("http://worker/metrics", 1) is None


def test_completion_parser_rejects_misaligned_logprobs() -> None:
    response = _completion_response([8, 9])
    response["choices"][0]["logprobs"]["token_logprobs"] = [-0.1]
    with pytest.raises(RuntimeError, match="1 token logprobs for 2"):
        generation_module._parse_dynamo_completion_response(
            response, request_url="http://dynamo/v1/completions"
        )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ({"http_status": 408}, True),
        ({"http_status": 429}, True),
        ({"http_status": 503}, True),
        ({"transport_error": "refused"}, True),
        ({"json_decode_error": True}, True),
        ({"http_status": 400}, False),
        ({"http_status": 404}, False),
    ],
)
def test_completion_retry_predicate(response, expected) -> None:
    assert generation_module._is_retryable_http_response(response) is expected


def test_completion_retry_eventually_succeeds(monkeypatch) -> None:
    _patch_runtime(monkeypatch)
    responses = iter(
        [{"status": "error", "http_status": 503}, _completion_response([8])]
    )
    calls = []

    async def fake_post(*args):
        calls.append(args)
        return next(responses)

    async def no_sleep(_):
        return None

    monkeypatch.setattr(generation_module, "async_http_post_json", fake_post)
    monkeypatch.setattr(generation_module.asyncio, "sleep", no_sleep)
    generation = DynamoGeneration(cluster=object(), config=_config())

    token_ids, _, _ = asyncio.run(
        generation._post_completion_request(
            prompt_token_ids=[1],
            greedy=False,
            stop_strings=None,
            max_new_tokens=1,
        )
    )

    assert token_ids == [8]
    assert len(calls) == 2


@pytest.mark.parametrize("status", [400, 503])
def test_completion_retry_stops_on_nonretryable_or_exhaustion(
    monkeypatch, status
) -> None:
    _patch_runtime(monkeypatch)
    calls = []

    async def fake_post(*args):
        calls.append(args)
        return {"status": "error", "http_status": status}

    async def no_sleep(_):
        return None

    monkeypatch.setattr(generation_module, "async_http_post_json", fake_post)
    monkeypatch.setattr(generation_module.asyncio, "sleep", no_sleep)
    generation = DynamoGeneration(cluster=object(), config=_config())

    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        asyncio.run(
            generation._post_completion_request(
                prompt_token_ids=[1],
                greedy=False,
                stop_strings=None,
                max_new_tokens=1,
            )
        )

    assert len(calls) == (1 if status == 400 else generation_module._HTTP_MAX_ATTEMPTS)


def test_direct_completions_are_not_limited_by_default_thread_pool(
    monkeypatch,
) -> None:
    _patch_runtime(monkeypatch)
    request_count = 40
    entered_count = 0
    all_entered = asyncio.Event()
    release = asyncio.Event()

    async def fake_post(*args):
        nonlocal entered_count
        entered_count += 1
        if entered_count == request_count:
            all_entered.set()
        await release.wait()
        return _completion_response([8])

    monkeypatch.setattr(generation_module, "async_http_post_json", fake_post)
    generation = DynamoGeneration(cluster=object(), config=_config())

    async def run_requests():
        tasks = [
            asyncio.create_task(
                generation._post_completion_request(
                    prompt_token_ids=[1],
                    greedy=False,
                    stop_strings=None,
                    max_new_tokens=1,
                )
            )
            for _ in range(request_count)
        ]
        await asyncio.wait_for(all_entered.wait(), timeout=1)
        release.set()
        return await asyncio.gather(*tasks)

    responses = asyncio.run(run_requests())

    assert entered_count == request_count
    assert len(responses) == request_count
