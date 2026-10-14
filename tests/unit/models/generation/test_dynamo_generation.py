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
from types import SimpleNamespace
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
from nemo_rl.models.generation.interfaces import (
    CollectiveSenderSpec,
    RefitFleetShape,
)
from nemo_rl.distributed.refit_watchdog import REFIT_ABORTED_TOKEN, RefitAborted
from nemo_rl.utils.staged_payload import decode_staged_payload
from nemo_rl.weight_sync.membership import desired_membership


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


@pytest.mark.parametrize("prompt_key", ["vllm_content", "vllm_multi_modal_data"])
def test_async_generate_rejects_multimodal_prompt_columns(
    monkeypatch: pytest.MonkeyPatch, prompt_key: str
) -> None:
    _patch_runtime(monkeypatch)
    post = MagicMock(
        side_effect=AssertionError("Multimodal prompts must not reach HTTP")
    )
    monkeypatch.setattr(generation_module, "async_http_post_json", post)
    generation = DynamoGeneration(cluster=object(), config=_config())
    data = _data()
    data[prompt_key] = ["prompt" if prompt_key == "vllm_content" else {"image": "img"}]

    async def collect() -> list:
        return [item async for item in generation.generate_async(data)]

    with pytest.raises(NotImplementedError, match="multimodal"):
        asyncio.run(collect())
    post.assert_not_called()


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
    # vLLM reads packing from the init info only (worker default is unpacked).
    for call in init_calls:
        init_info = call["payload"]["init_info"]
        assert init_info["packed"] is True
        assert init_info["packed_buffer_size_bytes"] == 1024**3
        assert init_info["packed_num_buffers"] == 2
    assert generation.update_weights_from_collective() == [True, True]
    assert update_calls[0]["update_info"] == {
        "names": ["weight"],
        "dtype_names": ["bfloat16"],
        "shapes": [[4, 8]],
    }
    assert all(call["timeout_s"] == 10 for call in update_calls)

    restored = pickle.loads(pickle.dumps(generation))
    assert restored.frontend_url == generation.frontend_url
    # SingleController prepares refit on the driver, then pickles the
    # generation client into its actor, which issues the weight updates.
    expected_update_info = update_calls[0]["update_info"]
    update_calls.clear()
    assert restored.update_weights_from_collective() == [True, True]
    assert [call["update_info"] for call in update_calls] == [expected_update_info] * 2
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


@pytest.mark.parametrize("clear_cache", [True, False])
def test_refit_pause_drains_every_worker_and_resume_restores_them(
    monkeypatch, clear_cache: bool
) -> None:
    workers = [
        {"instance_id": "a", "system_url": "http://10.0.0.2:4000"},
        {"instance_id": "b", "system_url": "http://10.0.0.3:4000"},
    ]
    _patch_runtime(monkeypatch, workers=workers)
    calls = []
    monkeypatch.setattr(
        refit_module._post_worker_route,
        "remote",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    monkeypatch.setattr(refit_module.ray, "get", lambda refs: refs)

    # SingleController drives the pause from its own pickled client copy.
    generation = pickle.loads(
        pickle.dumps(DynamoGeneration(cluster=object(), config=_config()))
    )
    assert generation.pause_generation_for_refit(clear_cache=clear_cache)
    assert [(call["route"], call["payload"]) for call in calls] == [
        ("pause_generation", {"mode": "wait", "clear_cache": clear_cache}),
        ("pause_generation", {"mode": "wait", "clear_cache": clear_cache}),
    ]

    calls.clear()
    assert generation.resume_generation_after_refit()
    assert [(call["route"], call["system_url"]) for call in calls] == [
        ("resume_generation", "http://10.0.0.2:4000"),
        ("resume_generation", "http://10.0.0.3:4000"),
    ]


def test_failed_pause_resumes_the_workers_that_paused(monkeypatch) -> None:
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

    with pytest.raises(RuntimeError, match="pause before weight update failed"):
        channel.pause(clear_cache=True)

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
    # vLLM's Worker.start_weight_update takes no arguments; Dynamo forwards
    # every non-reserved key, so any extra key would raise in the engine.
    assert set(calls[0]) == {"engine_rpc", "allow_unpaused", "reset_prefix_cache"}


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

    # Rollout telemetry drains to each worker's latest sample, which it keeps.
    sampler._samples["vllm_generation_tokens_total"][0].append(9.0)
    for _ in range(2):
        assert sampler.drain_latest()["generation_tokens"] == {0: [9.0]}


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


def _reshard_generation(monkeypatch, *, refit_transport="nccl_reshard"):
    """A two-engine, TP=2 DynamoGeneration whose engine RPCs are recorded, not sent."""
    workers = [
        {"instance_id": "a", "system_url": "http://10.0.0.2:4000"},
        {"instance_id": "b", "system_url": "http://10.0.0.3:4000"},
    ]
    _patch_runtime(monkeypatch, workers=workers)
    posts: list[dict[str, Any]] = []
    staged: list[dict[str, Any]] = []
    monkeypatch.setattr(
        refit_module._post_worker_route,
        "remote",
        lambda **kwargs: posts.append(kwargs) or True,
    )
    monkeypatch.setattr(
        refit_module._stage_and_post_worker_route,
        "remote",
        lambda **kwargs: staged.append(kwargs) or True,
    )
    monkeypatch.setattr(refit_module.ray, "get", lambda refs: refs)
    config = _config(tp=2)
    config["refit_transport"] = refit_transport
    generation = DynamoGeneration(cluster=object(), config=config)
    return generation, posts, staged


def _full_fleet():
    return desired_membership(
        absent_shards=[], dp_size=2, total_gen_workers=4, train_world_size=3
    )


def test_nccl_reshard_drives_the_worker_extension_on_every_engine(
    monkeypatch,
) -> None:
    generation, posts, staged = _reshard_generation(monkeypatch)
    membership = _full_fleet()

    assert generation.get_refit_fleet_shape() == RefitFleetShape(
        dp_size=2, total_gen_workers=4
    )
    # The trainer's misc broadcast must speak the extension's NeMo protocol.
    assert generation.get_collective_sender_spec() == CollectiveSenderSpec()
    generation.set_refit_membership(membership)

    assert generation.rebuild_collective(membership, "10.1.0.1", 1500) == [True] * 2
    assert [call["route"] for call in posts] == ["init_weights_update_group"] * 2
    assert [call["payload"] for call in posts] == [
        {
            "engine_rpc": "init_collective",
            "rank_prefix": prefix,
            "ip": "10.1.0.1",
            "port": 1500,
            "world_size": 7,
            "train_world_size": 3,
        }
        for prefix in (0, 2)
    ]
    # Past the server's init deadline (control_timeout_s=10), never before it.
    assert all(call["timeout_s"] == 40 for call in posts)

    posts.clear()
    generation.rebuild_nccl_reshard_comm_group(
        membership,
        pp_ips=["10.1.0.1"],
        pp_ports=[1600],
        pp_size=1,
        train_ranks_per_stage=3,
        sub_world_size=7,
    )
    assert [call["payload"] for call in posts] == [
        {
            "engine_rpc": "init_nccl_reshard_comm_group",
            "rank_prefix": prefix,
            "pp_ips": ["10.1.0.1"],
            "pp_ports": [1600],
            "pp_size": 1,
            "train_ranks_per_stage": 3,
            "sub_world_size": 7,
        }
        for prefix in (0, 2)
    ]

    refit_info = {
        "per_layer_params": {"layers.0": [{"name": "w", "global_shape": [4, 8]}]},
        "misc_meta": {"embed": {"shape": [8, 4], "dtype": "bfloat16"}},
    }
    generation.prepare_nccl_reshard_refit_info(refit_info)
    assert [call["system_url"] for call in staged] == [
        "http://10.0.0.2:4000",
        "http://10.0.0.3:4000",
    ]
    for call in staged:
        payload = call["payload"]
        assert payload["engine_rpc"] == "nrl_prepare_nccl_reshard_refit_info_from_stage"
        assert payload["key"] == call["key"]
        assert decode_staged_payload(call["chunks"], payload["sha256"]) == refit_info

    posts.clear()
    assert generation.nccl_reshard_refit(refit_timeout_s=100) == [True] * 2
    assert [call["route"] for call in posts] == ["update_weights_from_distributed"] * 2
    assert all(
        call["payload"]
        == {
            "engine_rpc": "nrl_nccl_reshard_refit",
            "allow_unpaused": True,
            "reset_prefix_cache": False,
            "refit_timeout_s": 100,
        }
        for call in posts
    )
    # The HTTP call outlives the refit's own deadline.
    assert all(call["timeout_s"] == 160 for call in posts)

    # SingleController issues refits from a pickled copy in its own actor.
    restored = pickle.loads(pickle.dumps(generation))
    posts.clear()
    assert restored.nccl_reshard_refit() == [True] * 2
    assert [call["system_url"] for call in posts] == [
        "http://10.0.0.2:4000",
        "http://10.0.0.3:4000",
    ]
    assert all(call["payload"]["refit_timeout_s"] is None for call in posts)
    assert all(call["timeout_s"] == 10 for call in posts)


def test_dynamo_refit_membership_must_be_the_whole_fixed_fleet(monkeypatch) -> None:
    generation, posts, _ = _reshard_generation(monkeypatch)
    one_engine_lost = desired_membership(
        absent_shards=[1], dp_size=2, total_gen_workers=4, train_world_size=3
    )

    with pytest.raises(NotImplementedError, match="fixed worker fleet"):
        generation.set_refit_membership(one_engine_lost)
    with pytest.raises(NotImplementedError, match="fixed worker fleet"):
        generation.rebuild_collective(one_engine_lost, "10.1.0.1", 1500)
    assert posts == []


def test_dynamo_collective_refit_keeps_the_builtin_vllm_protocol(monkeypatch) -> None:
    generation, _, _ = _reshard_generation(monkeypatch, refit_transport=None)

    assert generation.get_collective_sender_spec().nccl_peer == "vllm"
    with pytest.raises(NotImplementedError, match="built-in"):
        generation.rebuild_collective(_full_fleet(), "10.1.0.1", 1500)


def test_engine_route_errors_surface_and_aborts_stay_recognizable(
    monkeypatch,
) -> None:
    responses = iter(
        [
            {"status": "ok"},
            {"status": "error", "message": "boom"},
            {"status": "error", "message": f"{REFIT_ABORTED_TOKEN} deadline passed"},
        ]
    )
    monkeypatch.setattr(
        refit_module, "http_post_json", lambda url, payload, timeout_s: next(responses)
    )
    post = refit_module._post_worker_route._function
    kwargs = {
        "system_url": "http://w:4000",
        "route": "update_weights_from_distributed",
        "payload": {"engine_rpc": "nrl_nccl_reshard_refit"},
        "timeout_s": 1.0,
    }

    assert post(**kwargs) is True
    with pytest.raises(RuntimeError, match="nrl_nccl_reshard_refit.*boom"):
        post(**kwargs)
    with pytest.raises(RefitAborted):
        post(**kwargs)


def test_engine_rpc_tasks_are_never_retried() -> None:
    # A re-POST of a collective after a worker died can never complete.
    for task in (
        refit_module._post_worker_route,
        refit_module._stage_and_post_worker_route,
    ):
        assert task._default_options["max_retries"] == 0


def test_staging_posts_every_chunk_in_order_before_the_call(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(
        refit_module,
        "http_post_json",
        lambda url, payload, timeout_s: sent.append((url, payload)) or {"status": "ok"},
    )
    call = {"engine_rpc": "nrl_prepare_nccl_reshard_refit_info_from_stage"}

    refit_module._stage_and_post_worker_route._function(
        system_url="http://w:4000",
        key="k",
        chunks=["c0", "c1", "c2"],
        payload=call,
        timeout_s=1.0,
    )

    assert [url for url, _ in sent] == [
        "http://w:4000/engine/init_weights_update_group"
    ] * 4
    assert [payload for _, payload in sent] == [
        {
            "engine_rpc": "nrl_stage_chunk",
            "key": "k",
            "index": index,
            "total": 3,
            "data": f"c{index}",
        }
        for index in range(3)
    ] + [call]


def test_reshard_synchronizer_drives_dynamo_end_to_end(monkeypatch) -> None:
    """NcclReshardWeightSynchronizer over a real DynamoGeneration: 3 trainers, 2x TP2."""
    from nemo_rl.weight_sync.nccl_reshard_weight_synchronizer import (
        NcclReshardWeightSynchronizer,
    )

    generation, posts, staged = _reshard_generation(monkeypatch)
    trainer_calls: dict[str, Any] = {}
    policy = SimpleNamespace(
        cfg={
            "megatron_cfg": {
                "tensor_model_parallel_size": 1,
                "expert_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
            },
            "generation": generation.cfg,
        },
        init_collective=lambda *args, **kwargs: trainer_calls.setdefault(
            "init_collective", (args, kwargs)
        )
        and ["train-init"],
        init_nccl_reshard_comm_group=lambda **kwargs: trainer_calls.setdefault(
            "bulk", kwargs
        )
        and ["train-bulk"],
        prepare_nccl_reshard_refit_info=lambda *args,
        **kwargs: trainer_calls.setdefault("plan", (args, kwargs))
        and {
            "per_layer_params": {},
            "misc_meta": {"w": {"shape": [2], "dtype": "bfloat16"}},
        },
        nccl_reshard_refit=lambda **kwargs: trainer_calls.setdefault("refit", kwargs)
        and ["train-refit"],
    )
    train_cluster = SimpleNamespace(
        world_size=lambda: 3,
        num_gpus_per_node=8,
        get_master_address_and_port=lambda: ("10.1.0.1", 1500),
        get_available_address_and_port=lambda pg_idx, bundle_idx: ("10.1.0.1", 1600),
    )
    sync = NcclReshardWeightSynchronizer(
        policy=policy,
        generation=generation,
        train_cluster=train_cluster,
        inference_cluster=None,
        refit_timeout_s=30.0,
    )

    sync.init_communicator()

    _, init_kwargs = trainer_calls["init_collective"]
    assert init_kwargs["nccl_peer"] == "nemo"
    gen_rpcs = [call["payload"]["engine_rpc"] for call in posts]
    assert gen_rpcs == ["init_collective"] * 2 + ["init_nccl_reshard_comm_group"] * 2
    assert [call["payload"]["rank_prefix"] for call in posts] == [0, 2, 0, 2]
    # The reshard plan is built for every Dynamo rank: 2 engines x TP2.
    plan_args, _ = trainer_calls["plan"]
    assert plan_args[1]["tp_size"] == 2 and plan_args[3] == 4
    assert len(staged) == 2

    posts.clear()
    sync.sync_weights()

    assert trainer_calls["refit"]["refit_timeout_s"] == 30.0
    assert [call["payload"]["engine_rpc"] for call in posts] == [
        "nrl_nccl_reshard_refit"
    ] * 2
    assert all(call["payload"]["refit_timeout_s"] == 30.0 for call in posts)
    assert not sync.is_stale
