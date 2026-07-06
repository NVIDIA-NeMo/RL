# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm import vllm_worker_async
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
    _append_trace_jsonl,
    _extract_sse_json_dict,
    _generation_token_ids_from_choices,
)


class _FakeModel:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, mode="json"):
        return self.payload


class _FakeRequest(_FakeModel):
    def __init__(self, *, stream=False, request_id=None):
        payload = {
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
            "temperature": 1.0,
            "top_p": 1.0,
            "stream": stream,
        }
        super().__init__(payload)
        self.model = payload["model"]
        self.max_tokens = payload["max_tokens"]
        self.temperature = payload["temperature"]
        self.top_p = payload["top_p"]
        self.stream = stream
        self.required_prefix_token_ids = None
        self.request_id = request_id


class _FakeResponse(_FakeModel):
    pass


def _build_worker() -> VllmAsyncGenerationWorkerImpl:
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker.base_url = "http://node-1:31001/v1"
    worker.trace_worker_id = "worker-3"
    worker.trace_worker_seed = 3
    worker.trace_bundle_indices = [0, 1]
    return worker


def test_request_trace_helpers_record_flattened_metrics(monkeypatch):
    snapshots = iter(
        [
            {
                "num_requests_running": 128,
                "num_requests_waiting": 12,
                "kv_cache_usage_perc": 0.71,
                "generation_tokens": 9001,
            },
            {
                "num_requests_running": 64,
                "num_requests_waiting": 4,
                "kv_cache_usage_perc": 0.55,
                "generation_tokens": 9017,
            },
        ]
    )
    monkeypatch.setattr(
        vllm_worker_async,
        "_snapshot_vllm_trace_metrics",
        lambda: next(snapshots),
    )
    monkeypatch.setattr(
        vllm_worker_async, "_get_node_ip_local", lambda: "192.0.2.10"
    )

    worker = _build_worker()
    raw_request = SimpleNamespace(
        headers={"x-request-id": "trace-123"},
        client=SimpleNamespace(host="127.0.0.1", port=4321),
        url="http://node-1:31001/v1/chat/completions",
    )
    request = _FakeRequest()
    response = _FakeResponse(
        {
            "id": "chatcmpl-abc",
            "usage": {
                "prompt_tokens": 5426,
                "completion_tokens": 15049,
                "total_tokens": 20475,
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "prompt_token_ids": [1, 2, 3],
                        "generation_token_ids": [4, 5, 6],
                    }
                }
            ],
        }
    )

    record = worker._build_request_trace_base_record(
        raw_request,
        request,
        arrival_unix_ms=123.4,
        arrival_monotonic_ms=1000.0,
    )
    record = worker._finish_request_trace_record(
        record,
        completion_unix_ms=456.7,
        completion_monotonic_ms=1333.3,
        response=response,
        status_code=200,
    )

    assert record["trace_id"] == "trace-123"
    assert record["request_id"] == "trace-123"
    assert record["worker_id"] == "worker-3"
    assert record["chosen_worker"] == "worker-3"
    assert record["base_url"] == "http://node-1:31001/v1"
    assert record["arrival_timestamp_ms"] == 123.4
    assert record["completion_timestamp_ms"] == 456.7
    assert record["duration_ms"] == pytest.approx(333.3)
    assert record["num_requests_running_at_arrival"] == 128
    assert record["num_requests_waiting_at_arrival"] == 12
    assert record["kv_cache_usage_perc_at_arrival"] == 0.71
    assert record["num_requests_running_at_completion"] == 64
    assert record["num_requests_waiting_at_completion"] == 4
    assert record["kv_cache_usage_perc_at_completion"] == 0.55
    assert record["prompt_tokens"] == 5426
    assert record["completion_tokens"] == 15049
    assert record["output_tokens"] == 15049
    assert record["total_tokens"] == 20475
    assert record["prompt_token_ids"] == [1, 2, 3]
    assert record["generation_token_ids"] == [4, 5, 6]
    assert record["response_id"] == "chatcmpl-abc"
    assert record["trace_join_key"] == (
        f"{record['prompt_token_hash']}:{record['generation_token_hash']}"
    )


def test_request_trace_helper_falls_back_to_extracted_token_counts(monkeypatch):
    snapshots = iter(
        [
            {
                "num_requests_running": 1,
                "num_requests_waiting": 0,
                "kv_cache_usage_perc": 0.1,
                "generation_tokens": 0,
            },
            {
                "num_requests_running": 0,
                "num_requests_waiting": 0,
                "kv_cache_usage_perc": 0.2,
                "generation_tokens": 3,
            },
        ]
    )
    monkeypatch.setattr(
        vllm_worker_async,
        "_snapshot_vllm_trace_metrics",
        lambda: next(snapshots),
    )

    worker = _build_worker()
    raw_request = SimpleNamespace(
        headers={},
        client=SimpleNamespace(host="127.0.0.1", port=4321),
        url="http://node-1:31001/v1/chat/completions",
    )
    request = _FakeRequest(stream=True)
    response = {
        "choices": [
            {
                "delta": {
                    "prompt_token_ids": [11, 12, 13, 14],
                    "generation_token_ids": [21, 22, 23],
                }
            }
        ]
    }

    record = worker._build_request_trace_base_record(
        raw_request,
        request,
        arrival_unix_ms=10.0,
        arrival_monotonic_ms=100.0,
    )
    record = worker._finish_request_trace_record(
        record,
        completion_unix_ms=40.0,
        completion_monotonic_ms=160.0,
        response=response,
        status_code=200,
        streaming=True,
    )

    assert record["trace_id"]
    assert record["streaming"] is True
    assert record["prompt_tokens"] == 4
    assert record["completion_tokens"] == 3
    assert record["output_tokens"] == 3
    assert record["total_tokens"] == 7


def test_request_trace_helper_uses_preprocessed_prompt_and_logprob_token_ids(
    monkeypatch,
):
    snapshots = iter(
        [
            {
                "num_requests_running": 1,
                "num_requests_waiting": 0,
                "kv_cache_usage_perc": 0.1,
                "generation_tokens": 0,
            },
            {
                "num_requests_running": 0,
                "num_requests_waiting": 0,
                "kv_cache_usage_perc": 0.2,
                "generation_tokens": 2,
            },
        ]
    )
    monkeypatch.setattr(
        vllm_worker_async,
        "_snapshot_vllm_trace_metrics",
        lambda: next(snapshots),
    )

    worker = _build_worker()
    raw_request = SimpleNamespace(
        headers={"x-request-id": "trace-logprobs"},
        client=SimpleNamespace(host="127.0.0.1", port=4321),
        url="http://node-1:31001/v1/chat/completions",
    )
    request = _FakeRequest()
    response = {
        "id": "chatcmpl-logprobs",
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"token": "token_id:51"},
                        {"token": "token_id:52"},
                    ]
                }
            }
        ],
    }

    record = worker._build_request_trace_base_record(
        raw_request,
        request,
        arrival_unix_ms=10.0,
        arrival_monotonic_ms=100.0,
    )
    record = worker._finish_request_trace_record(
        record,
        completion_unix_ms=40.0,
        completion_monotonic_ms=160.0,
        response=response,
        status_code=200,
        prompt_token_ids=[41, 42],
    )

    assert record["extracted_prompt_tokens"] == 2
    assert record["extracted_generation_tokens"] == 2
    assert record["prompt_token_ids"] == [41, 42]
    assert record["generation_token_ids"] == [51, 52]
    assert record["prompt_token_hash"]
    assert record["generation_token_hash"]
    assert record["trace_join_key"] == (
        f"{record['prompt_token_hash']}:{record['generation_token_hash']}"
    )


def test_request_trace_helper_reads_vllm_choice_token_ids(monkeypatch):
    snapshots = iter([{}, {}])
    monkeypatch.setattr(
        vllm_worker_async,
        "_snapshot_vllm_trace_metrics",
        lambda: next(snapshots),
    )

    worker = _build_worker()
    raw_request = SimpleNamespace(
        headers={"x-request-id": "trace-choice-token-ids"},
        client=SimpleNamespace(host="127.0.0.1", port=4321),
        url="http://node-1:31001/v1/chat/completions",
    )
    request = _FakeRequest()
    response = {
        "id": "chatcmpl-choice-token-ids",
        "prompt_token_ids": [41, 42],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        "choices": [{"token_ids": [51, 52, 53]}],
    }

    record = worker._build_request_trace_base_record(
        raw_request,
        request,
        arrival_unix_ms=10.0,
        arrival_monotonic_ms=100.0,
    )
    record = worker._finish_request_trace_record(
        record,
        completion_unix_ms=40.0,
        completion_monotonic_ms=160.0,
        response=response,
        status_code=200,
    )

    assert record["prompt_token_ids"] == [41, 42]
    assert record["generation_token_ids"] == [51, 52, 53]
    assert _generation_token_ids_from_choices(response) == [51, 52, 53]


def test_request_trace_helper_hashes_empty_logprob_generation(monkeypatch):
    snapshots = iter(
        [
            {
                "num_requests_running": 1,
                "num_requests_waiting": 0,
                "kv_cache_usage_perc": 0.1,
                "generation_tokens": 0,
            },
            {
                "num_requests_running": 0,
                "num_requests_waiting": 0,
                "kv_cache_usage_perc": 0.2,
                "generation_tokens": 0,
            },
        ]
    )
    monkeypatch.setattr(
        vllm_worker_async,
        "_snapshot_vllm_trace_metrics",
        lambda: next(snapshots),
    )

    worker = _build_worker()
    raw_request = SimpleNamespace(
        headers={"x-request-id": "trace-empty"},
        client=SimpleNamespace(host="127.0.0.1", port=4321),
        url="http://node-1:31001/v1/chat/completions",
    )
    request = _FakeRequest()
    response = {
        "id": "chatcmpl-empty",
        "usage": {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2},
        "choices": [{"logprobs": {"content": []}}],
    }

    record = worker._build_request_trace_base_record(
        raw_request,
        request,
        arrival_unix_ms=10.0,
        arrival_monotonic_ms=100.0,
    )
    record = worker._finish_request_trace_record(
        record,
        completion_unix_ms=40.0,
        completion_monotonic_ms=160.0,
        response=response,
        status_code=200,
        prompt_token_ids=[41, 42],
    )

    assert record["completion_tokens"] == 0
    assert record["extracted_generation_tokens"] == 0
    assert record["prompt_token_ids"] == [41, 42]
    assert record["generation_token_ids"] == []
    assert record["generation_token_hash"]
    assert record["trace_join_key"] == (
        f"{record['prompt_token_hash']}:{record['generation_token_hash']}"
    )


def test_extract_sse_json_dict_reads_last_payload():
    chunk = (
        "data: {\"choices\":[{\"delta\":{\"generation_token_ids\":[1]}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"generation_token_ids\":[1,2]}}]}\n\n"
        "data: [DONE]\n\n"
    )

    assert _extract_sse_json_dict(chunk) == {
        "choices": [{"delta": {"generation_token_ids": [1, 2]}}]
    }


def test_append_trace_jsonl_writes_json_lines(tmp_path):
    trace_path = tmp_path / "server_trace.jsonl"
    record = {"trace_id": "trace-1", "worker_id": "worker-0", "prompt_tokens": 3}

    _append_trace_jsonl(str(trace_path), record)
    _append_trace_jsonl(str(trace_path), record)

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == [record, record]
