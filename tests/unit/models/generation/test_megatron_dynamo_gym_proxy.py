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

"""NeMo-Gym's server for Megatron engines behind Dynamo: the frontend-backed client."""

import asyncio
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from nemo_rl.models.generation.megatron import dynamo_gym_proxy as proxy
from nemo_rl.models.generation.megatron import dynamo_rollouts
from nemo_rl.models.generation.megatron.dynamo_gym_proxy import (
    FrontendInferenceClient,
    completion_record,
    completion_request,
    failed_record,
)

URL = "http://10.0.0.1:3000/v1"
COMPLETIONS = f"{URL}/completions"


def _params(**overrides) -> SimpleNamespace:
    """Megatron ``SamplingParams`` at its defaults, as the chat endpoint builds them."""
    params = dict(
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        return_log_probs=False,
        num_tokens_to_generate=16,
        num_tokens_total=None,
        termination_id=None,
        stop_token_ids=None,
        top_n_logprobs=0,
        stop_words=None,
        detokenize_stop_sequence=False,
        seed=None,
        streaming=False,
    )
    params.update(overrides)
    return SimpleNamespace(**params)


def _response(tokens, logprobs=None, *, finish="stop", cached=None, id="cmpl-1"):
    choice: dict[str, Any] = {"index": 0, "text": "", "finish_reason": finish}
    if logprobs is not None:
        choice["logprobs"] = {"token_logprobs": logprobs}
    usage: dict[str, Any] = {"prompt_tokens": 3, "completion_tokens": len(tokens)}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {
        "id": id,
        "choices": [choice],
        "usage": usage,
        "nvext": {"completion_token_ids": list(tokens)},
    }


# ---------------------------------------------------------------------------
# Request and reply shaping
# ---------------------------------------------------------------------------


def test_completion_request_maps_megatron_sampling_to_the_frontend():
    payload = completion_request([1, 2, 3], _params(), model_name="m")

    # Megatron disables top_p with 0.0 and top_k with 0; the frontend takes
    # 1.0 and -1. Nothing else is sent unless set.
    assert payload == {
        "model": "m",
        "prompt": [1, 2, 3],
        "max_tokens": 16,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "n": 1,
        "include_stop_str_in_output": False,
        "nvext": {"extra_fields": ["completion_token_ids"]},
    }


def test_completion_request_carries_every_set_sampling_field():
    payload = completion_request(
        [1, 2, 3],
        _params(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            return_log_probs=True,
            stop_words=["</s>", "\n\n"],
            detokenize_stop_sequence=True,
            termination_id=2,
            stop_token_ids=[7, 2],
            seed=11,
            num_tokens_to_generate=None,
            num_tokens_total=10,
        ),
        model_name="m",
    )
    assert payload["max_tokens"] == 7, "the total budget minus the prompt"
    assert (payload["temperature"], payload["top_p"], payload["top_k"]) == (
        0.7,
        0.9,
        40,
    )
    # Only the sampled tokens' log probabilities exist behind the frontend.
    assert payload["logprobs"] == 0
    assert payload["stop"] == ["</s>", "\n\n"]
    assert payload["include_stop_str_in_output"] is True
    assert payload["stop_token_ids"] == [2, 7]
    assert payload["seed"] == 11


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"top_n_logprobs": 5}, "top_logprobs"),
        ({"num_tokens_to_generate": None}, "token budget"),
        ({"num_tokens_to_generate": 0}, "token budget"),
    ],
)
def test_completion_request_refuses_what_the_adapter_cannot_serve(overrides, match):
    with pytest.raises(ValueError, match=match):
        completion_request([1, 2, 3], _params(**overrides), model_name="m")


def test_completion_record_reads_like_a_coordinator_reply():
    payload = completion_request(
        [1, 2, 3], _params(return_log_probs=True), model_name="m"
    )
    record = completion_record(
        [1, 2, 3],
        payload,
        _response([4, 5], [-0.5, -0.25], cached=2),
        request_id=7,
        request_url=COMPLETIONS,
        latency_s=0.5,
    )
    # The keys Megatron-LM's chat endpoint reads from a serialized request.
    assert record["status"] == "COMPLETED" and record["events"] == []
    assert record["uid"] == "cmpl-1" and record["request_id"] == 7
    assert record["generated_tokens"] == [4, 5]
    assert record["generated_log_probs"] == [-0.5, -0.25]
    assert record["prompt_tokens"] == [1, 2, 3] and record["prompt_length"] == 3
    assert record["compact_prompt_tokens"] is None
    assert record["num_cached_tokens"] == 2
    assert record["payload_offloaded"] is False
    assert record["payload_stage_metadata"] is None
    assert record["routing_indices"] is None
    # The finish reason is derived from the budget, as for an engine reply.
    assert record["sampling_params"] == {"num_tokens_to_generate": 16}
    assert record["latency"] == 0.5


def test_completion_record_omits_log_probs_the_request_did_not_ask_for():
    payload = completion_request([1], _params(), model_name="m")
    record = completion_record(
        [1], payload, _response([4]), request_id=0, request_url=COMPLETIONS, latency_s=0
    )
    assert record["generated_log_probs"] is None
    assert record["num_cached_tokens"] == 0
    assert record["uid"] == "cmpl-1"


def test_completion_record_with_log_probs_needs_them_in_the_reply():
    payload = completion_request([1], _params(return_log_probs=True), model_name="m")
    with pytest.raises(RuntimeError, match="token_logprobs"):
        completion_record(
            [1],
            payload,
            _response([4]),
            request_id=0,
            request_url=COMPLETIONS,
            latency_s=0,
        )


def test_failed_record_names_the_failure_kind():
    transient = failed_record([1], "HTTP 503: busy", request_id=3, transient=True)
    rejected = failed_record([1], "HTTP 400: bad", request_id=4, transient=False)
    assert transient["status"] == rejected["status"] == "FAILED"
    assert transient["events"] == [
        {"type": "ERROR_TRANSIENT", "payload": "HTTP 503: busy"}
    ]
    assert rejected["events"] == [
        {"type": "ERROR_NONTRANSIENT", "payload": "HTTP 400: bad"}
    ]
    assert rejected["generated_tokens"] == [] and rejected["prompt_tokens"] == [1]
    assert rejected["uid"] == "4"


# ---------------------------------------------------------------------------
# The client: submissions become HTTP completions on the running loop
# ---------------------------------------------------------------------------


@pytest.fixture
def http(monkeypatch):
    """Fake ``async_http_post_json``: scripted replies, every call recorded; no retry delay."""
    state = SimpleNamespace(calls=[], replies=[], release=None)

    async def post(url, payload, timeout_s):
        state.calls.append((url, payload, timeout_s))
        if state.release is not None:
            await state.release.wait()
        reply = state.replies.pop(0)
        return reply() if callable(reply) else reply

    monkeypatch.setattr(proxy, "async_http_post_json", post)
    monkeypatch.setattr(dynamo_rollouts, "_HTTP_RETRY_DELAY_S", 0.0)
    return state


def _client(**overrides) -> FrontendInferenceClient:
    kwargs = dict(frontend_url=URL, model_name="m", request_timeout_s=30.0)
    kwargs.update(overrides)
    return FrontendInferenceClient(**kwargs)


def test_add_request_with_id_posts_one_completion_and_shapes_the_reply(http):
    http.replies = [_response([4, 5], [-0.1, -0.2])]
    client = _client()

    async def run():
        request_id, future = client.add_request_with_id(
            torch.tensor([1, 2, 3]), _params(return_log_probs=True)
        )
        return request_id, await future

    request_id, record = asyncio.run(run())
    assert request_id == 0 and client.next_request_id == 1
    assert record["generated_tokens"] == [4, 5]
    assert record["generated_log_probs"] == [-0.1, -0.2]
    (call,) = http.calls
    assert call[0] == COMPLETIONS and call[2] == 30.0
    assert call[1]["prompt"] == [1, 2, 3] and call[1]["logprobs"] == 0
    assert client._tasks == {}, "a finished request is forgotten"


def test_add_request_returns_the_future_alone(http):
    http.replies = [_response([4])]

    async def run():
        return await _client().add_request([1], _params())

    assert asyncio.run(run())["generated_tokens"] == [4]


def test_streaming_delivers_the_whole_completion_as_the_final_item(http):
    pytest.importorskip("megatron.core.inference.async_stream")
    http.replies = [_response([4, 5])]

    async def run():
        stream = _client().add_request_streaming([1, 2], _params())
        return [item async for item in stream]

    items = asyncio.run(run())
    assert [list(item) for item in items] == [["final"]]
    assert items[0]["final"]["generated_tokens"] == [4, 5]


@pytest.mark.parametrize(
    ("reply", "kind", "message"),
    [
        (
            {"status": "error", "http_status": 400, "raw": "bad prompt"},
            "ERROR_NONTRANSIENT",
            "HTTP 400: bad prompt",
        ),
        (
            {"status": "error", "http_status": 503, "raw": "no engines"},
            "ERROR_TRANSIENT",
            "HTTP 503: no engines",
        ),
    ],
)
def test_frontend_errors_become_failed_replies(http, reply, kind, message):
    # One attempt for a 400; the transient 503 is retried to the attempt cap.
    http.replies = [reply] * dynamo_rollouts._HTTP_MAX_ATTEMPTS

    async def run():
        return await _client().add_request([1], _params())

    record = asyncio.run(run())
    assert record["status"] == "FAILED"
    assert record["events"] == [{"type": kind, "payload": message}]
    expected_calls = (
        1 if kind == "ERROR_NONTRANSIENT" else dynamo_rollouts._HTTP_MAX_ATTEMPTS
    )
    assert len(http.calls) == expected_calls


def test_transient_failures_are_retried_until_a_reply_arrives(http):
    http.replies = [
        {"status": "error", "transport_error": "ClientError: reset"},
        _response([4]),
    ]

    async def run():
        return await _client().add_request([1], _params())

    assert asyncio.run(run())["generated_tokens"] == [4]
    assert len(http.calls) == 2


def test_an_unreadable_reply_fails_without_a_retry(http):
    http.replies = [{"choices": []}]

    async def run():
        return await _client().add_request([1], _params())

    record = asyncio.run(run())
    assert record["status"] == "FAILED"
    assert record["events"][0]["type"] == "ERROR_NONTRANSIENT"
    assert "choices" in record["events"][0]["payload"]
    assert len(http.calls) == 1


def test_abort_cancels_the_request_in_flight(http):
    http.replies = [_response([4])]
    client = _client()

    async def run():
        http.release = asyncio.Event()
        request_id, future = client.add_request_with_id([1], _params())
        await asyncio.sleep(0)
        client.abort_request(request_id)
        client.abort_request(request_id)  # a second abort is not an error
        client.abort_request(99)  # nor an unknown id
        with pytest.raises(asyncio.CancelledError):
            await future
        return request_id

    assert asyncio.run(run()) == 0
    assert client._tasks == {}


def test_stop_cancels_every_request_in_flight(http):
    http.replies = [_response([4]), _response([5])]
    client = _client()

    async def run():
        http.release = asyncio.Event()
        futures = [client.add_request([1], _params()) for _ in range(2)]
        await asyncio.sleep(0)
        client.stop()
        results = await asyncio.gather(*futures, return_exceptions=True)
        return results

    results = asyncio.run(run())
    assert all(isinstance(r, asyncio.CancelledError) for r in results)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"prompt": "hello"}, ValueError, "token ids"),
        (
            {"prompt": [1], "multi_modal_data": {"image": b"..."}},
            NotImplementedError,
            "text prompts only",
        ),
        (
            {"prompt": [1], "request_metadata": {"ng_capture": {}}},
            NotImplementedError,
            "token capture",
        ),
    ],
)
def test_client_refuses_what_the_frontend_path_cannot_carry(http, kwargs, error, match):
    client = _client()
    prompt = kwargs.pop("prompt")

    async def run():
        client.add_request_with_id(prompt, _params(), **kwargs)

    with pytest.raises(error, match=match):
        asyncio.run(run())
    assert http.calls == [] and client.next_request_id == 0


def test_client_refusals_do_not_consume_request_ids_after_a_success(http):
    http.replies = [_response([4])]
    client = _client()

    async def run():
        await client.add_request([1], _params())
        with pytest.raises(ValueError):
            client.add_request([1], _params(top_n_logprobs=1))
        request_id, future = client.add_request_with_id([1], _params())
        client.abort_request(request_id)
        with pytest.raises(asyncio.CancelledError):
            await future
        return request_id

    assert asyncio.run(run()) == 1


# ---------------------------------------------------------------------------
# The app: Megatron-LM's endpoints over the client
# ---------------------------------------------------------------------------


def test_build_gym_proxy_app_assembles_the_endpoints_over_the_client(monkeypatch):
    registered = []

    class Quart:
        def __init__(self, name):
            self.name = name
            self.config = {}

        def register_blueprint(self, blueprint):
            registered.append(blueprint)

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)

    module("quart", Quart=Quart)
    module(
        "megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints",
        __all__=["chat", "completions", "health"],
    )
    client = _client()
    tokenizer = SimpleNamespace(name="tok")

    app = proxy.build_gym_proxy_app(
        client,
        tokenizer=tokenizer,
        parsers=["nano_v3"],
        multimodal_prompt_config="mm",
        default_temperature=0.6,
        default_top_p=0.95,
        default_top_k=20,
    )

    assert registered == ["chat", "completions", "health"]
    config = app.config
    assert config["client"] is client and config["tokenizer"] is tokenizer
    assert (
        config["parsers"] == ["nano_v3"] and config["multimodal_prompt_config"] == "mm"
    )
    assert (
        config["default_temperature"],
        config["default_top_p"],
        config["default_top_k"],
    ) == (
        0.6,
        0.95,
        20,
    )
    assert config["chat_template"] is None and config["eval_mode"] is False
    assert config["MAX_CONTENT_LENGTH"] == 2**30
    # A copy for the tokenize thread, as Megatron-LM's server keeps one.
    assert config["tokenizer_copy"] is not tokenizer
    assert config["tokenizer_copy"].name == "tok"
    config["tokenize_executor"].shutdown()


def test_bind_server_socket_shares_the_port_with_a_reservation():
    import socket

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    holder.bind(("0.0.0.0", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        sock = proxy.bind_server_socket(port)
    finally:
        holder.close()
    try:
        assert sock.getsockname()[1] == port
        assert sock.getblocking() is False
    finally:
        sock.close()


def test_wait_until_healthy_reports_a_server_that_died(monkeypatch):
    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, timeout):
            raise AssertionError("not reached")

    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(Session=Session, RequestException=Exception),
    )
    server = SimpleNamespace(
        done=lambda: True, exception=lambda: RuntimeError("bind failed")
    )
    with pytest.raises(RuntimeError, match="stopped before serving") as info:
        proxy.wait_until_healthy(URL, server=server, timeout_s=1.0)
    assert str(info.value.__cause__) == "bind failed"
