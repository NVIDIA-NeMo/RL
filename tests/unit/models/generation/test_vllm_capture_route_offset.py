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

"""Capture suffix serialization preserves canonical records and HTTP responses."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from functools import partial
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_rl.data_plane.gpu_token_payload import BoundGpuTokenSink  # noqa: E402
from nemo_rl.models.generation.vllm import utils  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl as Worker,
)
from nemo_rl.utils.routed_experts_codec import (  # noqa: E402
    decode_routed_experts,
    encode_routed_experts,
)
from tests.unit.models.generation import (  # noqa: E402
    test_vllm_chat_template_wiring as wiring,
    test_vllm_token_capture_hosting as hosting,
)

pytestmark = [pytest.mark.nemo_gym, pytest.mark.asyncio]
PROMPT, GENERATED = [10, 11, 12, 13, 14], [20, 21, 22]
UPSTREAM_ROUTES = "upstream-choice-level-NPY-is-unchanged"


def _prepare(prev_len: int) -> tuple[Any, Any, Any]:
    sink = hosting._MemorySink()
    worker = hosting._worker_with_capture(sink)
    admission = {"rollout_id": "r0", "model_call_id": "c1", "mode": "text"}
    if prev_len:
        admission.update(
            mode="token_in",
            prev_len=prev_len,
            parent_call_id="parent",
            parent_chain_hash="a" * 64,
            required_prefix_token_ids=PROMPT[:prev_len],
        )
    request = hosting._FakeRequest(
        ng_capture=admission,
        stream=False,
        top_k=None,
        top_p=1.0,
        temperature=1.0,
        return_tokens_as_token_ids=True,
        logprobs=True,
        top_logprobs=0,
    )
    Worker._begin_request_capture(worker, request, PROMPT)
    worker._finish_request_capture = partial(Worker._finish_request_capture, worker)
    worker._complete_request_capture = partial(Worker._complete_request_capture, worker)
    worker._abort_request_capture = partial(Worker._abort_request_capture, worker)
    return worker, request, sink


def _output(*, split: bool, missing: int) -> Any:
    routes = torch.arange((7 - missing) * 6, dtype=torch.int32).reshape(-1, 2, 3)
    return SimpleNamespace(
        finished=True,
        prompt_token_ids=PROMPT,
        num_cached_tokens=4,
        prompt_routed_experts=routes[:5] if split else None,
        outputs=[
            SimpleNamespace(
                index=0,
                token_ids=GENERATED,
                logprobs=[
                    {token: SimpleNamespace(logprob=-0.25)} for token in GENERATED
                ],
                routed_experts=routes[5:] if split else routes,
            )
        ],
    )


def _response(response_type: type = SimpleNamespace) -> Any:
    response = response_type()
    response.choices = [SimpleNamespace(index=0, message=SimpleNamespace())]
    content = hosting._served_content(GENERATED, [-0.25] * 3)
    content["choices"][0]["routed_experts"] = UPSTREAM_ROUTES
    response.model_dump = lambda: deepcopy(content)
    return response


async def _stage(
    prev_len: int,
    dtype: torch.dtype,
    *,
    offset: bool,
    split: bool = False,
    missing: int = 0,
    malformed: str | None = None,
) -> tuple[dict[str, Any], Any, str]:
    worker, request, sink = _prepare(prev_len)
    output, response = _output(split=split, missing=missing), _response()
    utils.attach_token_information_to_chat_response_choices(response, output)
    utils.attach_routed_experts_to_chat_response_choices(
        response,
        output,
        device=torch.device("cpu"),
        routed_experts_dtype=dtype,
        routed_experts_start=prev_len if offset else 0,
    )
    content = utils.model_dump_chat_response_with_dynamic_message_fields(response)
    message = content["choices"][0]["message"]
    envelope = message["routed_experts"]
    if malformed == "length":
        worker._capture_calls[id(request)].prompt_token_ids = PROMPT + [99]
    elif malformed == "base64":
        message["routed_experts"] = envelope.rsplit(":", 1)[0] + ":invalid!"
    elif malformed == "generation":
        message["generation_log_probs"] = []
    if offset:
        result = await worker._complete_request_capture(request, content)
    else:
        result = worker._finish_request_capture(request, content)
    assert not worker._capture_calls
    return result, sink, envelope


@pytest.mark.parametrize(
    ("prev_len", "dtype", "split"),
    [
        (0, torch.int8, False),
        (2, torch.int16, False),
        (5, torch.int32, False),
        (2, torch.int8, True),
        (5, torch.int16, True),
    ],
)
@pytest.mark.parametrize("missing", [0, 2])
async def test_offset_preserves_complete_record_and_response(
    prev_len: int, dtype: torch.dtype, split: bool, missing: int
) -> None:
    old, old_sink, old_wire = await _stage(
        prev_len, dtype, offset=False, split=split, missing=missing
    )
    new, new_sink, new_wire = await _stage(
        prev_len, dtype, offset=True, split=split, missing=missing
    )
    assert old == new
    assert old_sink.records[0].model_dump() == new_sink.records[0].model_dump()
    assert new["ng_commit_coords"]["disposition"] == "staged"
    assert new["choices"][0]["routed_experts"] == UPSTREAM_ROUTES
    assert "routed_experts" not in new["choices"][0]["message"]
    routes = decode_routed_experts(new_wire, dtype)
    assert torch.equal(routes, decode_routed_experts(old_wire, dtype)[prev_len:])
    assert torch.equal(routes[-1], torch.arange(3, dtype=dtype).expand(2, 3))
    if missing:
        assert torch.all(routes[7 - missing - prev_len : -1] == -1)


@pytest.mark.parametrize("malformed", ["length", "base64", "generation"])
async def test_offset_retains_route_drop_and_token_failure(malformed: str) -> None:
    old, old_sink, _ = await _stage(2, torch.int16, offset=False, malformed=malformed)
    new, new_sink, _ = await _stage(2, torch.int16, offset=True, malformed=malformed)
    assert old == new
    assert [r.model_dump() for r in old_sink.records] == [
        r.model_dump() for r in new_sink.records
    ]
    if malformed == "generation":
        assert new["ng_commit_coords"]["disposition"] == "capture_failed"
        assert not new_sink.records
    else:
        assert new_sink.records[0].extras is None


@pytest.mark.parametrize("legacy", [False, True])
async def test_default_full_routes_still_canonicalize(legacy: bool) -> None:
    routes = torch.arange(48, dtype=torch.int16).reshape(8, 2, 3)
    canonical = encode_routed_experts(routes)
    payload = {
        "choices": [
            {
                "message": {
                    "routed_experts": routes.tolist()
                    if legacy
                    else canonical.replace(":8x2x3:", ":08x02x03:")
                }
            }
        ]
    }
    Worker._delta_align_routed_experts(
        payload, prev_len=0, prompt_len=5, generated_len=3
    )
    assert payload["choices"][0]["message"]["routed_experts"] == canonical


@pytest.mark.parametrize("mode", ["captured", "cpu_fallback", "uncaptured"])
async def test_endpoint_propagates_offset_and_preserves_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    reference, reference_sink, _ = await _stage(2, torch.int16, offset=False)
    wiring._install_fake_vllm(monkeypatch)
    worker, request, sink = _prepare(2)
    worker.cfg = {
        "temperature": 1.0,
        "top_p": 1.0,
        "val_temperature": 1.0,
        "val_top_p": 1.0,
        "vllm_cfg": {},
    }
    worker._http_engine_client = SimpleNamespace(
        model_config="model", renderer="renderer"
    )
    worker.llm_async_engine_args = SimpleNamespace(
        create_model_config=lambda: SimpleNamespace(
            served_model_name="model", model="model"
        ),
    )
    worker.routed_experts_dtype = torch.int16
    worker._return_routed_experts_enabled = lambda: True
    response = _response(
        sys.modules[
            "vllm.entrypoints.openai.chat_completion.protocol"
        ].ChatCompletionResponse
    )

    async def formatter(
        self: Any, req: Any, results: Any, *args: Any, **kwargs: Any
    ) -> Any:
        async for _result in results:
            pass
        return response

    monkeypatch.setattr(
        wiring._OpenAIServingChat,
        "chat_completion_full_generator",
        formatter,
        raising=False,
    )
    if mode == "uncaptured":
        worker._capture_calls.clear()
    elif mode == "cpu_fallback":
        state = worker._capture_calls[id(request)]
        state.gpu_sink = BoundGpuTokenSink(sink)
        state.capture = hosting.RolloutTokenCapture(
            sink=state.gpu_sink,
            weight_version_fn=lambda: 0,
            adapter=state.capture.adapter,
        )

        async def bind(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("GPU preparation unavailable")

        async def finish(state: Any, operation: Any, *, finalize: Any = None) -> Any:
            result = await asyncio.to_thread(operation)
            return finalize(result) if finalize is not None else result

        worker._gpu_capture_host = SimpleNamespace(
            start_export=lambda *args, **kwargs: None, bind=bind, finish=finish
        )
    app = wiring._FakeApp()
    Worker._setup_vllm_openai_api_server(worker, app)
    serving = wiring._BUILT["chat"][0]

    async def outputs() -> Any:
        yield _output(split=False, missing=0)

    async def create(req: Any, raw: Any) -> Any:
        return await serving.chat_completion_full_generator(req, outputs())

    monkeypatch.setattr(serving, "create_chat_completion", create, raising=False)
    result = await dict(app.routes)["/v1/chat/completions"](request, None)
    content = json.loads(result.body)
    assert not worker._capture_calls
    if mode == "uncaptured":
        assert not sink.records and "ng_commit_coords" not in content
        assert content["choices"][0]["message"]["routed_experts"].startswith(
            "nrlre1:int16:8x2x3:"
        )
    else:
        assert content == reference
        assert sink.records[0].model_dump() == reference_sink.records[0].model_dump()
