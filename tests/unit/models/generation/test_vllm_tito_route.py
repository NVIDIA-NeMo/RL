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
import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_rl.models.generation.prefix_recovery import NoPrefixRecovery
from nemo_rl.models.generation.tito import TITO_GENERATE_ROUTE, TitoGenerateRequest
from nemo_rl.models.generation.vllm.tito_route import (
    register_tito_route,
    resolve_tito_parent,
)

MAX_MODEL_LEN = 16


def _admission(mode="token_in", *, prev_len=3, generation_cut=None):
    admission = types.SimpleNamespace(
        mode=mode,
        model_call_id="call-2",
        parent_call_id="call-1",
        prev_len=prev_len,
        generation_cut=generation_cut,
    )
    admission.model_copy = lambda update: types.SimpleNamespace(
        **{**vars(admission), **update}
    )
    return admission


def _worker(*, parent=(1, 2, 3)):
    worker = MagicMock()
    worker.cfg = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 20,
        "val_temperature": 0.0,
        "val_top_p": 1.0,
        "val_top_k": None,
        "stop_token_ids": [9],
    }
    worker._prefix_recovery = NoPrefixRecovery()
    worker._resolve_admission_prefix.return_value = list(parent)
    worker._context_refusal_headers.return_value = {"x-refused": "call-2"}
    worker._finish_request_capture.side_effect = lambda request, content: {
        **content,
        "ng_commit_coords": {"staged": True},
    }
    return worker


def test_parent_of_a_root_is_empty():
    worker = _worker()

    _, parent = asyncio.run(resolve_tito_parent(worker, _admission("text", prev_len=0)))

    assert parent == []
    worker._resolve_admission_prefix.assert_not_called()


@pytest.mark.parametrize(
    "admission,match",
    [
        (_admission("candidate"), "decided capture admission"),
        (_admission(prev_len=4), "resolved to 3 tokens"),
    ],
)
def test_parent_resolution_rejects_an_unusable_admission(admission, match):
    with pytest.raises(ValueError, match=match):
        asyncio.run(resolve_tito_parent(_worker(), admission))


def test_parent_resolution_rejects_a_restored_generation_prefix():
    worker = _worker()
    worker._prefix_recovery = MagicMock()
    worker._prefix_recovery.resolve_restore.side_effect = lambda admission, parent: (
        admission,
        "cut",
    )

    with pytest.raises(ValueError, match="does not restore generation prefixes"):
        asyncio.run(resolve_tito_parent(worker, _admission()))


class _App:
    def __init__(self):
        self.routes = {}

    def post(self, path):
        def decorator(handler):
            self.routes[path] = handler
            return handler

        return decorator


@pytest.fixture
def route(monkeypatch):
    """Build the route against stub vLLM modules; return its handler and parts."""

    class SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    for name, attrs in {
        "vllm": {},
        "vllm.entrypoints": {},
        "vllm.entrypoints.serve": {},
        "vllm.entrypoints.serve.utils": {},
        "vllm.entrypoints.serve.utils.api_utils": {
            "with_cancellation": lambda handler: handler
        },
        "vllm.sampling_params": {
            "SamplingParams": SamplingParams,
            "RequestOutputKind": types.SimpleNamespace(FINAL_ONLY="final"),
        },
    }.items():
        module = types.ModuleType(name)
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
        monkeypatch.setitem(sys.modules, name, module)

    worker = _worker()
    engine_client = MagicMock()
    engine_client.model_config.max_model_len = MAX_MODEL_LEN
    engine_client.renderer.render_cmpl_async = AsyncMock(return_value=["engine-input"])
    generation = types.SimpleNamespace(
        index=0,
        token_ids=[7, 9],
        logprobs=[
            {7: types.SimpleNamespace(logprob=-0.5)},
            {9: types.SimpleNamespace(logprob=-0.25)},
        ],
        finish_reason="stop",
        stop_reason=9,
    )
    calls = []

    async def generate(engine_input, sampling_params, request_id):
        calls.append((engine_input, sampling_params))
        yield types.SimpleNamespace(outputs=[generation])

    engine_client.generate = generate
    abort = AsyncMock()
    app = _App()
    register_tito_route(
        app, worker=worker, engine_client=engine_client, abort_request_capture=abort
    )
    return types.SimpleNamespace(
        handler=app.routes[TITO_GENERATE_ROUTE],
        worker=worker,
        engine_client=engine_client,
        calls=calls,
        abort=abort,
    )


def _request(**fields):
    return TitoGenerateRequest(
        **{"token_ids": [4, 5], "temperature": 1.0, "top_p": 1.0, **fields}
    )


def _body(response):
    return json.loads(response.body)


def test_uncaptured_call_generates_from_the_submitted_tokens(route):
    route.worker._finish_request_capture.side_effect = lambda request, content: content

    response = asyncio.run(route.handler(_request(max_tokens=3), MagicMock()))

    assert _body(response) == {
        "call_id": None,
        "prompt_token_count": 2,
        "token_ids": [7, 9],
        "logprobs": [-0.5, -0.25],
        "finish_reason": "stop",
        "stop_reason": 9,
        "ng_commit_coords": None,
    }
    route.worker._capture_admission.assert_not_called()
    route.engine_client.renderer.render_cmpl_async.assert_awaited_once_with(
        [{"prompt_token_ids": [4, 5]}]
    )
    ((_, sampling_params),) = route.calls
    assert (sampling_params.top_k, sampling_params.max_tokens) == (20, 3)
    assert sampling_params.stop_token_ids == [9]


def test_captured_call_continues_its_parent(route):
    admission = _admission()
    route.worker._capture_admission.return_value = admission
    request = _request(ng_capture={"model_call_id": "call-2"})

    response = asyncio.run(route.handler(request, MagicMock()))

    body = _body(response)
    assert (body["call_id"], body["prompt_token_count"]) == ("call-2", 5)
    route.worker._begin_request_capture.assert_called_once_with(
        request, [1, 2, 3, 4, 5], admission=admission, prefix_token_ids=[1, 2, 3]
    )
    route.engine_client.renderer.render_cmpl_async.assert_awaited_once_with(
        [{"prompt_token_ids": [1, 2, 3, 4, 5]}]
    )
    ((_, sampling_params),) = route.calls
    assert sampling_params.max_tokens == MAX_MODEL_LEN - 5
    staged = route.worker._finish_request_capture.call_args.args[1]
    assert staged["choices"][0]["message"]["generation_token_ids"] == [7, 9]


@pytest.mark.parametrize(
    "admission,token_ids,match,refused",
    [
        (None, [4, 5], "requires token capture", False),
        (_admission("candidate"), [4, 5], "decided capture admission", False),
        (_admission(), [4] * MAX_MODEL_LEN, "maximum context length", True),
    ],
)
def test_refused_call_generates_nothing(route, admission, token_ids, match, refused):
    route.worker._capture_admission.return_value = admission
    request = _request(token_ids=token_ids, ng_capture={"model_call_id": "call-2"})

    response = asyncio.run(route.handler(request, MagicMock()))

    assert response.status_code == 400
    assert match in _body(response)["error"]["message"]
    assert (response.headers.get("x-refused") == "call-2") is refused
    assert not route.calls
    route.worker._begin_request_capture.assert_not_called()


def test_engine_failure_fails_the_capture(route):
    route.worker._capture_admission.return_value = _admission()
    route.engine_client.renderer.render_cmpl_async.side_effect = RuntimeError("boom")
    request = _request(ng_capture={"model_call_id": "call-2"})

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(route.handler(request, MagicMock()))

    route.abort.assert_awaited_once_with(request, "engine_error")
    route.worker._finish_request_capture.assert_not_called()


@pytest.mark.nemo_gym
def test_token_only_calls_stage_a_training_row(route):
    """A root call and its continuation linearize into one exact training row."""
    from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter
    from nemo_gym.token_id_capture.staging import install_capture
    from nemo_gym.token_id_capture.staging.rebuild import verify_and_linearize
    from nemo_gym.token_id_capture.staging.records import (
        CallRecord,
        RolloutReceipt,
        StagedCallBaseSnapshot,
        StageResult,
    )

    from nemo_rl.models.generation.prefix_recovery import CaptureRegistry
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    class Sink:
        def __init__(self):
            self.records = []

        def stage(self, record, *, attachments=None):
            self.records.append(record)
            return StageResult(ok=True, staging_key=record.model_call_id)

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = route.worker.cfg
    worker._capture_registry = CaptureRegistry()
    worker._prefix_recovery = NoPrefixRecovery()
    worker._capture_top_logprobs = 0
    # Inline prefixes never read the staged-chain cache.
    worker._chain_prefix = None
    sink = Sink()
    install_capture(
        worker, sink=sink, weight_version_fn=lambda: 7, adapter=VLLMCaptureAdapter()
    )
    app = _App()
    register_tito_route(
        app,
        worker=worker,
        engine_client=route.engine_client,
        abort_request_capture=route.abort,
    )
    handler = app.routes[TITO_GENERATE_ROUTE]

    root = _body(
        asyncio.run(
            handler(
                _request(
                    token_ids=[1, 2, 3],
                    ng_capture={
                        "capture_key": "rollout",
                        "model_call_id": "first",
                        "mode": "text",
                    },
                ),
                MagicMock(),
            )
        )
    )
    parent = [1, 2, 3, *root["token_ids"]]
    continuation = _body(
        asyncio.run(
            handler(
                _request(
                    token_ids=[4, 5],
                    ng_capture={
                        "capture_key": "rollout",
                        "model_call_id": "second",
                        "mode": "token_in",
                        "parent_call_id": "first",
                        "prev_len": len(parent),
                        "parent_chain_hash": root["ng_commit_coords"]["chain_hash"],
                        "required_prefix_token_ids": parent,
                    },
                ),
                MagicMock(),
            )
        )
    )

    assert continuation["prompt_token_count"] == len(parent) + 2
    snapshots, manifest = [], []
    for record in sink.records:
        payload = record.model_dump()
        snapshots.append(
            StagedCallBaseSnapshot(
                **{
                    key: value
                    for key, value in payload.items()
                    if key in StagedCallBaseSnapshot.model_fields
                }
            )
        )
        manifest.append(
            CallRecord(
                staging_key=record.model_call_id,
                response_id="response-" + record.model_call_id,
                **{
                    key: value
                    for key, value in payload.items()
                    if key in CallRecord.model_fields and key != "staging_key"
                },
            )
        )
    row = verify_and_linearize(
        RolloutReceipt(
            capture_key="rollout",
            terminal_model_call_id="second",
            manifest=manifest,
            terminal_selection="declared",
        ),
        snapshots,
    )
    assert row.token_ids == [1, 2, 3, 7, 9, 4, 5, 7, 9]
    assert row.token_mask == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    assert row.logprobs == [0.0, 0.0, 0.0, -0.5, -0.25, 0.0, 0.0, -0.5, -0.25]
    assert row.model_call_ids == ["first", "second"]
