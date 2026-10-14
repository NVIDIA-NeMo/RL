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
"""CPU contract tests for native vLLM output capture."""

import asyncio
import threading
from types import SimpleNamespace as NS

import pytest

from nemo_rl.experience.direct_capture import (
    DirectCapture,
    DirectCaptureConfig,
    NativeCall,
    current_capture,
    extract_call,
    select_main_branch,
)
from nemo_rl.experience.vllm_http_client import _AsyncLLMHTTPClient


def output(tokens=(3, 4), *, prompt=(1, 2), finished=True, routes=None):
    return NS(
        prompt_token_ids=list(prompt),
        finished=finished,
        outputs=[
            NS(
                index=0,
                token_ids=list(tokens),
                logprobs=[{token: NS(logprob=-0.25)} for token in tokens],
                finish_reason="stop" if finished else None,
                routed_experts=routes,
            )
        ],
    )


def owner(stage, **limits):
    capture = DirectCapture(
        DirectCaptureConfig(**limits), stage, lambda: 7, clear=lambda keys: None
    )
    capture.register("key", "rollout", temperature=1, top_p=1)
    return capture


def test_main_branch_excludes_fork_and_compaction_without_inventing_ancestry():
    def call(i, prompt, generated):
        return NativeCall(
            "r",
            str(i),
            i,
            0,
            tuple(prompt),
            tuple(generated),
            (-0.2,) * len(generated),
            None,
        )

    parent = call(0, [1, 2], [3])
    fork = call(1, [1, 2], [4])
    child = call(2, [1, 2, 3, 5], [6])
    branch, excluded = select_main_branch([child, fork, parent])
    assert branch == [parent, child]
    assert excluded == [fork]
    compact = call(3, [9], [10])
    assert select_main_branch([parent, child, compact]) == ([compact], [parent, child])


def test_delta_logprobs_and_terminal_routes():
    routes = object()
    call = extract_call(
        "r",
        "c",
        0,
        5,
        [output((3,), finished=False), output((4,), routes=routes)],
        delta=True,
    )
    assert call.prompt == (1, 2)
    assert call.tokens == (3, 4)
    assert call.logprobs == (-0.25, -0.25)
    assert call.routed_experts is routes
    with pytest.raises(ValueError, match="finish"):
        extract_call("r", "c", 0, 5, [output(finished=False)], delta=True)


def test_submission_is_nonblocking_and_overflow_cannot_silently_drop_calls():
    entered, release = threading.Event(), threading.Event()

    def stage(call):
        entered.set()
        assert release.wait(5)
        return {"ordinal": call.ordinal, "staging_key": "staged"}

    capture = owner(stage, max_pending_calls=1)
    context = capture.enter("key")
    try:
        ticket = capture.begin(context)
        capture.complete(context, "call", *ticket, [output()], delta=False)
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="queue"):
            capture.begin(context)
        capture.leave(context)
        with pytest.raises(TimeoutError):
            capture.finalize("key", timeout=0.001)
        release.set()
        receipt = capture.finalize("key", timeout=5)
        assert receipt["failure"] == "capture_queue_full"
        assert len(receipt["manifest"]) == 1
        assert capture.finalize("key", timeout=0) == receipt
        capture.release("key")
    finally:
        release.set()
        capture.shutdown()


def test_engine_bridge_keeps_http_payload_native_and_captures_selected_logprobs():
    async def run():
        calls = []
        capture = owner(lambda call: calls.append(call) or {"ordinal": call.ordinal})

        class Engine:
            model_config = renderer = input_processor = vllm_config = None

            async def generate(self, prompt, sampling, request_id, **kwargs):
                assert sampling.logprobs == 0
                yield output((3,), finished=False)
                yield output((4,))

            async def abort(self, request_id):
                pytest.fail("completed generation should not abort")

        bridge = _AsyncLLMHTTPClient(Engine(), asyncio.get_running_loop())
        context = capture.enter("key")
        token = current_capture.set(context)
        sampling = NS(logprobs=None, output_kind=NS(name="DELTA"))
        try:
            outputs = [item async for item in bridge.generate(None, sampling, "call")]
            assert len(outputs) == 2
            assert sampling.logprobs is None
        finally:
            current_capture.reset(token)
            capture.leave(context)
        receipt = await asyncio.to_thread(capture.finalize, "key", timeout=5)
        assert receipt["failure"] is None
        assert calls[0].tokens == (3, 4)
        assert calls[0].weight_version == 7
        capture.shutdown()

    asyncio.run(run())


def test_failed_storage_invalidates_receipt_and_key_cannot_change_owner():
    def stage(call):
        raise ValueError("storage unavailable")

    capture = owner(stage)
    try:
        with pytest.raises(ValueError, match="another"):
            capture.register("key", "different", temperature=1, top_p=1)
        context = capture.enter("key")
        ticket = capture.begin(context)
        capture.complete(context, "call", *ticket, [output()], delta=False)
        capture.leave(context)
        assert (
            capture.finalize("key", timeout=5)["failure"] == "staging_failed:ValueError"
        )
    finally:
        capture.shutdown()


def test_http_identity_control_auth_and_no_training_payload_in_response():
    import httpx
    from fastapi import FastAPI
    from nemo_rl.experience.direct_http import install_direct_control

    async def run():
        records = []
        capture = DirectCapture(
            DirectCaptureConfig(),
            lambda call: records.append(call) or {"ordinal": call.ordinal},
            lambda: 0,
            clear=lambda keys: None,
        )
        app = FastAPI()
        install_direct_control(
            app,
            capture=lambda: capture,
            auth_token=lambda: "control",
            timeout=lambda: 5,
        )

        class Engine:
            model_config = renderer = input_processor = vllm_config = None

            async def generate(self, prompt, sampling, request_id, **kwargs):
                yield output(routes="large-native-array")

            async def abort(self, request_id):
                pytest.fail("unexpected abort")

        bridge = _AsyncLLMHTTPClient(Engine(), asyncio.get_running_loop())

        @app.post("/v1/chat/completions")
        async def chat(body: dict):
            async for item in bridge.generate(
                None, NS(logprobs=None, output_kind=NS(name="FINAL_ONLY")), "engine-id"
            ):
                assert item.outputs[0].routed_experts is None
            return {"text": "done"}

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://local"
            ) as http:
                payload = {
                    "api_key": "rollout-key",
                    "rollout_id": "r",
                    "temperature": 1,
                    "top_p": 1,
                }
                assert (
                    await http.post("/nrl/capture/register", json=payload)
                ).status_code == 401
                auth = {"Authorization": "Bearer control"}
                assert (
                    await http.post("/nrl/capture/register", json=payload, headers=auth)
                ).status_code == 200
                assert (
                    await http.post("/v1/chat/completions", json={})
                ).status_code == 401
                response = await http.post(
                    "/v1/chat/completions",
                    json={},
                    headers={"Authorization": "Bearer rollout-key"},
                )
                assert response.json() == {"text": "done"}
                sealed = await http.post(
                    "/nrl/capture/finalize",
                    json={"api_key": "rollout-key"},
                    headers=auth,
                )
                assert sealed.json()["failure"] is None
                assert records[0].routed_experts == "large-native-array"
                assert records[0].rollout_id == "r"
                assert current_capture.get() is None
        finally:
            capture.shutdown()

    asyncio.run(run())


def test_cancelled_stream_is_aborted_and_cannot_be_trained_as_complete():
    async def run():
        aborted = []
        capture = owner(lambda call: pytest.fail("partial stream must not stage"))

        class Engine:
            model_config = renderer = input_processor = vllm_config = None

            async def generate(self, prompt, sampling, request_id, **kwargs):
                yield output((3,), finished=False)
                await asyncio.Event().wait()

            async def abort(self, request_id):
                aborted.append(request_id)

        bridge = _AsyncLLMHTTPClient(Engine(), asyncio.get_running_loop())
        context = capture.enter("key")
        token = current_capture.set(context)
        stream = bridge.generate(
            None, NS(logprobs=None, output_kind=NS(name="DELTA")), "cancelled"
        )
        try:
            await anext(stream)
            await stream.aclose()
        finally:
            current_capture.reset(token)
            capture.leave(context)
        assert aborted == ["cancelled"]
        assert capture.finalize("key", timeout=5)["failure"] == "incomplete_generation"
        capture.shutdown()

    asyncio.run(run())
