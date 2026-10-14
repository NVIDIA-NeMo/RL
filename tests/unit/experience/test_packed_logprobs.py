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
"""Owned packed values, bounded streaming custody, and engine-loop integration."""

import asyncio
import sys
import threading
import weakref
from array import array
from types import SimpleNamespace as NS

import numpy as np
import pytest

from nemo_rl.experience.direct_capture import (
    DirectCapture,
    DirectCaptureConfig,
    current_capture,
)
from nemo_rl.experience.packed_logprobs import PackedLogprobBuffer, PackedLogprobCapture
from nemo_rl.experience.vllm_http_client import _AsyncLLMHTTPClient


def native(tokens, *, values=None, columns=1):
    ids = np.repeat(np.asarray(tokens, dtype=np.int64)[:, None], columns, axis=1)
    lps = np.full(ids.shape, -0.25, dtype=np.float32)
    if values is not None:
        lps[:, 0] = values
    return ids, lps, np.ones(len(tokens), dtype=np.int64), None


def terminal(tokens=(3,), *, finished=True, reason="length", routes=None):
    return NS(
        finished=finished,
        prompt_token_ids=[1, 2],
        outputs=[
            NS(
                index=0,
                token_ids=list(tokens),
                logprobs=None,
                finish_reason=reason,
                routed_experts=routes,
            )
        ],
    )


@pytest.mark.parametrize("packing", ["fused", "legacy"])
@pytest.mark.parametrize("columns", [1, 4])
def test_exact_values_source_reuse_and_terminal_routes(packing, columns):
    buffer = PackedLogprobBuffer(5)
    buffer.prompt = [1, 2]
    arrays = native([3, 4, 5], values=[-0.25, -0.5, -1.75], columns=columns)
    buffer.append([3, 4, 5], arrays)
    arrays[0][:] = 900
    arrays[1][:] = 100
    buffer.append([6], native([6], values=[-2]))
    routes = object()
    buffer.final = terminal([6], routes=routes)
    call = buffer.extract("rollout", ordinal=3, weight_version=7, packing=packing)
    assert list(call.logprobs) == [-0.25, -0.5, -1.75, -2]
    assert call.call_id == "rollout/3"
    assert call.weight_version == 7
    assert call.routed_experts is routes
    if packing == "fused":
        assert list(call.token_ids) == [1, 2, 3, 4, 5, 6]
        buffer.prompt[0] = 99
        assert call.token_ids[0] == 1
    else:
        assert call.prompt == (1, 2)
        assert call.tokens == (3, 4, 5, 6)


@pytest.mark.parametrize("generated", [65536, 131072])
def test_long_single_token_stream_releases_source_arrays_and_obeys_limit(generated):
    buffer = PackedLogprobBuffer(generated)
    buffer.prompt = [1]
    for _ in range(generated):
        source = native([7])
        ids_ref, values_ref = weakref.ref(source[0]), weakref.ref(source[1])
        buffer.append([7], source)
        del source
        assert ids_ref() is None and values_ref() is None
    buffer.final = terminal([7])
    call = buffer.extract("r", ordinal=0, weight_version=0, packing="fused")
    assert len(call.logprobs) == generated
    assert sum(call.logprobs) == -0.25 * generated
    # Flat storage includes only amortized array slack, not per-chunk objects.
    assert (
        sys.getsizeof(buffer.tokens) + sys.getsizeof(buffer.logprobs) < 14 * generated
    )
    buffer.append([7], native([7]))
    with pytest.raises(ValueError, match="limit"):
        buffer.extract("r", ordinal=0, weight_version=0, packing="fused")
    assert not buffer.tokens and not buffer.logprobs


@pytest.mark.parametrize(
    "kind", ["missing", "length", "alignment", "dtype", "nonfinite"]
)
def test_invalid_native_values_fail_at_barrier_without_killing_output_handler(kind):
    buffer = PackedLogprobBuffer(8)
    buffer.prompt = [1]
    source = native([3, 4])
    if kind == "missing":
        source = None
    elif kind == "length":
        source = native([3])
    elif kind == "alignment":
        source = native([3, 8])
    elif kind == "dtype":
        source = (source[0], source[1].astype(np.float64), source[2], None)
    else:
        source[1][0, 0] = np.nan
    buffer.append([3, 4], source)
    buffer.final = terminal([4])
    with pytest.raises(ValueError):
        buffer.extract("r", ordinal=0, weight_version=0, packing="fused")


class PythonProcessor:
    def __init__(self):
        self.logprobs = []
        self.cumulative_logprob = 0.0
        self.prompt_logprobs = None
        self.updates = 0

    def update_from_output(self, output):
        self.updates += 1
        for token, value in zip(output.new_token_ids, output.new_logprobs[1][:, 0]):
            self.logprobs.append({token: NS(logprob=float(value))})
            self.cumulative_logprob += float(value)

    def _update_prompt_logprobs(self, values):
        self.prompt_logprobs = values

    def pop_prompt_logprobs(self):
        result, self.prompt_logprobs = self.prompt_logprobs, None
        return result


class OutputProcessor:
    def __init__(self):
        self.request_states = {}

    def add_request(
        self, request, prompt, parent_req=None, request_index=0, queue=None
    ):
        self.request_states[request.request_id] = NS(
            prompt_token_ids=request.prompt_token_ids,
            logprobs_processor=PythonProcessor(),
            queue=queue,
        )


def request(external="call", **overrides):
    return NS(
        **(
            {
                "external_req_id": external,
                "request_id": "internal-" + external,
                "resumable": False,
                "prompt_token_ids": [1, 2],
                "prompt_embeds": None,
            }
            | overrides
        )
    )


@pytest.fixture
def vllm_version(monkeypatch):
    module = NS(__version__="0.29.0")
    monkeypatch.setitem(sys.modules, "vllm", module)
    return module


@pytest.mark.parametrize("export", [False, True])
def test_hook_is_request_scoped_and_preserves_explicit_api_and_prompt_logprobs(
    vllm_version, export
):
    processor = OutputProcessor()
    capture = PackedLogprobCapture(processor)
    buffer = PackedLogprobBuffer(8)
    capture.register("call", buffer, export_logprobs=export)
    processor.add_request(request(), None)
    processor.add_request(request("uncaptured"), None)
    wrapped = processor.request_states["internal-call"].logprobs_processor
    untouched = processor.request_states["internal-uncaptured"].logprobs_processor
    assert type(untouched) is PythonProcessor
    output = NS(
        new_token_ids=[3, 4],
        new_logprobs=native([3, 4], columns=3),
        new_prompt_logprobs_tensors=None,
    )
    wrapped.update_from_output(output)
    assert wrapped.original.updates == int(export)
    if export:
        assert wrapped.logprobs is wrapped.original.logprobs
        assert wrapped.cumulative_logprob == -0.5
    else:
        assert wrapped.logprobs is None
        assert wrapped.cumulative_logprob is None
        output.new_token_ids = []
        output.new_logprobs = None
        output.new_prompt_logprobs_tensors = object()
        wrapped.update_from_output(output)
        assert wrapped.pop_prompt_logprobs() is output.new_prompt_logprobs_tensors
        assert wrapped.prompt_logprobs is None
    assert list(buffer.logprobs) == [-0.25, -0.25]
    capture.unregister("call")
    assert not capture._bindings


def test_unknown_vllm_version_fails_before_installation(vllm_version):
    vllm_version.__version__ = "0.30.0"
    processor = OutputProcessor()
    original = processor.add_request
    with pytest.raises(RuntimeError, match="0.29.0"):
        PackedLogprobCapture(processor)
    assert processor.add_request == original


@pytest.mark.parametrize(
    "failure", ["cancel", "engine_error", "overflow", "stage_error"]
)
def test_failure_drains_bindings_and_slots_without_training_partial_calls(
    vllm_version, failure
):
    async def run():
        stage_calls = []

        def stage(call):
            stage_calls.append(call)
            if failure == "stage_error":
                raise ValueError("storage unavailable")
            pytest.fail("incomplete capture must not stage")

        capture = DirectCapture(
            DirectCaptureConfig(max_pending_calls=1, max_generated_tokens_per_call=1),
            stage,
            lambda: 0,
            clear=lambda keys: None,
        )
        capture.register("key", "rollout", temperature=1, top_p=1)
        aborted = []

        class Engine:
            model_config = renderer = input_processor = vllm_config = None

            def __init__(self):
                self.output_processor = OutputProcessor()

            async def generate(self, prompt, sampling, request_id, **kwargs):
                self.output_processor.add_request(
                    request(request_id), None, queue=NS(put=lambda output: None)
                )
                processor = self.output_processor.request_states[
                    "internal-" + request_id
                ].logprobs_processor
                processor.update_from_output(
                    NS(
                        new_token_ids=[3],
                        new_logprobs=native([3]),
                        new_prompt_logprobs_tensors=None,
                    )
                )
                if failure == "engine_error":
                    raise RuntimeError("engine failed")
                if failure == "cancel":
                    yield terminal([3], finished=False)
                    await asyncio.Event().wait()
                if failure == "overflow":
                    processor.update_from_output(
                        NS(
                            new_token_ids=[4],
                            new_logprobs=native([4]),
                            new_prompt_logprobs_tensors=None,
                        )
                    )
                final = terminal([3])
                self.output_processor.request_states[
                    "internal-" + request_id
                ].queue.put(final)
                yield final
                self.output_processor.request_states.clear()

            async def abort(self, request_id):
                aborted.append(request_id)
                self.output_processor.request_states.clear()

        bridge = _AsyncLLMHTTPClient(Engine(), asyncio.get_running_loop())
        bridge.enable_packed_capture()
        context = capture.enter("key")
        token = current_capture.set(context)
        stream = bridge.generate(
            None,
            NS(
                logprobs=None,
                num_logprobs=None,
                max_tokens=4,
                output_kind=NS(name="DELTA"),
            ),
            "call",
        )
        try:
            if failure == "cancel":
                await anext(stream)
                pending = asyncio.create_task(anext(stream))
                await asyncio.sleep(0)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            elif failure == "engine_error":
                with pytest.raises(RuntimeError, match="engine failed"):
                    await anext(stream)
            else:
                async for _ in stream:
                    pass
        finally:
            await stream.aclose()
            current_capture.reset(token)
            capture.leave(context)
        try:
            receipt = await asyncio.to_thread(capture.finalize, "key", timeout=5)
            assert receipt["failure"] is not None
            assert not receipt["manifest"]
            assert len(stage_calls) == int(failure == "stage_error")
            assert aborted == (
                ["call"] if failure in {"cancel", "engine_error"} else []
            )
            assert not bridge._packed_capture._bindings
            # Failure must release admission, including worker-side failures.
            capture.register("next", "next", temperature=1, top_p=1)
            next_context = capture.enter("next")
            capture.begin(next_context)
            capture.abort(next_context)
            capture.leave(next_context)
        finally:
            capture.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize(
    "overrides",
    [{"resumable": True}, {"prompt_embeds": object()}, {"prompt_token_ids": None}],
)
def test_unsupported_requests_fail_before_engine_admission(vllm_version, overrides):
    processor = OutputProcessor()
    capture = PackedLogprobCapture(processor)
    capture.register("call", PackedLogprobBuffer(8), export_logprobs=False)
    with pytest.raises(ValueError):
        processor.add_request(request(**overrides), None)
    assert not processor.request_states


@pytest.mark.parametrize("output_kind", ["DELTA", "CUMULATIVE", "FINAL_ONLY"])
@pytest.mark.parametrize("public_logprobs", [None, 3, "token_ids"])
@pytest.mark.parametrize("concurrent", [1, 2])
def test_cross_loop_capture_uses_engine_batches_independent_of_http_chunks(
    vllm_version, output_kind, public_logprobs, concurrent
):
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_engine():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run_engine)
    thread.start()
    assert ready.wait(5)
    calls = []
    capture = DirectCapture(
        DirectCaptureConfig(),
        lambda call: calls.append(call) or {"ordinal": call.ordinal},
        lambda: 7,
        clear=lambda keys: None,
    )
    capture.register("key", "rollout", temperature=1, top_p=1)
    logprobs_arg = None if public_logprobs == "token_ids" else public_logprobs
    num_logprobs = 3 if public_logprobs == "token_ids" else public_logprobs

    class Engine:
        model_config = renderer = input_processor = vllm_config = None

        def __init__(self):
            self.output_processor = OutputProcessor()

        async def generate(self, prompt, sampling, request_id, **kwargs):
            assert asyncio.get_running_loop() is loop
            assert sampling.logprobs == (0 if public_logprobs is None else logprobs_arg)
            self.output_processor.add_request(
                request(request_id), None, queue=NS(put=lambda output: None)
            )
            processor = self.output_processor.request_states[
                "internal-" + request_id
            ].logprobs_processor
            # Four native batches, two emitted chunks (or only the final one).
            for index in range(4):
                token = index + 3
                processor.update_from_output(
                    NS(
                        new_token_ids=[token],
                        new_logprobs=native([token]),
                        new_prompt_logprobs_tensors=None,
                    )
                )
                await asyncio.sleep(0)
                if index in (1, 3) and (output_kind != "FINAL_ONLY" or index == 3):
                    tokens = (
                        [token - 1, token]
                        if output_kind == "DELTA"
                        else list(range(3, token + 1))
                    )
                    output = terminal(
                        tokens,
                        finished=index == 3,
                        routes="routes" if index == 3 else None,
                    )
                    output.outputs[0].logprobs = processor.logprobs
                    self.output_processor.request_states[
                        "internal-" + request_id
                    ].queue.put(output)
                    yield output
            del self.output_processor.request_states["internal-" + request_id]

        async def abort(self, request_id):
            pytest.fail("completed generation should not abort")

    engine = Engine()
    bridge = _AsyncLLMHTTPClient(engine, loop)

    async def install():
        bridge.enable_packed_capture()

    asyncio.run_coroutine_threadsafe(install(), loop).result(5)

    async def consume(call_id):
        context = capture.enter("key")
        token = current_capture.set(context)
        sampling = NS(
            logprobs=logprobs_arg,
            num_logprobs=num_logprobs,
            max_tokens=16,
            output_kind=NS(name=output_kind),
        )
        try:
            outputs = [item async for item in bridge.generate(None, sampling, call_id)]
            assert sampling.logprobs == logprobs_arg
            assert all(item.outputs[0].routed_experts is None for item in outputs)
            assert (outputs[-1].outputs[0].logprobs is None) == (
                public_logprobs is None
            )
        finally:
            current_capture.reset(token)
            capture.leave(context)

    async def consume_all():
        await asyncio.gather(*(consume(str(index)) for index in range(concurrent)))
        receipt = await asyncio.to_thread(capture.finalize, "key", timeout=5)
        assert receipt["failure"] is None

    try:
        asyncio.run(consume_all())
        assert {call.ordinal for call in calls} == set(range(concurrent))
        assert len(calls) == concurrent
        for call in calls:
            assert list(call.token_ids) == [1, 2, 3, 4, 5, 6]
            assert isinstance(call.logprobs, array)
            assert list(call.logprobs) == [-0.25] * 4
            assert call.routed_experts == "routes"
        assert not bridge._packed_capture._bindings
    finally:
        capture.shutdown()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()
