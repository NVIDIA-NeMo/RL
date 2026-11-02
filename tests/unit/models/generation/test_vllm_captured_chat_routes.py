# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Capture-only serving views preserve the original replay route arrays."""

import ast
import asyncio
import io
import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from nemo_rl.models.generation.vllm.captured_chat_logprobs import (
    CapturedChatLogprobsMixin,
)
from nemo_rl.models.generation.vllm.utils import (
    attach_routed_experts_to_chat_response_choices,
    attach_token_information_to_chat_response_choices,
    without_native_chat_routed_experts,
)
from nemo_rl.utils.routed_experts_codec import decode_routed_experts

_ROOT = Path(__file__).resolve().parents[4]
_WORKER_SOURCE = _ROOT / "nemo_rl/models/generation/vllm/vllm_worker_async.py"


class _Response:
    def __init__(self, output: Any, native_routes: str | None) -> None:
        self.choices = [
            SimpleNamespace(
                index=output.index,
                message=SimpleNamespace(content=output.text),
                routed_experts=native_routes,
            )
        ]
        self.usage = None


def _request(*, stream: bool = False, logprobs: bool = True) -> Any:
    return SimpleNamespace(
        stream=stream,
        logprobs=logprobs,
        top_logprobs=0,
        return_tokens_as_token_ids=True,
        _restored_generation_token_ids=[],
        _restored_prefix_terminal=None,
    )


def _output(seed: int = 0) -> Any:
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        request_id="engine-request",
        prompt_token_ids=[1, 2, 3],
        outputs=[
            SimpleNamespace(
                index=0,
                text="tail",
                token_ids=[4, 5, 6, 7],
                logprobs=[
                    {token: SimpleNamespace(logprob=-0.25)} for token in [4, 5, 6, 7]
                ],
                routed_experts=rng.integers(0, 128, (6, 2, 4), dtype=np.int16),
            )
        ],
    )


@pytest.fixture
def serving_factory():
    """Execute the actual nested serving mixin without Ray/GPU initialization."""
    tree = ast.parse(_WORKER_SOURCE.read_text())
    mixin = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name == "NeMoRLOpenAIServingChatMixin"
    )

    def factory(captured: list[Any]):
        observed = []
        restored = []
        worker = SimpleNamespace(
            _capture_calls={id(request): object() for request in captured},
            _observe_request_capture=lambda request, raw: observed.append(raw),
            _restore_response_prefix=lambda request, raw, **kwargs: restored.append(
                raw
            ),
            _return_routed_experts_enabled=lambda: True,
            routed_experts_dtype=torch.int16,
        )
        namespace = {
            "asyncio": asyncio,
            "worker_self": worker,
            "ChatCompletionResponse": _Response,
            "_RestoredPrefixTerminal": type("RestoredTerminal", (), {}),
            "attach_token_information_to_chat_response_choices": attach_token_information_to_chat_response_choices,
            "attach_routed_experts_to_chat_response_choices": attach_routed_experts_to_chat_response_choices,
            "without_native_chat_routed_experts": without_native_chat_routed_experts,
            "torch": torch,
            "LOGGER": logging.getLogger(__name__),
        }
        exec(
            compile(
                ast.Module(body=[mixin], type_ignores=[]), str(_WORKER_SOURCE), "exec"
            ),
            namespace,
        )
        calls = []
        upstream_outputs = []

        class UpstreamFormatter:
            async def chat_completion_full_generator(
                self, request, result_generator, *args, **kwargs
            ):
                async for result in result_generator:
                    await asyncio.sleep(0)
                    upstream_outputs.append(result)
                    output = result.outputs[0]
                    native = None
                    if output.routed_experts is not None:
                        calls.append(output.routed_experts)
                        native = "native-npy-base64"
                    response = _Response(output, native)
                return response

        class Serving(
            namespace["NeMoRLOpenAIServingChatMixin"],
            CapturedChatLogprobsMixin,
            UpstreamFormatter,
        ):
            renderer = SimpleNamespace(tokenizer=None)
            return_tokens_as_token_ids = True

        return Serving(), observed, restored, calls, upstream_outputs

    return factory


def test_serving_view_preserves_all_original_arrays_and_metadata():
    raw = _output()
    raw.outputs.append(_output(1).outputs[0])
    view = without_native_chat_routed_experts(raw)
    assert view is not raw and view.outputs is not raw.outputs
    assert view.request_id == raw.request_id
    assert view.prompt_token_ids is raw.prompt_token_ids
    for original, formatted in zip(raw.outputs, view.outputs, strict=True):
        assert formatted is not original
        assert formatted.routed_experts is None
        assert original.routed_experts is not None
        assert formatted.token_ids is original.token_ids
        assert formatted.logprobs is original.logprobs


@pytest.mark.asyncio
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("logprobs", [False, True])
async def test_bypass_scope_and_original_route_capture(
    serving_factory, captured, stream, logprobs
):
    request = _request(stream=stream, logprobs=logprobs)
    raw = _output()
    original = raw.outputs[0].routed_experts.copy()
    serving, observed, restored, calls, upstream_outputs = serving_factory(
        [request] if captured else []
    )

    async def results():
        yield raw

    response = await serving.chat_completion_full_generator(request, results())
    bypassed = captured and not stream
    assert len(calls) == (0 if bypassed else 1)
    assert observed == [raw] and restored == [raw]
    assert np.array_equal(raw.outputs[0].routed_experts, original)
    assert (upstream_outputs[0] is raw) is (not bypassed)
    assert response.choices[0].message.content == "tail"
    routes = decode_routed_experts(response.choices[0].message.routed_experts)
    assert torch.equal(routes[:-1], torch.from_numpy(original))
    assert routes[-1].eq(-1).all()
    if logprobs:
        assert response.choices[0].message.generation_token_ids == [4, 5, 6, 7]
        assert response.choices[0].message.generation_log_probs == [-0.25] * 4


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(20))
async def test_concurrent_ordinary_request_preserves_identical_capture_routes(
    serving_factory, seed
):
    captured, ordinary = _request(), _request()
    raw = _output(seed)
    serving, _, _, calls, upstream_outputs = serving_factory([captured])

    async def results():
        yield raw

    optimized, baseline = await asyncio.gather(
        serving.chat_completion_full_generator(captured, results()),
        serving.chat_completion_full_generator(ordinary, results()),
    )
    assert len(calls) == 1
    assert optimized.choices[0].routed_experts is None
    assert baseline.choices[0].routed_experts == "native-npy-base64"
    encoded = optimized.choices[0].message.routed_experts
    assert encoded == baseline.choices[0].message.routed_experts
    payloads = [
        {
            "choices": [
                {"message": {"routed_experts": encoded}, "routed_experts": native}
            ]
        }
        for native in [None, "native-npy-base64"]
    ]
    adapter = pytest.importorskip(
        "nemo_gym.token_id_capture.adapters.vllm"
    ).VLLMCaptureAdapter()
    assert adapter.extract_extras(payloads[0]) == adapter.extract_extras(payloads[1])
    assert any(result is raw for result in upstream_outputs)
    assert raw.outputs[0].routed_experts is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RuntimeError("engine failed"), asyncio.CancelledError()]
)
async def test_generation_failure_preserves_engine_routes(serving_factory, error):
    request, raw = _request(), _output()
    routes = raw.outputs[0].routed_experts
    serving, _, _, calls, _ = serving_factory([request])

    async def results():
        yield raw
        raise error

    with pytest.raises(type(error)):
        await serving.chat_completion_full_generator(request, results())
    assert raw.outputs[0].routed_experts is routes
    assert not calls


def test_terminal_cleanup_strips_native_routes_only_after_staging():
    tree = ast.parse(_WORKER_SOURCE.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_finish_request_capture_with_lifecycle_owned"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, method], type_ignores=[])
            ),
            str(_WORKER_SOURCE),
            "exec",
        ),
        namespace,
    )
    staged = []
    coords = SimpleNamespace(model_dump=lambda: {"test": "coordinates"})

    def stage(call, payload, **kwargs):
        # The existing record extraction still sees the full message envelope.
        staged.append(payload["choices"][0]["message"]["routed_experts"])
        return coords

    worker = SimpleNamespace(
        token_capture=SimpleNamespace(adapter=None, complete_call_from_response=stage),
        _remember_completed_capture=lambda *args, **kwargs: None,
        _pop_request_capture=lambda *args: None,
    )
    state = SimpleNamespace(
        call=SimpleNamespace(model_call_id="call", generation_cut=None),
        prompt_token_ids=[1],
        media=None,
        effective_output_limit=8,
        terminal_finish_reason="length",
        terminal_stop_reason=None,
        lock=threading.Lock(),
        generation_cut_staging_keys=[],
    )
    content = {
        "choices": [
            {
                "routed_experts": "unused-native",
                "message": {"content": "text", "routed_experts": "capture-envelope"},
            }
        ]
    }
    result, _ = namespace["_finish_request_capture_with_lifecycle_owned"](
        worker, state, _request(), content
    )
    assert staged == ["capture-envelope"]
    assert "routed_experts" not in result["choices"][0]
    assert "routed_experts" not in result["choices"][0]["message"]
    assert result["choices"][0]["message"]["content"] == "text"


def test_bypass_avoids_actual_pinned_vllm_native_encoding():
    path = os.environ.get("NRL_TEST_VLLM_CHAT_SERVING_SOURCE")
    if path is None:
        spec = pytest.importorskip("vllm.entrypoints.openai.chat_completion.serving")
        path = spec.__file__
    tree = ast.parse(Path(path).read_text())
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "routed_experts_b64"
            for target in node.targets
        )
    )
    code = compile(ast.Module(body=[assignment], type_ignores=[]), path, "exec")
    calls = []

    def encode(array):
        calls.append(array)
        with io.BytesIO() as buffer:
            np.save(buffer, array, allow_pickle=False)
            return buffer.getvalue()

    raw = _output()
    namespace = {"output": raw.outputs[0], "numpy2base64": encode}
    exec(code, namespace)
    assert len(calls) == 1
    namespace["output"] = without_native_chat_routed_experts(raw).outputs[0]
    exec(code, namespace)
    assert namespace["routed_experts_b64"] is None
    assert len(calls) == 1
    assert raw.outputs[0].routed_experts is not None
