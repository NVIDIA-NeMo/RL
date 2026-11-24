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
"""The token-in/token-out core of the vLLM HTTP server."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nemo_rl.models.generation.tito import (
    TITO_GENERATE_ROUTE,
    TitoGenerateRequest,
    TitoGenerateResponse,
    TitoRootRejected,
)
from nemo_rl.models.generation.vllm.utils import (
    chat_generation_tokens,
    check_sampling_param,
)


async def resolve_tito_parent(worker: Any, admission: Any) -> tuple[Any, list[int]]:
    """Resolve the tokens a captured call continues.

    Args:
        worker: The generation worker that hosts token capture.
        admission: The call's ``CaptureAdmission``, already decided by the
            caller: ``token_in`` continues a parent and ``text`` starts a root.

    Returns:
        The admission to record the call under and the parent's prompt followed
        by its generated tokens, which is empty for a root.

    Raises:
        ValueError: The admission is undecided, its parent's tokens differ from
            the recorded length, or it restores a generation prefix.
    """
    if admission.mode == "candidate":
        raise ValueError(
            "A token-in/token-out call needs a decided capture admission; "
            "candidate admissions are resolved by the chat adapter"
        )
    parent: list[int] = []
    if admission.mode == "token_in":
        parent = await asyncio.to_thread(worker._resolve_admission_prefix, admission)
        if len(parent) != admission.prev_len:
            raise ValueError(
                f"Parent {admission.parent_call_id} resolved to {len(parent)} "
                f"tokens but the admission records {admission.prev_len}"
            )
    admission, generation_cut = await asyncio.to_thread(
        worker._prefix_recovery.resolve_restore, admission, parent
    )
    if generation_cut is not None:
        raise ValueError(
            "The token-in/token-out interface does not restore generation prefixes"
        )
    return admission, parent


def register_tito_route(
    app: FastAPI,
    *,
    worker: Any,
    engine_client: Any,
    abort_request_capture: Callable[[Any, str], Awaitable[None]],
    reject_root_reasons: frozenset[str],
) -> None:
    """Serve token-in/token-out model calls from the worker's engine.

    Args:
        app: The worker's HTTP application.
        worker: The generation worker that hosts token capture.
        engine_client: The worker's HTTP engine client.
        abort_request_capture: Fails a request's in-flight capture.
        reject_root_reasons: Gym root reasons that refuse a call.
    """
    # vLLM is optional outside the serving worker.
    from vllm.entrypoints.serve.utils.api_utils import with_cancellation
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    def refusal(message: str, *, headers: dict[str, str] | None = None) -> JSONResponse:
        return JSONResponse(
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": "token_ids",
                    "code": 400,
                }
            },
            status_code=400,
            headers=headers,
        )

    @app.post(TITO_GENERATE_ROUTE)
    @with_cancellation
    async def tito_generate(request: TitoGenerateRequest, raw_request: Request):
        top_k = check_sampling_param(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            generation_config=worker.cfg,
        )
        admission = None
        parent: list[int] = []
        if request.ng_capture is not None:
            admission = worker._capture_admission(request)
            if admission is None:
                return refusal("ng_capture requires token capture on this worker")
            if admission.root_reason in reject_root_reasons:
                return refusal(str(TitoRootRejected(admission.root_reason)))
            try:
                admission, parent = await resolve_tito_parent(worker, admission)
            except ValueError as error:
                return refusal(str(error))
        prompt_token_ids = parent + request.token_ids
        max_model_len = engine_client.model_config.max_model_len
        remaining = max_model_len - len(prompt_token_ids)
        if remaining <= 0:
            # Gym matches the literal "context length" to handle overflow.
            return refusal(
                f"Prompt length ({len(prompt_token_ids)}) fills or exceeds this "
                f"model's maximum context length ({max_model_len}). No room for "
                "output tokens.",
                headers=worker._context_refusal_headers(request),
            )
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=top_k,
            min_p=0.0,
            repetition_penalty=1.0,
            max_tokens=min(request.max_tokens or remaining, remaining),
            seed=request.seed,
            logprobs=0,
            stop_token_ids=worker.cfg["stop_token_ids"],
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        worker._begin_request_capture(
            request,
            prompt_token_ids,
            admission=admission,
            prefix_token_ids=parent or None,
        )
        try:
            (engine_input,) = await engine_client.renderer.render_cmpl_async(
                [{"prompt_token_ids": prompt_token_ids}]
            )
            final_output = None
            async for output in engine_client.generate(
                engine_input, sampling_params, f"tito-{uuid.uuid4().hex}"
            ):
                final_output = output
            if final_output is None:
                raise RuntimeError("The engine returned no output")
            generation = final_output.outputs[0]
            token_ids, logprobs = chat_generation_tokens(generation)
            # Capture stages a call from a chat-completion payload.
            content = await asyncio.to_thread(
                worker._finish_request_capture,
                request,
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": generation.finish_reason,
                            "stop_reason": generation.stop_reason,
                            "message": {
                                "role": "assistant",
                                "generation_token_ids": list(token_ids),
                                "generation_log_probs": list(logprobs),
                            },
                        }
                    ]
                },
            )
        except asyncio.CancelledError:
            await abort_request_capture(request, "client_disconnect")
            raise
        except BaseException:
            await abort_request_capture(request, "engine_error")
            raise
        return JSONResponse(
            content=TitoGenerateResponse(
                call_id=admission.model_call_id if admission is not None else None,
                prompt_token_count=len(prompt_token_ids),
                token_ids=token_ids,
                logprobs=logprobs,
                finish_reason=generation.finish_reason,
                stop_reason=generation.stop_reason,
                ng_commit_coords=content.get("ng_commit_coords"),
            ).model_dump()
        )
