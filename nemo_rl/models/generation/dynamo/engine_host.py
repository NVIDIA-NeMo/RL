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

"""NeMo RL request handling inside a managed ``dynamo.vllm`` engine process.

The engine process owns vLLM's AsyncLLM, as NeMo RL's async vLLM worker does,
so per-request features of the native backend run here through the same shared
code (:class:`TokenCaptureHostMixin`). Dynamo's Rust frontend replaces vLLM's
Python OpenAI layer, so requests arrive pre-tokenized; anything that needs the
chat messages stays out of reach.

This module imports no Dynamo code, so it is testable without the Dynamo venv;
``engine_main`` binds it to Dynamo's handler.
"""

import asyncio
import functools
import json
import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import ray

from nemo_rl.models.generation.openai_server_utils import (
    derive_prefix_eos_count,
    splice_prefix_tokens,
)
from nemo_rl.models.generation.vllm.token_capture_host import (
    TokenCaptureHostMixin,
    parse_capture_admission,
)
from nemo_rl.models.generation.vllm.utils import off_policy_sampling_error

LOGGER = logging.getLogger(__name__)

# Request annotations set by NeMo Gym's Dynamo dialect (vllm_model with
# dynamo_frontend). Dynamo forwards nvext.annotations to the engine verbatim.
ROLLOUT_ANNOTATION = "nemo_gym:rollout"
CAPTURE_ANNOTATION_PREFIX = "ng_capture:"

# Engine routes, served at {system_url}/engine/<name>.
SETUP_TOKEN_CAPTURE_ROUTE = "nrl_setup_token_capture"
SET_ROLLOUT_WEIGHT_VERSION_ROUTE = "nrl_set_rollout_weight_version"

# Environment the managing actor sets for the engine process.
SAMPLING_PROFILE_ENV = "NRL_DYNAMO_SAMPLING_PROFILE"
RAY_ADDRESS_ENV = "NRL_DYNAMO_RAY_ADDRESS"

_SAMPLING_PROFILE_KEYS = ("temperature", "top_p", "val_temperature", "val_top_p")
_ENGINE_DATA_TOKEN_FIELDS = (
    "completion_token_ids",
    "completion_logprobs",
    "prompt_token_ids",
)

ParentGenerate = Callable[[dict[str, Any], Any], AsyncIterator[dict[str, Any]]]


class EngineRequestError(Exception):
    """A request refused before generation; the handler returns it as HTTP."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def sampling_profile_env(generation_config: Mapping[str, Any]) -> str:
    """Serialize the train and validation sampling the engine enforces on rollouts."""
    return json.dumps({key: generation_config[key] for key in _SAMPLING_PROFILE_KEYS})


def _annotation_value(annotations: list[str], prefix: str) -> str | None:
    for annotation in annotations:
        if annotation.startswith(prefix):
            return annotation[len(prefix) :]
    return None


class NemoRLEngineHostMixin(TokenCaptureHostMixin):
    """NeMo-Gym rollout handling for a ``dynamo.vllm`` decode handler.

    Requests without NeMo Gym's annotations pass through untouched. Rollout
    requests get the native server's checks and output clamp; captured
    requests additionally splice their staged prefix, stage the finished call
    to TransferQueue from this process, and return only commit coordinates.
    """

    _nrl_max_model_len: int
    _nrl_eos_token_id: int
    _nrl_sampling_profile: dict[str, Any]
    _nrl_executor: ThreadPoolExecutor

    def _init_nemo_rl_engine_host(
        self, *, max_model_len: int, eos_token_id: int, sampling_profile: str
    ) -> None:
        """Set up dormant host state; capture stays off until its route runs."""
        self._init_token_capture_host()
        self._nrl_max_model_len = max_model_len
        self._nrl_eos_token_id = eos_token_id
        self._nrl_sampling_profile = json.loads(sampling_profile)
        # Data-plane reads and writes block; the engine serves every stream from
        # one event loop, so they run here instead.
        self._nrl_executor = ThreadPoolExecutor(thread_name_prefix="nrl-capture")

    async def _run_off_loop(self, fn: Callable[..., Any], *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._nrl_executor, functools.partial(fn, *args)
        )

    async def nrl_setup_token_capture(self, body: dict[str, Any]) -> dict[str, Any]:
        """Engine route: attach to the data plane and install token capture."""

        def setup() -> None:
            # The engine process is not a Ray worker. Attach explicitly, so a
            # missing address fails here instead of starting a local cluster.
            if not ray.is_initialized():
                ray.init(address=os.environ[RAY_ADDRESS_ENV], log_to_driver=False)
            self._install_token_capture_host(
                body["dp_cfg"],
                body["staging_partition"],
                capture_media=False,
                media_pixel_dtype=None,
            )

        await self._run_off_loop(setup)
        return {"status": "ok"}

    async def nrl_set_rollout_weight_version(
        self, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Engine route: rotate the weight version stamped on captured calls."""
        await self.set_rollout_weight_version(int(body["version"]))
        return {"status": "ok"}

    def _check_rollout_request(self, request: dict[str, Any]) -> None:
        if (request.get("stop_conditions") or {}).get("stop"):
            raise EngineRequestError(
                400,
                "stop strings are not supported for NeMo-Gym rollouts on managed "
                "Dynamo: the frontend matches them itself and drops the final "
                "chunk that carries the generated token ids.",
            )
        sampling_options: dict[str, Any] = request.get("sampling_options") or {}
        error = off_policy_sampling_error(
            temperature=sampling_options.get("temperature"),
            top_p=sampling_options.get("top_p"),
            top_k=sampling_options.get("top_k"),
            generation_config=self._nrl_sampling_profile,
        )
        if error is not None:
            raise EngineRequestError(400, error)

    def _clamp_max_tokens(self, request: dict[str, Any], prompt_len: int) -> None:
        """Cap output so prompt + output fit max_model_len, as the native server does."""
        remaining = self._nrl_max_model_len - prompt_len
        if remaining <= 0:
            # Keep "maximum context length": Gym treats a 400 naming the context
            # window as a length refusal rather than an engine failure.
            raise EngineRequestError(
                400,
                f"This model's maximum context length is {self._nrl_max_model_len} "
                f"tokens. However, your messages resulted in {prompt_len} tokens. "
                "Please reduce the length of the messages.",
            )
        stop_conditions = request.get("stop_conditions") or {}
        max_tokens = stop_conditions.get("max_tokens")
        if max_tokens is not None and max_tokens > remaining:
            stop_conditions["max_tokens"] = remaining
            request["stop_conditions"] = stop_conditions

    async def _nemo_rl_generate(
        self, request: dict[str, Any], context: Any, generate: ParentGenerate
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one request through ``generate`` (Dynamo's handler) with NeMo RL's hooks."""
        annotations = request.get("annotations") or []
        capture_context = _annotation_value(annotations, CAPTURE_ANNOTATION_PREFIX)
        if capture_context is None and ROLLOUT_ANNOTATION not in annotations:
            async for chunk in generate(request, context):
                yield chunk
            return

        self._check_rollout_request(request)
        admission = None
        if self.token_capture is not None and capture_context is not None:
            admission = parse_capture_admission(json.loads(capture_context))
        token_ids = list(request["token_ids"])
        prefix_token_ids = None
        if admission is not None and admission.mode == "token_in":
            prefix_token_ids = await self._run_off_loop(
                self._resolve_admission_prefix, admission
            )
            # The frontend rendered the full conversation; replace the history
            # with the exact tokens staged for it.
            token_ids = splice_prefix_tokens(
                tokenizer=None,
                model_prefix_token_ids=prefix_token_ids,
                template_token_ids=token_ids,
                template_prefix_eos_count=derive_prefix_eos_count(
                    prefix_token_ids, eos_token_id=self._nrl_eos_token_id
                ),
                eos_token_id=self._nrl_eos_token_id,
            ).token_ids
            request["token_ids"] = token_ids
        self._clamp_max_tokens(request, len(token_ids))

        if admission is None:
            async for chunk in generate(request, context):
                yield chunk
            return

        # Sampled-token logprobs feed the staged record through engine_data.
        output_options = request.get("output_options") or {}
        if output_options.get("logprobs") is None:
            output_options["logprobs"] = 0
        request["output_options"] = output_options
        self._begin_request_capture(
            request, token_ids, admission=admission, prefix_token_ids=prefix_token_ids
        )
        finished = False
        try:
            async for chunk in generate(request, context):
                # Staged to TransferQueue; keep them off the frontend hop.
                chunk.pop("log_probs", None)
                chunk.pop("top_logprobs", None)
                if chunk.get("finish_reason") is not None:
                    chunk = await self._finish_engine_capture(request, chunk, context)
                    finished = True
                yield chunk
        finally:
            if not finished:
                self._abort_request_capture(
                    request,
                    reason=(
                        "client_disconnect" if context.is_stopped() else "engine_error"
                    ),
                )

    async def _finish_engine_capture(
        self, request: dict[str, Any], chunk: dict[str, Any], context: Any
    ) -> dict[str, Any]:
        """Stage the finished call and replace engine_data with its coordinates."""
        finish_reason = chunk["finish_reason"]
        engine_data = dict(chunk.get("engine_data") or {})
        failure = None
        if context.is_stopped() or context.is_killed():
            failure = "client_disconnect"
        elif finish_reason == "cancelled" or str(finish_reason).startswith("error"):
            failure = "engine_error"
        elif "completion_token_ids" not in engine_data:
            LOGGER.error(
                "Captured request %s returned no engine_data; the client must "
                "request nvext.extra_fields=['engine_data']",
                context.id(),
            )
            failure = "engine_error"
        if failure is not None:
            self._abort_request_capture(request, reason=failure)
            chunk["engine_data"] = {
                key: value
                for key, value in engine_data.items()
                if key not in _ENGINE_DATA_TOKEN_FIELDS
            }
            return chunk

        message: dict[str, Any] = {
            "generation_token_ids": engine_data["completion_token_ids"]
        }
        if "completion_logprobs" in engine_data:
            message["generation_log_probs"] = engine_data["completion_logprobs"]
        # The shape the native server stages from (choices[0].message token
        # fields); the prompt is the one recorded at begin_call.
        content = {
            "choices": [
                {"index": 0, "finish_reason": finish_reason, "message": message}
            ]
        }
        content = await self._run_off_loop(
            self._finish_request_capture, request, content
        )
        chunk["engine_data"] = {
            "finished": engine_data.get("finished", True),
            "ng_commit_coords": content["ng_commit_coords"],
        }
        return chunk
