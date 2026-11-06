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
"""Ledger-authoritative token capture hosted by the TRT-LLM PD disagg frontend.

Gym's policy-model server forwards every admitted model call with an
``ng_capture`` admission over the vLLM-worker capture wire. Under PD
disaggregation the replica's frontend is the one hop that sees both that
request and the finished response, so it hosts capture: it resolves the
admission's staged prefix, renders the exact engine prompt with it, and once
the decode leg returns, stages the call's token delta to TransferQueue and
answers with ``ng_commit_coords`` in place of the token arrays.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from nemo_rl.models.generation.trtllm.trtllm_http_server import (
    decode_prompt_token_ids_b64,
)

logger = logging.getLogger(__name__)

_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# Token arrays the capture stages to TransferQueue; they never travel to Gym.
_STAGED_MESSAGE_FIELDS = (
    "prompt_token_ids",
    "generation_token_ids",
    "generation_log_probs",
)

# ``payload -> prompt_token_ids_b64``, rendering with an explicit prefix.
TokenizeFn = Callable[..., Awaitable[str]]


def _ensure_nemo_gym_importable() -> None:
    """Make Gym's capture modules importable from the TRT-LLM venv.

    The ``trtllm`` and ``nemo_gym`` extras conflict, so this venv cannot
    install Gym, but ``nemo_gym.token_id_capture`` needs nothing the venv
    lacks. Append the in-tree checkout, as ``nemo_rl/__init__.py`` does for
    Megatron-LM.
    """
    try:
        import nemo_gym  # noqa: F401
    except ImportError:
        import sys

        gym_root = (
            Path(__file__).resolve().parents[4] / "3rdparty" / "Gym-workspace" / "Gym"
        )
        if not (gym_root / "nemo_gym").is_dir():
            raise RuntimeError(
                f"TRT-LLM token capture needs NeMo-Gym's source at {gym_root}"
            ) from None
        sys.path.append(str(gym_root))


@dataclass
class _CapturedCall:
    """An admitted call and the exact prompt ids the engine was handed."""

    call: Any
    prompt_token_ids: list[int]


class TrtllmCaptureHost:
    """One frontend's capture state; the ``CaptureHost`` seam ``install_capture`` wires."""

    def __init__(self) -> None:
        self.token_capture: Any = None
        self._chain_prefix: Any = None
        self._rollout_weight_version = 0

    @property
    def installed(self) -> bool:
        return self.token_capture is not None

    def install_token_capture(self, capture: Any) -> None:
        """Gym's ``install_capture`` seam (the ``CaptureHost`` contract)."""
        self.token_capture = capture

    def setup(self, dp_cfg: Any, staging_partition: str) -> None:
        """Build the frontend's TransferQueue sink/source and install Gym's capture core.

        Args:
            dp_cfg: Data-plane config for the in-process TransferQueue client.
            staging_partition: Partition that captured calls are staged in.
        """
        _ensure_nemo_gym_importable()
        # Deferred: Gym is importable only after the call above.
        from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter
        from nemo_gym.token_id_capture.staging import install_capture

        from nemo_rl.data_plane import build_data_plane_client
        from nemo_rl.data_plane.tq_token_sink import (
            ChainPrefixCache,
            TQTokenSink,
            TQTokenSource,
        )

        # Own no segment, as the vLLM and Megatron capture hosts do: a save
        # would otherwise wait on this generation-only process.
        dp_client = build_data_plane_client(dp_cfg, bootstrap=False, segment_size=0)
        self._chain_prefix = ChainPrefixCache(
            TQTokenSource(dp_client, staging_partition=staging_partition)
        )
        install_capture(
            self,
            sink=TQTokenSink(dp_client, staging_partition=staging_partition),
            weight_version_fn=lambda: self._rollout_weight_version,
            # The frontend's responses carry the vLLM-shaped rollout fields
            # (see trtllm_disagg_server.attach_rollout_fields).
            adapter=VLLMCaptureAdapter(),
        )

    def set_rollout_weight_version(self, version: int) -> None:
        """Stamp subsequently admitted calls with the trainer weight version."""
        if type(version) is not int or version < 0:
            raise ValueError(
                f"rollout weight version must be a non-negative int, got {version!r}"
            )
        self._rollout_weight_version = version

    async def begin(
        self, payload: dict[str, Any], context: dict[str, Any], tokenize_fn: TokenizeFn
    ) -> _CapturedCall:
        """Admit one call and pin its engine prompt on ``payload``.

        Resolves the admission's prefix (a staged chain is a blocking TQ read,
        so it runs off the event loop), renders the conversation with that
        exact prefix spliced in, and sets ``prompt_token_ids_b64`` so the
        engine prefills those ids verbatim.

        Raises:
            ValueError: The admission is malformed, the request streams, or
                the render does not start with the authorized prefix.
            CaptureError: Gym rejected the admission.
        """
        # Deferred: Gym is optional outside capture runs.
        from nemo_gym.token_id_capture.staging.records import CaptureAdmission

        from nemo_rl.data_plane.tq_token_sink import resolve_admission_prefix
        from nemo_rl.models.generation.capture_context import decide_capture_input

        if payload.get("stream"):
            raise ValueError("token capture requires a non-streaming request")
        admission = CaptureAdmission.model_validate(context)
        serving = admission
        if admission.mode == "candidate" or admission.request_replay is not None:
            # allow_empty_tool_content stays False: no renderer here has proven
            # null and empty tool content render alike, so such a cut roots a
            # new segment instead of splicing.
            decision = decide_capture_input(
                admission, messages=payload.get("messages") or []
            )
            admission, serving = decision.storage, decision.serving
        prefix: list[int] = []
        if serving is not None and serving.mode == "token_in":
            prefix = await asyncio.to_thread(
                resolve_admission_prefix, serving, self._chain_prefix
            )
        # The admission alone decides the prefix; an inline override would not
        # be the one Gym verifies.
        payload.pop("required_prefix_token_ids", None)
        encoded = await tokenize_fn(payload, required_prefix_ids=prefix)
        prompt_token_ids = decode_prompt_token_ids_b64(encoded)
        if prompt_token_ids[: len(prefix)] != prefix:
            raise ValueError(
                "the rendered prompt does not start with the authorized capture prefix"
            )
        call = self.token_capture.begin_call(
            admission,
            prefix_token_ids=prefix if admission.mode == "token_in" else None,
        )
        payload["prompt_token_ids_b64"] = encoded
        return _CapturedCall(call, prompt_token_ids)

    def complete(
        self, captured: _CapturedCall, response: dict[str, Any]
    ) -> dict[str, Any]:
        """Stage a finished call and swap its token arrays for ``ng_commit_coords``.

        Blocking (a TransferQueue write); call it off the event loop.
        """
        from nemo_gym.token_id_capture import NG_COMMIT_COORDS_FIELD

        capture = self.token_capture
        if response.get("prompt_token_ids") != captured.prompt_token_ids:
            coords = capture.fail_call(
                captured.call, reason="engine prompt differs from the captured prompt"
            )
        else:
            coords = capture.complete_call_from_response(captured.call, response)
        response.pop("prompt_token_ids", None)
        for choice in response.get("choices") or []:
            choice.pop("logprobs", None)
            choice.pop("token_ids", None)
            message = choice.get("message")
            if isinstance(message, dict):
                for field_name in _STAGED_MESSAGE_FIELDS:
                    message.pop(field_name, None)
        response[NG_COMMIT_COORDS_FIELD] = coords.model_dump(mode="json")
        return response

    def fail(self, captured: _CapturedCall, *, reason: str) -> None:
        """Poison an admitted call that produced no completion."""
        self.token_capture.fail_call(captured.call, reason=reason)


async def _read_body(receive: Any) -> Optional[bytes]:
    """The full request body, or None if the client disconnected first."""
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            return b"".join(chunks)


def _replay(body: bytes) -> Any:
    delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


def _with_content_length(
    headers: list[tuple[bytes, bytes]], length: int
) -> list[tuple[bytes, bytes]]:
    kept = [(name, value) for name, value in headers if name.lower() != b"content-length"]
    return [*kept, (b"content-length", str(length).encode())]


class _BufferedResponse:
    """Collects a downstream ASGI response so it can be rewritten before sending."""

    def __init__(self) -> None:
        self.status = 500
        self.headers: list[tuple[bytes, bytes]] = []
        self.chunks: list[bytes] = []

    async def send(self, message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            self.status = message["status"]
            self.headers = list(message.get("headers", []))
        elif message["type"] == "http.response.body":
            self.chunks.append(message.get("body", b""))


async def _send(
    send: Any, status: int, headers: list[tuple[bytes, bytes]], body: bytes
) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": _with_content_length(headers, len(body)),
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


class TokenCaptureMiddleware:
    """ASGI middleware hosting token capture around the frontend's chat route.

    Installed outermost, so the prompt ids it pins (``prompt_token_ids_b64``)
    make the frontend-tokenize middleware below skip its own render. Until
    :meth:`TrtllmCaptureHost.setup` runs -- including on a frontend Ray
    restarted, which comes back without capture -- requests pass through
    untouched, so an ``ng_capture`` request fails TRT-LLM's strict request
    validation instead of being served uncaptured.
    """

    def __init__(
        self, app: Any, *, host: TrtllmCaptureHost, tokenize_fn: TokenizeFn
    ) -> None:
        self.app = app
        self._host = host
        self._tokenize_fn = tokenize_fn

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("path") != _CHAT_COMPLETIONS_PATH
            or not self._host.installed
        ):
            await self.app(scope, receive, send)
            return
        # Deferred: Gym is importable once the host is installed.
        from nemo_gym.token_id_capture import NG_CAPTURE_FIELD
        from nemo_gym.token_id_capture.protocols import CAPTURE_REFUSAL_HEADER
        from nemo_gym.token_id_capture.staging import CaptureError

        body = await _read_body(receive)
        if body is None:
            return
        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
        context = (
            payload.pop(NG_CAPTURE_FIELD, None) if isinstance(payload, dict) else None
        )
        if context is None:
            await self.app(scope, _replay(body), send)
            return
        try:
            captured = await self._host.begin(payload, context, self._tokenize_fn)
        except (ValueError, CaptureError) as error:
            logger.warning("Rejected captured call before inference: %s", error)
            message = {"message": str(error), "type": "invalid_request_error"}
            await _send(
                send,
                400,
                [(b"content-type", b"application/json")],
                json.dumps({"error": message}).encode(),
            )
            return

        body = json.dumps(payload).encode()
        scope = {**scope, "headers": _with_content_length(scope["headers"], len(body))}
        response = _BufferedResponse()
        try:
            await self.app(scope, _replay(body), response.send)
        except BaseException:
            self._host.fail(captured, reason="engine_error")
            raise
        body = b"".join(response.chunks)
        headers = response.headers
        try:
            completed = json.loads(body) if response.status == 200 else None
        except ValueError:
            completed = None
        if completed is not None:
            completed = await asyncio.to_thread(self._host.complete, captured, completed)
            body = json.dumps(completed).encode()
        else:
            # The engines' context-length rejection reaches here as a 400 with
            # its text intact (OpenAIDisaggServerAdaptor._handle_exception).
            refused = response.status == 400 and b"context length" in body
            self._host.fail(
                captured, reason="context_length" if refused else "engine_error"
            )
            if refused:
                # Acknowledges the refusal so Gym records it rather than a
                # missing commit.
                model_call_id = captured.call.admission.model_call_id
                headers = [
                    *headers,
                    (CAPTURE_REFUSAL_HEADER.encode(), model_call_id.encode()),
                ]
        await _send(send, response.status, headers, body)
