# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Token-capture hooks for Megatron's forked OpenAI HTTP frontends."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)

_ACTIVE_CALL: contextvars.ContextVar["_ActiveCapture | None"] = contextvars.ContextVar(
    "megatron_active_token_capture", default=None
)
_HOOKS_INSTALLED = False
_DP_CONFIG: dict[str, Any] | None = None
_STAGING_PARTITION: str | None = None
_WEIGHT_VERSION: Any = None
_PROCESS_STATE: tuple[int, Any, Any] | None = None


class MegatronCaptureAdapter:
    """Extract exact token material from Megatron chat-completion responses."""

    @staticmethod
    def _choice(response_payload: dict[str, Any]) -> dict[str, Any]:
        choices = response_payload.get("choices") or []
        if len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError(
                "Megatron token capture requires exactly one object choice"
            )
        return choices[0]

    @classmethod
    def _message(cls, response_payload: dict[str, Any]) -> dict[str, Any]:
        message = cls._choice(response_payload).get("message") or {}
        if not isinstance(message, dict):
            raise ValueError("Megatron response choice.message must be an object")
        return message

    def enter_prefix(
        self, request_payload: dict[str, Any], prefix_ids: list[int]
    ) -> dict[str, Any]:
        """Inject a captured prefix through Megatron's existing stitch seam."""
        messages = request_payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Megatron token-in capture requires messages")
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                # Megatron retokenizes the visible history to locate the old
                # prefix, then replaces it with these exact model-input IDs.
                message["prompt_token_ids"] = list(prefix_ids)
                message["compact_prompt_token_ids"] = list(prefix_ids)
                message["generation_token_ids"] = []
                return request_payload
        raise ValueError(
            "Megatron token-in capture requires a prior assistant message"
        )

    def extract_prompt_ids(self, response_payload: dict[str, Any]) -> list[int]:
        prompt_ids = self._message(response_payload).get("prompt_token_ids")
        if prompt_ids is None:
            raise ValueError("Megatron response carries no prompt_token_ids")
        return [int(token_id) for token_id in prompt_ids]

    def extract_generation(
        self, response_payload: dict[str, Any]
    ) -> tuple[list[int], list[float]]:
        message = self._message(response_payload)
        token_ids = [int(token_id) for token_id in message["generation_token_ids"]]
        logprobs = [float(value) for value in message["generation_log_probs"]]
        if len(token_ids) != len(logprobs):
            raise ValueError(
                "Megatron generated token and log-probability lengths differ"
            )
        return token_ids, logprobs

    def extract_extras(
        self, response_payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        routed_experts = self._choice(response_payload).get("moe_topk_indices")
        return (
            {"routed_experts": routed_experts}
            if routed_experts is not None
            else None
        )


@dataclass
class _ActiveCapture:
    capture: Any
    call: Any


def _delta_align_routed_experts(
    payload: dict[str, Any], *, prev_len: int
) -> None:
    """Normalize MCore's next-token routes to the staged token delta."""
    choices = payload.get("choices") or []
    if len(choices) != 1 or not isinstance(choices[0], dict):
        return
    choice = choices[0]
    routed = choice.get("moe_topk_indices")
    if routed is None:
        return
    try:
        message = choice["message"]
        full_len = len(message["prompt_token_ids"]) + len(
            message["generation_token_ids"]
        )
        if not isinstance(routed, list):
            raise ValueError("moe_topk_indices must be a list")
        if len(routed) == full_len - 1:
            if not routed:
                raise ValueError("cannot infer routed-expert shape from an empty list")
            sentinel = [
                [-1 for _ in layer]
                for layer in routed[-1]
            ]
            routed = [*routed, sentinel]
        if len(routed) != full_len:
            raise ValueError(
                f"route length {len(routed)} does not match sequence length {full_len}"
            )
        choice["moe_topk_indices"] = routed[prev_len:]
    except (KeyError, TypeError, ValueError) as error:
        LOGGER.warning("dropping invalid Megatron routed experts: %s", error)
        choice.pop("moe_topk_indices", None)


def _weight_version() -> int:
    if _WEIGHT_VERSION is None:
        raise RuntimeError("Megatron token-capture weight version is not initialized")
    with _WEIGHT_VERSION.get_lock():
        return int(_WEIGHT_VERSION.value)


def _process_state() -> tuple[Any, Any]:
    """Lazily construct process-local TQ clients after the HTTP fork."""
    global _PROCESS_STATE
    pid = os.getpid()
    if _PROCESS_STATE is not None and _PROCESS_STATE[0] == pid:
        return _PROCESS_STATE[1], _PROCESS_STATE[2]
    if _DP_CONFIG is None or _STAGING_PARTITION is None:
        raise RuntimeError("Megatron token capture is not configured")

    from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture

    from nemo_rl.data_plane import build_data_plane_client
    from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource

    dp_client = build_data_plane_client(_DP_CONFIG, bootstrap=False)
    source = TQTokenSource(dp_client, staging_partition=_STAGING_PARTITION)
    capture = RolloutTokenCapture(
        sink=TQTokenSink(dp_client, staging_partition=_STAGING_PARTITION),
        weight_version_fn=_weight_version,
        adapter=MegatronCaptureAdapter(),
    )
    _PROCESS_STATE = (pid, capture, source)
    return capture, source


async def _before_chat_completion() -> Any:
    from quart import jsonify, request

    payload = await request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload.get("ng_capture"):
        return None
    if payload.get("stream", False):
        return jsonify({"error": "token capture does not support streaming"}), 400

    try:
        from nemo_gym.token_id_capture.staging.records import CaptureAdmission

        capture, source = _process_state()
        admission = CaptureAdmission.model_validate(payload["ng_capture"])
        prefix_ids: list[int] | None = None
        if admission.mode == "token_in":
            prefix_ids = (
                list(admission.required_prefix_token_ids)
                if admission.required_prefix_token_ids
                else await asyncio.to_thread(
                    source.fetch_prefix_token_ids, list(admission.staging_chain)
                )
            )
            capture.adapter.enter_prefix(payload, prefix_ids)
        call = capture.begin_call(admission, prefix_token_ids=prefix_ids)
        _ACTIVE_CALL.set(_ActiveCapture(capture=capture, call=call))
    except Exception:
        # Missing coordinates make the Gym ledger mark only this call failed;
        # generation itself remains available.
        LOGGER.exception("Megatron request could not enter token capture")
    return None


async def _after_chat_completion(response: Any) -> Any:
    active = _ACTIVE_CALL.get()
    _ACTIVE_CALL.set(None)
    if active is None:
        return response

    try:
        if response.status_code >= 400:
            coords = active.capture.fail_call(
                active.call, reason=f"HTTP {response.status_code}"
            )
        else:
            payload = await response.get_json()
            if not isinstance(payload, dict):
                raise ValueError("Megatron response is not a JSON object")
            _delta_align_routed_experts(
                payload, prev_len=active.call.admission.prev_len
            )
            coords = await asyncio.to_thread(
                active.capture.complete_call_from_response,
                active.call,
                payload,
            )
            # Gym strips the standard token fields after recording commit
            # coordinates; this Megatron-only stitch helper is not part of
            # that inventory and must not ride onward to the agent.
            for choice in payload.get("choices") or []:
                message = choice.get("message") if isinstance(choice, dict) else None
                if isinstance(message, dict):
                    message.pop("compact_prompt_token_ids", None)
            payload["ng_commit_coords"] = coords.model_dump(mode="json")
            response.set_data(json.dumps(payload))
            return response
    except Exception:
        LOGGER.exception("Megatron response token capture failed")
        try:
            coords = active.capture.fail_call(
                active.call, reason="response capture failed"
            )
        except Exception:
            return response

    try:
        payload = await response.get_json()
        if isinstance(payload, dict):
            payload["ng_commit_coords"] = coords.model_dump(mode="json")
            response.set_data(json.dumps(payload))
    except Exception:
        LOGGER.exception("Could not attach Megatron capture coordinates")
    return response


def install_megatron_token_capture_hooks(
    dp_config: dict[str, Any], staging_partition: str, weight_version: Any
) -> None:
    """Install hooks on the blueprint before MCore forks HTTP replicas."""
    global _DP_CONFIG, _HOOKS_INSTALLED, _STAGING_PARTITION, _WEIGHT_VERSION

    _DP_CONFIG = dict(dp_config)
    _STAGING_PARTITION = str(staging_partition)
    _WEIGHT_VERSION = weight_version
    if _HOOKS_INSTALLED:
        return

    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
        ChatCompletions,
    )

    ChatCompletions.before_request(_before_chat_completion)
    ChatCompletions.after_request(_after_chat_completion)
    _HOOKS_INSTALLED = True
