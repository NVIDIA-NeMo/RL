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

"""Worker-local token capture lifecycle, independent of the vLLM engine."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Protocol

import torch

from nemo_rl.data_plane.interfaces import DataPlaneRuntimeConfig

if TYPE_CHECKING:
    from nemo_gym.token_id_capture.staging.capture import (
        ActiveCall,
        RolloutTokenCapture,
    )
    from nemo_gym.token_id_capture.staging.records import CaptureAdmission

LOGGER = logging.getLogger(__name__)


class _PrefixSource(Protocol):
    def fetch_prefix_token_ids(self, staging_keys: list[str]) -> list[int]: ...


class TokenCaptureHost:
    """Own capture state and staging prefixes for one serving worker.

    Gym imports remain deferred so constructing a disabled host does not require
    the optional Gym dependency. Engine interception belongs to the worker.
    """

    def __init__(self) -> None:
        self.token_capture: RolloutTokenCapture | None = None
        self._rollout_weight_version = 0
        self._capture_calls: dict[int, tuple[ActiveCall, list[int]]] = {}
        self._staging_source: _PrefixSource | None = None
        self._prefix_cache: dict[str, list[int]] = {}
        self._prefix_cache_lock = threading.Lock()

    def install_token_capture(self, capture: RolloutTokenCapture) -> None:
        """Gym's ``install_capture`` seam (the ``CaptureHost`` contract)."""
        self.token_capture = capture

    def setup(self, dp_cfg: DataPlaneRuntimeConfig, staging_partition: str) -> bool:
        """Install Gym capture with a worker-local staging client and adapter."""
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter
        from nemo_gym.token_id_capture.staging import install_capture

        from nemo_rl.data_plane import build_data_plane_client
        from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource

        dp_client = build_data_plane_client(dp_cfg, bootstrap=False)
        sink = TQTokenSink(dp_client, staging_partition=staging_partition)
        self._staging_source = TQTokenSource(
            dp_client, staging_partition=staging_partition
        )
        self._prefix_cache.clear()
        install_capture(
            self,
            sink=sink,
            weight_version_fn=lambda: self._rollout_weight_version,
            adapter=VLLMCaptureAdapter(),
        )
        return True

    def set_weight_version(self, version: int) -> None:
        """Rotate the weight version stamped on subsequent captured calls."""
        self._rollout_weight_version = int(version)

    def admission(self, request: Any) -> CaptureAdmission | None:
        """Parse the ledger's ``ng_capture`` context into a ``CaptureAdmission``.

        Returns None unless capture is installed and the request carries the
        context. The dict itself is never mutated: the admission is the typed,
        read-only contract that the prefix resolution and ``begin_call`` share.
        """
        context = getattr(request, "ng_capture", None)
        if self.token_capture is None or not context:
            return None
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.staging.records import CaptureAdmission

        return CaptureAdmission.model_validate(context)

    def begin_request(
        self,
        request: Any,
        prompt_token_ids: list[int],
        *,
        admission: CaptureAdmission | None = None,
        prefix_token_ids: list[int] | None = None,
    ) -> None:
        """Admit one ledger-forwarded call into the capture layer.

        Called from preprocess_chat once the exact engine prompt is known
        (post-splice in token-in mode, full render in text mode). No-op
        unless capture is installed and the request carries the ledger's
        ``ng_capture`` context.

        ``prefix_token_ids`` is the prefix resolved by
        :meth:`resolve_prefix`; Gym's ``begin_call`` checks it
        against the admission (length == ``prev_len``, equal to an inline
        prefix) and requires it for a ``staging_chain`` admission.
        """
        capture = self.token_capture
        if capture is None:
            return
        if admission is None:
            admission = self.admission(request)
            if admission is None:
                return
        call = capture.begin_call(
            admission,
            prefix_token_ids=prefix_token_ids,
            stream=bool(getattr(request, "stream", False)),
        )
        self._capture_calls[id(request)] = (call, list(prompt_token_ids))

    def fetch_chain_prefix(self, staging_chain: list[str]) -> list[int]:
        """Assemble prefix token ids from staging_chain, with a worker-local LRU cache."""
        cache = self._prefix_cache
        with self._prefix_cache_lock:
            cached_ids: list[int] = []
            miss_start = 0
            for i, key in enumerate(staging_chain):
                if key in cache:
                    cached_ids = cache[key]
                    miss_start = i + 1
            miss_keys = staging_chain[miss_start:]
        if not miss_keys:
            return list(cached_ids)
        if self._staging_source is None:
            raise RuntimeError(
                "_staging_source not initialized; call setup_token_capture() first"
            )
        # TQ read stays outside the lock so concurrent fetches overlap.
        fetched = self._staging_source.fetch_prefix_token_ids(miss_keys)
        result = cached_ids + fetched
        last_key = staging_chain[-1]
        with self._prefix_cache_lock:
            cache[last_key] = result
            if len(cache) > 256:
                del cache[next(iter(cache))]
        return result

    def resolve_prefix(self, admission: CaptureAdmission) -> list[int]:
        """Resolve a ``CaptureAdmission`` to the flat prefix the engine prompt starts with.

        A ``staging_chain`` is fetched through the cached TransferQueue read;
        an inline ``required_prefix_token_ids`` is used as is; a text root has
        no prefix. Length checks are Gym's: ``begin_call`` rejects a prefix
        that does not match ``prev_len``.
        """
        if admission.mode == "text":
            return []
        if admission.staging_chain:
            return self.fetch_chain_prefix(list(admission.staging_chain))
        return list(admission.required_prefix_token_ids)

    def enter_prefix(self, request: Any, prefix_token_ids: list[int]) -> None:
        """Attach the resolved prefix to the request through the capture adapter.

        ``VLLMCaptureAdapter.enter_prefix`` writes the engine-native field
        (``required_prefix_token_ids``) into a payload; the same fields are
        applied to the pydantic request so the existing prefix-splice branch
        of preprocess_chat handles staged and inline prefixes alike.
        """
        capture = self.token_capture
        assert capture is not None
        adapter = capture.adapter
        assert adapter is not None
        for field_name, value in adapter.enter_prefix({}, prefix_token_ids).items():
            setattr(request, field_name, value)

    @staticmethod
    def _delta_align_routed_experts(
        payload: dict[str, Any], *, prev_len: int, prompt_len: int, generated_len: int
    ) -> None:
        """Normalize optional vLLM routes to the exact staged token delta."""
        choices = payload.get("choices") or []
        if len(choices) != 1 or not isinstance(choices[0], dict):
            return
        choice = dict(choices[0])
        message = dict(choice.get("message") or {})
        routed = message.get("routed_experts")
        if routed is None:
            return
        try:
            from nemo_rl.utils.routed_experts_codec import (
                decode_routed_experts,
                encode_routed_experts,
            )

            if isinstance(routed, str):
                dtype_name = routed.split(":", 3)[1]
                dtype = {
                    "int8": torch.int8,
                    "int16": torch.int16,
                    "int32": torch.int32,
                }.get(dtype_name)
                if dtype is None:
                    raise ValueError(f"unsupported routed_experts dtype {dtype_name!r}")
            else:
                dtype = torch.int16
            experts = decode_routed_experts(routed, dtype)
            expected_full_len = prompt_len + generated_len
            if experts.dim() != 3 or experts.shape[0] != expected_full_len:
                raise ValueError(
                    f"route length {experts.shape[0]} does not match engine sequence "
                    f"length {expected_full_len}"
                )
            message["routed_experts"] = encode_routed_experts(experts[prev_len:])
        except (IndexError, TypeError, ValueError) as error:
            LOGGER.warning(
                "dropping invalid routed_experts from staged capture: %s", error
            )
            message.pop("routed_experts", None)
        choice["message"] = message
        payload["choices"] = [choice]

    def finish_request(self, request: Any, content: dict) -> dict:
        """Stage the finished call and ride its coords on the response.

        Fail-closed: the sink write happens inside complete_call —
        the coords exist only after the bytes are durable, and any capture
        failure degrades to capture_failed coords without breaking the
        completion. Token ids and logprobs are stripped: the staged delta is
        the only token store on this path, so the worker->gate hop carries
        text + delta ids + coords only.
        """
        state = self._capture_calls.pop(id(request), None)
        if state is None:
            return content
        call, prompt_token_ids = state
        payload = dict(content)
        # vLLM's OpenAI response carries no prompt ids; the adapter reads the
        # preprocess-time engine prompt off the payload (see
        # nemo_gym.token_id_capture.adapters.vllm.extract_prompt_ids).
        payload["prompt_token_ids"] = prompt_token_ids
        capture = self.token_capture
        assert capture is not None
        adapter = capture.adapter
        if adapter is not None:
            try:
                generated_token_ids, _ = adapter.extract_generation(payload)
            except Exception:  # capture core will report the authoritative failure
                generated_token_ids = []
            self._delta_align_routed_experts(
                payload,
                prev_len=call.admission.prev_len,
                prompt_len=len(prompt_token_ids),
                generated_len=len(generated_token_ids),
            )
        coords = capture.complete_call_from_response(call, payload)
        for choice in content.get("choices") or []:
            choice.pop("logprobs", None)
            # The delta-aligned routes were staged to TQ above; the served
            # full-length copy is dead weight the gate strips on arrival.
            message = choice.get("message")
            if isinstance(message, dict):
                message.pop("routed_experts", None)
        content["ng_commit_coords"] = coords.model_dump()
        return content

    def abort_request(self, request: Any, *, reason: str) -> None:
        """Drop the in-flight capture state for a request that errored."""
        state = self._capture_calls.pop(id(request), None)
        if state is not None and self.token_capture is not None:
            self.token_capture.fail_call(state[0], reason=reason)
