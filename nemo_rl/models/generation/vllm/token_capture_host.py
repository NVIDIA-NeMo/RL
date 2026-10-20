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

"""Backend-neutral token-capture host for processes that own a vLLM AsyncLLM.

NeMo RL's async vLLM worker and the managed Dynamo engine handler both host
Gym's ``RolloutTokenCapture`` next to the engine. This mixin carries the state
and the call lifecycle they share: admission parsing, staged-prefix
resolution, ``begin_call``, staging the finished call to TransferQueue, and
failure. Prompt preparation and media stay with each host, because they depend
on how that host receives requests.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from nemo_rl.data.captured_media import CapturedMedia
from nemo_rl.data_plane.tq_token_sink import ChainPrefixCache, resolve_admission_prefix
from nemo_rl.experience.route_assembly import boundary_route_rows
from nemo_rl.utils.routed_experts_codec import (
    decode_routed_experts,
    encode_routed_experts,
)

if TYPE_CHECKING:
    from nemo_gym.token_id_capture.staging.capture import ActiveCall

    from nemo_rl.data_plane.interfaces import DataPlaneConfig

LOGGER = logging.getLogger(__name__)


def parse_capture_admission(context: Any) -> Any | None:
    """Parse a raw ``ng_capture`` context into a ``CaptureAdmission``.

    Returns None when no context was sent. The context is never mutated: the
    admission is the typed, read-only contract that the prefix resolution and
    ``begin_call`` share.
    """
    if not context:
        return None
    # Deferred: nemo_gym is an optional extra absent in non-gym runs.
    from nemo_gym.token_id_capture.staging.records import CaptureAdmission

    return CaptureAdmission.model_validate(context)


@dataclass
class CapturedRequest:
    """Request-local ownership of tokens and processed-media snapshots."""

    call: "ActiveCall"
    prompt_token_ids: list[int]
    media: CapturedMedia | None


class TokenCaptureHostMixin:
    """Gym ``CaptureHost`` state and call lifecycle shared by vLLM-owning hosts.

    Hosts call :meth:`_init_token_capture_host` from their constructor and
    :meth:`_install_token_capture_host` once ``token_capture.enabled`` setup
    reaches them. In-flight calls are keyed by ``id(request)``: the host must
    pass the same request object to begin, finish, and abort.
    """

    token_capture: Any
    _rollout_weight_version: int
    _capture_calls: dict[int, CapturedRequest]
    _staging_source: Any
    _chain_prefix: ChainPrefixCache

    def _init_token_capture_host(self) -> None:
        """Initialize dormant capture state; nothing is captured until install."""
        # Ledger-authoritative token capture (dormant until the
        # setup_token_capture fan-out runs). The weight
        # version is stamped per model call at begin_call time and rotated by
        # the set_rollout_weight_version fan-out from the SC's _sync_weights.
        self.token_capture = None
        self._rollout_weight_version = 0
        # In-flight captured calls keyed by id(request): the ActiveCall, the
        # exact engine prompt ids recorded at preprocess time, and the media
        # snapshot the engine ran on (see CapturedRequest).
        self._capture_calls = {}
        # TQTokenSource installed by setup_token_capture. The media path reads
        # parent-chain rows through it directly; prefix resolution goes through
        # the shared ChainPrefixCache below.
        self._staging_source = None
        # Resolved staging-chain prefixes, shared implementation with the Megatron
        # preparer. Installed by setup_token_capture; fetch runs on executor threads.
        self._chain_prefix = ChainPrefixCache()

    def _install_token_capture_host(
        self,
        dp_cfg: "DataPlaneConfig",
        staging_partition: str,
        *,
        capture_media: bool,
        media_pixel_dtype: torch.dtype | None,
    ) -> None:
        """Build the in-process data-plane client and install Gym's capture core.

        Makes the single ``install_capture`` call wiring Gym's engine-blind
        capture core and vLLM adapter into this host.
        """
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter
        from nemo_gym.token_id_capture.staging import install_capture

        # Resolved at call time, like the rest of this setup: the client and
        # sink are only built when token_capture.enabled reaches this host.
        from nemo_rl.data_plane import build_data_plane_client
        from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource

        # Own no segment: a save would otherwise wait on this host, whose event
        # loop is busy serving rollouts. Capture writes land in other segments.
        dp_client = build_data_plane_client(dp_cfg, bootstrap=False, segment_size=0)
        sink = TQTokenSink(
            dp_client,
            staging_partition=staging_partition,
            capture_media=capture_media,
            media_pixel_dtype=media_pixel_dtype,
        )
        source = TQTokenSource(
            dp_client, staging_partition=staging_partition, capture_media=capture_media
        )
        self._staging_source = source
        self._chain_prefix.install(source)
        install_capture(
            self,
            sink=sink,
            weight_version_fn=lambda: self._rollout_weight_version,
            adapter=VLLMCaptureAdapter(),
        )

    def install_token_capture(self, capture: Any) -> None:
        """Gym's ``install_capture`` seam (the ``CaptureHost`` contract)."""
        self.token_capture = capture

    async def set_rollout_weight_version(self, version: int) -> None:
        """Rotate the weight version stamped on subsequent captured calls."""
        self._rollout_weight_version = int(version)

    def _capture_admission(self, request: Any) -> Any | None:
        """Parse the ledger's ``ng_capture`` context off a request object.

        Returns None unless capture is installed and the request carries the
        context.
        """
        if self.token_capture is None:
            return None
        return parse_capture_admission(getattr(request, "ng_capture", None))

    def _begin_request_capture(
        self,
        request: Any,
        prompt_token_ids: list[int],
        *,
        admission: Any | None = None,
        prefix_token_ids: list[int] | None = None,
        media: CapturedMedia | None = None,
    ) -> None:
        """Admit one ledger-forwarded call into the capture layer.

        Called once the exact engine prompt is known (post-splice in token-in
        mode, full render in text mode). No-op unless capture is installed and
        the request carries the ledger's ``ng_capture`` context.

        ``prefix_token_ids`` is the prefix resolved by
        :meth:`_resolve_admission_prefix`. Gym's ``begin_call`` requires it for
        a ``staging_chain`` admission and checks its length against
        ``prev_len``; violations raise ``CaptureError`` before any state is kept.
        """
        capture = self.token_capture
        if capture is None:
            return
        if admission is None:
            admission = self._capture_admission(request)
            if admission is None:
                return
        call = capture.begin_call(
            admission,
            prefix_token_ids=prefix_token_ids,
            stream=bool(getattr(request, "stream", False)),
        )
        self._capture_calls[id(request)] = CapturedRequest(
            call, list(prompt_token_ids), media
        )

    def _fetch_chain_prefix(self, staging_chain: list[str]) -> list[int]:
        """Resolve a staging chain through the shared, cached TQ read."""
        return self._chain_prefix.fetch(staging_chain)

    def _resolve_admission_prefix(self, admission: Any) -> list[int]:
        """Resolve a ``CaptureAdmission`` to the flat prefix the engine prompt starts with."""
        return resolve_admission_prefix(admission, self._chain_prefix)

    @staticmethod
    def _delta_align_routed_experts(
        payload: dict[str, Any], *, prev_len: int, prompt_len: int, generated_len: int
    ) -> None:
        """Normalize optional vLLM routes to the staged token delta (plus boundary row)."""
        choices = payload.get("choices") or []
        if len(choices) != 1 or not isinstance(choices[0], dict):
            return
        choice = dict(choices[0])
        message = dict(choice.get("message") or {})
        routed = message.get("routed_experts")
        if routed is None:
            return
        try:
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
            start = prev_len - boundary_route_rows(prev_len)
            message["routed_experts"] = encode_routed_experts(experts[start:])
        except (IndexError, TypeError, ValueError) as error:
            LOGGER.warning(
                "dropping invalid routed_experts from staged capture: %s", error
            )
            message.pop("routed_experts", None)
        choice["message"] = message
        payload["choices"] = [choice]

    def _finish_request_capture(self, request: Any, content: dict) -> dict:
        """Stage the finished call and ride its coords on the response.

        Tokens and the call's new media tensors go to TQ in one write (the
        media ride as opaque attachments beside the record), so ``staged``
        coords vouch for both and any failure is ``capture_failed`` at call
        time. Token ids, logprobs, and routes are stripped after staging, so
        the worker->gate hop carries the completion and coords only.
        """
        state = self._capture_calls.pop(id(request), None)
        if state is None:
            return content
        call, prompt_token_ids = state.call, state.prompt_token_ids
        payload = dict(content)
        # vLLM's OpenAI response carries no prompt ids; the adapter reads the
        # preprocess-time engine prompt off the payload (see
        # nemo_gym.token_id_capture.adapters.vllm.extract_prompt_ids).
        payload["prompt_token_ids"] = prompt_token_ids
        if state.media is not None:
            # Optional Gym dependency: only captured media calls reach here.
            from nemo_gym.token_id_capture.adapters.vllm import MEDIA_SPANS_FIELD

            # Placeholder metadata (offsets, token hashes, sizes) rides the
            # digest-covered extras; the pixels themselves are attachments.
            payload[MEDIA_SPANS_FIELD] = [item.to_dict() for item in state.media.items]
        adapter = self.token_capture.adapter
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
        coords = self.token_capture.complete_call_from_response(
            call,
            payload,
            attachments=state.media.tensors if state.media is not None else None,
        )
        for choice in content.get("choices") or []:
            choice.pop("logprobs", None)
            # Token arrays and delta-aligned routes were staged to TQ above;
            # remove the serializer's message fields before the worker->gate hop.
            message = choice.get("message")
            if isinstance(message, dict):
                for field in (
                    "prompt_token_ids",
                    "generation_token_ids",
                    "generation_log_probs",
                    "routed_experts",
                ):
                    message.pop(field, None)
        content["ng_commit_coords"] = coords.model_dump()
        return content

    def _abort_request_capture(self, request: Any, *, reason: str) -> None:
        """Drop the in-flight capture state for a request that errored."""
        state = self._capture_calls.pop(id(request), None)
        if state is not None and self.token_capture is not None:
            self.token_capture.fail_call(state.call, reason=reason)
