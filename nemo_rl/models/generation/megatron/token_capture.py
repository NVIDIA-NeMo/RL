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
"""Megatron Inference (MInf) hooks for NeMo-Gym token capture.

The Megatron generation worker installs these two adapters on the dynamic
inference engine of the model-parallel coordinator:

- ``TQMegatronPromptPreparer`` resolves a Gym-authorized ``staging_chain``
  prefix from TransferQueue and splices it into the rendered prompt before
  the engine admits the request.
- ``TQMegatronTokenStager`` canonicalizes the finished completion through
  Gym's capture core and writes the same TQ row the vLLM worker writes.

Both reach TransferQueue only through the backend-neutral ``TQTokenSink`` /
``TQTokenSource`` in ``nemo_rl.data_plane.tq_token_sink``; this module is
the Megatron analog of the capture glue in ``vllm_worker_async.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from nemo_rl.data_plane.tq_token_sink import (
    COMPACT_PREV_LEN_KEY,
    MEDIA_PREV_COUNT_KEY,
    MINF_CAPTURE_PARAMS_FIELD,
    ChainPrefixCache,
    TQTokenSink,
    TQTokenSource,
    resolve_admission_prefix_chains,
    slice_media_tensors,
)
from nemo_rl.models.generation.openai_server_utils import replace_prefix_tokens

if TYPE_CHECKING:
    from megatron.core.inference.inference_request import (
        RequestPayloadStageResult,
        RequestPromptPreparationResult,
    )


class TQMegatronPromptPreparer:
    """Resolve a Gym-authorized staged prefix before MInf admits a request.

    Same shape as the vLLM worker's ``_resolve_admission_prefix``:
    ``prepare_prompt`` resolves the admission through
    ``resolve_admission_prefix_chains`` over a worker-local ``ChainPrefixCache``,
    then splices the result with the shared ``replace_prefix_tokens`` using the
    rendered prior-turn tokens and EOS id the Megatron endpoint carried in
    ``offload_params``.
    """

    def __init__(self, source: TQTokenSource) -> None:
        # Same cached chain resolution as the vLLM worker (see ChainPrefixCache).
        self._chain_prefix = ChainPrefixCache(source)

    def prepare_prompt(
        self,
        prompt: str | list[int] | torch.Tensor,
        *,
        offload_params: dict[str, Any] | None = None,
    ) -> RequestPromptPreparationResult:
        """Fetch a chained prefix, splice it into the prompt, and update admission."""
        # Deferred because the prompt preparer is optional and requires the
        # Megatron-LM hooks from NVIDIA/Megatron-LM#7015. The two field names
        # are the request-metadata keys the Megatron chat endpoint writes when
        # it defers the prefix splice to this preparer.
        from megatron.core.inference.inference_request import (
            PREFIX_EOS_TOKEN_ID_FIELD,
            PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
            RequestPromptPreparationResult,
        )

        if offload_params is None:
            return RequestPromptPreparationResult(prompt=prompt)
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture import NG_CAPTURE_FIELD
        from nemo_gym.token_id_capture.staging.records import CaptureAdmission

        capture_payload = offload_params.get(NG_CAPTURE_FIELD)
        if capture_payload is None:
            return RequestPromptPreparationResult(
                prompt=prompt, offload_params=offload_params
            )

        admission = CaptureAdmission.model_validate(capture_payload)
        if admission.mode == "text":
            return RequestPromptPreparationResult(
                prompt=prompt, offload_params=offload_params
            )
        if not isinstance(prompt, list):
            raise TypeError("MInf token-in capture requires a token-id list prompt")

        chains = resolve_admission_prefix_chains(admission, self._chain_prefix)
        prefix_token_ids = chains.expanded
        if len(prefix_token_ids) != admission.prev_len:
            raise ValueError(
                "MInf capture prefix length mismatch: "
                f"expected {admission.prev_len}, got {len(prefix_token_ids)}"
            )

        updated_offload_params = dict(offload_params)
        # Gym verifies the engine's *expanded* prompt against this prefix.
        updated_admission = admission.model_copy(
            update={"required_prefix_token_ids": prefix_token_ids}
        )
        updated_offload_params[NG_CAPTURE_FIELD] = updated_admission.model_dump(
            mode="json"
        )
        # The stager needs the compact length of the spliced chain to cut this
        # call's compact delta (Gym's MegatronCaptureAdapter reads it off the
        # payload the stager assembles).
        updated_offload_params[MINF_CAPTURE_PARAMS_FIELD] = {
            **(updated_offload_params.get(MINF_CAPTURE_PARAMS_FIELD) or {}),
            COMPACT_PREV_LEN_KEY: len(chains.compact),
            MEDIA_PREV_COUNT_KEY: chains.media_count,
        }

        template_prefix_token_ids = updated_offload_params.get(
            PREFIX_TEMPLATE_TOKEN_IDS_FIELD
        )
        eos_token_id = updated_offload_params.get(PREFIX_EOS_TOKEN_ID_FIELD)
        if template_prefix_token_ids is not None or eos_token_id is not None:
            if not isinstance(template_prefix_token_ids, list) or any(
                type(token_id) is not int for token_id in template_prefix_token_ids
            ):
                raise ValueError(
                    "MInf capture request carries no valid template prefix tokens"
                )
            if type(eos_token_id) is not int:
                raise ValueError("MInf capture request carries no valid EOS token id")
            # Same splice as the vLLM worker (vllm_worker_async.py), but in the
            # *compact* token space: the chat endpoint renders one media token
            # per image and the engine expands every media token it is handed
            # (Megatron-LM ``_build_vlm_request``), so splicing the expanded
            # chain here would expand the previous turn twice. The engine's
            # expanded prompt is then checked against ``chains.expanded`` by
            # Gym's capture core when the call is staged.
            prompt = replace_prefix_tokens(
                tokenizer=None,
                model_prefix_token_ids=chains.compact,
                template_prefix_token_ids=template_prefix_token_ids,
                template_token_ids=prompt,
                eos_token_id=eos_token_id,
            )
        elif admission.staging_chain:
            raise ValueError(
                "MInf staged-prefix request carries no prompt splice metadata"
            )

        if prompt[: len(chains.compact)] != chains.compact:
            raise ValueError("MInf failed to apply the authorized token prefix")
        return RequestPromptPreparationResult(
            prompt=prompt, offload_params=updated_offload_params
        )


@dataclass(frozen=True)
class _MegatronCapturePayload:
    """The MInf offloaded payload plus the worker-side context Gym's adapter reads."""

    prompt_token_ids: Any
    generated_token_ids: Any
    generated_log_probs: Any
    compact_prompt_token_ids: Any
    compact_prev_len: int
    # The engine's media tensors minus what the parent chain already staged.
    media_tensors: dict[str, Any] | None

    @classmethod
    def from_offloaded(
        cls, payload: Any, minf_params: Any
    ) -> "_MegatronCapturePayload":
        if minf_params is not None and not isinstance(minf_params, dict):
            raise TypeError(
                f"MInf capture params must be a dict, got {type(minf_params).__name__}"
            )

        def _count(key: str) -> int:
            value = minf_params.get(key) if minf_params is not None else None
            if value is None:
                return 0
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"MInf capture request carries an invalid {key}: {value!r}"
                )
            return value

        media_tensors = getattr(payload, "media_tensors", None)
        if media_tensors is not None and not isinstance(media_tensors, Mapping):
            raise TypeError(
                "MInf payload media_tensors must be a mapping, got "
                f"{type(media_tensors).__name__}"
            )
        media: dict[str, Any] | None = (
            None if media_tensors is None else dict(media_tensors)
        )
        media = slice_media_tensors(media, _count(MEDIA_PREV_COUNT_KEY))
        return cls(
            prompt_token_ids=getattr(payload, "prompt_token_ids", None),
            generated_token_ids=getattr(payload, "generated_token_ids", None),
            generated_log_probs=getattr(payload, "generated_log_probs", None),
            compact_prompt_token_ids=getattr(payload, "compact_prompt_token_ids", None),
            compact_prev_len=_count(COMPACT_PREV_LEN_KEY),
            media_tensors=media,
        )


class TQMegatronTokenStager:
    """Canonicalize one admitted MInf completion through Gym's capture core.

    MInf owns the exact prompt/output material and its per-request policy epoch.
    Gym owns the lineage admission carried opaquely as ``ng_capture``. This
    adapter joins them before the response leaves MInf, writes the same
    canonical TQ row as vLLM, and returns lightweight commit coordinates.
    """

    def __init__(self, sink: TQTokenSink) -> None:
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.adapters.megatron import (
            MegatronCaptureAdapter,
        )
        from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture

        self._capture = RolloutTokenCapture(
            sink=sink,
            # MInf passes the authoritative version explicitly for every call.
            weight_version_fn=lambda: 0,
            adapter=MegatronCaptureAdapter(),
        )
        # Requests that straddled a refit (more than one policy_epoch boundary).
        # Metered here because they are stamped, not masked; see _weight_version.
        self._epoch_span_count = 0

    @property
    def epoch_span_count(self) -> int:
        """Number of staged calls whose generation spanned more than one policy epoch."""
        return self._epoch_span_count

    def _weight_version(self, finished_metadata: Any) -> int:
        """Stamp the policy epoch the request was admitted under.

        The engine records ``policy_epoch`` as ``(token_index, epoch)`` boundaries:
        one at admission, plus one appended on every ``set_generation_epoch``
        while the request is active, so a request that straddles a refit carries
        several. vLLM stamps the version in effect at ``begin_call`` and never
        re-checks, so the admission epoch (first boundary) is the matching choice
        here. Spans are counted and logged rather than masked;
        ``_abort_stale_inflight`` is skipped on the Gym path (#2625), so they are
        routine under async rollouts.
        """
        policy_epoch = getattr(finished_metadata, "policy_epoch", None)
        if not isinstance(policy_epoch, list) or not policy_epoch:
            raise ValueError("MInf captured request carries no policy_epoch boundaries")
        try:
            versions = {int(boundary[1]) for boundary in policy_epoch}
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(
                "MInf captured request carries invalid policy_epoch metadata"
            ) from error
        # Admission epoch (first boundary); later boundaries only mark refits.
        version = int(policy_epoch[0][1])
        if version < 0:
            raise ValueError(
                f"MInf captured request has negative policy epoch {version}"
            )
        if len(versions) > 1:
            self._epoch_span_count += 1
            logging.getLogger(__name__).warning(
                "MInf captured request spans policy epochs %s; stamping admission "
                "epoch %d (span count %d)",
                sorted(versions),
                version,
                self._epoch_span_count,
            )
        return version

    def stage(
        self,
        uid: str,
        payload: Any,
        *,
        finished_metadata: Any,
        offload_params: dict[str, Any] | None = None,
    ) -> RequestPayloadStageResult | None:
        """Stage an admitted request, or decline ordinary non-capture traffic."""
        if not isinstance(uid, str) or not uid:
            raise ValueError("MInf request UID must be a non-empty string")
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture import NG_CAPTURE_FIELD

        capture_payload = (offload_params or {}).get(NG_CAPTURE_FIELD)
        if capture_payload is None:
            return None
        try:
            return self._stage_admitted(
                payload,
                capture_payload=capture_payload,
                finished_metadata=finished_metadata,
                minf_params=(offload_params or {}).get(MINF_CAPTURE_PARAMS_FIELD),
            )
        except Exception:  # noqa: BLE001 — capture failure must not fail generation
            logging.getLogger(__name__).exception(
                "MInf canonical token capture failed for request %s", uid
            )
            return None

    def _stage_admitted(
        self,
        payload: Any,
        *,
        capture_payload: Any,
        finished_metadata: Any,
        minf_params: Any = None,
    ) -> RequestPayloadStageResult:
        """Validate and stage traffic that carries a Gym capture admission."""
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.staging.records import CaptureAdmission

        admission = CaptureAdmission.model_validate(capture_payload)
        call = self._capture.begin_call(
            admission,
            weight_version=self._weight_version(finished_metadata),
        )
        # Deferred: Megatron-LM's inference hooks are only present on the
        # Megatron generation backend (see prepare_prompt); nemo_gym is an
        # optional extra absent in non-gym runs.
        from megatron.core.inference.inference_request import (
            RequestPayloadStageResult,
        )
        from nemo_gym.token_id_capture import NG_COMMIT_COORDS_FIELD

        # Gym's MegatronCaptureAdapter reads prompt/generated ids and log
        # probs off the offloaded payload, plus the multimodal material
        # (compact prompt and the compact length of the spliced chain the
        # preparer recorded). A malformed payload poisons
        # the call with ``capture_failed`` coordinates (surfacing in Gym as
        # ``worker_capture_failed``, matching vLLM) instead of raising here,
        # which would leave Gym with no coordinates at all. The payload view
        # is derived before Gym's extraction (media delta slicing and the
        # preparer's counts), so its failures are routed through the same
        # poison path explicitly.
        try:
            capture_payload_view = _MegatronCapturePayload.from_offloaded(
                payload, minf_params
            )
        except (TypeError, ValueError, RuntimeError) as error:
            coords = self._capture.fail_call(
                call, reason=f"{type(error).__name__}: {error}"
            )
            return RequestPayloadStageResult(
                response_metadata={
                    NG_COMMIT_COORDS_FIELD: coords.model_dump(mode="json")
                }
            )
        # Gym's record cannot carry tensors; they ride beside it as opaque
        # attachments and land in the same put as the token columns
        # (TQTokenSink.stage). None means a text call.
        coords = self._capture.complete_call_from_response(
            call,
            capture_payload_view,
            attachments=capture_payload_view.media_tensors or None,
        )
        return RequestPayloadStageResult(
            response_metadata={
                NG_COMMIT_COORDS_FIELD: coords.model_dump(mode="json"),
            }
        )
