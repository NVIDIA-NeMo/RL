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
from typing import TYPE_CHECKING, Any

import torch

from nemo_rl.data_plane.tq_token_sink import (
    ChainPrefixCache,
    TQTokenSink,
    TQTokenSource,
    resolve_admission_prefix,
)
from nemo_rl.experience.route_assembly import ROUTE_MISSING_SENTINEL
from nemo_rl.models.generation.openai_server_utils import replace_prefix_tokens
from nemo_rl.utils.routed_experts_codec import encode_routed_experts

if TYPE_CHECKING:
    from megatron.core.inference.inference_request import (
        RequestPayloadStageResult,
        RequestPromptPreparationResult,
    )


def _delta_align_minf_routing_indices(
    routing_indices: Any,
    *,
    total_tokens: int,
    prev_len: int,
    expected_route_dims: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Convert MInf ``[T - 1, L, K]`` routes to Gym's delta-token layout.

    ``expected_route_dims`` is the model-owned ``(L, K)``; when given, a payload
    whose layer or top-k axis differs is rejected here, at the first request,
    rather than a full rollout later in the reassembler.
    """
    if not 0 <= prev_len <= total_tokens:
        raise ValueError(
            f"MInf route prev_len must be in [0, {total_tokens}], got {prev_len}"
        )
    routes = torch.as_tensor(routing_indices)
    if routes.dim() != 3:
        raise ValueError(
            "MInf routing_indices must have shape [tokens, layers, topk], "
            f"got {tuple(routes.shape)}"
        )
    expected_routes = total_tokens - 1
    if routes.shape[0] != expected_routes:
        raise ValueError(
            "MInf routing_indices must contain one row for every non-final token: "
            f"got {routes.shape[0]}, expected {expected_routes}"
        )
    if routes.shape[1] <= 0 or routes.shape[2] <= 0:
        raise ValueError(
            "MInf routing_indices layer and top-k dimensions must be positive"
        )
    if expected_route_dims is not None and tuple(routes.shape[1:]) != tuple(
        expected_route_dims
    ):
        raise ValueError(
            "MInf routing_indices (layers, top_k) does not match the served "
            f"model: got {tuple(routes.shape[1:])}, expected "
            f"{tuple(expected_route_dims)}. MInf records routes per pipeline "
            "stage; check the generation model's parallel layout."
        )
    aligned = torch.full(
        (total_tokens, routes.shape[1], routes.shape[2]),
        ROUTE_MISSING_SENTINEL,
        dtype=routes.dtype,
        device=routes.device,
    )
    if expected_routes:
        aligned[:-1].copy_(routes)
    return aligned[prev_len:]


class TQMegatronPromptPreparer:
    """Resolve a Gym-authorized staged prefix before MInf admits a request.

    Mirrors the vLLM worker: ``prepare_prompt`` resolves the prefix through the
    shared ``resolve_admission_prefix`` / ``ChainPrefixCache`` pair, then splices
    it with the shared ``replace_prefix_tokens`` using the rendered prior-turn
    tokens and EOS id the Megatron endpoint carried in ``offload_params``.
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

        prefix_token_ids = resolve_admission_prefix(admission, self._chain_prefix)
        if len(prefix_token_ids) != admission.prev_len:
            raise ValueError(
                "MInf capture prefix length mismatch: "
                f"expected {admission.prev_len}, got {len(prefix_token_ids)}"
            )

        updated_offload_params = dict(offload_params)
        updated_admission = admission.model_copy(
            update={"required_prefix_token_ids": prefix_token_ids}
        )
        updated_offload_params[NG_CAPTURE_FIELD] = updated_admission.model_dump(
            mode="json"
        )

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
            # Same splice as the vLLM worker (vllm_worker_async.py); the
            # post-condition below verifies the result.
            prompt = replace_prefix_tokens(
                tokenizer=None,
                model_prefix_token_ids=prefix_token_ids,
                template_prefix_token_ids=template_prefix_token_ids,
                template_token_ids=prompt,
                eos_token_id=eos_token_id,
            )
        elif admission.staging_chain:
            raise ValueError(
                "MInf staged-prefix request carries no prompt splice metadata"
            )

        if prompt[: admission.prev_len] != prefix_token_ids:
            raise ValueError("MInf failed to apply the authorized token prefix")
        return RequestPromptPreparationResult(
            prompt=prompt, offload_params=updated_offload_params
        )


class TQMegatronTokenStager:
    """Canonicalize one admitted MInf completion through Gym's capture core.

    MInf owns the exact prompt/output material and its per-request policy epoch.
    Gym owns the lineage admission carried opaquely as ``ng_capture``. This
    adapter joins them before the response leaves MInf, writes the same
    canonical TQ row as vLLM, and returns lightweight commit coordinates.
    """

    def __init__(
        self,
        sink: TQTokenSink,
        *,
        require_routed_experts: bool = False,
        expected_route_dims: tuple[int, int] | None = None,
    ) -> None:
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.adapters.megatron import (
            MegatronCaptureAdapter,
        )
        from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture

        # Gym's adapter validates the offloaded payload's prompt/generated ids
        # and log probs; _stage_admitted reads them through it.
        self._adapter = MegatronCaptureAdapter()
        self._capture = RolloutTokenCapture(
            sink=sink,
            # MInf passes the authoritative version explicitly for every call.
            weight_version_fn=lambda: 0,
            adapter=self._adapter,
        )
        self._require_routed_experts = require_routed_experts
        self._expected_route_dims = expected_route_dims
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
        call = None
        try:
            # Deferred: nemo_gym is an optional extra absent in non-gym runs.
            from nemo_gym.token_id_capture.staging.records import CaptureAdmission

            admission = CaptureAdmission.model_validate(capture_payload)
            call = self._capture.begin_call(
                admission,
                weight_version=self._weight_version(finished_metadata),
            )
            return self._stage_admitted(payload, call=call)
        except Exception:  # noqa: BLE001 — capture failure must not fail generation
            logging.getLogger(__name__).exception(
                "MInf canonical token capture failed for request %s", uid
            )
            if call is not None and not call.completed:
                coords = self._capture.fail_call(
                    call, reason="megatron_payload_staging_failed"
                )
                return self._stage_result(coords)
            return None

    def _stage_admitted(
        self,
        payload: Any,
        *,
        call: Any,
    ) -> RequestPayloadStageResult:
        """Validate and stage traffic that carries a Gym capture admission."""
        # Gym's MegatronCaptureAdapter reads prompt/generated ids and log
        # probs off the offloaded payload and rejects malformed material. It
        # stages no extras: MInf routes are ``[T - 1, L, K]`` while the
        # staging contract is delta-token aligned and needs the admission's
        # ``prev_len``, so the route conversion lives here and the call is
        # completed through ``complete_call`` rather than
        # ``complete_call_from_response``. Errors raised here are poisoned by
        # ``stage`` with ``capture_failed`` coordinates (surfacing in Gym as
        # ``worker_capture_failed``, matching vLLM) instead of leaving Gym
        # with no coordinates at all.
        prompt_token_ids = self._adapter.extract_prompt_ids(payload)
        generated_token_ids, generated_logprobs = self._adapter.extract_generation(
            payload
        )
        routing_indices = getattr(payload, "routing_indices", None)
        if routing_indices is None and self._require_routed_experts:
            raise ValueError(
                "MInf offloaded payload carries no routing_indices while router "
                "replay is enabled"
            )
        extras = None
        if routing_indices is not None:
            routed_experts = _delta_align_minf_routing_indices(
                routing_indices,
                total_tokens=len(prompt_token_ids) + len(generated_token_ids),
                prev_len=call.admission.prev_len,
                expected_route_dims=self._expected_route_dims,
            )
            extras = {"routed_experts": encode_routed_experts(routed_experts)}
        coords = self._capture.complete_call(
            call,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            generated_logprobs=generated_logprobs,
            extras=extras,
        )
        return self._stage_result(coords)

    @staticmethod
    def _stage_result(coords: Any) -> RequestPayloadStageResult:
        """Wrap Gym commit coordinates in MInf's stage result."""
        # Deferred: Megatron-LM's inference hooks are only present on the
        # Megatron generation backend (see prepare_prompt); nemo_gym is an
        # optional extra absent in non-gym runs.
        from megatron.core.inference.inference_request import (
            RequestPayloadStageResult,
        )
        from nemo_gym.token_id_capture import NG_COMMIT_COORDS_FIELD

        return RequestPayloadStageResult(
            response_metadata={
                NG_COMMIT_COORDS_FIELD: coords.model_dump(mode="json"),
            }
        )
