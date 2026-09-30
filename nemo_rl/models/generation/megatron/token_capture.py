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
    MEDIA_PREV_COUNT_KEY,
    MINF_CAPTURE_PARAMS_FIELD,
    ChainPrefixCache,
    TQTokenSink,
    TQTokenSource,
    resolve_admission_prefix_chains,
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
    rendered prior-turn tokens and EOS ids the Megatron endpoint carried in
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
        """Resolve the admission's staged prefix and splice it into the prompt.

        Returns the prompt unchanged when ``offload_params`` carries no
        ``ng_capture`` admission or the admission is text mode; a text-mode
        return also drops the endpoint's ``_prefix_media_count`` (if any), since
        nothing was spliced and the engine must expand every media placeholder
        itself. Otherwise the
        admission's ``staging_chain`` (or inline prefix) is resolved through the
        worker-local ``ChainPrefixCache`` into expanded token ids plus the number
        of media items the parent chain staged. Those ids are spliced over the
        endpoint's re-rendered history with ``replace_prefix_tokens``, using the
        template prefix tokens and EOS ids the endpoint placed in
        ``offload_params``; the splice metadata is skipped only when the
        endpoint sent none and the admission has no ``staging_chain``.

        The returned ``offload_params`` copy is updated with:

        - ``ng_capture.required_prefix_token_ids``: the expanded prefix Gym
          verifies the engine's prompt against.
        - ``ng_capture_minf.media_prev_count``: media items the parent chain
          staged; the stager slices the engine's whole-conversation
          ``media_tensors`` there so each row carries only this call's media.
        - ``_prefix_expanded_token_count``: written only when the endpoint
          reported ``_prefix_media_count``; tells the engine where the
          already-expanded prefix ends so it expands only the media
          placeholders after it.

        Args:
            prompt: The endpoint's rendered prompt. Must be a token-id list
                when a token-in admission is present.
            offload_params: Request metadata the endpoint attached; may carry
                the Gym admission and the prompt splice metadata.

        Returns:
            The (possibly spliced) prompt and the updated ``offload_params``.

        Raises:
            ValueError: The resolved prefix length differs from the admission's
                ``prev_len``, the splice metadata is malformed or missing for a
                chained admission, the splice did not yield the authorized
                prefix, or the endpoint's prefix media count disagrees with the
                chain's.
            TypeError: The prompt is not a token-id list.
        """
        # Deferred import: megatron-core is a heavy, optional dependency.
        from megatron.core.inference.inference_request import (
            PREFIX_EOS_TOKEN_ID_FIELD,
            PREFIX_EXPANDED_TOKEN_COUNT_FIELD,
            PREFIX_MEDIA_COUNT_FIELD,
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
            if PREFIX_MEDIA_COUNT_FIELD in offload_params:
                offload_params = dict(offload_params)
                offload_params.pop(PREFIX_MEDIA_COUNT_FIELD)
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
        # Gym verifies the engine's prompt against this prefix.
        updated_admission = admission.model_copy(
            update={"required_prefix_token_ids": prefix_token_ids}
        )
        updated_offload_params[NG_CAPTURE_FIELD] = updated_admission.model_dump(
            mode="json"
        )
        updated_offload_params[MINF_CAPTURE_PARAMS_FIELD] = {
            **(updated_offload_params.get(MINF_CAPTURE_PARAMS_FIELD) or {}),
            MEDIA_PREV_COUNT_KEY: chains.media_count,
        }

        template_prefix_token_ids = updated_offload_params.get(
            PREFIX_TEMPLATE_TOKEN_IDS_FIELD
        )
        eos_token_ids = updated_offload_params.get(PREFIX_EOS_TOKEN_ID_FIELD)
        if template_prefix_token_ids is not None or eos_token_ids is not None:
            if not isinstance(template_prefix_token_ids, list) or any(
                type(token_id) is not int for token_id in template_prefix_token_ids
            ):
                raise ValueError(
                    "MInf capture request carries no valid template prefix tokens"
                )
            if type(eos_token_ids) is not int and (
                not isinstance(eos_token_ids, list)
                or not eos_token_ids
                or any(type(token_id) is not int for token_id in eos_token_ids)
            ):
                raise ValueError("MInf capture request carries no valid EOS token ids")
            prompt = replace_prefix_tokens(
                tokenizer=None,
                model_prefix_token_ids=prefix_token_ids,
                template_prefix_token_ids=template_prefix_token_ids,
                template_token_ids=prompt,
                eos_token_id=eos_token_ids,
            )
        elif admission.staging_chain:
            raise ValueError(
                "MInf staged-prefix request carries no prompt splice metadata"
            )

        if prompt[: len(prefix_token_ids)] != prefix_token_ids:
            raise ValueError("MInf failed to apply the authorized token prefix")
        endpoint_media_count = updated_offload_params.get(PREFIX_MEDIA_COUNT_FIELD)
        if (endpoint_media_count or 0) != chains.media_count:
            raise ValueError(
                "MInf capture prefix media count mismatch: the chat request's history "
                f"carries {endpoint_media_count or 0}, the staged chain "
                f"{chains.media_count}"
            )
        if endpoint_media_count is not None:
            updated_offload_params[PREFIX_EXPANDED_TOKEN_COUNT_FIELD] = len(
                prefix_token_ids
            )
        return RequestPromptPreparationResult(
            prompt=prompt, offload_params=updated_offload_params
        )


def slice_media_tensors(
    media_tensors: dict[str, Any] | None, prev_count: int
) -> dict[str, Any] | None:
    """Drop the first ``prev_count`` media items from the engine's media tensors.

    Every chat request carries the whole conversation, so the engine hands the
    stager pixels for every image in the prompt. The parent chain already
    staged the first ``prev_count`` of them; this keeps only the rest so media
    columns are per-call deltas like the token columns.

    Item boundaries come from the tensors themselves: ``num_frames`` (frames per
    video) when present, else one row of ``imgs_sizes`` per image. ``imgs`` must
    be packed patches ``[1, total_patches, C*P*P]`` (the only layout
    ``validate_media_tensors`` accepts); the patch count per row is
    ``h*w/P**2`` with ``P**2`` recovered from the totals.

    Raises:
        ValueError: ``imgs_sizes`` is missing, ``prev_count`` exceeds the items
            present, ``imgs`` is not packed patches, or the geometry does not
            tile into whole patches at the parent boundary.
    """
    if not media_tensors or prev_count <= 0:
        return media_tensors
    imgs = media_tensors.get("imgs")
    imgs_sizes = media_tensors.get("imgs_sizes")
    num_frames = media_tensors.get("num_frames")
    if imgs is None:
        return media_tensors
    if imgs_sizes is None:
        raise ValueError("media delta requires imgs_sizes to locate items")

    if num_frames is not None:
        total_items = int(num_frames.numel())
    else:
        total_items = int(imgs_sizes.reshape(-1, 2).shape[0])
    if prev_count > total_items:
        raise ValueError(
            f"media_prev_count {prev_count} exceeds the {total_items} media items "
            "the engine saw"
        )
    if prev_count == total_items:
        return None

    # Rows of imgs_sizes / imgs covered by the parent chain.
    if num_frames is not None:
        prev_rows = int(num_frames.reshape(-1)[:prev_count].sum().item())
    else:
        prev_rows = prev_count

    sliced: dict[str, Any] = {}
    if imgs.ndim == 3 and imgs.shape[0] == 1:
        # Packed patches: recover patches-per-row from sizes and the total.
        sizes = imgs_sizes.reshape(-1, 2).to(torch.int64)
        areas = sizes[:, 0] * sizes[:, 1]
        total_area = int(areas.sum().item())
        total_patches = int(imgs.shape[1])
        if total_patches == 0 or total_area % total_patches:
            raise ValueError(
                f"packed patches {total_patches} do not divide the media area {total_area}"
            )
        patch_area = total_area // total_patches
        prev_area = int(areas[:prev_rows].sum().item())
        if prev_area % patch_area:
            raise ValueError("parent media does not end on a patch boundary")
        sliced["imgs"] = imgs[:, prev_area // patch_area :, :]
    else:
        raise ValueError(
            "media imgs must be packed patches [1, total_patches, F], "
            f"got shape {tuple(imgs.shape)}"
        )
    sliced["imgs_sizes"] = imgs_sizes.reshape(-1, 2)[prev_rows:]
    if num_frames is not None:
        sliced["num_frames"] = num_frames.reshape(-1)[prev_count:]
    return sliced


@dataclass(frozen=True)
class _MegatronCapturePayload:
    """The MInf offloaded payload plus the worker-side context Gym's adapter reads."""

    prompt_token_ids: Any
    generated_token_ids: Any
    generated_log_probs: Any
    # The engine's media tensors minus what the parent chain already staged.
    media_tensors: dict[str, Any] | None

    @classmethod
    def from_offloaded(
        cls, payload: Any, minf_params: Any
    ) -> "_MegatronCapturePayload":
        """Build the adapter-facing view of one finished MInf payload.

        Copies the ``prompt_token_ids`` / ``generated_token_ids`` /
        ``generated_log_probs`` attributes Gym's ``MegatronCaptureAdapter``
        reads (missing ones become ``None``) and slices
        ``payload.media_tensors`` at ``minf_params["media_prev_count"]`` so
        only the media new to this call remains.

        Args:
            payload: The engine's ``OffloadedRequestPayload`` (or equivalent).
            minf_params: The ``ng_capture_minf`` mapping the prompt preparer
                wrote, or ``None`` for requests it did not touch.

        Raises:
            TypeError: ``minf_params`` is not a dict or ``media_tensors`` is not
                a mapping.
            ValueError: ``media_prev_count`` is not a non-negative int, or the
                media geometry cannot be sliced there.

        The stager maps both to ``capture_failed`` coordinates.
        """
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
