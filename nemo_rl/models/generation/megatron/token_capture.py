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
from typing import TYPE_CHECKING, Any

import torch

from nemo_rl.data_plane.tq_token_sink import (
    ChainPrefixCache,
    StagedMediaTensors,
    TQTokenSink,
    TQTokenSource,
    resolve_admission_prefix_chains,
    validate_media_tensors,
)
from nemo_rl.experience.route_assembly import (
    ROUTE_MISSING_SENTINEL,
    boundary_route_rows,
)
from nemo_rl.models.generation.openai_server_utils import replace_prefix_tokens
from nemo_rl.utils.routed_experts_codec import encode_routed_experts

if TYPE_CHECKING:
    from megatron.core.inference.inference_request import (
        RequestPayloadStageResult,
        RequestPromptPreparationResult,
    )

# offload_params sub-dict the Megatron preparer writes and the stager reads.
MINF_CAPTURE_PARAMS_FIELD = "ng_capture_minf"
MEDIA_PREV_COUNT_KEY = "media_prev_count"
# ``(height, width)`` rows per staged item (``StagedPrefix.media_sizes`` as
# nested lists): the stager checks the engine's retained media against them.
MEDIA_PREV_SIZES_KEY = "media_prev_sizes"


def _delta_align_minf_routing_indices(
    routing_indices: Any,
    *,
    total_tokens: int,
    prev_len: int,
    expected_route_dims: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Convert MInf ``[T - 1, L, K]`` routes to Gym's delta-token layout.

    A continuation also keeps the row for the previous turn's last token: the
    previous call sampled it without feeding it in, so only this call's
    prefill routed it.

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
    return aligned[prev_len - boundary_route_rows(prev_len) :]


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
        - ``ng_capture_minf.media_prev_count`` and ``media_prev_sizes``: the
          media items the parent chain staged and their geometry; the stager
          checks the leading items of the engine's whole-conversation
          ``media_tensors`` against the sizes (a resent image at a new size
          fails the call) and slices at the count so each row carries only
          this call's media.
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
            MEDIA_PREV_SIZES_KEY: [
                [list(size) for size in item] for item in chains.media_sizes
            ],
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


def _prev_sizes(minf_params: Any, prev_count: int) -> list[list[list[int]]]:
    """Validate ``media_prev_sizes``: one list of ``[h, w]`` rows per counted item."""
    value = minf_params.get(MEDIA_PREV_SIZES_KEY) if minf_params is not None else None
    if value is None:
        value = []
    if (
        not isinstance(value, list)
        or len(value) != prev_count
        or any(
            not isinstance(item, list)
            or not item
            or any(
                not isinstance(size, list)
                or len(size) != 2
                or any(type(v) is not int or v <= 0 for v in size)
                for size in item
            )
            for item in value
        )
    ):
        raise ValueError(
            f"MInf capture request carries an invalid {MEDIA_PREV_SIZES_KEY} for "
            f"{prev_count} staged media items: {value!r}"
        )
    return value


def slice_media_tensors(
    media: StagedMediaTensors, prev_count: int
) -> dict[str, torch.Tensor] | None:
    """Drop the first ``prev_count`` media items from the engine's media.

    Every chat request carries the whole conversation, so the engine hands the
    stager pixels for every image in the prompt. The parent chain already
    staged the first ``prev_count`` of them; this keeps only the rest so media
    columns are per-call deltas like the token columns.

    ``media`` must already have passed ``validate_media_tensors``, which
    pins the packed-patch layout and gives ``patch_size``; item boundaries
    come from ``StagedMediaTensors.item_sizes``, the same rule the sink uses
    to count a chain's staged items. Returns the remaining media as the
    attachments mapping Gym's capture core takes, or ``None`` when every item
    was already staged.

    Raises:
        ValueError: ``prev_count`` exceeds the items present.
    """
    items = media.item_sizes
    if prev_count > len(items):
        raise ValueError(
            f"media_prev_count {prev_count} exceeds the {len(items)} media items "
            "the engine saw"
        )
    if prev_count == len(items):
        return None
    sliced: dict[str, torch.Tensor] = {
        "imgs": media.imgs,
        "imgs_sizes": media.imgs_sizes,
    }
    if media.num_frames is not None:
        sliced["num_frames"] = media.num_frames
    if prev_count <= 0:
        return sliced

    prev_rows = sum(len(item) for item in items[:prev_count])
    patch_area = media.patch_size**2
    prev_patches = sum(h * w for item in items[:prev_count] for h, w in item)
    # validate_media_tensors checked every row divides into whole patches.
    prev_patches //= patch_area
    sliced["imgs"] = media.imgs[:, prev_patches:, :]
    sliced["imgs_sizes"] = media.imgs_sizes[prev_rows:]
    if media.num_frames is not None:
        sliced["num_frames"] = media.num_frames[prev_count:]
    return sliced


def _new_media_tensors(payload: Any, minf_params: Any) -> dict[str, Any] | None:
    """Return the media of one finished MInf payload that is new to this call.

    Checks the first ``minf_params["media_prev_count"]`` items of
    ``payload.media_tensors`` against ``minf_params["media_prev_sizes"]`` (the
    geometry the parent chain staged), and slices the tensors there so only the
    media new to this call remains. ``media_tensors`` is validated against the
    sink's media contract (``validate_media_tensors``) first.

    Args:
        payload: The engine's ``OffloadedRequestPayload`` (or equivalent).
        minf_params: The ``ng_capture_minf`` mapping the prompt preparer
            wrote, or ``None`` for requests it did not touch.

    Returns:
        The sliced media tensors, or ``None`` when the call adds no media.

    Raises:
        TypeError: ``minf_params`` is not a dict, ``media_tensors`` is not
            a mapping, or a media tensor is not a ``torch.Tensor``.
        ValueError: ``media_tensors`` violates the media contract,
            ``media_prev_count`` is not a non-negative int,
            ``media_prev_sizes`` is malformed or does not have one entry per
            counted item, a retained item's geometry differs from the
            staged one, or the media cannot be sliced at the count.

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
    prev_count = _count(MEDIA_PREV_COUNT_KEY)
    staged_sizes = _prev_sizes(minf_params, prev_count)
    # The sink's contract check runs here, before slicing, so the layout
    # is known good and a malformed bundle fails this call, not the write.
    validated = validate_media_tensors(media_tensors or None)
    items = () if validated is None else validated.item_sizes
    if prev_count > len(items):
        raise ValueError(
            f"media_prev_count {prev_count} exceeds the {len(items)} media "
            "items the engine saw"
        )
    for index, (staged, item) in enumerate(
        zip(staged_sizes, items[:prev_count], strict=True)
    ):
        seen = [list(size) for size in item]
        if staged != seen:
            raise ValueError(
                f"MInf retained media geometry changed: item {index} was "
                f"staged at {staged}, the engine saw {seen}"
            )
    return None if validated is None else slice_media_tensors(validated, prev_count)


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
        # MInf hands over float32 pixels; the sink's column is pinned to the
        # vision encoder's weight dtype (the trainer casts pixels to it before
        # encoding anyway), so cast once here and stage half the bytes.
        self._media_pixel_dtype = sink.media_pixel_dtype
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
            return self._stage_admitted(
                payload,
                call=call,
                minf_params=(offload_params or {}).get(MINF_CAPTURE_PARAMS_FIELD),
            )
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

    def _cast_media_pixels(
        self, media_tensors: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Cast the sliced ``imgs`` to the sink's pinned pixel dtype.

        Only the pixels change dtype; ``imgs_sizes`` / ``num_frames`` stay
        int32. A no-op when the dtypes already agree or the sink is text-only.
        """
        if not media_tensors:
            return None
        imgs = media_tensors["imgs"]
        if self._media_pixel_dtype is None or imgs.dtype == self._media_pixel_dtype:
            return media_tensors
        return {**media_tensors, "imgs": imgs.to(dtype=self._media_pixel_dtype)}

    def _stage_admitted(
        self,
        payload: Any,
        *,
        call: Any,
        minf_params: Any = None,
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
        try:
            media = _new_media_tensors(payload, minf_params)
        except (TypeError, ValueError, RuntimeError) as error:
            coords = self._capture.fail_call(
                call, reason=f"{type(error).__name__}: {error}"
            )
            return self._stage_result(coords)
        coords = self._capture.complete_call(
            call,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            generated_logprobs=generated_logprobs,
            extras=extras,
            attachments=self._cast_media_pixels(media),
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
