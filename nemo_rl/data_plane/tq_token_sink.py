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
"""TransferQueue implementations of NeMo-Gym's token staging protocols.

``TQStagingStore`` is NeMo RL's single keyed-row transport for token custody.
Both vLLM and MInf write canonical Gym call deltas through ``TQTokenSink``.
This module is the only hot-path file that knows tokens live in TQ; Gym sees
opaque staging keys.

Each staged row carries three jagged columns (``token_ids_delta``,
``token_mask_delta``, ``generation_logprobs_delta``), the complete receipt
identity/lineage metadata, and all digest inputs so it round-trips to a
normally validated ``StagedCallBaseSnapshot``. Masks/logprobs are float32 on
the wire, matching ``compute_staging_digest``'s float32-bit-pattern scheme, so
digest recomputation over fetched values is byte-exact. Route payloads never
ride inside snapshots: the source returns them as separate ``RouteFragment``
values keyed by staging key, digest-verified by the plan executor at point of
use.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ray
import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from megatron.core.inference.inference_request import (
        RequestPromptPreparationResult,
    )

    # Deferred: nemo_gym is an optional extra absent in non-gym runs; runtime
    # uses import locally so this module (and the finalizer actor importing
    # it) stays importable without it.
    from nemo_gym.token_id_capture.staging.records import (
        StagedCallBaseSnapshot,
        StagedCallRecord,
        StageResult,
    )

from nemo_rl.data_plane.schema import (
    ROUTE_ENCODING_ENVELOPE,
    ROUTE_ENCODING_LIST,
    ROUTE_ENCODING_NONE,
    ROUTED_EXPERTS_ENCODING_FIELD,
    ROUTED_EXPERTS_FIELD,
    ROUTED_EXTRAS_METADATA_FIELD,
    ROUTED_LEN_FIELD,
)
from nemo_rl.experience.route_assembly import RouteFragment
from nemo_rl.models.generation.openai_server_utils import replace_prefix_tokens

# These names come from nemo_gym.token_id_capture.staging.records.StagedCallRecord,
# transformed by stage() below. Adding a field means editing both this list and
# stage(); a mismatch is caught by test_tq_sink_source_passes_conformance's
# round-trip equality check -- but only for required StagedCallRecord fields. An
# optional field Gym adds that this sink never stages will default identically
# on both sides and pass that check silently.
# Compact-space token delta for calls that carried media (see Gym's
# ``nemo_gym.token_id_capture.staging.media``). ``token_ids_delta`` stays the
# expanded sequence the trainer needs; the compact form is what the next turn's
# chat render must be spliced against. ``COMPACT_LEN_FIELD`` is 0 when the call
# staged no compact form, in which case the compact and expanded deltas coincide
# and readers fall back to ``token_ids_delta``.
COMPACT_TOKEN_IDS_FIELD = "compact_token_ids_delta"
COMPACT_LEN_FIELD = "compact_len"
# Extras key Gym uses for the compact delta (popped into the column above, like
# routed_experts) and for the media summary (left in the extras JSON).
COMPACT_TOKEN_IDS_EXTRAS_KEY = "compact_token_ids_delta"
MEDIA_EXTRAS_KEY = "media"
# offload_params sibling key the MInf preparer writes for the stager: the
# compact length of the parent chain it spliced in. Reserved-prefix-free ('_'
# keys are engine-owned) and never read by Megatron itself.
MINF_CAPTURE_PARAMS_FIELD = "ng_capture_minf"
COMPACT_PREV_LEN_KEY = "compact_prev_len"
# Media items (images, or videos) already staged by the parent chain. The stager
# slices the engine's media tensors at this boundary so each call row holds only
# the media new to that call, the same way token columns hold only new tokens.
MEDIA_PREV_COUNT_KEY = "media_prev_count"

# Media the engine's vision encoder consumed, staged as extra columns on the
# call row (a second put onto the same staging key, after the token row is
# durable). ``imgs`` is MInf's packed-patch tensor ``[1, total_patches, C*P*P]``
# (or padded pixels), ``imgs_sizes`` ``[N, 2]``, ``num_frames`` / ``num_tiles``
# when the request carried them. Each tensor is flattened to one ``[1, numel]``
# row; ``media_geometry_json`` records shape and dtype per name. Registered on
# the staging partition only for multimodal runs; each row holds only the media
# new to its call, and the finalizer concatenates them along the terminal chain.
MEDIA_IMGS_FIELD = "media_imgs"
MEDIA_IMGS_SIZES_FIELD = "media_imgs_sizes"
MEDIA_NUM_FRAMES_FIELD = "media_num_frames"
MEDIA_NUM_TILES_FIELD = "media_num_tiles"
MEDIA_GEOMETRY_FIELD = "media_geometry_json"
MEDIA_TENSOR_COLUMNS: dict[str, str] = {
    "imgs": MEDIA_IMGS_FIELD,
    "imgs_sizes": MEDIA_IMGS_SIZES_FIELD,
    "num_frames": MEDIA_NUM_FRAMES_FIELD,
    "num_tiles": MEDIA_NUM_TILES_FIELD,
}
MEDIA_STAGING_FIELDS = [*MEDIA_TENSOR_COLUMNS.values(), MEDIA_GEOMETRY_FIELD]

STAGING_FIELDS = [
    "token_ids_delta",
    "token_mask_delta",
    "generation_logprobs_delta",
    COMPACT_TOKEN_IDS_FIELD,
    COMPACT_LEN_FIELD,
    "schema_version",
    "digest_version",
    "extras_digest_version",
    "rollout_id_utf8",
    "model_call_id_utf8",
    "parent_call_id_utf8",
    "parent_call_id_present",
    "capture_mode",
    "prev_len",
    "delta_len",
    "cum_len",
    "weight_version",
    "digest_bytes",
    "extras_digest_bytes",
    "chain_hash_bytes",
    "chain_hash_present",
    "cumulative_hash_bytes",
    "cumulative_hash_present",
    ROUTED_EXTRAS_METADATA_FIELD,
    ROUTED_EXPERTS_ENCODING_FIELD,
    ROUTED_LEN_FIELD,
]

_MODE_TO_CODE = {"text": 0, "token_in": 1}
_CODE_TO_MODE = {code: mode for mode, code in _MODE_TO_CODE.items()}


def _bytes_tensor(value: bytes) -> torch.Tensor:
    """Encode non-empty bytes as one jagged TQ row."""
    if not value:
        raise ValueError("staging byte fields must be non-empty")
    return torch.tensor([list(value)], dtype=torch.uint8)


def _optional_digest_fields(value: str | None) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _bytes_tensor(bytes.fromhex(value) if value is not None else bytes(32)),
        torch.tensor([value is not None], dtype=torch.bool),
    )


@dataclass(frozen=True)
class StagedMediaTensors:
    """The media tensors one staged call ran on, restored to their original shapes."""

    imgs: torch.Tensor
    imgs_sizes: torch.Tensor | None
    num_frames: torch.Tensor | None
    num_tiles: torch.Tensor | None


@dataclass(frozen=True)
class FetchedStagedCall:
    """One explicitly identified small-column finalization fetch result.

    ``fragment`` is populated only when the fetch requested route payloads
    (direct mode); deferred finalization leaves route bytes in TQ and carries
    only ``routed_len`` transport metadata.
    """

    staging_key: str
    snapshot: StagedCallBaseSnapshot
    routed_len: int
    fragment: RouteFragment | None = None
    # Decoded extras JSON (minus the columns the sink popped out), None when
    # the call staged no extras. Carries Gym's media summary for VLM calls.
    extras: dict[str, Any] | None = None


def _call_dp(dp_client: Any, method_name: str, **kwargs: Any) -> Any:
    """Call a DataPlaneClient method on a local client or a Ray actor handle."""
    method = getattr(dp_client, method_name)
    remote = getattr(method, "remote", None)
    if remote is not None:
        return ray.get(remote(**kwargs))
    return method(**kwargs)


class TQStagingStore:
    """Shared keyed-row transport for all token-capture TQ codecs."""

    def __init__(self, dp_client: Any, *, staging_partition: str) -> None:
        self._dp_client = dp_client
        self._staging_partition = staging_partition

    def put(
        self,
        key: str,
        field_dict: dict[str, torch.Tensor],
        *,
        tags: dict[str, Any] | None = None,
    ) -> None:
        _call_dp(
            self._dp_client,
            "put_samples",
            sample_ids=[key],
            partition_id=self._staging_partition,
            fields=TensorDict(field_dict, batch_size=[1]),
            tags=[tags or {}],
        )

    def get(self, keys: list[str], *, select_fields: list[str]) -> TensorDict:
        return _call_dp(
            self._dp_client,
            "get_samples",
            sample_ids=list(keys),
            partition_id=self._staging_partition,
            select_fields=list(select_fields),
        )

    def clear(self, keys: list[str]) -> None:
        if not keys:
            return
        _call_dp(
            self._dp_client,
            "clear_samples",
            sample_ids=list(keys),
            partition_id=self._staging_partition,
        )


class TQTokenSink:
    """Gym ``StagingSink`` over ``DataPlaneClient.put_samples``.

    ``stage`` is synchronous and returns only after TQ acknowledged the
    write, so the capture layer's fail-closed ordering (bytes durable before
    the model call is acked) holds by construction. Failures are reported in
    the ``StageResult``; the finalizer turns a poisoned rollout into a
    placeholder row (see ``RolloutReassembler.finalize_group``).

    ``stage`` is thread-safe per the ``StagingSink`` contract: it holds no
    per-call mutable state, so the capture host may run writes for unrelated
    calls concurrently.
    """

    def __init__(self, dp_client: Any, *, staging_partition: str) -> None:
        self._store = TQStagingStore(dp_client, staging_partition=staging_partition)

    def stage(self, record: StagedCallRecord) -> StageResult:
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.staging.records import StageResult

        key = record.staging_key
        try:
            field_dict = {
                "token_ids_delta": torch.tensor(
                    [record.token_ids_delta], dtype=torch.int64
                ),
                "token_mask_delta": torch.tensor(
                    [record.token_mask_delta], dtype=torch.float32
                ),
                "generation_logprobs_delta": torch.tensor(
                    [record.generation_log_probs_delta], dtype=torch.float32
                ),
                "schema_version": torch.tensor(
                    [record.schema_version], dtype=torch.int64
                ),
                "digest_version": torch.tensor(
                    [record.digest_version], dtype=torch.int64
                ),
                "extras_digest_version": torch.tensor(
                    [record.extras_digest_version], dtype=torch.int64
                ),
                "rollout_id_utf8": _bytes_tensor(record.rollout_id.encode("utf-8")),
                "model_call_id_utf8": _bytes_tensor(
                    record.model_call_id.encode("utf-8")
                ),
                "parent_call_id_utf8": _bytes_tensor(
                    (record.parent_call_id or "\0").encode("utf-8")
                ),
                "parent_call_id_present": torch.tensor(
                    [record.parent_call_id is not None], dtype=torch.bool
                ),
                "capture_mode": torch.tensor(
                    [_MODE_TO_CODE[record.mode]], dtype=torch.int64
                ),
                "prev_len": torch.tensor([record.prev_len], dtype=torch.int64),
                "delta_len": torch.tensor([record.delta_len], dtype=torch.int64),
                "cum_len": torch.tensor([record.cum_len], dtype=torch.int64),
                "weight_version": torch.tensor(
                    [record.weight_version], dtype=torch.int64
                ),
                "digest_bytes": _bytes_tensor(bytes.fromhex(record.digest)),
                "extras_digest_bytes": _bytes_tensor(
                    bytes.fromhex(record.extras_digest)
                ),
            }
            chain_hash, chain_hash_present = _optional_digest_fields(record.chain_hash)
            cumulative_hash, cumulative_hash_present = _optional_digest_fields(
                record.cumulative_hash
            )
            field_dict.update(
                {
                    "chain_hash_bytes": chain_hash,
                    "chain_hash_present": chain_hash_present,
                    "cumulative_hash_bytes": cumulative_hash,
                    "cumulative_hash_present": cumulative_hash_present,
                }
            )
            extras_metadata = dict(record.extras) if record.extras is not None else None
            routed = (
                extras_metadata.pop("routed_experts", None)
                if extras_metadata is not None
                else None
            )
            compact_delta = (
                extras_metadata.pop(COMPACT_TOKEN_IDS_EXTRAS_KEY, None)
                if extras_metadata is not None
                else None
            )
            if compact_delta is not None:
                if (
                    not isinstance(compact_delta, list)
                    or not compact_delta
                    or any(type(token_id) is not int for token_id in compact_delta)
                ):
                    raise ValueError(
                        "compact_token_ids_delta must be a non-empty list of ints"
                    )
                field_dict[COMPACT_TOKEN_IDS_FIELD] = torch.tensor(
                    [compact_delta], dtype=torch.int64
                )
                field_dict[COMPACT_LEN_FIELD] = torch.tensor(
                    [len(compact_delta)], dtype=torch.int64
                )
            else:
                # Sentinel row: jagged columns cannot be empty. compact_len 0
                # tells readers the compact delta equals token_ids_delta.
                field_dict[COMPACT_TOKEN_IDS_FIELD] = torch.tensor(
                    [[0]], dtype=torch.int64
                )
                field_dict[COMPACT_LEN_FIELD] = torch.tensor([0], dtype=torch.int64)
            field_dict[ROUTED_EXTRAS_METADATA_FIELD] = _bytes_tensor(
                json.dumps(
                    extras_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            )
            routed_len = 0
            routed_encoding = ROUTE_ENCODING_NONE
            if routed is not None:
                delta_len = len(record.token_ids_delta)
                if isinstance(routed, str):
                    from nemo_rl.utils.routed_experts_codec import (
                        decode_routed_experts,
                    )

                    dtype_name = routed.split(":", 3)[1]
                    dtype = {
                        "int8": torch.int8,
                        "int16": torch.int16,
                        "int32": torch.int32,
                    }.get(dtype_name)
                    if dtype is None:
                        raise ValueError(
                            f"unsupported routed_experts dtype {dtype_name!r}"
                        )
                    experts = decode_routed_experts(routed, dtype)
                    routed_encoding = ROUTE_ENCODING_ENVELOPE
                else:
                    experts = torch.tensor(routed, dtype=torch.int16)
                    routed_encoding = ROUTE_ENCODING_LIST
                if experts.dim() != 3 or experts.shape[0] != delta_len:
                    raise ValueError(
                        "routed_experts must already be delta-aligned: "
                        f"got shape {tuple(experts.shape)} for delta_len={delta_len}"
                    )
                field_dict[ROUTED_EXPERTS_FIELD] = experts.unsqueeze(0)
                routed_len = int(experts.shape[0])
            field_dict[ROUTED_EXPERTS_ENCODING_FIELD] = torch.tensor(
                [routed_encoding], dtype=torch.int64
            )
            field_dict[ROUTED_LEN_FIELD] = torch.tensor([routed_len], dtype=torch.int64)
            tags = [
                {
                    "rollout_id": record.rollout_id,
                    "model_call_id": record.model_call_id,
                    "parent_call_id": record.parent_call_id,
                    "prev_len": record.prev_len,
                    "delta_len": record.delta_len,
                    "cum_len": record.cum_len,
                    "weight_version": record.weight_version,
                    "digest": record.digest,
                    "schema_version": record.schema_version,
                }
            ]
            self._store.put(key, field_dict, tags=tags[0])
        except Exception as error:  # noqa: BLE001 — any failure must poison, not crash serving
            # The reason string is dropped downstream (_failed_coords carries
            # only the disposition) — this log line is the only place the
            # actual stage failure is visible.
            logging.getLogger(__name__).warning(
                "TQTokenSink.stage failed for %s: %s: %s",
                key,
                type(error).__name__,
                error,
            )
            return StageResult(
                ok=False, staging_key=key, error=f"{type(error).__name__}: {error}"
            )
        return StageResult(ok=True, staging_key=key)

    def clear(self, staging_keys: list[str]) -> None:
        """Drop staged rows (finalizer / eviction cleanup)."""
        self._store.clear(staging_keys)

    def stage_media(self, staging_key: str, media_tensors: dict[str, Any]) -> None:
        """Add the engine's media tensors to an already-staged call row.

        A second put onto the same key: TQ tracks field readiness per field, so
        the media columns land beside the token columns without rewriting them.
        Raises on failure; the caller decides how to report it (the finalizer
        rejects the rollout as ``media_columns_missing`` either way).
        """
        self._store.put(staging_key, media_field_dict(media_tensors))


@dataclass(frozen=True)
class MegatronPayloadStageResult:
    """Structural MInf staging acknowledgement returned to the engine."""

    response_metadata: dict[str, Any]


# Request-metadata keys the Megatron chat endpoint writes when it defers the
# prefix splice to the engine's prompt preparer. Must match the constants of
# the same name in Megatron-LM's ``megatron/core/inference/inference_request.py``;
# the names mirror the ``replace_prefix_tokens`` arguments they feed.
PREFIX_TEMPLATE_TOKEN_IDS_FIELD = "template_prefix_token_ids"
PREFIX_EOS_TOKEN_ID_FIELD = "eos_token_id"


@dataclass(frozen=True)
class PrefixChains:
    """One resolved ``staging_chain`` in both token spaces.

    ``expanded`` is the concatenated ``token_ids_delta`` chain: what the engine
    prompt must start with and what Gym's capture core verifies. ``compact`` is
    the concatenated compact deltas (falling back to the expanded delta for
    calls that staged none): what a multimodal chat render is spliced against.
    They are identical for text-only chains. ``media_count`` is how many media
    items (images, or videos) the chain's rows staged, so the next call can
    stage only the media new to it.
    """

    expanded: list[int]
    compact: list[int]
    media_count: int = 0

    def __add__(self, other: "PrefixChains") -> "PrefixChains":
        return PrefixChains(
            expanded=self.expanded + other.expanded,
            compact=self.compact + other.compact,
            media_count=self.media_count + other.media_count,
        )


_EMPTY_CHAINS = PrefixChains(expanded=[], compact=[])


def media_item_count(geometry: dict[str, Any] | None) -> int:
    """Number of media items a staged row's geometry describes.

    Videos count once each (``num_frames`` has one entry per video), static
    tiling counts images by ``num_tiles`` entries, and stills by ``imgs_sizes``.
    """
    if not geometry:
        return 0
    for name in ("num_frames", "num_tiles", "imgs_sizes"):
        value = geometry.get(name)
        if value is not None:
            return len(value)
    return 0


class ChainPrefixCache:
    """Worker-local cache of resolved ``staging_chain`` prefixes."""

    def __init__(self, source: TQTokenSource | None = None) -> None:
        self._source: TQTokenSource | None = source
        self._cache: dict[str, PrefixChains] = {}
        self._lock = threading.Lock()

    def install(self, source: TQTokenSource) -> None:
        """Attach (or replace) the ``TQTokenSource`` and drop cached chains."""
        with self._lock:
            self._source = source
            self._cache.clear()

    def fetch(self, staging_chain: list[str]) -> list[int]:
        """Assemble the expanded prefix from staging_chain (see :meth:`fetch_chains`)."""
        return list(self.fetch_chains(staging_chain).expanded)

    def fetch_chains(self, staging_chain: list[str]) -> PrefixChains:
        """Assemble both prefix spaces from staging_chain, with a worker-local FIFO (256-entry) cache."""
        cache = self._cache
        with self._lock:
            source = self._source
            cached: PrefixChains = _EMPTY_CHAINS
            miss_start = 0
            for i, key in enumerate(staging_chain):
                if key in cache:
                    cached = cache[key]
                    miss_start = i + 1
            miss_keys = staging_chain[miss_start:]
        if not miss_keys:
            return PrefixChains(
                list(cached.expanded), list(cached.compact), cached.media_count
            )
        if source is None:
            raise RuntimeError(
                "staging source not initialized; call setup_token_capture() first"
            )
        # TQ read stays outside the lock so concurrent fetches overlap.
        fetched = source.fetch_prefix_chains(miss_keys)
        result = cached + fetched
        last_key = staging_chain[-1]
        with self._lock:
            cache[last_key] = result
            if len(cache) > 256:
                del cache[next(iter(cache))]
        return PrefixChains(
            list(result.expanded), list(result.compact), result.media_count
        )


def resolve_admission_prefix(
    admission: Any, chain_prefix: ChainPrefixCache
) -> list[int]:
    """Resolve a ``CaptureAdmission`` to the flat prefix the engine prompt starts with."""
    return resolve_admission_prefix_chains(admission, chain_prefix).expanded


def resolve_admission_prefix_chains(
    admission: Any, chain_prefix: ChainPrefixCache
) -> PrefixChains:
    """Resolve a ``CaptureAdmission`` to its prefix in both token spaces.

    An inline ``required_prefix_token_ids`` prefix has no separate compact form:
    Gym only inlines prefixes for text chains.
    """
    if admission.mode == "text":
        return PrefixChains(expanded=[], compact=[])
    if admission.staging_chain:
        return chain_prefix.fetch_chains(list(admission.staging_chain))
    inline = list(admission.required_prefix_token_ids)
    return PrefixChains(expanded=inline, compact=list(inline))


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
        # Megatron-LM hooks from NVIDIA/Megatron-LM#7015.
        from megatron.core.inference.inference_request import (
            RequestPromptPreparationResult,
        )

        if offload_params is None:
            return RequestPromptPreparationResult(prompt=prompt)
        capture_payload = offload_params.get("ng_capture")
        if capture_payload is None:
            return RequestPromptPreparationResult(
                prompt=prompt, offload_params=offload_params
            )

        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.staging.records import CaptureAdmission

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
        updated_offload_params["ng_capture"] = updated_admission.model_dump(mode="json")
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


def slice_media_tensors(
    media_tensors: dict[str, Any] | None, prev_count: int
) -> dict[str, Any] | None:
    """Drop the first ``prev_count`` media items from the engine's media tensors.

    Every chat request carries the whole conversation, so the engine hands the
    stager pixels for every image in the prompt. The parent chain already
    staged the first ``prev_count`` of them; this keeps only the rest so media
    columns are per-call deltas like the token columns.

    Item boundaries come from the tensors themselves: ``num_frames`` (frames per
    video) or ``num_tiles`` (tiles per image) when present, else one row of
    ``imgs_sizes`` per image. For packed patches (``imgs`` as
    ``[1, total_patches, C*P*P]``) the patch count per row is ``h*w/P**2`` with
    ``P**2`` recovered from the totals.
    """
    if not media_tensors or prev_count <= 0:
        return media_tensors
    imgs = media_tensors.get("imgs")
    imgs_sizes = media_tensors.get("imgs_sizes")
    num_frames = media_tensors.get("num_frames")
    num_tiles = media_tensors.get("num_tiles")
    if imgs is None:
        return media_tensors
    if imgs_sizes is None and num_tiles is None:
        raise ValueError("media delta requires imgs_sizes or num_tiles to locate items")

    if num_frames is not None:
        total_items = int(num_frames.numel())
    elif num_tiles is not None:
        total_items = int(num_tiles.numel())
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
    elif num_tiles is not None:
        prev_rows = int(num_tiles.reshape(-1)[:prev_count].sum().item())
    else:
        prev_rows = prev_count

    sliced: dict[str, Any] = {}
    if imgs.ndim == 3 and imgs.shape[0] == 1 and imgs_sizes is not None:
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
        # Padded pixels [N, C, H, W] (or tiles): one row per frame / tile.
        sliced["imgs"] = imgs[prev_rows:]
    if imgs_sizes is not None:
        sliced["imgs_sizes"] = imgs_sizes.reshape(-1, 2)[prev_rows:]
    if num_frames is not None:
        sliced["num_frames"] = num_frames.reshape(-1)[prev_count:]
    if num_tiles is not None:
        sliced["num_tiles"] = num_tiles.reshape(-1)[prev_count:]
    return sliced


def media_geometry(media_tensors: dict[str, Any] | None) -> dict[str, Any] | None:
    """Gym's digest-covered media geometry, read off the engine's media tensors.

    ``imgs_sizes`` / ``num_frames`` / ``num_tiles`` become plain int lists; the
    modality is ``video`` when the engine carried frame counts. The finalizer
    checks the staged media columns against this before publishing a row.
    """
    if not media_tensors or media_tensors.get("imgs") is None:
        return None

    def as_list(name: str) -> list | None:
        tensor = media_tensors.get(name)
        return None if tensor is None else tensor.detach().cpu().tolist()

    num_frames = as_list("num_frames")
    return {
        "modality": "video" if num_frames is not None else "image",
        "imgs_sizes": as_list("imgs_sizes"),
        "num_frames": num_frames,
        "num_tiles": as_list("num_tiles"),
    }


def media_field_dict(media_tensors: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Encode the engine's media tensors as the media columns of one call row.

    The encoder half of the media column codec; ``row_to_media_tensors`` is
    its inverse. Each present tensor is flattened to one ``[1, numel]`` row
    under its ``MEDIA_TENSOR_COLUMNS`` field, absent tensors get an int64
    ``zeros((1, 1))`` sentinel (jagged columns cannot be empty), and
    ``MEDIA_GEOMETRY_FIELD`` records shape and dtype per present name.

    Args:
        media_tensors: The engine's media tensors keyed by
            ``MEDIA_TENSOR_COLUMNS`` name; ``imgs`` is mandatory.

    Returns:
        The field dict to ``put`` onto the call's staging key.

    Raises:
        ValueError: ``imgs`` is missing / empty, or a key is not a known
            media tensor name.
        TypeError: A present value is not a ``torch.Tensor``.
    """
    imgs = media_tensors.get("imgs")
    if not isinstance(imgs, torch.Tensor) or imgs.numel() == 0:
        raise ValueError("media_tensors must carry a non-empty imgs tensor")
    unknown = sorted(set(media_tensors) - set(MEDIA_TENSOR_COLUMNS))
    if unknown:
        raise ValueError(f"unsupported media tensors: {unknown}")
    geometry: dict[str, Any] = {}
    field_dict: dict[str, torch.Tensor] = {}
    for name, column in MEDIA_TENSOR_COLUMNS.items():
        tensor = media_tensors.get(name)
        if tensor is None:
            # Sentinel row: jagged columns cannot be empty; absence is
            # recorded by the missing geometry entry.
            field_dict[column] = torch.zeros((1, 1), dtype=torch.int64)
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"media tensor {name!r} must be a torch.Tensor")
        flat = tensor.detach().cpu().contiguous().reshape(1, -1)
        field_dict[column] = flat
        geometry[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
    field_dict[MEDIA_GEOMETRY_FIELD] = _bytes_tensor(
        json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return field_dict


@dataclass(frozen=True)
class _MegatronCapturePayload:
    """The MInf offloaded payload plus the worker-side context Gym's adapter reads."""

    prompt_token_ids: Any
    generated_token_ids: Any
    generated_log_probs: Any
    compact_prompt_token_ids: Any
    media: dict[str, Any] | None
    compact_prev_len: int
    # The engine's media tensors minus what the parent chain already staged.
    media_tensors: dict[str, Any] | None

    @classmethod
    def from_offloaded(
        cls, payload: Any, minf_params: Any
    ) -> "_MegatronCapturePayload":
        def _count(key: str) -> int:
            value = minf_params.get(key) if isinstance(minf_params, dict) else None
            if value is None:
                return 0
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"MInf capture request carries an invalid {key}: {value!r}"
                )
            return value

        media_tensors = slice_media_tensors(
            getattr(payload, "media_tensors", None), _count(MEDIA_PREV_COUNT_KEY)
        )
        return cls(
            prompt_token_ids=getattr(payload, "prompt_token_ids", None),
            generated_token_ids=getattr(payload, "generated_token_ids", None),
            generated_log_probs=getattr(payload, "generated_log_probs", None),
            compact_prompt_token_ids=getattr(payload, "compact_prompt_token_ids", None),
            media=media_geometry(media_tensors),
            compact_prev_len=_count(COMPACT_PREV_LEN_KEY),
            media_tensors=media_tensors,
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

        self._sink = sink
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
    ) -> MegatronPayloadStageResult | None:
        """Stage an admitted request, or decline ordinary non-capture traffic."""
        if not isinstance(uid, str) or not uid:
            raise ValueError("MInf request UID must be a non-empty string")
        capture_payload = (offload_params or {}).get("ng_capture")
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
    ) -> MegatronPayloadStageResult:
        """Validate and stage traffic that carries a Gym capture admission."""
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.staging.records import CaptureAdmission

        admission = CaptureAdmission.model_validate(capture_payload)
        call = self._capture.begin_call(
            admission,
            weight_version=self._weight_version(finished_metadata),
        )
        # Gym's MegatronCaptureAdapter reads prompt/generated ids and log
        # probs off the offloaded payload, plus the multimodal material
        # (compact prompt, media summary, and the compact length of the
        # spliced chain the preparer recorded). A malformed payload poisons
        # the call with ``capture_failed`` coordinates (surfacing in Gym as
        # ``worker_capture_failed``, matching vLLM) instead of raising here,
        # which would leave Gym with no coordinates at all.
        capture_payload_view = _MegatronCapturePayload.from_offloaded(
            payload, minf_params
        )
        coords = self._capture.complete_call_from_response(call, capture_payload_view)
        # Media tensors ride the same row as extra columns once the token row
        # is durable. They are outside Gym's digest; the digest-covered geometry
        # in extras (media_geometry above) names them, and the finalizer checks
        # the columns against it before publishing.
        media_tensors = capture_payload_view.media_tensors
        if coords.disposition == "staged" and media_tensors:
            try:
                self._sink.stage_media(coords.staging_key, media_tensors)
            except Exception:  # noqa: BLE001 — finalizer rejects media_columns_missing
                logging.getLogger(__name__).exception(
                    "MInf media staging failed for %s", coords.staging_key
                )
        return MegatronPayloadStageResult(
            response_metadata={
                "ng_commit_coords": coords.model_dump(mode="json"),
            }
        )


class TQTokenSource:
    """Gym ``StagingSource`` over ``DataPlaneClient.get_samples``.

    All requested rows are fetched in a single batched ``get_samples`` call
    (TQ returns jagged delta columns as nested tensors; ``_from_wire``
    preserves the raggedness), in the order requested. A missing or
    unreadable row raises ``KeyError`` per the protocol — the finalizer maps
    that to a placeholder, never a silent skip. TQ's field-readiness check
    is all-or-nothing across a batch, so the extras fallback is batch-level:
    extras-free runs land in the base schema exactly like the old per-key
    probe, but a batch with *mixed* extras presence degrades every row to
    the base schema (worker feature-gating makes presence uniform per run).
    """

    def __init__(self, dp_client: Any, *, staging_partition: str) -> None:
        self._store = TQStagingStore(dp_client, staging_partition=staging_partition)
        self._staging_partition = staging_partition

    def fetch(self, staging_keys: list[str]) -> list[StagedCallBaseSnapshot]:
        """Gym ``StagingSource`` conformance: base snapshots only, in order."""
        return [item.snapshot for item in self.fetch_for_finalization(staging_keys)]

    def fetch_prefix_token_ids(self, staging_keys: list[str]) -> list[int]:
        """Bulk-fetch ordered delta chain and concatenate token_ids_delta into a prefix."""
        return self.fetch_prefix_chains(staging_keys).expanded

    def fetch_prefix_chains(self, staging_keys: list[str]) -> PrefixChains:
        """Bulk-fetch the ordered delta chain in both token spaces.

        The compact chain uses each row's ``compact_token_ids_delta`` when the
        call staged one (``compact_len > 0``) and its ``token_ids_delta``
        otherwise, so text calls contribute the same ids to both chains.
        """
        if not staging_keys:
            return PrefixChains(expanded=[], compact=[])
        if len(set(staging_keys)) != len(staging_keys):
            raise KeyError("prefix fetch: staging_keys contains duplicates")
        try:
            rows = self._store.get(
                staging_keys,
                select_fields=[
                    "token_ids_delta",
                    COMPACT_TOKEN_IDS_FIELD,
                    COMPACT_LEN_FIELD,
                    ROUTED_EXTRAS_METADATA_FIELD,
                ],
            )
        except Exception as error:  # noqa: BLE001 — protocol maps any miss to KeyError
            raise KeyError(
                f"prefix fetch: staged rows for {len(staging_keys)} keys could "
                f"not be fetched from {self._staging_partition!r}: {error}"
            ) from error
        n_rows = int(rows.batch_size[0]) if rows.batch_size else 0
        if n_rows != len(staging_keys):
            raise KeyError(
                f"prefix fetch incomplete: requested {len(staging_keys)} keys, got {n_rows}"
            )
        expanded: list[int] = []
        compact: list[int] = []
        media_count = 0
        for index in range(n_rows):
            row = _select_row(rows, index)
            delta = [int(t) for t in row["token_ids_delta"].squeeze(0).tolist()]
            expanded.extend(delta)
            media_count += media_item_count(
                (_row_extras(row) or {}).get(MEDIA_EXTRAS_KEY)
            )
            compact_len = _row_scalar_int(row, COMPACT_LEN_FIELD)
            if compact_len > 0:
                compact_delta = [
                    int(t) for t in row[COMPACT_TOKEN_IDS_FIELD].squeeze(0).tolist()
                ]
                if len(compact_delta) != compact_len:
                    raise ValueError(
                        f"compact_token_ids_delta length {len(compact_delta)} does not "
                        f"match compact_len {compact_len}"
                    )
                compact.extend(compact_delta)
            else:
                compact.extend(delta)
        return PrefixChains(expanded=expanded, compact=compact, media_count=media_count)

    def fetch_media(self, staging_key: str) -> StagedMediaTensors:
        """Read one call's media columns; ``KeyError`` when the row carries none."""
        try:
            rows = self._store.get([staging_key], select_fields=MEDIA_STAGING_FIELDS)
        except Exception as error:  # noqa: BLE001 — protocol maps misses to KeyError
            raise KeyError(
                f"media columns for {staging_key!r} could not be fetched from "
                f"{self._staging_partition!r}: {error}"
            ) from error
        n_rows = int(rows.batch_size[0]) if len(rows.batch_size) else 0
        if n_rows != 1:
            raise KeyError(f"media columns for {staging_key!r} missing")
        return row_to_media_tensors(_select_row(rows, 0))

    def fetch_for_finalization(
        self,
        staging_keys: list[str],
        *,
        include_route_fragments: bool = False,
    ) -> list[FetchedStagedCall]:
        """Fetch digest-covered base columns, plus route payloads when requested.

        Deferred mode (the default) never selects ``routed_experts`` — route
        bytes stay in TQ for the policy worker. Direct mode passes
        ``include_route_fragments=True`` to pull the payloads in the same
        batched read and receives them as ``RouteFragment`` values beside the
        base snapshots, never inside them.
        """
        if not staging_keys:
            return []
        if len(set(staging_keys)) != len(staging_keys):
            raise KeyError("finalization staging request contains duplicate keys")
        try:
            if include_route_fragments:
                # Route payloads are optional per run (feature-gated at the
                # worker); fall back to the base schema so extras-free rows
                # keep fetching.
                try:
                    rows = self._store.get(
                        staging_keys,
                        select_fields=STAGING_FIELDS + [ROUTED_EXPERTS_FIELD],
                    )
                except Exception:  # noqa: BLE001 — field-not-present probe
                    rows = self._store.get(staging_keys, select_fields=STAGING_FIELDS)
            else:
                rows = self._store.get(staging_keys, select_fields=STAGING_FIELDS)
        except Exception as error:  # noqa: BLE001 — protocol maps misses to KeyError
            raise KeyError(
                f"staged rows for {len(staging_keys)} keys could not be "
                f"fetched from {self._staging_partition!r}: {error}"
            ) from error
        # TQ's kv path only errors when *zero* keys resolve; a partial miss
        # returns fewer rows with no error. Guard explicitly so a lost row
        # rejects the rollout as missing_staging_row instead of surfacing
        # later as a misleading digest mismatch from misaligned zipping.
        n_rows = int(rows.batch_size[0]) if len(rows.batch_size) else 0
        if n_rows != len(staging_keys):
            raise KeyError(
                f"staged rows missing: requested {len(staging_keys)} keys "
                f"from {self._staging_partition!r}, got {n_rows} rows"
            )
        # Row order mirrors the requested key order; digest recomputation at
        # snapshot validation is the byte-exact backstop if that ever breaks.
        fetched: list[FetchedStagedCall] = []
        for index, key in enumerate(staging_keys):
            row = _select_row(rows, index)
            snapshot = _row_to_base_snapshot(row)
            if snapshot.staging_key != key:
                raise KeyError(
                    f"staged row identity mismatch: requested {key!r}, got {snapshot.staging_key!r}"
                )
            fetched.append(
                FetchedStagedCall(
                    staging_key=key,
                    snapshot=snapshot,
                    routed_len=_row_scalar_int(row, ROUTED_LEN_FIELD),
                    fragment=(
                        _row_to_route_fragment(row) if include_route_fragments else None
                    ),
                    extras=_row_extras(row),
                )
            )
        return fetched


def _select_row(rows: TensorDict, index: int) -> dict[str, torch.Tensor]:
    """Slice one row out of a batched fetch, restoring single-row shapes.

    ``_row_to_base_snapshot`` predates batching and expects each field with a
    leading batch dim of 1 (the shape a single-key ``get_samples`` returns),
    so re-add it after indexing. Indexing a nested tensor yields that row's
    dense component, which is exactly the jagged-row payload.
    """
    row: dict[str, torch.Tensor] = {}
    for field in rows.keys():
        value = rows.get(field)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"staging field {field!r} must be a tensor, got {type(value).__name__}"
            )
        row[str(field)] = value[index].unsqueeze(0)
    return row


def _row_leaf(row: Any, name: str) -> torch.Tensor:
    value = row[name]
    tensor = value[0] if value.dim() > 1 or value.numel() > 1 else value
    return tensor.reshape(-1)


def _row_text(row: Any, name: str) -> str:
    return bytes(int(value) for value in _row_leaf(row, name).tolist()).decode("utf-8")


def _row_to_base_snapshot(row: Any) -> StagedCallBaseSnapshot:
    """Rebuild one normally validated base snapshot; route bytes never enter it."""
    # Deferred: nemo_gym is an optional extra absent in non-gym runs.
    from nemo_gym.token_id_capture.staging.records import StagedCallBaseSnapshot

    def _digest(name: str) -> str:
        value = bytes(int(item) for item in _row_leaf(row, name).tolist())
        if len(value) != 32:
            raise ValueError(f"{name} must contain exactly 32 bytes")
        return value.hex()

    def _optional_digest(name: str, present_name: str) -> str | None:
        return _digest(name) if bool(_row_leaf(row, present_name)[0].item()) else None

    parent_call_id = (
        _row_text(row, "parent_call_id_utf8")
        if bool(_row_leaf(row, "parent_call_id_present")[0].item())
        else None
    )
    mode_code = int(_row_leaf(row, "capture_mode")[0].item())
    try:
        mode = _CODE_TO_MODE[mode_code]
    except KeyError as error:
        raise ValueError(f"unknown capture_mode code {mode_code}") from error
    routed_encoding = int(_row_leaf(row, ROUTED_EXPERTS_ENCODING_FIELD)[0].item())
    if routed_encoding not in (
        ROUTE_ENCODING_NONE,
        ROUTE_ENCODING_ENVELOPE,
        ROUTE_ENCODING_LIST,
    ):
        raise ValueError(f"unknown routed_experts_encoding {routed_encoding}")

    return StagedCallBaseSnapshot(
        schema_version=int(_row_leaf(row, "schema_version")[0].item()),
        digest_version=int(_row_leaf(row, "digest_version")[0].item()),
        extras_digest_version=int(_row_leaf(row, "extras_digest_version")[0].item()),
        rollout_id=_row_text(row, "rollout_id_utf8"),
        model_call_id=_row_text(row, "model_call_id_utf8"),
        parent_call_id=parent_call_id,
        mode=mode,
        prev_len=int(_row_leaf(row, "prev_len")[0].item()),
        delta_len=int(_row_leaf(row, "delta_len")[0].item()),
        cum_len=int(_row_leaf(row, "cum_len")[0].item()),
        weight_version=int(_row_leaf(row, "weight_version")[0].item()),
        digest=_digest("digest_bytes"),
        token_ids_delta=[int(t) for t in _row_leaf(row, "token_ids_delta").tolist()],
        token_mask_delta=[
            float(m) for m in _row_leaf(row, "token_mask_delta").tolist()
        ],
        generation_log_probs_delta=[
            float(p) for p in _row_leaf(row, "generation_logprobs_delta").tolist()
        ],
        extras_digest=_digest("extras_digest_bytes"),
        chain_hash=_optional_digest("chain_hash_bytes", "chain_hash_present"),
        cumulative_hash=_optional_digest(
            "cumulative_hash_bytes", "cumulative_hash_present"
        ),
    )


def _row_extras(row: Any) -> dict[str, Any] | None:
    """Decode the staged extras JSON (None for ``null`` / absent extras)."""
    decoded = json.loads(_row_text(row, ROUTED_EXTRAS_METADATA_FIELD))
    if decoded is None:
        return None
    if not isinstance(decoded, dict):
        raise ValueError("staged extras metadata must be a JSON object or null")
    return decoded


def _row_to_route_fragment(row: Any) -> RouteFragment | None:
    """Extract one staged route payload beside (never inside) the snapshot."""
    routed_encoding = int(_row_leaf(row, ROUTED_EXPERTS_ENCODING_FIELD)[0].item())
    if routed_encoding == ROUTE_ENCODING_NONE:
        return None
    try:
        routed = row[ROUTED_EXPERTS_FIELD]
    except KeyError as error:
        raise KeyError(
            "staged row metadata names routed_experts but its field is absent"
        ) from error
    experts = routed[0] if routed.dim() > 3 or routed.shape[0] == 1 else routed
    return RouteFragment(
        routes=experts,
        encoding=routed_encoding,
        extras_metadata_json=_row_text(row, ROUTED_EXTRAS_METADATA_FIELD).encode(
            "utf-8"
        ),
    )


def row_to_media_tensors(row: Any) -> StagedMediaTensors:
    """Restore one call row's media columns to the engine's media tensors.

    The decoder half of the media column codec; ``media_field_dict`` is its
    inverse. ``MEDIA_GEOMETRY_FIELD`` names which tensors are present and
    their shape / dtype; each named ``MEDIA_TENSOR_COLUMNS`` column is cast
    and reshaped accordingly, unnamed ones (sentinel rows) come back ``None``.

    Args:
        row: One row as returned by ``_select_row``, holding
            ``MEDIA_STAGING_FIELDS``.

    Returns:
        The staged media tensors in their original shapes.

    Raises:
        ValueError: The geometry does not describe ``imgs``, names an unknown
            dtype, or disagrees with a column's element count.
    """
    geometry = json.loads(_row_text(row, MEDIA_GEOMETRY_FIELD))
    if not isinstance(geometry, dict) or "imgs" not in geometry:
        raise ValueError("staged media geometry must describe imgs")
    tensors: dict[str, torch.Tensor | None] = {}
    for name, column in MEDIA_TENSOR_COLUMNS.items():
        spec = geometry.get(name)
        if spec is None:
            tensors[name] = None
            continue
        flat = row[column].reshape(-1)
        dtype = getattr(torch, spec["dtype"], None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"media column {column!r} names unknown dtype {spec['dtype']!r}"
            )
        shape = torch.Size(spec["shape"])
        if flat.numel() != shape.numel():
            raise ValueError(
                f"media column {column!r} holds {flat.numel()} values but its "
                f"geometry describes {shape.numel()}"
            )
        tensors[name] = flat.to(dtype).reshape(shape)
    assert tensors["imgs"] is not None
    return StagedMediaTensors(
        imgs=tensors["imgs"],
        imgs_sizes=tensors["imgs_sizes"],
        num_frames=tensors["num_frames"],
        num_tiles=tensors["num_tiles"],
    )


def _row_scalar_int(row: Any, field_name: str) -> int:
    """Read one required scalar from a single-row TQ result."""
    value = row[field_name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"staging field {field_name!r} must be a tensor, got {type(value).__name__}"
        )
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if value.dtype not in integer_dtypes:
        raise TypeError(
            f"staging field {field_name!r} must use an integer dtype, got {value.dtype}"
        )
    tensor = value[0] if value.dim() > 1 or value.numel() > 1 else value
    flattened = tensor.reshape(-1)
    if flattened.numel() != 1:
        raise ValueError(
            f"staging field {field_name!r} must contain one scalar, got "
            f"shape {tuple(value.shape)}"
        )
    return int(flattened[0].item())
