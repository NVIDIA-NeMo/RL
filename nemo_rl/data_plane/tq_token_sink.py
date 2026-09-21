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
Inference workers write canonical Gym call deltas through ``TQTokenSink``.
This module is the only hot-path file that knows tokens live in TQ; Gym sees
opaque staging keys.

Each staged row carries three jagged columns (``token_ids_delta``,
``token_mask_delta``, ``generation_logprobs_delta``), the complete receipt
identity/lineage metadata, and all digest inputs so it round-trips to a
normally validated ``StagedCallBaseSnapshot``. On media-enabled partitions the
same ``put`` also carries the processed media the engine ran on (see
``MEDIA_STAGING_FIELDS``), so ``staged`` coordinates acknowledge tokens and
pixels together. Masks/logprobs are float32 on
the wire, matching ``compute_staging_digest``'s float32-bit-pattern scheme, so
digest recomputation over fetched values is byte-exact. Route payloads never
ride inside snapshots: the source returns them as separate ``RouteFragment``
values keyed by staging key, digest-verified by the plan executor at point of
use.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ray
import torch
from tensordict import TensorDict

if TYPE_CHECKING:
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

# These names come from nemo_gym.token_id_capture.staging.records.StagedCallRecord,
# transformed by stage() below. Adding a field means editing both this list and
# stage(); a mismatch is caught by test_tq_sink_source_passes_conformance's
# round-trip equality check -- but only for required StagedCallRecord fields. An
# optional field Gym adds that this sink never stages will default identically
# on both sides and pass that check silently.
# Media the engine's vision encoder consumed, staged in the *same* put as the
# token columns (``TQTokenSink.stage`` with attachments). Every row of a
# media-enabled staging partition carries all of these columns: two bool flags
# and three tensor columns. Tensors keep their native shape and dtype on the
# wire (TQ adds its row dimension): ``media_imgs`` is ``[total_patches, 3*P*P]``
# per row in the engine's float dtype, ``media_imgs_sizes`` ``[N, 2]`` int32,
# ``media_num_frames`` ``[N_videos]`` int32. Rows without media (text calls,
# continuations with no new media) and stills without frame counts write a
# ``[1]`` int64 sentinel in place of the tensor; ``media_present`` /
# ``media_has_frames`` say which columns carry real data, so the finalizer
# never batch-reads a sentinel beside a real tensor (one nested column needs
# one dtype). Text-only partitions register none of these columns.
MEDIA_PRESENT_FIELD = "media_present"
MEDIA_HAS_FRAMES_FIELD = "media_has_frames"
MEDIA_IMGS_FIELD = "media_imgs"
MEDIA_IMGS_SIZES_FIELD = "media_imgs_sizes"
MEDIA_NUM_FRAMES_FIELD = "media_num_frames"
MEDIA_TENSOR_COLUMNS: dict[str, str] = {
    "imgs": MEDIA_IMGS_FIELD,
    "imgs_sizes": MEDIA_IMGS_SIZES_FIELD,
    "num_frames": MEDIA_NUM_FRAMES_FIELD,
}
MEDIA_FLAG_FIELDS = [MEDIA_PRESENT_FIELD, MEDIA_HAS_FRAMES_FIELD]
MEDIA_STAGING_FIELDS = [*MEDIA_FLAG_FIELDS, *MEDIA_TENSOR_COLUMNS.values()]
_MEDIA_REQUIRED = ("imgs", "imgs_sizes")
_MEDIA_PIXEL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MEDIA_INDEX_DTYPES = (torch.int32, torch.int64)


STAGING_FIELDS = [
    "token_ids_delta",
    "token_mask_delta",
    "generation_logprobs_delta",
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
    """One staged call's validated media bundle, in the engine's native shapes.

    ``imgs`` is ``[1, total_patches, 3*P*P]`` packed patches, ``imgs_sizes``
    ``[N, 2]`` per-frame ``[height, width]``, ``num_frames`` ``[N_videos]``
    frame counts partitioning ``imgs_sizes`` (``None`` for still images).
    """

    imgs: torch.Tensor
    imgs_sizes: torch.Tensor
    num_frames: torch.Tensor | None

    @property
    def patch_size(self) -> int:
        return int(math.isqrt(int(self.imgs.shape[-1]) // 3))


def _sentinel() -> torch.Tensor:
    """Placeholder for an absent media tensor: TQ rows cannot be empty."""
    return torch.zeros((1,), dtype=torch.int64)


def validate_media_tensors(
    attachments: Mapping[str, Any] | None,
) -> StagedMediaTensors | None:
    """Check a media bundle against the Omni capture contract.

    Shared by the sink (before any write) and the source (after every read),
    so malformed geometry is rejected at both ends instead of surfacing as a
    reshape error in the learner. ``None`` (no attachments) returns ``None``;
    anything else must be a mapping with ``imgs`` and ``imgs_sizes`` and an
    optional ``num_frames``. Raises ``TypeError`` / ``ValueError``; supported
    dtypes are preserved, never cast.
    """
    if attachments is None:
        return None
    if not isinstance(attachments, Mapping):
        raise TypeError(
            f"media attachments must be a mapping, got {type(attachments).__name__}"
        )
    if not attachments:
        raise ValueError("media attachments must not be empty")
    unknown = sorted(set(attachments) - set(MEDIA_TENSOR_COLUMNS))
    if unknown:
        raise ValueError(f"unsupported media attachments: {unknown}")
    for name in _MEDIA_REQUIRED:
        if attachments.get(name) is None:
            raise ValueError(f"media attachments require {name!r}")
    for name, value in attachments.items():
        if value is None and name not in _MEDIA_REQUIRED:
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"media attachment {name!r} must be a torch.Tensor, got "
                f"{type(value).__name__}"
            )

    imgs: torch.Tensor = attachments["imgs"]
    if imgs.dtype not in _MEDIA_PIXEL_DTYPES:
        raise ValueError(
            f"media imgs must be float16/bfloat16/float32, got {imgs.dtype}"
        )
    if imgs.ndim != 3 or imgs.shape[0] != 1 or imgs.shape[1] == 0:
        raise ValueError(
            "media imgs must be [1, total_patches, 3*P*P] with total_patches > 0, "
            f"got shape {tuple(imgs.shape)}"
        )
    feature = int(imgs.shape[2])
    if feature <= 0 or feature % 3:
        raise ValueError(f"media imgs feature dim {feature} is not 3*P*P")
    patch_size = math.isqrt(feature // 3)
    if patch_size <= 0 or 3 * patch_size * patch_size != feature:
        raise ValueError(f"media imgs feature dim {feature} is not a square RGB patch")

    sizes: torch.Tensor = attachments["imgs_sizes"]
    if sizes.dtype not in _MEDIA_INDEX_DTYPES:
        raise ValueError(f"media imgs_sizes must be int32/int64, got {sizes.dtype}")
    if sizes.ndim != 2 or sizes.shape[1] != 2 or sizes.shape[0] == 0:
        raise ValueError(
            f"media imgs_sizes must be [N, 2] with N > 0, got shape {tuple(sizes.shape)}"
        )
    _check_int32_positive(sizes, "imgs_sizes")
    sizes64 = sizes.to(torch.int64)
    if bool((sizes64 % patch_size).any()):
        raise ValueError(
            f"media imgs_sizes must be divisible by the patch size {patch_size}"
        )
    total_patches = int((sizes64[:, 0] * sizes64[:, 1]).sum().item()) // (
        patch_size * patch_size
    )
    if total_patches != int(imgs.shape[1]):
        raise ValueError(
            f"media imgs_sizes describe {total_patches} patches but imgs holds "
            f"{int(imgs.shape[1])}"
        )

    num_frames = attachments.get("num_frames")
    if num_frames is not None:
        if num_frames.dtype not in _MEDIA_INDEX_DTYPES:
            raise ValueError(
                f"media num_frames must be int32/int64, got {num_frames.dtype}"
            )
        if num_frames.ndim != 1 or num_frames.numel() == 0:
            raise ValueError(
                f"media num_frames must be a nonempty 1-D tensor, got shape "
                f"{tuple(num_frames.shape)}"
            )
        _check_int32_positive(num_frames, "num_frames")
        if int(num_frames.to(torch.int64).sum().item()) != int(sizes.shape[0]):
            raise ValueError(
                f"media num_frames sum {int(num_frames.sum().item())} does not "
                f"partition the {int(sizes.shape[0])} imgs_sizes rows"
            )
    return StagedMediaTensors(imgs=imgs, imgs_sizes=sizes, num_frames=num_frames)


def _check_int32_positive(tensor: torch.Tensor, name: str) -> None:
    if bool((tensor <= 0).any()) or bool((tensor > torch.iinfo(torch.int32).max).any()):
        raise ValueError(f"media {name} values must be positive and fit in int32")


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
    # the call staged no extras. Carries vLLM's ``media_spans`` for VLM calls.
    extras: dict[str, Any] | None = None
    # Media presence flags read with the base columns (always False on a
    # text-only partition). ``fetch_media`` reads the tensors for rows whose
    # ``media_present`` is True; ``media_has_frames`` selects the video shape.
    media_present: bool = False
    media_has_frames: bool = False


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
            fields=TensorDict(
                {name: tensor for name, tensor in field_dict.items()}, batch_size=[1]
            ),
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

    ``capture_media`` mirrors the staging partition's schema: a media-enabled
    partition registers ``MEDIA_STAGING_FIELDS`` and every row written here
    carries them (flags False + sentinels for text calls); a text-only
    partition rejects attachments outright.
    """

    def __init__(
        self, dp_client: Any, *, staging_partition: str, capture_media: bool = False
    ) -> None:
        self._store = TQStagingStore(dp_client, staging_partition=staging_partition)
        self._capture_media = capture_media

    def stage(
        self,
        record: StagedCallRecord,
        *,
        attachments: Mapping[str, Any] | None = None,
    ) -> StageResult:
        """Write the token row and its media attachments in one ``put``.

        Success is returned only after the combined write was acknowledged,
        so ``staged`` coordinates vouch for tokens and pixels together. TQ has
        no transactional rollback: if the write raises, the attempted key is
        discarded best-effort before the failure is reported (see
        ``_discard_failed_write``).
        """
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.staging.records import StageResult

        key = record.staging_key
        write_started = False
        try:
            if attachments is not None and not self._capture_media:
                raise ValueError(
                    "media attachments require a media-enabled staging partition"
                )
            media = validate_media_tensors(attachments)
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
            if self._capture_media:
                field_dict.update(_media_columns(media))
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
            write_started = True
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
            if write_started:
                self._discard_failed_write(key)
            return StageResult(
                ok=False, staging_key=key, error=f"{type(error).__name__}: {error}"
            )
        return StageResult(ok=True, staging_key=key)

    def _discard_failed_write(self, key: str) -> None:
        """Reclaim whatever a failed combined write may have left behind.

        TQ writes a row field by field and only then publishes readiness, so
        a raise mid-write can leave partial field keys with no logical row.
        Gym's ``capture_failed`` coordinates carry no staging key, so the
        finalizer never learns about this key; the sink is the only owner
        able to clean it. The discard is idempotent (clearing an unknown key
        is a no-op). If it fails too the storage state is uncertain and is
        logged at ERROR for the operator; the call is still reported failed.
        """
        try:
            self._store.clear([key])
        except Exception as error:  # noqa: BLE001 — cleanup must not mask the stage failure
            logging.getLogger(__name__).error(
                "TQTokenSink could not discard the failed write for %s: %s: %s; "
                "the staging partition may retain orphaned field keys",
                key,
                type(error).__name__,
                error,
            )

    def clear(self, staging_keys: list[str]) -> None:
        """Drop staged rows (finalizer / eviction cleanup)."""
        self._store.clear(staging_keys)


def _media_columns(media: StagedMediaTensors | None) -> dict[str, torch.Tensor]:
    """Encode one row's media columns for a media-enabled partition.

    Tensors are written with TQ's row dimension prepended and otherwise native:
    ``imgs`` drops its own leading 1 so the row is ``[total_patches, F]`` and
    the patch dim is the row's leading (ragged) dim, exactly like
    ``token_ids_delta`` / ``routed_experts``, which is what a batched nested
    read requires. ``fetch_media`` restores ``[1, total_patches, F]``.
    """
    present = media is not None
    has_frames = present and media.num_frames is not None
    columns: dict[str, torch.Tensor] = {
        MEDIA_PRESENT_FIELD: torch.tensor([present], dtype=torch.bool),
        MEDIA_HAS_FRAMES_FIELD: torch.tensor([has_frames], dtype=torch.bool),
    }
    tensors: dict[str, torch.Tensor | None] = (
        {"imgs": None, "imgs_sizes": None, "num_frames": None}
        if media is None
        else {
            "imgs": media.imgs.reshape(media.imgs.shape[1], media.imgs.shape[2]),
            "imgs_sizes": media.imgs_sizes,
            "num_frames": media.num_frames,
        }
    )
    for name, column in MEDIA_TENSOR_COLUMNS.items():
        tensor = tensors[name]
        if tensor is None:
            # Only absent media or optional frame counts use sentinels; a
            # missing required tensor was rejected by validate_media_tensors.
            columns[column] = _sentinel().unsqueeze(0)
        else:
            columns[column] = tensor.detach().cpu().contiguous().unsqueeze(0)
    return columns


def slice_media_tensors(
    media_tensors: dict[str, Any] | None, prev_count: int
) -> dict[str, Any] | None:
    """Drop the first ``prev_count`` media items from the engine's media tensors.

    Every chat request carries the whole conversation, so the engine hands the
    stager pixels for every image in the prompt. The parent chain already
    staged the first ``prev_count`` of them; this keeps only the rest so media
    columns are per-call deltas like the token columns.

    Item boundaries come from the tensors themselves: ``num_frames`` (frames per
    video) when present, else one row of ``imgs_sizes`` per image. For packed
    patches (``imgs`` as ``[1, total_patches, C*P*P]``) the patch count per row
    is ``h*w/P**2`` with ``P**2`` recovered from the totals.
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
        # Padded pixels [N, C, H, W]: one row per frame.
        sliced["imgs"] = imgs[prev_rows:]
    sliced["imgs_sizes"] = imgs_sizes.reshape(-1, 2)[prev_rows:]
    if num_frames is not None:
        sliced["num_frames"] = num_frames.reshape(-1)[prev_count:]
    return sliced


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

    def __init__(
        self, dp_client: Any, *, staging_partition: str, capture_media: bool = False
    ) -> None:
        self._store = TQStagingStore(dp_client, staging_partition=staging_partition)
        self._staging_partition = staging_partition
        # Mirrors the partition schema: only a media-enabled partition has the
        # flag/tensor columns, so selection is gated rather than probed.
        self._capture_media = capture_media

    def fetch(self, staging_keys: list[str]) -> list[StagedCallBaseSnapshot]:
        """Gym ``StagingSource`` conformance: base snapshots only, in order."""
        return [item.snapshot for item in self.fetch_for_finalization(staging_keys)]

    def fetch_prefix_token_ids(self, staging_keys: list[str]) -> list[int]:
        """Bulk-fetch ordered delta chain and concatenate token_ids_delta into a prefix."""
        if not staging_keys:
            return []
        if len(set(staging_keys)) != len(staging_keys):
            raise KeyError("prefix fetch: staging_keys contains duplicates")
        try:
            rows = self._store.get(
                list(staging_keys), select_fields=["token_ids_delta"]
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
        result: list[int] = []
        for index in range(n_rows):
            row = _select_row(rows, index)
            delta = row["token_ids_delta"].squeeze(0).tolist()
            result.extend(int(t) for t in delta)
        return result

    def fetch_media(self, items: list[FetchedStagedCall]) -> list[StagedMediaTensors]:
        """One batched read of the media tensor columns for rows known to carry media.

        ``items`` come from ``fetch_for_finalization`` (their flags were read
        with the base columns) and must all have ``media_present=True`` and
        the same ``media_has_frames``: a nested column needs one dtype, and a
        sentinel row (int64) cannot batch beside a real tensor (bf16 / int32).
        Results are returned in request order. A transport miss raises
        ``KeyError``; malformed columns raise ``TypeError`` / ``ValueError``.
        """
        if not self._capture_media:
            raise ValueError("media reads require a media-enabled staging source")
        if not items:
            return []
        if any(not item.media_present for item in items):
            raise ValueError("fetch_media only accepts rows with media_present=True")
        if len({item.media_has_frames for item in items}) != 1:
            raise ValueError("fetch_media requires uniform media_has_frames")
        has_frames = items[0].media_has_frames
        keys = [item.staging_key for item in items]
        if len(set(keys)) != len(keys):
            raise KeyError("media fetch: staging keys contain duplicates")
        columns = list(MEDIA_TENSOR_COLUMNS.values())
        try:
            rows = self._store.get(keys, select_fields=columns)
        except Exception as error:  # noqa: BLE001 — protocol maps misses to KeyError
            raise KeyError(
                f"media columns for {len(keys)} keys could not be fetched from "
                f"{self._staging_partition!r}: {error}"
            ) from error
        n_rows = int(rows.batch_size[0]) if len(rows.batch_size) else 0
        if n_rows != len(keys):
            raise KeyError(
                f"media rows missing: requested {len(keys)}, got {n_rows} from "
                f"{self._staging_partition!r}"
            )
        parts: list[StagedMediaTensors] = []
        for index in range(n_rows):
            row = _select_row(rows, index)

            def column(name: str) -> torch.Tensor:
                value = row[MEDIA_TENSOR_COLUMNS[name]]
                if value.ndim < 2 or value.shape[0] != 1:
                    raise ValueError(
                        f"invalid media column shape {tuple(value.shape)} for {name!r}"
                    )
                return value[0]  # remove exactly TQ's row dimension

            imgs = column("imgs")
            if imgs.ndim != 2:
                raise ValueError(
                    f"media imgs column must be [total_patches, F], got {tuple(imgs.shape)}"
                )
            parts.append(
                validate_media_tensors(
                    {
                        "imgs": imgs.unsqueeze(0),
                        "imgs_sizes": column("imgs_sizes"),
                        "num_frames": column("num_frames") if has_frames else None,
                    }
                )
            )
        return parts

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
        # Read 1 of (at most) 2: the media presence flags ride with the base
        # columns so the finalizer can select tensor rows without a probe.
        select_fields = list(STAGING_FIELDS)
        if self._capture_media:
            select_fields += MEDIA_FLAG_FIELDS
        try:
            if include_route_fragments:
                # Route payloads are optional per run (feature-gated at the
                # worker); fall back to the base schema so extras-free rows
                # keep fetching.
                try:
                    rows = self._store.get(
                        staging_keys,
                        select_fields=select_fields + [ROUTED_EXPERTS_FIELD],
                    )
                except Exception:  # noqa: BLE001 — field-not-present probe
                    rows = self._store.get(staging_keys, select_fields=select_fields)
            else:
                rows = self._store.get(staging_keys, select_fields=select_fields)
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
            media_present = media_has_frames = False
            if self._capture_media:
                media_present = _row_scalar_bool(row, MEDIA_PRESENT_FIELD)
                media_has_frames = _row_scalar_bool(row, MEDIA_HAS_FRAMES_FIELD)
                if media_has_frames and not media_present:
                    raise ValueError(
                        f"frame counts require media_present=True for {key!r}"
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
                    media_present=media_present,
                    media_has_frames=media_has_frames,
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


def _row_scalar_bool(row: Any, field_name: str) -> bool:
    """Read one required bool flag from a single-row TQ result."""
    value = row[field_name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"staging field {field_name!r} must be a tensor, got {type(value).__name__}"
        )
    if value.dtype is not torch.bool:
        raise TypeError(
            f"staging field {field_name!r} must use torch.bool, got {value.dtype}"
        )
    flattened = value.reshape(-1)
    if flattened.numel() != 1:
        raise ValueError(
            f"staging field {field_name!r} must contain one flag, got "
            f"shape {tuple(value.shape)}"
        )
    return bool(flattened[0].item())
