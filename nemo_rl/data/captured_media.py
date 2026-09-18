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
"""Capture processed Omni image/video inputs with exact token and tensor identity."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import torch

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.experience.route_assembly import verify_route_fragment_integrity

if TYPE_CHECKING:
    from nemo_gym.token_id_capture.staging.protocols import TensorAttachment

    from nemo_rl.data_plane.tq_token_sink import FetchedStagedCall, TQTokenSource
    from nemo_rl.models.generation.openai_server_utils import PrefixSplice

MEDIA_CAPTURE_FIELD = "media_capture"
MEDIA_CAPTURE_ADAPTER = "omni_media_v1"
MEDIA_STAGING_FIELDS = ("pixel_values", "imgs_sizes", "num_frames")
_TENSOR_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


@dataclass(frozen=True)
class CapturedTensor:
    """One owned tensor, with shape/dtype/content committed by the call extras."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    digest: str

    def __post_init__(self) -> None:
        if (
            self.name not in MEDIA_STAGING_FIELDS
            or self.dtype not in _TENSOR_DTYPES
            or not self.shape
            or any(type(n) is not int or n <= 0 for n in self.shape)
            or not _is_digest(self.digest)
        ):
            raise ValueError("Malformed captured tensor descriptor")

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    def verify(self, tensor: torch.Tensor) -> None:
        if (
            tuple(tensor.shape) != self.shape
            or tensor.dtype != _TENSOR_DTYPES[self.dtype]
        ):
            raise ValueError("Captured tensor shape or dtype mismatch")
        if hashlib.sha256(_tensor_bytes(tensor)).hexdigest() != self.digest:
            raise ValueError("Captured tensor content digest mismatch")


@dataclass(frozen=True)
class CapturedMediaItem:
    """One image or video occurrence, including discontiguous visual token runs.

    The entire placeholder is committed too: video timestamps and separators
    must survive a prefix splice even though only visual tokens consume pixels.
    """

    modality: Literal["image", "video"]
    layout: Literal["pixels", "packed_patches"]
    placeholder_offset: int
    placeholder_length: int
    placeholder_digest: str
    token_id: int
    embedding_spans: tuple[tuple[int, int], ...]
    tensors: tuple[CapturedTensor, ...]

    def __post_init__(self) -> None:
        if (
            self.modality not in ("image", "video")
            or self.layout not in ("pixels", "packed_patches")
            or type(self.placeholder_offset) is not int
            or self.placeholder_offset < 0
            or type(self.placeholder_length) is not int
            or self.placeholder_length <= 0
            or type(self.token_id) is not int
            or self.token_id < 0
            or not _is_digest(self.placeholder_digest)
            or not self.embedding_spans
            or tuple(t.name for t in self.tensors) != MEDIA_STAGING_FIELDS
        ):
            raise ValueError("Malformed captured media occurrence")
        previous_end = self.placeholder_offset
        for start, length in self.embedding_spans:
            if (
                type(start) is not int
                or type(length) is not int
                or length <= 0
                or start < previous_end
                or start + length > self.end
            ):
                raise ValueError("Invalid captured media embedding span")
            previous_end = start + length

    @property
    def end(self) -> int:
        return self.placeholder_offset + self.placeholder_length

    def verify_tokens(self, tokens: list[int], *, origin: int) -> None:
        start, end = self.placeholder_offset - origin, self.end - origin
        if (
            start < 0
            or end > len(tokens)
            or _json_digest(tokens[start:end]) != self.placeholder_digest
        ):
            raise ValueError(
                "Media placeholder does not match the exact carried prompt"
            )
        for offset, length in self.embedding_spans:
            if any(
                t != self.token_id
                for t in tokens[offset - origin : offset - origin + length]
            ):
                raise ValueError(
                    "Media embeddings do not match the exact carried prompt"
                )


def _media_digest(
    items: list[CapturedMediaItem] | tuple[CapturedMediaItem, ...],
) -> str:
    return _json_digest([asdict(item) for item in items])


@dataclass(frozen=True)
class CapturedMedia:
    """Small versioned descriptor; named tensor bytes travel as sink attachments."""

    prefix_digest: str
    items: tuple[CapturedMediaItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": MEDIA_CAPTURE_ADAPTER,
            "prefix_digest": self.prefix_digest,
            "items": [
                {
                    **asdict(item),
                    "embedding_spans": [list(span) for span in item.embedding_spans],
                    "tensors": [
                        {**asdict(tensor), "shape": list(tensor.shape)}
                        for tensor in item.tensors
                    ],
                }
                for item in self.items
            ],
        }

    @classmethod
    def from_dict(cls, value: Any) -> CapturedMedia:
        if (
            not isinstance(value, dict)
            or set(value) != {"adapter", "prefix_digest", "items"}
            or value["adapter"] != MEDIA_CAPTURE_ADAPTER
            or not _is_digest(value["prefix_digest"])
            or not isinstance(value["items"], list)
        ):
            raise ValueError("Missing or unsupported worker media capture descriptor")
        try:
            items = []
            for encoded in value["items"]:
                fields = dict(encoded)
                tensors = []
                for tensor in fields["tensors"]:
                    tensor_fields: dict[str, Any] = {
                        **tensor,
                        "shape": tuple(tensor["shape"]),
                    }
                    tensors.append(CapturedTensor(**tensor_fields))
                fields["tensors"] = tuple(tensors)
                fields["embedding_spans"] = tuple(
                    tuple(span) for span in fields["embedding_spans"]
                )
                items.append(CapturedMediaItem(**fields))
        except (TypeError, KeyError) as error:
            raise ValueError("Malformed captured media occurrence") from error
        return cls(value["prefix_digest"], tuple(items))

    def decode_tensors(
        self, fields: dict[str, torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """Validate flat staging columns and restore each item's owned inputs."""
        if set(fields) != (set(MEDIA_STAGING_FIELDS) if self.items else set()):
            raise ValueError("Captured media columns do not match the descriptor")
        cursors = dict.fromkeys(fields, 0)
        result = []
        for item in self.items:
            tensors = {}
            for spec in item.tensors:
                flat = fields[spec.name]
                start = cursors[spec.name]
                if flat.ndim != 1 or flat.numel() < start + spec.numel:
                    raise ValueError("Captured media tensor length mismatch")
                tensor = flat[start : start + spec.numel].reshape(spec.shape)
                spec.verify(tensor)
                tensors[spec.name] = tensor
                cursors[spec.name] += spec.numel
            _validate_omni_tensors(tensors, modality=item.modality, layout=item.layout)
            result.append(tensors)
        if any(fields[name].numel() != end for name, end in cursors.items()):
            raise ValueError("Unexpected trailing captured tensor values")
        return result


def _validate_omni_tensors(
    tensors: dict[str, torch.Tensor], *, modality: str, layout: str
) -> None:
    pixels, sizes, frames = (tensors[name] for name in MEDIA_STAGING_FIELDS)
    if (
        pixels.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or sizes.dtype not in (torch.int32, torch.int64)
        or sizes.ndim != 2
        or sizes.shape[1] != 2
        or sizes.numel() == 0
        or bool((sizes <= 0).any())
        or frames.dtype not in (torch.int32, torch.int64)
        or frames.shape != (1,)
        or int(frames[0]) != sizes.shape[0]
        or (modality == "image" and int(frames[0]) != 1)
    ):
        raise ValueError("Invalid Omni image/video frame geometry")
    if layout == "pixels":
        if (
            pixels.ndim != 4
            or pixels.shape[0] != sizes.shape[0]
            or pixels.shape[1] != 3
            or bool((sizes[:, 0] > pixels.shape[-2]).any())
            or bool((sizes[:, 1] > pixels.shape[-1]).any())
        ):
            raise ValueError("Omni pixels require [frames, 3, height, width] geometry")
    else:
        if pixels.ndim != 3 or pixels.shape[0] != 1:
            raise ValueError("Omni packed patches require [1, patches, features]")
        patch = math.isqrt(pixels.shape[-1] // 3)
        if (
            patch <= 0
            or 3 * patch * patch != pixels.shape[-1]
            or bool((sizes % patch != 0).any())
            or int((sizes[:, 0] // patch * (sizes[:, 1] // patch)).sum())
            != pixels.shape[1]
        ):
            raise ValueError("Omni packed patches disagree with frame geometry")


def _geometry_tensor(value: Any) -> torch.Tensor:
    """Normalize integer geometry without truncating floats or overflowing int32."""
    tensor = torch.as_tensor(value)
    if (
        tensor.dtype not in (torch.int32, torch.int64)
        or bool((tensor <= 0).any())
        or bool((tensor > torch.iinfo(torch.int32).max).any())
    ):
        raise ValueError("Omni media geometry requires positive int32 values")
    return tensor.to(dtype=torch.int32)


def _processed_omni_tensors(
    data: dict[str, Any], modality: str
) -> tuple[Literal["pixels", "packed_patches"], dict[str, torch.Tensor]]:
    """Normalize vLLM processor items or one Omni encoder-input bundle.

    vLLM 0.25.1 uses pixel_values_flat_video/video_num_patches for native
    videos. Its frames are already sampled, resized and normalized. Encoder
    bundles use imgs/imgs_sizes/num_frames, including packed patches.
    """
    if data.get("num_tiles") is not None:
        raise ValueError("Canonical Omni capture does not support static num_tiles")
    layout: Literal["pixels", "packed_patches"]
    if "imgs" in data:
        pixels = data["imgs"]
        if not isinstance(pixels, torch.Tensor):
            raise ValueError("Omni capture requires an imgs tensor")
        sizes = _geometry_tensor(data["imgs_sizes"])
        if sizes.ndim != 2 or sizes.shape[1] != 2:
            raise ValueError("Omni imgs_sizes must contain [height, width] rows")
        frames = (
            _geometry_tensor(data["num_frames"]).reshape(-1)
            if "num_frames" in data
            else torch.ones(1, dtype=torch.int32)
        )
        if modality == "video" and "num_frames" not in data:
            raise ValueError("Omni video capture requires num_frames")
        layout = "packed_patches" if pixels.ndim in (2, 3) else "pixels"
        if pixels.ndim == 2:
            pixels = pixels.unsqueeze(0)
    elif modality == "video":
        pixels = data.get("pixel_values_flat_video")
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
            raise ValueError(
                "Video capture requires processed [frames, 3, height, width] pixels"
            )
        frames = _geometry_tensor(data["video_num_patches"]).reshape(-1)
        sizes = torch.tensor(
            [list(pixels.shape[-2:])] * pixels.shape[0], dtype=torch.int32
        )
        layout = "pixels"
    else:
        pixels = data.get("pixel_values_flat")
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 3:
            raise ValueError("Image capture requires processed CHW pixels")
        pixels = pixels.unsqueeze(0)
        sizes = _geometry_tensor(data["imgs_sizes"])
        if tuple(sizes.shape) != (2,) or sizes.tolist() != list(pixels.shape[-2:]):
            raise ValueError("Image capture requires exact CHW/HW geometry")
        sizes = sizes.unsqueeze(0)
        frames = torch.ones(1, dtype=torch.int32)
        layout = "pixels"
    tensors = {"pixel_values": pixels, "imgs_sizes": sizes, "num_frames": frames}
    _validate_omni_tensors(tensors, modality=modality, layout=layout)
    return layout, tensors


def capture_processed_media(
    engine_prompt: dict[str, Any],
    *,
    prev_len: int,
    retained: tuple[CapturedMediaItem, ...] = (),
    splice: PrefixSplice | None = None,
    image_token_id: int | None = None,
) -> tuple[CapturedMedia, tuple[TensorAttachment, ...]]:
    """Snapshot new Omni images/videos and verify retained occurrences exactly."""
    # Gym is optional outside captured rollouts.
    from nemo_gym.token_id_capture.staging.protocols import TensorAttachment

    tokens = engine_prompt["prompt_token_ids"]
    placeholders = engine_prompt.get("mm_placeholders") or {}
    kwargs = engine_prompt.get("mm_kwargs") or {}
    if (set(placeholders) | set(kwargs)) - {"image", "video"}:
        raise ValueError("Omni token capture supports images and native video only")
    occurrences = []
    modalities: tuple[Literal["image", "video"], ...] = ("image", "video")
    for modality in modalities:
        spans, items = placeholders.get(modality, []), kwargs.get(modality, [])
        if len(spans) != len(items):
            raise ValueError("Media placeholders and processor occurrences disagree")
        occurrences.extend(
            (span.offset, modality, span, item)
            for span, item in zip(spans, items, strict=True)
        )
    occurrences.sort(key=lambda occurrence: occurrence[0])
    added, seen = [], []
    buffers: dict[str, list[bytes]] = {name: [] for name in MEDIA_STAGING_FIELDS}
    dtypes: dict[str, str] = {}
    corrected = {name: [] for name in placeholders}
    previous_end = 0
    for _, modality, span, item in occurrences:
        if item is None:
            raise ValueError(
                "Media capture requires processor data, not cache references"
            )
        if (
            type(span.offset) is not int
            or type(span.length) is not int
            or span.offset < previous_end
            or span.length <= 0
            or span.offset + span.length > len(tokens)
        ):
            raise ValueError("Invalid media placeholder range")
        previous_end = span.offset + span.length
        data = item.get_data()
        layout, tensors = _processed_omni_tensors(data, modality)
        local_tokens = tokens[span.offset : previous_end]
        if modality == "video":
            if image_token_id is None:
                raise ValueError(
                    "Native video capture requires the Omni image-context token ID"
                )
            token_id = image_token_id
            positions = [i for i, token in enumerate(local_tokens) if token == token_id]
        else:
            positions = list(range(span.length))
            if span.is_embed is not None:
                if span.is_embed.dtype != torch.bool or tuple(span.is_embed.shape) != (
                    span.length,
                ):
                    raise ValueError("Invalid image embedding mask")
                positions = span.is_embed.nonzero().flatten().tolist()
            if not positions or positions != list(
                range(positions[0], positions[-1] + 1)
            ):
                raise ValueError(
                    "Image capture requires contiguous image embedding positions"
                )
            token_id = local_tokens[positions[0]]
            if data.get("num_tokens_per_image") is not None and int(
                data["num_tokens_per_image"]
            ) != len(positions):
                raise ValueError("Image embedding count changed")
        if not positions or any(local_tokens[i] != token_id for i in positions):
            raise ValueError("Invalid media embedding tokens")
        offset = span.offset
        if splice is not None:
            if span.offset + span.length <= splice.template_cut_start:
                if len(seen) >= len(retained):
                    raise ValueError(
                        "Rendered prefix has an unexpected media occurrence"
                    )
                offset = retained[len(seen)].placeholder_offset
            elif span.offset >= splice.template_cut_start:
                offset = splice.model_cut_end + span.offset - splice.template_cut_start
            else:
                raise ValueError("Media placeholder crosses the token splice boundary")
        runs: list[tuple[int, int]] = []
        for position in positions:
            if runs and runs[-1][0] + runs[-1][1] == offset + position:
                runs[-1] = (runs[-1][0], runs[-1][1] + 1)
            else:
                runs.append((offset + position, 1))
        specs, raw_tensors = [], {}
        for name, tensor in tensors.items():
            raw = _tensor_bytes(tensor)
            dtype = str(tensor.dtype).removeprefix("torch.")
            if name in dtypes and dtypes[name] != dtype:
                raise ValueError("Media tensor dtypes changed within a call")
            dtypes[name] = dtype
            specs.append(
                CapturedTensor(
                    name, tuple(tensor.shape), dtype, hashlib.sha256(raw).hexdigest()
                )
            )
            raw_tensors[name] = raw
        captured = CapturedMediaItem(
            modality,
            layout,
            offset,
            span.length,
            _json_digest(local_tokens),
            token_id,
            tuple(runs),
            tuple(specs),
        )
        captured.verify_tokens(
            splice.token_ids if splice is not None else tokens, origin=0
        )
        if offset < prev_len < captured.end:
            raise ValueError("Media placeholder crosses the captured prefix boundary")
        if captured.end <= prev_len:
            if len(seen) >= len(retained) or captured != retained[len(seen)]:
                raise ValueError("Retained media geometry, tokens or pixels changed")
            seen.append(captured)
        else:
            added.append(captured)
            for name, raw in raw_tensors.items():
                buffers[name].append(raw)
        corrected[modality].append(replace(span, offset=offset))
    if tuple(seen) != retained:
        raise ValueError("Rendered prefix dropped captured media")
    descriptor = CapturedMedia(_media_digest(seen), tuple(added))
    attachments = (
        tuple(
            TensorAttachment(
                name,
                dtypes[name],
                (
                    sum(
                        t.numel
                        for item in added
                        for t in item.tensors
                        if t.name == name
                    ),
                ),
                b"".join(buffers[name]),
            )
            for name in MEDIA_STAGING_FIELDS
        )
        if added
        else ()
    )
    if occurrences:
        engine_prompt["mm_placeholders"] = corrected
    return descriptor, attachments


def attachment_tensors(
    descriptor: CapturedMedia, attachments: tuple[TensorAttachment, ...]
) -> dict[str, torch.Tensor]:
    """Verify every required attachment before any token/media write."""
    expected = set(MEDIA_STAGING_FIELDS) if descriptor.items else set()
    if {a.name for a in attachments} != expected or len(attachments) != len(expected):
        raise ValueError(
            "Media capture requires exactly its declared tensor attachments"
        )
    fields = {}
    for attachment in attachments:
        if attachment.dtype not in _TENSOR_DTYPES:
            raise ValueError("Unsupported captured attachment dtype")
        tensor = torch.frombuffer(
            bytearray(attachment.data), dtype=_TENSOR_DTYPES[attachment.dtype]
        )
        if tuple(tensor.shape) != attachment.shape:
            raise ValueError("Media attachment shape mismatch")
        fields[attachment.name] = tensor
    descriptor.decode_tensors(fields)
    return fields


def verify_media_chain(
    calls: list[FetchedStagedCall], *, required: bool
) -> list[CapturedMedia]:
    """Authenticate selected-call extras and exact media/prompt alignment."""
    # Gym is optional outside captured rollouts.
    from nemo_gym.token_id_capture.staging.digest import compute_extras_digest

    metadata = [json.loads(call.extras_metadata_json) for call in calls]
    if not required and not any(
        isinstance(extra, dict) and MEDIA_CAPTURE_FIELD in extra for extra in metadata
    ):
        return []
    retained: list[CapturedMediaItem] = []
    descriptors = []
    dtypes: dict[str, str] = {}
    for call, extra in zip(calls, metadata, strict=True):
        snapshot = call.snapshot
        if call.fragment is not None:
            integrity = (
                verify_route_fragment_integrity(
                    call.fragment,
                    extras_digest_version=snapshot.extras_digest_version,
                    expected_extras_digest=snapshot.extras_digest,
                )
                and call.fragment.extras_metadata_json == call.extras_metadata_json
            )
        else:
            integrity = compute_extras_digest(extra) == snapshot.extras_digest
        if not integrity or not isinstance(extra, dict):
            raise ValueError("Media capture extras commitment mismatch")
        descriptor = CapturedMedia.from_dict(extra.get(MEDIA_CAPTURE_FIELD))
        if descriptor.prefix_digest != _media_digest(retained):
            raise ValueError("Retained media changed within the rollout")
        carry_len = next(
            (i for i, mask in enumerate(snapshot.token_mask_delta) if mask == 1.0),
            snapshot.delta_len,
        )
        previous_end = snapshot.prev_len
        for item in descriptor.items:
            if (
                item.placeholder_offset < previous_end
                or item.end > snapshot.prev_len + carry_len
            ):
                raise ValueError("Media span is outside the carried prompt")
            item.verify_tokens(snapshot.token_ids_delta, origin=snapshot.prev_len)
            for tensor in item.tensors:
                if tensor.name in dtypes and dtypes[tensor.name] != tensor.dtype:
                    raise ValueError("Media tensor dtypes changed within the rollout")
                dtypes[tensor.name] = tensor.dtype
            previous_end = item.end
            retained.append(item)
        descriptors.append(descriptor)
    return descriptors


def _frame_pixels(
    item: CapturedMediaItem, tensors: dict[str, torch.Tensor]
) -> list[torch.Tensor | None]:
    """Restore CHW frames for the existing learner, without media preprocessing.

    Packed patches are only reshaped/permuted: no sampling, resizing,
    normalization or precision conversion occurs. This also allows pixel and
    packed-patch siblings to share the existing padded PackedTensor transport.
    """
    pixels, sizes = tensors["pixel_values"], tensors["imgs_sizes"]
    if item.layout == "pixels":
        return [
            frame[:, : int(size[0]), : int(size[1])].unsqueeze(0)
            for frame, size in zip(pixels, sizes, strict=True)
        ]
    patch = math.isqrt(pixels.shape[-1] // 3)
    frames: list[torch.Tensor | None] = []
    cursor = 0
    for height, width in sizes.tolist():
        rows, columns = height // patch, width // patch
        count = rows * columns
        frame = (
            pixels[0, cursor : cursor + count]
            .reshape(rows, columns, 3, patch, patch)
            .permute(2, 0, 3, 1, 4)
            .reshape(1, 3, height, width)
        )
        frames.append(frame)
        cursor += count
    return frames


def assemble_captured_media(
    calls: list[FetchedStagedCall], *, source: TQTokenSource, required: bool
) -> dict[str, PackedTensor]:
    """Assemble selected new media with exact frame grouping into one training row."""
    descriptors = verify_media_chain(calls, required=required)
    if not descriptors:
        return {}
    fields: dict[str, list[torch.Tensor | None]] = {
        name: [] for name in MEDIA_STAGING_FIELDS
    }
    for call, descriptor in zip(calls, descriptors, strict=True):
        if not descriptor.items:
            continue
        tensors_by_item = descriptor.decode_tensors(
            source.fetch_media(call.staging_key)
        )
        for item, tensors in zip(descriptor.items, tensors_by_item, strict=True):
            pixels = PackedTensor(
                _frame_pixels(item, tensors), 0, pad_to_max_shape=True
            ).as_tensor()
            fields["pixel_values"].append(pixels)
            fields["imgs_sizes"].append(tensors["imgs_sizes"])
            fields["num_frames"].append(tensors["num_frames"])
    if not fields["pixel_values"]:
        return {}
    return {
        name: PackedTensor(
            values,
            0,
            pad_to_max_shape=name == "pixel_values",
            _row_offsets=[0, len(values)],
            _segment_indices=list(range(len(values))),
        )
        for name, values in fields.items()
    }
