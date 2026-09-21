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
"""vLLM media extraction and placeholder remapping for shared token capture.

Pixels use the same packed ``imgs`` bundle as Megatron inference. Only the
small ``media_spans`` extras are vLLM-specific: they restore placeholder
positions when expanded-token history is spliced into a rendered prompt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from nemo_rl.models.generation.openai_server_utils import PrefixSplice

MEDIA_SPANS_FIELD = "media_spans"


def _token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CapturedMediaItem:
    """One new placeholder occurrence in the expanded token sequence."""

    modality: Literal["image", "video"]
    placeholder_offset: int
    placeholder_length: int
    placeholder_digest: str
    token_id: int
    embedding_spans: tuple[tuple[int, int], ...]
    imgs_sizes: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if (
            self.modality not in ("image", "video")
            or type(self.placeholder_offset) is not int
            or self.placeholder_offset < 0
            or type(self.placeholder_length) is not int
            or self.placeholder_length <= 0
            or type(self.token_id) is not int
            or self.token_id < 0
            or len(self.placeholder_digest) != 64
            or not self.embedding_spans
            or not self.imgs_sizes
            or any(
                len(size) != 2 or any(type(v) is not int or v <= 0 for v in size)
                for size in self.imgs_sizes
            )
        ):
            raise ValueError("Malformed media placeholder metadata")
        previous_end = self.placeholder_offset
        for start, length in self.embedding_spans:
            if (
                type(start) is not int
                or type(length) is not int
                or start < previous_end
                or length <= 0
                or start + length > self.end
            ):
                raise ValueError("Invalid media embedding span")
            previous_end = start + length

    @property
    def end(self) -> int:
        return self.placeholder_offset + self.placeholder_length

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["embedding_spans"] = [list(span) for span in self.embedding_spans]
        value["imgs_sizes"] = [list(size) for size in self.imgs_sizes]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CapturedMediaItem:
        fields = dict(value)
        fields["embedding_spans"] = tuple(
            tuple(span) for span in fields["embedding_spans"]
        )
        fields["imgs_sizes"] = tuple(tuple(size) for size in fields["imgs_sizes"])
        return cls(**fields)

    def verify_tokens(self, tokens: list[int], *, origin: int) -> None:
        start, end = self.placeholder_offset - origin, self.end - origin
        if (
            start < 0
            or end > len(tokens)
            or _token_digest(tokens[start:end]) != self.placeholder_digest
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


@dataclass(frozen=True)
class CapturedMedia:
    """New placeholder metadata and owned tensors for the common staging sink."""

    items: tuple[CapturedMediaItem, ...]
    tensors: dict[str, torch.Tensor] | None


def _geometry_tensor(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if (
        tensor.dtype not in (torch.int32, torch.int64)
        or bool((tensor <= 0).any())
        or bool((tensor > torch.iinfo(torch.int32).max).any())
    ):
        raise ValueError("Omni media geometry requires positive int32 values")
    return tensor.to(dtype=torch.int32)


def pack_images(
    pixels: torch.Tensor, sizes: torch.Tensor, *, patch_size: int
) -> torch.Tensor:
    """Losslessly match Bridge Omni's dynamic-image patch layout.

    This only rearranges processed pixels, preserving dtype and values. Frames
    remain in order and their temporal grouping travels separately in num_frames.
    """
    if (
        patch_size <= 0
        or pixels.ndim != 4
        or pixels.shape[1] != 3
        or pixels.shape[0] != sizes.shape[0]
    ):
        raise ValueError("Omni capture requires [frames, 3, height, width] pixels")
    patches = []
    for image, (height, width) in zip(pixels, sizes.tolist(), strict=True):
        if (
            height > image.shape[-2]
            or width > image.shape[-1]
            or height % patch_size
            or width % patch_size
        ):
            raise ValueError("Image geometry does not match the patch layout")
        patch = image[:, :height, :width].reshape(
            3, height // patch_size, patch_size, width // patch_size, patch_size
        )
        patches.append(patch.permute(1, 3, 0, 2, 4).reshape(-1, 3 * patch_size**2))
    return torch.cat(patches, dim=0).unsqueeze(0)


def _processed_omni_tensors(
    data: dict[str, Any], modality: str, *, patch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if data.get("num_tiles") is not None:
        raise ValueError("Omni capture does not support static num_tiles")
    if modality == "video":
        pixels = data.get("pixel_values_flat_video")
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
            raise ValueError("Video capture requires processed frame pixels")
        count = _geometry_tensor(data["video_num_patches"]).reshape(-1)
        if count.shape != (1,) or int(count[0]) != pixels.shape[0]:
            raise ValueError("Video frame count disagrees with processed pixels")
        sizes = torch.tensor(
            [list(pixels.shape[-2:])] * pixels.shape[0], dtype=torch.int32
        )
    else:
        pixels = data.get("pixel_values_flat")
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 3:
            raise ValueError("Image capture requires processed CHW pixels")
        sizes = _geometry_tensor(data["imgs_sizes"])
        if tuple(sizes.shape) != (2,) or sizes.tolist() != list(pixels.shape[-2:]):
            raise ValueError("Image capture requires exact CHW/HW geometry")
        pixels, sizes = pixels.unsqueeze(0), sizes.unsqueeze(0)
    if pixels.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Unsupported processed pixel dtype")
    return pack_images(pixels, sizes, patch_size=patch_size), sizes


def capture_processed_media(
    engine_prompt: dict[str, Any],
    *,
    prev_len: int,
    retained: tuple[CapturedMediaItem, ...] = (),
    splice: PrefixSplice | None = None,
    image_token_id: int | None = None,
    patch_size: int | None = None,
) -> CapturedMedia:
    """Remap vLLM placeholders and snapshot per-call media in Megatron's layout."""
    # The data-plane module has optional Gym dependencies and is only needed
    # when captured rollouts normalize their engine-owned media.
    from nemo_rl.data_plane.tq_token_sink import slice_media_tensors

    tokens = engine_prompt["prompt_token_ids"]
    placeholders = engine_prompt.get("mm_placeholders") or {}
    kwargs = engine_prompt.get("mm_kwargs") or {}
    if (set(placeholders) | set(kwargs)) - {"image", "video"}:
        raise ValueError("Omni token capture supports images and native video only")
    occurrences: list[tuple[int, Literal["image", "video"], Any, Any]] = []
    modalities: tuple[Literal["image"], Literal["video"]] = ("image", "video")
    for modality in modalities:
        spans, items = placeholders.get(modality, []), kwargs.get(modality, [])
        if len(spans) != len(items):
            raise ValueError("Media placeholders and processor occurrences disagree")
        occurrences.extend(
            (span.offset, modality, span, item)
            for span, item in zip(spans, items, strict=True)
        )
    occurrences.sort(key=lambda occurrence: occurrence[0])
    if len({modality for _, modality, _, _ in occurrences}) > 1:
        raise ValueError(
            "Shared media capture currently requires image-only or video-only conversations"
        )
    if occurrences and (patch_size is None or patch_size <= 0):
        raise ValueError("Media capture requires the model patch size")
    added, seen, packed, sizes_parts, frame_counts = [], [], [], [], []
    corrected = {name: [] for name in placeholders}
    previous_end = 0
    pixel_dtype = None
    for _, occurrence_modality, span, item in occurrences:
        modality: Literal["image", "video"] = occurrence_modality
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
        assert patch_size is not None
        patches, sizes = _processed_omni_tensors(data, modality, patch_size=patch_size)
        if pixel_dtype is not None and pixel_dtype != patches.dtype:
            raise ValueError("Media tensor dtypes changed within a call")
        pixel_dtype = patches.dtype
        packed.append(patches)
        sizes_parts.append(sizes)
        frame_counts.append(sizes.shape[0])
        local_tokens = tokens[span.offset : previous_end]
        if modality == "video":
            if image_token_id is None:
                raise ValueError(
                    "Native video capture requires the image-context token ID"
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
        runs = []
        for position in positions:
            if runs and runs[-1][0] + runs[-1][1] == offset + position:
                runs[-1] = (runs[-1][0], runs[-1][1] + 1)
            else:
                runs.append((offset + position, 1))
        captured = CapturedMediaItem(
            modality,
            offset,
            span.length,
            _token_digest(local_tokens),
            token_id,
            tuple(runs),
            tuple(tuple(size) for size in sizes.tolist()),
        )
        captured.verify_tokens(
            splice.token_ids if splice is not None else tokens, origin=0
        )
        if offset < prev_len < captured.end:
            raise ValueError("Media placeholder crosses the captured prefix boundary")
        if captured.end <= prev_len:
            if len(seen) >= len(retained) or captured != retained[len(seen)]:
                raise ValueError("Retained media geometry or tokens changed")
            seen.append(captured)
        else:
            added.append(captured)
        corrected[modality].append(replace(span, offset=offset))
    if tuple(seen) != retained:
        raise ValueError("Rendered prefix dropped captured media")
    tensors = None
    if packed:
        full = {
            "imgs": torch.cat(packed, dim=1),
            "imgs_sizes": torch.cat(sizes_parts, dim=0),
        }
        if occurrences[0][1] == "video":
            full["num_frames"] = torch.tensor(frame_counts, dtype=torch.int32)
        delta = slice_media_tensors(full, len(retained))
        if delta is not None:
            tensors = {
                name: tensor.detach().cpu().clone() for name, tensor in delta.items()
            }
        engine_prompt["mm_placeholders"] = corrected
    return CapturedMedia(tuple(added), tensors)
