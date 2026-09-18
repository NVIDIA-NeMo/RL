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
"""Worker-captured Nemotron dynamic images, bound to ordinary token records.

Geometry validation and packed-row assembly are adapted from RL #4124. Pixels
come from the engine's processor outputs instead of a second image processing
pass. Only new occurrences are staged; retained images are verified by digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.experience.route_assembly import verify_route_fragment_integrity

if TYPE_CHECKING:
    from nemo_gym.token_id_capture.staging.protocols import TensorAttachment

    from nemo_rl.data_plane.tq_token_sink import FetchedStagedCall, TQTokenSource
    from nemo_rl.models.generation.openai_server_utils import PrefixSplice

IMAGE_CAPTURE_FIELD = "image_capture"
IMAGE_CAPTURE_ADAPTER = "nemotron_dynamic_chw_v1"
STAGED_PIXEL_FIELD = "pixel_values"
_PIXEL_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class CapturedImage:
    """One image occurrence in the actual engine prompt, including pixel identity."""

    offset: int
    length: int
    height: int
    width: int
    token_id: int
    placeholder_offset: int
    placeholder_length: int
    dtype: str
    pixel_digest: str

    def __post_init__(self) -> None:
        integers = (
            self.offset,
            self.length,
            self.height,
            self.width,
            self.token_id,
            self.placeholder_offset,
            self.placeholder_length,
        )
        if (
            any(type(value) is not int or value < 0 for value in integers)
            or min(self.length, self.height, self.width, self.placeholder_length) <= 0
            or self.placeholder_offset > self.offset
            or self.offset + self.length
            > self.placeholder_offset + self.placeholder_length
            or self.dtype not in _PIXEL_DTYPES
            or not isinstance(self.pixel_digest, str)
            or len(self.pixel_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.pixel_digest)
        ):
            raise ValueError("Malformed captured image occurrence")

    @property
    def numel(self) -> int:
        return 3 * self.height * self.width


def _image_digest(images: list[CapturedImage] | tuple[CapturedImage, ...]) -> str:
    # Gym is optional outside captured rollouts.
    from nemo_gym.token_id_capture.staging.digest import compute_extras_digest

    return compute_extras_digest({"images": [asdict(image) for image in images]})


def _pixel_digest(data: bytes, *, dtype: str, height: int, width: int) -> str:
    header = f"{IMAGE_CAPTURE_ADAPTER}:{dtype}:3:{height}:{width}:".encode()
    digest = hashlib.sha256(header)
    digest.update(data)
    return digest.hexdigest()


@dataclass(frozen=True)
class CapturedMedia:
    """Small JSON descriptor; the pixel column is an explicit sink attachment."""

    prefix_digest: str
    images: tuple[CapturedImage, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": IMAGE_CAPTURE_ADAPTER,
            "prefix_digest": self.prefix_digest,
            "images": [asdict(image) for image in self.images],
        }

    @classmethod
    def from_dict(cls, value: Any) -> CapturedMedia:
        if (
            not isinstance(value, dict)
            or set(value) != {"adapter", "prefix_digest", "images"}
            or value["adapter"] != IMAGE_CAPTURE_ADAPTER
            or not isinstance(value["prefix_digest"], str)
            or not isinstance(value["images"], list)
        ):
            raise ValueError("Missing or unsupported worker image capture descriptor")
        try:
            images = tuple(CapturedImage(**image) for image in value["images"])
        except TypeError as error:
            raise ValueError("Malformed captured image occurrence") from error
        if len({image.dtype for image in images}) > 1:
            raise ValueError("Image pixel dtypes changed within a call")
        return cls(value["prefix_digest"], images)

    def decode_pixels(self, pixels: torch.Tensor) -> list[torch.Tensor]:
        """Check the flat column and restore exact CHW tensors without processing."""
        if (
            not self.images
            or pixels.ndim != 1
            or pixels.dtype != _PIXEL_DTYPES[self.images[0].dtype]
            or pixels.numel() != sum(image.numel for image in self.images)
        ):
            raise ValueError("Captured pixel payload does not match its descriptor")
        result = []
        cursor = 0
        for image in self.images:
            tensor = pixels[cursor : cursor + image.numel].reshape(
                3, image.height, image.width
            )
            data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            if (
                _pixel_digest(
                    data, dtype=image.dtype, height=image.height, width=image.width
                )
                != image.pixel_digest
            ):
                raise ValueError("Captured image pixel digest mismatch")
            result.append(tensor)
            cursor += image.numel
        return result


def capture_processed_images(
    engine_prompt: dict[str, Any],
    *,
    prev_len: int,
    retained: tuple[CapturedImage, ...] = (),
    splice: PrefixSplice | None = None,
) -> tuple[CapturedMedia, tuple[TensorAttachment, ...]]:
    """Capture processor pixels and correct ranges using exact splice coordinates.

    Uses #4124's CHW/HW and contiguous-embedding checks. Retained ranges come
    from the parent record, so repeated placeholder runs cannot alias each other.
    The returned immutable bytes are independent of processor/cache storage.
    """
    # Gym is optional outside captured rollouts.
    from nemo_gym.token_id_capture.staging.protocols import TensorAttachment

    tokens = engine_prompt["prompt_token_ids"]
    placeholders = engine_prompt.get("mm_placeholders") or {}
    kwargs = engine_prompt.get("mm_kwargs") or {}
    if (set(placeholders) | set(kwargs)) - {"image"}:
        raise ValueError("Token capture currently supports dynamic images only")
    ranges = placeholders.get("image", [])
    items = kwargs.get("image", [])
    if len(ranges) != len(items):
        raise ValueError("Image placeholders and processor occurrences disagree")
    added, seen, buffers, corrected = [], [], [], []
    previous_end = 0
    for span, item in zip(ranges, items, strict=True):
        if item is None:
            raise ValueError(
                "Image capture requires processor data, not cache references"
            )
        data = item.get_data()
        pixels = data.get("pixel_values_flat")
        sizes = data.get("imgs_sizes")
        if isinstance(sizes, torch.Tensor):
            sizes = sizes.tolist()
        if (
            not isinstance(pixels, torch.Tensor)
            or pixels.ndim != 3
            or pixels.shape[0] != 3
            or not isinstance(sizes, (list, tuple))
            or len(sizes) != 2
            or any(type(size) is not int or size <= 0 for size in sizes)
            or tuple(pixels.shape[-2:]) != tuple(sizes)
        ):
            raise ValueError(
                "Image capture requires exact dynamic-image CHW/HW geometry"
            )
        dtype = str(pixels.dtype).removeprefix("torch.")
        if dtype not in _PIXEL_DTYPES:
            raise ValueError(f"Unsupported captured pixel dtype {dtype!r}")
        embedding_start, length = 0, span.length
        if span.is_embed is not None:
            mask = span.is_embed
            if mask.dtype != torch.bool or tuple(mask.shape) != (length,):
                raise ValueError("Invalid image embedding mask")
            positions = mask.nonzero().flatten().tolist()
            if not positions or positions != list(
                range(positions[0], positions[-1] + 1)
            ):
                raise ValueError(
                    "Image capture requires contiguous image embedding positions"
                )
            embedding_start, length = positions[0], len(positions)
        original_offset = span.offset + embedding_start
        if (
            type(span.offset) is not int
            or type(span.length) is not int
            or span.offset < previous_end
            or span.length <= 0
            or span.offset + span.length > len(tokens)
            or data.get("num_tokens_per_image") != length
        ):
            raise ValueError("Invalid image placeholder range")
        previous_end = span.offset + span.length
        token_id = tokens[original_offset]
        if any(
            token != token_id
            for token in tokens[original_offset : original_offset + length]
        ):
            raise ValueError(
                "Image embedding span must contain one placeholder token ID"
            )
        corrected_span = span
        if splice is not None:
            if span.offset + span.length <= splice.template_cut_start:
                if len(seen) >= len(retained):
                    raise ValueError(
                        "Rendered prefix has an unexpected image occurrence"
                    )
                corrected_span = replace(
                    span, offset=retained[len(seen)].placeholder_offset
                )
            elif span.offset >= splice.template_cut_start:
                corrected_span = replace(
                    span,
                    offset=splice.model_cut_end
                    + span.offset
                    - splice.template_cut_start,
                )
            else:
                raise ValueError("Image placeholder crosses the token splice boundary")
        raw = pixels.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        image = CapturedImage(
            offset=corrected_span.offset + embedding_start,
            length=length,
            height=sizes[0],
            width=sizes[1],
            token_id=token_id,
            placeholder_offset=corrected_span.offset,
            placeholder_length=span.length,
            dtype=dtype,
            pixel_digest=_pixel_digest(
                raw, dtype=dtype, height=sizes[0], width=sizes[1]
            ),
        )
        final_tokens = splice.token_ids if splice is not None else tokens
        end = image.offset + image.length
        if end > len(final_tokens) or any(
            token != token_id for token in final_tokens[image.offset : end]
        ):
            raise ValueError("Remapped image does not match exact engine tokens")
        if (
            image.placeholder_offset
            < prev_len
            < image.placeholder_offset + image.placeholder_length
        ):
            raise ValueError("Image placeholder crosses the captured prefix boundary")
        if end <= prev_len:
            if len(seen) >= len(retained) or image != retained[len(seen)]:
                raise ValueError("Retained image geometry or pixels changed")
            seen.append(image)
        else:
            added.append(image)
            buffers.append(raw)
        corrected.append(corrected_span)
    if tuple(seen) != retained:
        raise ValueError("Rendered prefix dropped captured images")
    descriptor = CapturedMedia(_image_digest(seen), tuple(added))
    if len({image.dtype for image in seen + added}) > 1:
        raise ValueError("Image pixel dtypes changed within a call")
    if ranges:
        engine_prompt["mm_placeholders"] = {"image": corrected}
    attachments = (
        (
            TensorAttachment(
                STAGED_PIXEL_FIELD,
                added[0].dtype,
                (sum(image.numel for image in added),),
                b"".join(buffers),
            ),
        )
        if added
        else ()
    )
    return descriptor, attachments


def attachment_pixels(
    descriptor: CapturedMedia, attachments: tuple[TensorAttachment, ...]
) -> torch.Tensor | None:
    """Validate the sink payload before any token or media field is written."""
    if not descriptor.images:
        if attachments:
            raise ValueError("Pixel attachment without image occurrences")
        return None
    if len(attachments) != 1:
        raise ValueError("Image capture requires exactly one pixel attachment")
    attachment = attachments[0]
    if (
        attachment.name != STAGED_PIXEL_FIELD
        or attachment.dtype != descriptor.images[0].dtype
        or attachment.shape != (sum(image.numel for image in descriptor.images),)
    ):
        raise ValueError("Pixel attachment metadata mismatch")
    # bytearray gives torch writable owned storage; do not expose immutable bytes
    # through frombuffer (which otherwise permits writes into a Python bytes).
    pixels = torch.frombuffer(
        bytearray(attachment.data), dtype=_PIXEL_DTYPES[attachment.dtype]
    )
    descriptor.decode_pixels(pixels)
    return pixels


def verify_image_chain(
    calls: list[FetchedStagedCall], *, required: bool
) -> list[CapturedMedia]:
    """Authenticate selected-call extras and image/prompt alignment (#4124)."""
    # Gym is optional outside captured rollouts.
    from nemo_gym.token_id_capture.staging.digest import compute_extras_digest

    metadata = [json.loads(call.extras_metadata_json) for call in calls]
    if not required and not any(
        isinstance(extra, dict) and IMAGE_CAPTURE_FIELD in extra for extra in metadata
    ):
        return []
    retained: list[CapturedImage] = []
    descriptors = []
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
            raise ValueError("Image capture extras commitment mismatch")
        descriptor = CapturedMedia.from_dict(extra.get(IMAGE_CAPTURE_FIELD))
        if descriptor.prefix_digest != _image_digest(retained):
            raise ValueError(
                "Retained image geometry or pixels changed within the rollout"
            )
        carry_len = next(
            (i for i, mask in enumerate(snapshot.token_mask_delta) if mask == 1.0),
            snapshot.delta_len,
        )
        previous_end = snapshot.prev_len
        for image in descriptor.images:
            if retained and image.dtype != retained[0].dtype:
                raise ValueError("Image pixel dtypes changed within the rollout")
            start, end = image.offset, image.offset + image.length
            if (
                image.placeholder_offset < previous_end
                or image.placeholder_offset + image.placeholder_length
                > snapshot.prev_len + carry_len
                or any(
                    token != image.token_id
                    for token in snapshot.token_ids_delta[
                        start - snapshot.prev_len : end - snapshot.prev_len
                    ]
                )
            ):
                raise ValueError("Image span does not match the exact carried prompt")
            previous_end = image.placeholder_offset + image.placeholder_length
            retained.append(image)
        descriptors.append(descriptor)
    return descriptors


def assemble_captured_media(
    calls: list[FetchedStagedCall], *, source: TQTokenSource, required: bool
) -> dict[str, PackedTensor]:
    """Fetch selected new-image columns and assemble one ordinary training row."""
    descriptors = verify_image_chain(calls, required=required)
    images: list[CapturedImage] = []
    tensors: list[torch.Tensor] = []
    for call, descriptor in zip(calls, descriptors):
        if descriptor.images:
            tensors.extend(
                descriptor.decode_pixels(source.fetch_pixels(call.staging_key))
            )
            images.extend(descriptor.images)
    if not images:
        return {}
    # Same packed-row geometry as #4124, with per-image rather than per-action
    # segments. The Megatron dynamic-image path crops padding using imgs_sizes.
    fields: dict[str, list[torch.Tensor | None]] = {
        "pixel_values": [tensor.unsqueeze(0) for tensor in tensors],
        "imgs_sizes": [
            torch.tensor([[image.height, image.width]], dtype=torch.int32)
            for image in images
        ],
        "num_frames": [torch.ones(1, dtype=torch.int32) for _ in images],
    }
    return {
        key: PackedTensor(
            values,
            0,
            pad_to_max_shape=key == "pixel_values",
            _row_offsets=[0, len(values)],
            _segment_indices=list(range(len(values))),
        )
        for key, values in fields.items()
    }
