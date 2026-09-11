# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Per-segment image processing using the ordinary packed-media TQ columns."""

import json
from contextlib import ExitStack, closing
from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDict

from nemo_rl.data.multimodal_utils import (
    PACKED_MULTIMODAL_FIELDS,
    PackedTensor,
    attach_image_model_inputs_to_message,
    encode_multimodal_for_wire,
    get_responses_content_part_url,
    multimodal_row_tags,
    reassemble_packed_multimodal,
    resolve_to_image,
    uses_image_placeholder,
)
from nemo_rl.data_plane.interfaces import DataPlaneClient
from nemo_rl.data_plane.tq_token_sink import FetchedStagedCall
from nemo_rl.experience.route_assembly import verify_route_fragment_integrity


@dataclass(frozen=True)
class SegmentMedia:
    """Metadata only; tensor columns share the selected terminal's staging key.

    Existing receipt ownership therefore also owns media cleanup. Counts retain
    the per-action processing groups needed to compare worker image geometry.
    """

    field_names: tuple[str, ...]
    row_tags: dict[str, Any]
    occurrence_counts: tuple[int, ...]


def stage_segment_media(
    data_plane: DataPlaneClient,
    *,
    staging_partition: str,
    terminal_staging_key: str,
    media_assets: dict[str, dict[str, Any]],
    action_occurrences: list[list[str]],
    processor: Any,
    pad_dynamic_image_shapes: bool,
) -> SegmentMedia | None:
    """Process each action's new images as in #3910, then stage one physical row.

    Only media columns are added to the existing terminal call; captured token
    and route columns are never rewritten. The caller has already resolved this
    key from the selected response's manifest and must not retry uncertain puts.
    """
    if not any(action_occurrences):
        return None
    if processor is None or not uses_image_placeholder(processor):
        raise ValueError("CC images require the existing placeholder-style processor")
    turns: list[dict[str, PackedTensor]] = []
    for occurrences in action_occurrences:
        if not occurrences:
            continue
        message: dict[str, Any] = {}
        with ExitStack() as cleanup:
            images = []
            for media_id in occurrences:
                part = media_assets[media_id]["source_part"]
                source = get_responses_content_part_url(
                    part, "image", "image_url", "url"
                )
                if not source:
                    raise ValueError(f"Media asset {media_id!r} has no image source")
                images.append(cleanup.enter_context(closing(resolve_to_image(source))))
            attach_image_model_inputs_to_message(
                message,
                images=images,
                processor=processor,
                pad_dynamic_image_shapes=pad_dynamic_image_shapes,
            )
        if not message or set(message) - PACKED_MULTIMODAL_FIELDS:
            raise ValueError(
                "Processor returned no media or unregistered packed fields"
            )
        if turns and set(message) != set(turns[0]):
            raise ValueError("Processor media fields changed between actions")
        turns.append(message)

    # One physical row contains several original per-action tensor segments.
    # Keep their shapes; flattening through as_tensor here would pad them early.
    packed = {}
    for key in turns[0]:
        values = [turn[key] for turn in turns]
        merged = PackedTensor.concat(values)
        tensors = [tensor for tensor in merged.tensors if tensor is not None]
        packed[key] = PackedTensor(
            tensors,
            merged.dim_to_pack,
            pad_to_max_shape=merged.pad_to_max_shape,
            _row_offsets=[0, len(tensors)],
            _segment_indices=list(range(len(tensors))),
        )
    tags = multimodal_row_tags(packed, 1)
    if tags is None:
        raise ValueError("Selected images produced no media tensor segments")
    wire = {
        key: encode_multimodal_for_wire(key, value) for key, value in packed.items()
    }
    if any(value is None for value in wire.values()):
        raise ValueError("Selected images produced an empty media field")
    data_plane.put_samples(
        sample_ids=[terminal_staging_key],
        partition_id=staging_partition,
        fields=TensorDict(wire, batch_size=[1]),
        tags=tags,
    )
    return SegmentMedia(
        tuple(sorted(wire)), tags[0], tuple(map(len, action_occurrences))
    )


def fetch_segment_media(
    data_plane: DataPlaneClient,
    *,
    staging_partition: str,
    terminal_staging_key: str,
    descriptor: SegmentMedia,
) -> dict[str, PackedTensor]:
    """Read only registered image columns; restore their original segment shapes."""
    names = descriptor.field_names
    if (
        not names
        or len(set(names)) != len(names)
        or set(names) - PACKED_MULTIMODAL_FIELDS
    ):
        raise ValueError("Invalid processed-media field names")
    wire = data_plane.get_samples(
        sample_ids=[terminal_staging_key],
        partition_id=staging_partition,
        select_fields=list(names),
    )
    if tuple(wire.batch_size) != (1,):
        raise ValueError("Missing selected terminal media row")
    # A one-row/equal-length read may be dense even though the writer was
    # jagged. Restore that wire layout before the ordinary media decoder.
    fields = {
        key: torch.nested.as_nested_tensor([wire[key][0]], layout=torch.jagged)
        for key in names
    }
    try:
        reassemble_packed_multimodal(fields, [descriptor.row_tags])
    except RuntimeError as error:
        # Torch reports malformed split/reshape sizes as RuntimeError. This is
        # bad row geometry, not an uncertain TQ operation or a group failure.
        raise ValueError(
            "Processed media shape metadata does not match its payload"
        ) from error
    return fields


def verify_image_alignment(
    calls: list[FetchedStagedCall],
    *,
    descriptor: SegmentMedia | None,
    media: dict[str, PackedTensor],
) -> None:
    """Match selected, already token-verified calls to per-action training images.

    Existing extras commitments authenticate geometry. Prefix digests check
    retained engine images against earlier deltas, including image-free turns.
    New ranges must lie wholly in the call's carried prompt, never its output.
    """
    # Gym is optional outside captured rollouts.
    from nemo_gym.token_id_capture.staging.digest import compute_extras_digest

    metadata = [json.loads(call.extras_metadata_json) for call in calls]
    if descriptor is None and not any(
        isinstance(extra, dict) and "cc_image_geometry" in extra for extra in metadata
    ):
        return  # Text-only receipt predating the optional media contract.
    counts = (
        descriptor.occurrence_counts if descriptor is not None else (0,) * len(calls)
    )
    if len(counts) != len(calls) or any(
        type(count) is not int or count < 0 for count in counts
    ):
        raise ValueError("Image occurrence counts must cover selected actions")
    retained = []
    for call, extra, count in zip(calls, metadata, counts, strict=True):
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
            raise ValueError("Image geometry extras commitment mismatch")
        geometry = extra.get("cc_image_geometry")
        if not isinstance(geometry, dict) or set(geometry) != {
            "prefix_digest",
            "images",
        }:
            raise ValueError("Missing or malformed worker image geometry")
        if geometry["prefix_digest"] != compute_extras_digest({"images": retained}):
            raise ValueError(
                "Retained engine image geometry changed within the segment"
            )
        images = geometry["images"]
        if not isinstance(images, list) or len(images) != count:
            raise ValueError("Worker/agent image occurrence counts disagree")
        carry_len = snapshot.token_mask_delta.index(1.0)
        previous_end = snapshot.prev_len
        for image in images:
            if (
                not isinstance(image, dict)
                or set(image) != {"offset", "length", "height", "width", "token_id"}
                or any(type(value) is not int for value in image.values())
                or min(image["length"], image["height"], image["width"]) <= 0
                or image["token_id"] < 0
            ):
                raise ValueError("Malformed worker image occurrence")
            start, end = image["offset"], image["offset"] + image["length"]
            if start < previous_end or end > snapshot.prev_len + carry_len:
                raise ValueError("Image span lies outside the new carried prompt")
            if any(
                token != image["token_id"]
                for token in snapshot.token_ids_delta[
                    start - snapshot.prev_len : end - snapshot.prev_len
                ]
            ):
                raise ValueError("Image span does not match exact captured tokens")
            previous_end = end
            retained.append(image)
    if not retained:
        if media or descriptor is not None:
            raise ValueError("Processed media exists without engine image occurrences")
        return
    if set(media) != {"pixel_values", "imgs_sizes", "num_frames"}:
        raise ValueError(
            "CC image alignment requires dynamic-image pixels, sizes and frame counts"
        )
    groups = [count for count in counts if count]
    fields = [media[key] for key in ("pixel_values", "imgs_sizes", "num_frames")]
    if any(len(value) != 1 or len(value.tensors) != len(groups) for value in fields):
        raise ValueError(
            "Processed media must preserve each selected action's tensor group"
        )
    cursor = 0
    for count, pixels, sizes, frames in zip(
        groups, *(value.tensors for value in fields), strict=True
    ):
        if (
            pixels is None
            or sizes is None
            or frames is None
            or pixels.ndim != 4
            or tuple(pixels.shape[:2]) != (count, 3)
            or tuple(sizes.shape) != (count, 2)
            or tuple(frames.shape) != (count,)
            or sizes.is_floating_point()
            or frames.is_floating_point()
            or not bool((frames == 1).all())
        ):
            raise ValueError("Processed per-action image geometry is malformed")
        expected = [
            [image["height"], image["width"]]
            for image in retained[cursor : cursor + count]
        ]
        if sizes.tolist() != expected or tuple(pixels.shape[-2:]) != tuple(
            map(max, zip(*expected))
        ):
            raise ValueError(
                "Training image geometry differs from the actual engine input"
            )
        cursor += count
