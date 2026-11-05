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

from io import BytesIO
from typing import Any, Literal, cast

import torch


def decode_selected_av_bytes(
    value: bytes | bytearray | memoryview,
    *,
    modality: Literal["audio", "video"],
) -> Any:
    """Decode one selected raw AV member after packing has selected its row."""
    from megatron.energon.av import AVDecoder

    decoder = AVDecoder(BytesIO(bytes(value)))
    if modality == "audio":
        clips = decoder.get_audio().audio_clips
        if not clips:
            raise ValueError("Selected audio member must decode to at least one clip.")
        return clips

    clips = decoder.get_video().video_clips
    if len(clips) != 1:
        raise ValueError(
            f"Selected {modality} member must decode to one clip; got {len(clips)}."
        )
    return clips[0]


def materialize_media_value(
    value: Any,
    *,
    modality: str,
    sample: Any = None,
) -> Any:
    """Resolve one Energon lazy value and decode AV payloads for a processor."""
    payload = value.get(sample) if callable(getattr(value, "get", None)) else value
    if isinstance(payload, tuple) and len(payload) == 1:
        payload = payload[0]
    if modality == "image":
        return payload
    if modality not in {"audio", "video"}:
        raise ValueError(f"Unsupported media modality {modality!r}.")
    if isinstance(payload, (bytes, bytearray, memoryview)):
        payload = decode_selected_av_bytes(
            payload,
            modality=cast(Literal["audio", "video"], modality),
        )
    elif modality == "video" and callable(getattr(payload, "get_video", None)):
        clips = payload.get_video().video_clips
        if len(clips) != 1:
            raise ValueError(
                f"Selected video must decode to one clip; got {len(clips)}."
            )
        payload = clips[0]
    elif modality == "audio" and callable(getattr(payload, "get_audio", None)):
        payload = payload.get_audio().audio_clips
    if modality == "audio" and isinstance(payload, (tuple, list)):
        if not payload:
            raise ValueError("Selected audio must decode to at least one clip.")
        payload = torch.cat([torch.as_tensor(clip) for clip in payload], dim=-1)
    return payload


__all__ = ["decode_selected_av_bytes", "materialize_media_value"]
