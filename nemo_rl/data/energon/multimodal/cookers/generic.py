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

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any

from megatron.energon import CrudeSample, basic_sample_keys, stateless

from nemo_rl.data.energon.multimodal.model_families import (
    ALL_MODEL_FAMILIES,
    supports_model_families,
)
from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    MediaRef,
    freeze_media_metadata,
)

_MEDIA_TYPES = frozenset({"image", "video", "audio"})


def _decode_json_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        decoded = json.loads(value)
        if isinstance(decoded, dict):
            return decoded
    raise ValueError("The SFT payload must decode to a JSON object.")


def _get_crude_payload(sample: CrudeSample) -> dict[str, Any]:
    if "messages" in sample:
        return dict(sample)
    for key in ("json", "data", "metadata"):
        if key in sample:
            return _decode_json_payload(sample[key])
    raise ValueError("Energon SFT samples must contain 'messages' or a JSON payload.")


def _infer_modality(member: str) -> str:
    suffix = PurePosixPath(member).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return "image"
    if suffix in {".mp4", ".webm", ".mov", ".mkv"}:
        return "video"
    if suffix in {".wav", ".flac", ".mp3", ".ogg"}:
        return "audio"
    raise ValueError(f"Cannot infer media type from member {member!r}.")


def _get_media_value(sample: CrudeSample, entry: dict[str, Any]) -> Any:
    modality = entry.get("type")
    for key in ("value", modality, "image", "video", "audio"):
        if key and key in entry:
            return entry[key]

    member = entry.get("member")
    if not isinstance(member, str) or not member:
        raise ValueError("Each media entry needs a value or shard member name.")

    sample_key = str(sample.get("__key__", ""))
    member_path = PurePosixPath(member)
    candidates = [member, member_path.name]
    if sample_key and member.startswith(f"{sample_key}."):
        candidates.append(member.removeprefix(f"{sample_key}."))
    # The bare extension is the single-media-per-sample fallback. It cannot tell
    # two members sharing a suffix apart, so it must be tried last: otherwise
    # "<key>.first.jpg" and "<key>.second.jpg" both resolve to a bare "jpg".
    candidates.append(member_path.suffix.removeprefix("."))
    for candidate in candidates:
        if candidate and candidate in sample:
            return sample[candidate]
    raise ValueError(
        f"Media member {member!r} is absent from Energon sample {sample_key!r}."
    )


@supports_model_families(ALL_MODEL_FAMILIES)
@stateless
def cook_conversation(sample: CrudeSample) -> CanonicalSFTSample:
    """Convert one crude JSON-plus-media sample into canonical SFT data."""
    payload = _get_crude_payload(sample)
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Energon SFT samples require a non-empty messages list.")

    media: list[MediaRef] = []
    raw_media = payload.get("media", [])
    if not isinstance(raw_media, list):
        raise ValueError("The media manifest must be a list.")
    for raw_entry in raw_media:
        entry = {"member": raw_entry} if isinstance(raw_entry, str) else raw_entry
        if not isinstance(entry, dict):
            raise ValueError("Each media manifest entry must be a string or object.")
        member = entry.get("member")
        modality = entry.get("type")
        if modality is None and isinstance(member, str):
            modality = _infer_modality(member)
        if modality not in _MEDIA_TYPES:
            raise ValueError(f"Unsupported media type {modality!r}.")
        assert isinstance(modality, str)
        media.append(
            MediaRef(
                modality=modality,
                value=_get_media_value(sample, entry),
                metadata=freeze_media_metadata(entry.get("metadata")),
            )
        )

    tools = payload.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise ValueError("The tools field must be a list when present.")
    return CanonicalSFTSample(
        **basic_sample_keys(sample),
        messages=deepcopy(messages),
        media=media,
        tools=deepcopy(tools),
    )


__all__ = ["cook_conversation"]
