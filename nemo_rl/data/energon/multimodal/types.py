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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from megatron.energon import Sample, edataclass

MediaMetadataValue: TypeAlias = str | int | float | bool | None
FrozenMediaMetadata: TypeAlias = tuple[tuple[str, MediaMetadataValue], ...]


def freeze_media_metadata(metadata: object = None) -> FrozenMediaMetadata:
    """Validate and freeze media metadata without reading the media payload."""
    if metadata is None:
        return ()
    if not isinstance(metadata, Mapping):
        raise ValueError("Media metadata must be an object when present.")

    frozen: list[tuple[str, MediaMetadataValue]] = []
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Media metadata keys must be non-empty strings.")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError(
                f"Media metadata value for {key!r} must be a scalar or null."
            )
        frozen.append((key, value))
    return tuple(sorted(frozen))


@dataclass(frozen=True)
class MediaRef:
    """One ordered media occurrence in a conversation."""

    modality: str
    value: Any
    metadata: FrozenMediaMetadata = ()


@edataclass
class CanonicalSFTSample(Sample):
    """Model-neutral conversation produced by an Energon cooker."""

    messages: list[dict[str, Any]]
    media: list[MediaRef]
    tools: list[dict[str, Any]] | None


@edataclass
class EncodedSFTSample(Sample):
    """One tokenized message log before batching."""

    message_log: list[dict[str, Any]]
    length: int
    packing_cost: int
    loss_multiplier: float
    group_key: tuple[Any, ...]
    sample_key: str
    pending_sample: CanonicalSFTSample | None = None


__all__ = [
    "CanonicalSFTSample",
    "EncodedSFTSample",
    "FrozenMediaMetadata",
    "MediaRef",
    "MediaMetadataValue",
    "freeze_media_metadata",
]
