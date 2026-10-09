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

"""Shared-prefix prompt-length row tags for work-weighted sharding.

The cost model itself is owned by :mod:`megatron.rl.shared_prefix_cost`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nemo_rl.data.packing.shared_prefix_metadata import SHARED_PREFIX_PROMPT_LENGTHS

__all__ = ["with_prompt_length_tags"]


def with_prompt_length_tags(
    tags: Sequence[Mapping[str, Any]] | None,
    *,
    prompt_lengths: Sequence[int],
    sequence_lengths: Sequence[int],
) -> list[dict[str, Any]]:
    """Carry prompt lengths through the metadata's existing row transforms."""
    if len(prompt_lengths) != len(sequence_lengths):
        raise ValueError("Prompt lengths must align with sequence lengths")
    if tags is not None and len(tags) != len(prompt_lengths):
        raise ValueError("Prompt lengths must align with row tags")
    result = (
        [dict(tag) for tag in tags]
        if tags is not None
        else [{} for _ in prompt_lengths]
    )
    for tag, prefix, length in zip(
        result, prompt_lengths, sequence_lengths, strict=True
    ):
        if type(prefix) is not int or not 0 <= prefix <= length:
            raise ValueError(
                "Prompt length must be an integer within the real sequence"
            )
        if (
            SHARED_PREFIX_PROMPT_LENGTHS in tag
            and tag[SHARED_PREFIX_PROMPT_LENGTHS] != prefix
        ):
            raise ValueError("Prompt length tag disagrees with its tensor field")
        tag[SHARED_PREFIX_PROMPT_LENGTHS] = prefix
    return result
