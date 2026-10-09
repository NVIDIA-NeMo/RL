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

"""Shared-prefix row metadata carried through NeMo RL batches.

Planning is owned by :mod:`megatron.rl.shared_prefix_metadata`; callers
import it lazily so driver-side modules that only route batches stay
importable without the optional Megatron installation.
"""

from __future__ import annotations

__all__ = [
    "SHARED_PREFIX_EXECUTION_SLOT",
    "SHARED_PREFIX_GROUP_ID",
    "SHARED_PREFIX_PROMPT_LENGTHS",
    "group_id_from_sample_id",
    "parse_grouped_sample_id",
]

SHARED_PREFIX_GROUP_ID = "shared_prefix_group_id"
SHARED_PREFIX_PROMPT_LENGTHS = "shared_prefix_prompt_lengths"
SHARED_PREFIX_EXECUTION_SLOT = "_shared_prefix_execution_slot"


def parse_grouped_sample_id(sample_id: str) -> tuple[str, int]:
    """Parse a TQ sample ID following the ``{group_id}_g{index}`` contract."""
    if not isinstance(sample_id, str):
        raise TypeError(
            f"shared-prefix sample IDs must be strings, got {type(sample_id).__name__}"
        )
    group_id, separator, generation_index = sample_id.rpartition("_g")
    if (
        not separator
        or not group_id
        or not generation_index.isascii()
        or not generation_index.isdigit()
    ):
        raise ValueError(
            "shared-prefix sample IDs must use the form '{group_id}_g{index}', "
            f"got {sample_id!r}"
        )
    return group_id, int(generation_index)


def group_id_from_sample_id(sample_id: str) -> str:
    """Recover the TQ prompt-group prefix from ``{group_id}_g{index}``."""
    group_id, _ = parse_grouped_sample_id(sample_id)
    return group_id
