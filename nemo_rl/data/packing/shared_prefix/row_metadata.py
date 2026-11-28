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
"""Row metadata of shared-prefix execution.

The rollout reassembler stamps two tags on every row it writes. The driver plans forwards from
those tags and ships one tree id per row to each worker. The worker turns the tags and tree ids
into batch columns.
"""

import hashlib
from array import array
from collections.abc import Sequence

PROMPT_LENGTH_TAG = "shared_prefix_prompt_length"
"""Row tag: tokens at the start of the row that it may share with rows of its group; 0 for none."""

PROMPT_HASH_TAG = "shared_prefix_prompt_hash"
"""Row tag: ``prompt_hash`` of those tokens, so the driver can match prompts without reading them."""

SHARED_PREFIX_TREE_IDS = "shared_prefix_tree_ids"
"""Plan entry in ``KVBatchMeta.extra_info`` and batch column: each row's tree within its forward."""

SHARED_PREFIX_PROMPT_LENGTHS = "shared_prefix_prompt_lengths"
"""Batch column the worker builds from ``PROMPT_LENGTH_TAG``."""

UNSHARED_ROW = -1
"""Tree id of a row that runs as an ordinary packed sequence."""


def prompt_hash(token_ids: Sequence[int]) -> int:
    """Hash prompt tokens to a non-negative int64 that is the same in every process."""
    digest = hashlib.blake2b(array("q", token_ids).tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "little") >> 1
