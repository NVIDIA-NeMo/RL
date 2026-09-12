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

"""Stable sample identities shared by draft-training input paths."""

import hashlib
from collections.abc import Sequence


def stable_draft_sample_ids(sample_ids: Sequence[str]) -> list[int]:
    """Map unique string identities to deterministic signed-int64 values."""
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("draft training requires unique stable sample IDs")
    return [
        int.from_bytes(
            hashlib.blake2b(
                sample_id.encode("utf-8"),
                digest_size=8,
                person=b"NRLdraft",
            ).digest(),
            "little",
        )
        & ((1 << 63) - 1)
        for sample_id in sample_ids
    ]
