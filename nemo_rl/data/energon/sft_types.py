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

"""Lightweight value types shared by the SFTv2 driver and policy workers."""

from dataclasses import dataclass

from nemo_rl.data_plane.interfaces import KVBatchMeta


@dataclass(frozen=True)
class StepEnvelope:
    """Metadata for one prepared batch owned by one logical data rank."""

    meta: KVBatchMeta
    logical_rank: int
    logical_world_size: int
    source_ids: tuple[str, ...]
    field_names: tuple[str, ...]
    sequence_lengths: tuple[int, ...]
    load_seconds: float
    valid_tokens: int


__all__ = ["StepEnvelope"]
