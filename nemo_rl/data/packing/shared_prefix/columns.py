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
"""Names of the row metadata that shared-prefix packing reads and writes."""

SHARED_PREFIX_PROMPT_LENGTHS = "shared_prefix_prompt_lengths"
"""Column: tokens at the start of a row that it may share with rows of its group; 0 when none."""

SHARED_PREFIX_PROMPT_HASHES = "shared_prefix_prompt_hashes"
"""Column: hash of those prompt tokens, so the driver can match prompts without reading them."""

SHARED_PREFIX_TREE_IDS = "shared_prefix_tree_ids"
"""Per-row plan entry: the row's prefix tree within its forward, or ``UNSHARED_ROW``."""

UNSHARED_ROW = -1
"""Tree id of a row that runs as an ordinary packed sequence."""
