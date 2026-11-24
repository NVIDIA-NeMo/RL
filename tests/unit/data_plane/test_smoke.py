# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Tier-0 smoke tests — pre-commit gates.

Cheapest tier: catches drift in module paths, registry keys, and the
public ABC surface. Each test runs in milliseconds and never touches
real Ray / vLLM / TQ.
"""

from __future__ import annotations

import inspect


def test_kvbatchmeta_schema_unchanged() -> None:
    """Schema break check — KVBatchMeta is the cross-process boundary;
    adding/removing a field silently would break adapters that pickle it."""
    from nemo_rl.data_plane.interfaces import KVBatchMeta

    expected_fields = {
        "partition_id",
        "task_name",
        "sample_ids",
        "fields",
        "sequence_lengths",
        "extra_info",
        "tags",
    }
    actual_fields = {f.name for f in KVBatchMeta.__dataclass_fields__.values()}
    assert actual_fields == expected_fields, (
        f"KVBatchMeta schema drifted. expected={expected_fields}, "
        f"actual={actual_fields}"
    )


def test_dataplane_client_abc_surface() -> None:
    """Catches accidental ABC method removal / rename — e.g. dropping
    ``clear_samples`` would break step-end teardown silently."""
    from nemo_rl.data_plane.interfaces import DataPlaneClient

    expected_methods = {
        # task-mediated
        "register_partition",
        "claim_meta",
        "get_data",
        "check_consumption_status",
        # direct-by-key
        "put_samples",
        "get_samples",
        "list_sample_ids",
        "clear_samples",
        # lifecycle
        "close",
    }
    actual_methods = {
        name
        for name, member in inspect.getmembers(DataPlaneClient, callable)
        if not name.startswith("_") and getattr(member, "__isabstractmethod__", False)
    }
    assert expected_methods.issubset(actual_methods), (
        f"DataPlaneClient ABC missing methods: {expected_methods - actual_methods}"
    )
