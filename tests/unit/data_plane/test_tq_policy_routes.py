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
"""TQPolicy direct/deferred route field selection tests."""

from __future__ import annotations

import pytest

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import (
    ROUTE_PASSTHROUGH_FLAG,
    ROUTE_PLAN_TAG,
    ROUTED_EXPERTS_FIELD,
)
from nemo_rl.models.policy.tq_policy import TQPolicy


def _policy() -> TQPolicy:
    policy = object.__new__(TQPolicy)
    policy._router_replay_enabled = True
    return policy


def _meta(tags) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="canonical",
        task_name="train",
        sample_ids=[f"row{i}" for i in range(len(tags))],
        fields=["input_ids"],
        sequence_lengths=[2] * len(tags),
        tags=tags,
    )


def test_deferred_prev_lp_and_train_omit_canonical_route_field() -> None:
    meta = _meta([{ROUTE_PLAN_TAG: {"plan": 0}}, {ROUTE_PLAN_TAG: {"plan": 1}}])

    result = _policy()._with_route_fields(
        meta,
        ("input_ids",),
        task_name="prev_lp",
        want_routes=True,
    )

    assert ROUTED_EXPERTS_FIELD not in result.fields
    assert result.extra_info[ROUTE_PASSTHROUGH_FLAG] is True


def test_reference_lp_never_requests_or_materializes_routes() -> None:
    meta = _meta([{ROUTE_PLAN_TAG: {"plan": 0}}])

    result = _policy()._with_route_fields(
        meta,
        ("input_ids",),
        task_name="ref_lp",
        want_routes=False,
    )

    assert ROUTED_EXPERTS_FIELD not in result.fields
    assert ROUTE_PASSTHROUGH_FLAG not in result.extra_info


def test_direct_storage_keeps_canonical_route_field() -> None:
    result = _policy()._with_route_fields(
        _meta([{"weight_version": 1}]),
        ("input_ids",),
        task_name="train",
        want_routes=True,
    )
    assert ROUTED_EXPERTS_FIELD in result.fields
    assert ROUTE_PASSTHROUGH_FLAG not in result.extra_info


def test_mixed_direct_and_deferred_rows_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="mixed direct/deferred"):
        _policy()._with_route_fields(
            _meta([{ROUTE_PLAN_TAG: {}}, {"weight_version": 1}]),
            ("input_ids",),
            task_name="train",
            want_routes=True,
        )
