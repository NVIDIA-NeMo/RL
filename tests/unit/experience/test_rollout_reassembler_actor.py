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
"""Metadata-only finalizer actor boundary tests."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.experience.rollout_reassembler import FinalizedGroup
from nemo_rl.experience.rollout_reassembler_actor import (
    _FORBIDDEN_RPC_KEYS,
    ReassemblyRequest,
    assert_metadata_only,
)


def _request() -> ReassemblyRequest:
    return ReassemblyRequest(
        group_id="group",
        prompt_idx=17,
        rollout_ids=("group_g0",),
        canonical_sample_ids=("group_g0",),
        receipts=(
            {
                "rollout_id": "group_g0",
                "manifest": [
                    {
                        "call_id": "call",
                        "staging_key": "group_g0/call",
                        "delta_len": 2,
                    }
                ],
            },
        ),
        rewards=(1.0,),
        mask_sample=(False,),
        fallback_weight_version=4,
    )


def test_finalizer_request_and_result_are_metadata_only() -> None:
    assert_metadata_only(_request())
    result = FinalizedGroup(
        meta=KVBatchMeta(
            partition_id="canonical",
            task_name="train",
            sample_ids=["group_g0"],
            fields=["input_ids"],
            sequence_lengths=[3],
            tags=[{"weight_version": 4}],
        ),
        group_min_wv=4,
        group_max_wv=4,
        staging_keys=["group_g0/call"],
        metrics={"finalize/total_ms": 1.0},
    )
    assert_metadata_only(result)


@pytest.mark.parametrize(
    "payload",
    [
        torch.ones(2),
        {"input_ids": [1, 2]},
        {"routed_experts": [[[[1, 2]]]]},
    ],
)
def test_metadata_guard_rejects_tensor_and_heavy_row_payloads(payload) -> None:
    with pytest.raises(TypeError):
        assert_metadata_only(payload)


def test_rpc_dataclass_fields_are_classified() -> None:
    """A new field on either RPC dataclass must be a deliberate choice.

    assert_metadata_only cannot tell a heavy list[int] of token ids from a short
    list of metadata, so _FORBIDDEN_RPC_KEYS is maintained by hand. Pinning the
    inventory makes a new field fail here until someone decides whether it is
    light enough to cross the wire.
    """
    assert {f.name for f in fields(ReassemblyRequest)} == {
        "group_id",
        "rollout_ids",
        "receipts",
        "rewards",
        "fallback_weight_version",
        "prompt_idx",
        "mask_sample",
    }
    assert {f.name for f in fields(FinalizedGroup)} == {
        "meta",
        "group_min_wv",
        "group_max_wv",
        "staging_keys",
        "metrics",
        "dropped",
        "drop_reason",
        "valid_row_count",
        "total_row_count",
    }


@pytest.mark.parametrize("key", sorted(_FORBIDDEN_RPC_KEYS))
def test_every_forbidden_key_is_rejected(key) -> None:
    """Removing an entry from the denylist should fail loudly."""
    with pytest.raises(TypeError, match="forbidden heavy field"):
        assert_metadata_only({key: [1, 2, 3]})
