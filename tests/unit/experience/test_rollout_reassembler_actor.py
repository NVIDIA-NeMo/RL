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

from dataclasses import fields, replace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.experience.cc_media import SegmentMedia
from nemo_rl.experience.rollout_reassembler import (
    ActionOutputFlags,
    FinalizedGroup,
    SegmentReceipt,
)
from nemo_rl.experience.rollout_reassembler_actor import (
    _FORBIDDEN_RPC_KEYS,
    ReassemblyRequest,
    RolloutReassemblerActor,
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


def test_finalize_forwards_loss_multiplier_to_reassembler() -> None:
    actor_cls = RolloutReassemblerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._finalizer = MagicMock()
    result = FinalizedGroup(
        meta=None,
        group_min_wv=4,
        group_max_wv=4,
        staging_keys=[],
        dropped=True,
        drop_reason="test",
    )
    actor._finalizer.finalize_group.return_value = result
    request = replace(_request(), loss_multiplier=0.25)

    assert actor.finalize(request) is result
    actor._finalizer.finalize_group.assert_called_once_with(
        "group",
        ["group_g0"],
        [request.receipts[0]],
        [1.0],
        mask_sample=[False],
        fallback_weight_version=4,
        prompt_idx=17,
        loss_multiplier=0.25,
        canonical_sample_ids=["group_g0"],
        logical_segments=None,
        execution_row_multiple=1,
    )


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
        "canonical_sample_ids",
        "receipts",
        "rewards",
        "fallback_weight_version",
        "prompt_idx",
        "mask_sample",
        "loss_multiplier",
        "logical_segments",
        "execution_row_multiple",
    }
    assert {f.name for f in fields(FinalizedGroup)} == {
        "meta",
        "group_min_wv",
        "group_max_wv",
        "staging_keys",
        "canonical_output_tokens",
        "metrics",
        "dropped",
        "drop_reason",
        "valid_row_count",
        "total_row_count",
    }
    assert {f.name for f in fields(SegmentReceipt)} == {
        "capture_rollout_id",
        "receipt",
        "selected_response_ids",
        "truncated",
        "media",
        "action_flags",
    }
    assert {f.name for f in fields(SegmentMedia)} == {
        "field_names",
        "row_tags",
        "occurrence_counts",
    }
    assert {f.name for f in fields(ActionOutputFlags)} == {
        "invalid_tool_call",
        "malformed_thinking",
    }


@pytest.mark.parametrize("key", sorted(_FORBIDDEN_RPC_KEYS))
def test_every_forbidden_key_is_rejected(key) -> None:
    """Removing an entry from the denylist should fail loudly."""
    with pytest.raises(TypeError, match="forbidden heavy field"):
        assert_metadata_only({key: [1, 2, 3]})


def test_ordinary_request_cleanup_coordinates_are_unchanged() -> None:
    request = _request()
    assert request.canonical_sample_ids == request.rollout_ids
    assert request.capture_receipts == request.receipts


@pytest.mark.nemo_gym
def test_logical_request_enumerates_declared_rows_and_empty_owner_placeholder() -> None:
    request = replace(
        _request(),
        rollout_ids=("group_g0", "group_g1"),
        canonical_sample_ids=("group_g0", "group_g1"),
        receipts=(None, None),
        rewards=(1.0, 0.0),
        mask_sample=(False, False),
        logical_segments=(
            (
                SegmentReceipt("group_g0_s0", None, ("response-0",)),
                SegmentReceipt("group_g0_s1", None, ("response-1",)),
            ),
            (),
        ),
    )
    assert_metadata_only(request)
    assert request.capture_receipts == (None, None)
    assert request.cleanup_sample_ids == ("group_g0_s0", "group_g0_s1", "group_g1_s0")
    padded = replace(request, execution_row_multiple=4)
    assert padded.cleanup_sample_ids == request.cleanup_sample_ids + (
        "group_pad0",
        "group_pad1",
        "group_pad2",
    )
    heavy = replace(
        request,
        logical_segments=(
            (SegmentReceipt("group_g0_s0", {"token_ids": [1, 2]}, ("response-0",)),),
            (),
        ),
    )
    with pytest.raises(TypeError, match="forbidden heavy field"):
        assert_metadata_only(heavy)


@pytest.mark.nemo_gym
@pytest.mark.parametrize(
    "mismatch", ["owner", "scope", "ordinal", "receipt", "staging_key"]
)
def test_cleanup_coordinates_reject_foreign_ownership(mismatch: str) -> None:
    # Optional Gym fixtures are required only for logical capture requests.
    from tests.unit.data_plane.token_capture_test_fixtures import (
        build_fixture_artifacts,
    )

    _, receipt, _ = build_fixture_artifacts("single_call", rollout_id="group_g0_s0")
    segment = SegmentReceipt("group_g0_s0", receipt.model_dump(), ("chatcmpl-c1",))
    request = replace(_request(), logical_segments=((segment,),))
    if mismatch == "owner":
        request = replace(request, rollout_ids=("other_g0",))
    elif mismatch == "scope":
        request = replace(
            request,
            logical_segments=((replace(segment, capture_rollout_id="other_g0_s0"),),),
        )
    elif mismatch == "ordinal":
        request = replace(request, logical_segments=((segment, segment),))
    elif mismatch == "receipt":
        segment.receipt["rollout_id"] = "other_g0_s0"
    else:
        segment.receipt["manifest"][0]["staging_key"] = "other_g0_s0/c1"
    with pytest.raises(ValueError):
        _ = request.capture_receipts
    with pytest.raises(ValueError):
        _ = request.cleanup_sample_ids


@pytest.mark.nemo_gym
def test_actual_actor_method_forwards_segments_to_real_finalizer() -> None:
    # These optional Gym/staging fixtures avoid starting a Ray cluster while
    # exercising the original decorated actor method, not an extracted copy.
    from nemo_rl.data_plane.tq_token_sink import TQTokenSink
    from nemo_rl.experience.rollout_reassembler import RolloutReassembler
    from tests.unit.data_plane.token_capture_test_fixtures import (
        build_fixture_artifacts,
    )
    from tests.unit.experience.test_logical_owner_finalization import (
        PublicationDataPlane,
    )

    data_plane = PublicationDataPlane()
    segments = []
    for scope in ("group_g0_s0", "group_g0_s1"):
        records, receipt, _ = build_fixture_artifacts("single_call", rollout_id=scope)
        for record in records:
            assert TQTokenSink(data_plane, staging_partition="staged").stage(record).ok
        segments.append(SegmentReceipt(scope, receipt.model_dump(), ("chatcmpl-c1",)))
    # Response IDs must be unique across the selected actions. They are custody
    # metadata, not digest-covered worker evidence.
    segments[1].receipt["manifest"][0]["response_id"] = "chatcmpl-second"
    segments[1] = replace(segments[1], selected_response_ids=("chatcmpl-second",))
    request = replace(
        _request(),
        receipts=(None,),
        logical_segments=(tuple(segments),),
        execution_row_multiple=4,
    )
    actor_class = RolloutReassemblerActor.__ray_metadata__.modified_class
    actor = actor_class.__new__(actor_class)
    actor._finalizer = RolloutReassembler(
        data_plane,
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=100,
    )
    result = actor.finalize(request)
    assert_metadata_only(result)
    assert result.meta.sample_ids == [
        "group_g0_s0",
        "group_g0_s1",
        "group_pad0",
        "group_pad1",
    ]
    assert set(result.meta.sample_ids) <= set(request.cleanup_sample_ids)
    assert result.meta.tags[1]["logical_rollout_id"] == "group_g0"
    assert result.meta.tags[1]["segment_index"] == 1
    assert len(request.capture_receipts) == 2
    assert data_plane.rows["canonical", "group_g0_s1"]["input_ids"][0].tolist() == [
        10,
        11,
        12,
        13,
    ]
    assert data_plane.events[-2][:2] == ("put", "canonical")
    assert data_plane.events[-1][:2] == ("clear", "staged")
