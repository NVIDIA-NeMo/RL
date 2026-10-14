# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Production native codec and row assembly using CPU tensors and a memory DP."""

from array import array
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from nemo_rl.data_plane.direct_capture import NativeCallStore
from nemo_rl.experience.direct_capture import NativeCall, PackedNativeCall, extract_call
from nemo_rl.experience.rollout_reassembler import RolloutReassembler


class MemoryDataPlane:
    def __init__(self):
        self.rows = {}
        self.tags = {}

    def put_samples(self, *, sample_ids, partition_id, fields, tags):
        for index, key in enumerate(sample_ids):
            self.rows[partition_id, key] = {
                name: value[index].clone() for name, value in fields.items()
            }
            self.tags[partition_id, key] = tags[index]

    def get_samples(self, *, sample_ids, partition_id, select_fields):
        return {
            field: [self.rows[partition_id, key][field] for key in sample_ids]
            for field in select_fields
        }

    def clear_samples(self, *, sample_ids, partition_id):
        for key in sample_ids:
            self.rows.pop((partition_id, key))


@pytest.fixture(params=["legacy", "fused_list", "fused_buffer", "packed_logprobs"])
def token_packing(request):
    return request.param


def captured_call(call, packing):
    prompt = list(call.prompt) if packing != "fused_buffer" else array("q", call.prompt)
    result = extract_call(
        call.rollout_id,
        call.call_id,
        call.ordinal,
        call.weight_version,
        [
            NS(
                prompt_token_ids=prompt,
                finished=True,
                outputs=[
                    NS(
                        index=0,
                        token_ids=list(call.tokens),
                        finish_reason="stop",
                        logprobs=[
                            {token: NS(logprob=value)}
                            for token, value in zip(call.tokens, call.logprobs)
                        ],
                        routed_experts=call.routed_experts,
                    )
                ],
            )
        ],
        delta=False,
        packing="legacy" if packing == "legacy" else "fused",
    )
    if packing == "packed_logprobs":
        result = replace(result, logprobs=array("f", result.logprobs))
    return result


def stage_rollout(dp, *, routed=False, packing="legacy"):
    store = NativeCallStore(dp, "staging", router_replay=routed)
    calls = [
        NativeCall(
            "r",
            "a",
            0,
            4,
            (1, 2),
            (3,),
            (-0.25,),
            torch.zeros(2, 2, 2, dtype=torch.int16) if routed else None,
        ),
        NativeCall(
            "r",
            "fork",
            1,
            4,
            (1, 2),
            (8,),
            (-0.5,),
            torch.ones(2, 2, 2, dtype=torch.int16) if routed else None,
        ),
        NativeCall(
            "r",
            "b",
            2,
            5,
            (1, 2, 3, 4),
            (5, 6),
            (-0.75, -0.5),
            torch.full((5, 2, 2), 2, dtype=torch.int16) if routed else None,
        ),
    ]
    return {
        "format": "nrl_direct_v1",
        "rollout_id": "r",
        "failure": None,
        "manifest": [store.stage(captured_call(call, packing)) for call in calls],
    }


def test_native_roundtrip_main_branch_masks_and_excluded_calls(token_packing):
    dp = MemoryDataPlane()
    receipt = stage_rollout(dp, routed=True, packing=token_packing)
    assembler = RolloutReassembler(
        dp,
        partition_id="train",
        staging_partition="staging",
        pad_token_id=0,
        max_seq_len=32,
        router_replay_enabled=True,
    )
    row = assembler.finalize_rollout("r", receipt, reward=1)
    assert row.valid, row.rejection_reason
    assert row.token_ids == [1, 2, 3, 4, 5, 6]
    assert row.token_mask == [0, 0, 1, 0, 1, 1]
    assert row.logprobs == [0, 0, -0.25, 0, -0.75, -0.5]
    assert row.reward == 1
    assert (row.min_wv, row.max_wv) == (4, 5)
    assert row.reward_metrics["finalize/direct_excluded_calls"] == 1
    assert torch.equal(row.routed_experts[1], torch.zeros(2, 2, dtype=torch.int16))
    assert torch.equal(row.routed_experts[3], torch.full((2, 2), 2, dtype=torch.int16))
    assert row.routed_experts[-1].eq(-1).all()


def test_native_group_publishes_train_rows_and_clears_all_calls_including_excluded(
    token_packing,
):
    dp = MemoryDataPlane()
    receipt = stage_rollout(dp, packing=token_packing)
    assembler = RolloutReassembler(
        dp,
        partition_id="train",
        staging_partition="staging",
        pad_token_id=0,
        max_seq_len=32,
    )
    group = assembler.finalize_group(
        "group",
        ["r", "missing"],
        [receipt, None],
        [1, 0],
        mask_sample=[False, False],
        fallback_weight_version=0,
        prompt_idx=0,
        canonical_sample_ids=["group_g0", "group_g1"],
    )
    assert group.valid_row_count == 1
    assert group.total_row_count == 2
    assert group.canonical_output_tokens == 3
    assert group.meta.sample_ids == ["group_g0", "group_g1"]
    assert not any(partition == "staging" for partition, _ in dp.rows)
    assert dp.rows["train", "group_g1"]["sample_mask"].item() == 0


def test_native_receipt_rejects_tampered_identity(token_packing):
    dp = MemoryDataPlane()
    receipt = stage_rollout(dp, packing=token_packing)
    receipt["manifest"][0]["generation_length"] = 900
    assembler = RolloutReassembler(
        dp,
        partition_id="train",
        staging_partition="staging",
        pad_token_id=0,
        max_seq_len=32,
    )
    row = assembler.finalize_rollout("r", receipt, reward=1)
    assert not row.valid
    assert "identity mismatch" in row.rejection_reason


@pytest.mark.parametrize("invalid", ["dtype", "prompt", "logprobs", "empty_generation"])
def test_packed_staging_rejects_inconsistent_token_layout(invalid):
    call = PackedNativeCall("r", "c", 0, 0, array("q", [1, 2, 3]), 2, (-0.25,), None)
    if invalid == "dtype":
        call = replace(call, token_ids=array("i", [1, 2, 3]))
    elif invalid == "prompt":
        call = replace(call, prompt_length=-1)
    elif invalid == "logprobs":
        call = replace(call, logprobs=())
    else:
        call = replace(call, prompt_length=3)
    with pytest.raises(ValueError, match="packed native call"):
        NativeCallStore(MemoryDataPlane(), "staging", router_replay=False).stage(call)


def test_packed_logprobs_stage_as_a_view_and_reject_wrong_dtype():
    values = array("f", [-0.25, -0.5])
    call = PackedNativeCall("r", "c", 0, 0, array("q", [1, 2, 3]), 1, values, None)

    class InspectDataPlane(MemoryDataPlane):
        def put_samples(self, **kwargs):
            assert (
                kwargs["fields"]["direct_logprobs"].data_ptr()
                == values.buffer_info()[0]
            )
            super().put_samples(**kwargs)

    store = NativeCallStore(InspectDataPlane(), "staging", router_replay=False)
    manifest = store.stage(call)
    values[0] = 99
    assert store.fetch([manifest], "r")[0].logprobs == (-0.25, -0.5)
    with pytest.raises(ValueError, match="float32"):
        store.stage(replace(call, logprobs=array("d", [-0.25, -0.5])))
