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

from unittest.mock import MagicMock, patch

import torch

from nemo_rl.algorithms.loss.wrapper import SequencePackingLossWrapper
from nemo_rl.algorithms.opd_packed import (
    OPD_TEACHER_TOPK_PACKED_KEY,
    materialize_teacher_topk_microbatch,
    pack_teacher_topk_for_replay,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_pack_teacher_topk_stores_only_real_next_token_positions():
    indices = torch.arange(2 * 5 * 2).reshape(2, 5, 2)
    logprobs = -indices.float()

    with patch(
        "nemo_rl.algorithms.opd_packed.ray.put", side_effect=lambda value: value
    ):
        packed = pack_teacher_topk_for_replay(indices, logprobs, torch.tensor([5, 3]))

    assert [entry["seq_len"] for entry in packed] == [4, 2]
    assert packed[0]["topk_indices_ref"].dtype == torch.int32
    torch.testing.assert_close(
        packed[0]["topk_indices_ref"], indices[0, :4].to(torch.int32)
    )
    torch.testing.assert_close(packed[1]["topk_logprobs_ref"], logprobs[1, :2])


def test_materialize_teacher_topk_builds_only_the_current_microbatch():
    entries = [
        {
            "seq_len": 3,
            "topk": 2,
            "topk_indices": torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32),
            "topk_logprobs": torch.tensor([[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]]),
        },
        {
            "seq_len": 1,
            "topk": 2,
            "topk_indices": torch.tensor([[7, 8]], dtype=torch.int32),
            "topk_logprobs": torch.tensor([[-0.7, -0.8]]),
        },
    ]
    data = BatchedDataDict(
        {
            "input_ids": torch.zeros(2, 5, dtype=torch.int64),
            OPD_TEACHER_TOPK_PACKED_KEY: entries,
        }
    )

    materialize_teacher_topk_microbatch(data)

    assert OPD_TEACHER_TOPK_PACKED_KEY not in data
    assert data["opd_support_indices"].shape == (2, 5, 2)
    assert data["opd_support_indices"].dtype == torch.int64
    assert data["teacher_support_logprobs"].shape == (2, 5, 2)
    torch.testing.assert_close(
        data["opd_support_indices"][0, :3],
        entries[0]["topk_indices"].to(torch.int64),
    )
    torch.testing.assert_close(
        data["teacher_support_logprobs"][1, :1], entries[1]["topk_logprobs"]
    )
    torch.testing.assert_close(
        data["teacher_support_logprobs"][0, 3:], torch.zeros(2, 2)
    )


def test_replay_collation_flattens_metadata_without_topk_materialization():
    entries = [
        {
            "seq_len": 1,
            "topk": 2,
            "topk_indices_ref": f"indices-{idx}",
            "topk_logprobs_ref": f"logprobs-{idx}",
        }
        for idx in range(4)
    ]
    combined = BatchedDataDict.from_batches(
        [
            {
                "input_ids": torch.zeros(2, 3, dtype=torch.int64),
                OPD_TEACHER_TOPK_PACKED_KEY: entries[:2],
            },
            {
                "input_ids": torch.zeros(2, 5, dtype=torch.int64),
                OPD_TEACHER_TOPK_PACKED_KEY: entries[2:],
            },
        ]
    )

    assert combined[OPD_TEACHER_TOPK_PACKED_KEY] == entries
    assert combined["input_ids"].shape == (4, 5)
    assert "opd_support_indices" not in combined


def test_non_fused_packing_materializes_teacher_topk_one_sequence_at_a_time():
    entries = [
        {
            "seq_len": 2,
            "topk": 2,
            "topk_indices": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            "topk_logprobs": torch.tensor([[-0.1, -0.2], [-0.3, -0.4]]),
        },
        {
            "seq_len": 1,
            "topk": 2,
            "topk_indices": torch.tensor([[5, 6]], dtype=torch.int32),
            "topk_logprobs": torch.tensor([[-0.5, -0.6]]),
        },
    ]
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[10, 11, 12], [20, 21, 0]], dtype=torch.int64),
            "input_lengths": torch.tensor([3, 2], dtype=torch.int32),
            OPD_TEACHER_TOPK_PACKED_KEY: entries,
        }
    )
    standard_prepare = MagicMock(side_effect=AssertionError("unexpected dense prep"))
    prepared_shapes = []

    def per_sequence_packed_prepare_fn(**kwargs):
        sequence_data = kwargs["data"]
        sequence_length = sequence_data["input_ids"].shape[1]
        prepared_shapes.append(
            (
                tuple(sequence_data["opd_support_indices"].shape),
                tuple(sequence_data["teacher_support_logprobs"].shape),
                kwargs["cu_seqlens_q"].tolist(),
                kwargs["cu_seqlens_q_padded"].tolist(),
            )
        )
        assert OPD_TEACHER_TOPK_PACKED_KEY not in sequence_data
        return (
            {
                "next_token_logprobs": torch.zeros(1, sequence_length - 1),
                "current_support_logprobs": torch.zeros(1, sequence_length - 1, 2),
            },
            sequence_data,
        )

    class RecordingLoss:
        use_fused_linear_logprobs = False

        def __call__(self, *, next_token_logprobs, **kwargs):
            loss = next_token_logprobs.sum()
            return loss, {"loss": loss.item()}

    wrapper = SequencePackingLossWrapper(
        loss_fn=RecordingLoss(),
        prepare_fn=standard_prepare,
        per_sequence_packed_prepare_fn=per_sequence_packed_prepare_fn,
        cu_seqlens_q=torch.tensor([0, 3, 5], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
    )

    loss, metrics = wrapper(
        next_token_logits=torch.zeros(1, 8, 16),
        data=data,
        global_valid_seqs=None,
        global_valid_toks=None,
    )

    standard_prepare.assert_not_called()
    assert prepared_shapes == [
        ((1, 3, 2), (1, 3, 2), [0, 3], [0, 4]),
        ((1, 2, 2), (1, 2, 2), [0, 2], [0, 4]),
    ]
    assert loss.item() == 0
    assert metrics["loss"] == 0
