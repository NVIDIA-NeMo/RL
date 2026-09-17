# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.algorithms.x_token.packing_loss import (
    XTOKEN_LOGICAL_METRICS_KEY,
    XTokenSequencePackingLossWrapper,
    restore_packed_student_logits,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_restore_packed_student_logits_uses_padded_offsets_and_raw_lengths():
    packed = torch.tensor(
        [
            [
                [1.0],
                [2.0],
                [3.0],
                [99.0],
                [10.0],
                [11.0],
                [12.0],
                [13.0],
                [14.0],
                [98.0],
                [97.0],
                [96.0],
            ]
        ]
    )
    raw_cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    padded_cu = torch.tensor([0, 4, 12], dtype=torch.int32)

    first = restore_packed_student_logits(
        packed,
        sequence_index=0,
        cu_seqlens_q=raw_cu,
        cu_seqlens_q_padded=padded_cu,
    )
    second = restore_packed_student_logits(
        packed,
        sequence_index=1,
        cu_seqlens_q=raw_cu,
        cu_seqlens_q_padded=padded_cu,
    )

    assert first.shape == (1, 4, 1)
    assert second.shape == (1, 8, 1)
    assert first[:, :, 0].tolist() == [[1.0, 2.0, 3.0, 0.0]]
    assert second[:, :, 0].tolist() == [[10.0, 11.0, 12.0, 13.0, 14.0, 0.0, 0.0, 0.0]]


@pytest.mark.parametrize(
    ("cp_rank", "expected"),
    [
        (0, [[10.0, 11.0, 0.0, 0.0]]),
        (1, [[12.0, 13.0, 14.0, 0.0]]),
    ],
)
def test_restore_packed_student_logits_cp2_reapplies_logical_head_tail_layout(
    cp_rank, expected
):
    import nemo_rl.algorithms.x_token.packing_loss as packing_loss

    cp_group = MagicMock()
    packed_local = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    gathered_physical = torch.tensor(
        [[[10.0], [11.0], [12.0], [13.0], [14.0], [99.0], [98.0], [97.0]]]
    )
    with (
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(torch.distributed, "get_rank", return_value=cp_rank),
        patch.object(
            packing_loss,
            "allgather_cp_sharded_tensor",
            return_value=gathered_physical,
        ) as mock_allgather,
    ):
        restored = restore_packed_student_logits(
            packed_local,
            sequence_index=1,
            cu_seqlens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
            cu_seqlens_q_padded=torch.tensor([0, 4, 12], dtype=torch.int32),
            context_parallel_group=cp_group,
        )

    assert restored[:, :, 0].tolist() == expected
    # Sequence 1 begins at padded offset 4 globally, i.e. local offset 2.
    assert mock_allgather.call_args.args[0].shape == (1, 4, 1)


def test_xtoken_packing_wrapper_preserves_asymmetric_teacher_and_pair_axes():
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(16).reshape(2, 8),
            "input_lengths": torch.tensor([3, 5]),
            "token_mask": torch.tensor(
                [
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0],
                ]
            ),
            "kd_token_mask": torch.tensor(
                [
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 1, 0, 0, 0],
                ]
            ),
            "sample_mask": torch.ones(2),
            # Deliberately asymmetric with T_student=8.
            "teacher_0_input_ids": torch.arange(18).reshape(2, 9),
            "teacher_0_token_mask": torch.ones(2, 9),
            "alignment_0_teacher_chunk_id": torch.arange(18).reshape(2, 9),
            "alignment_0_student_chunk_id": torch.arange(16).reshape(2, 8),
            "alignment_0_student_exact_partition_mask": torch.ones(
                2, 8, dtype=torch.bool
            ),
            # Pair axis P=4 must not be truncated to either raw student length.
            "alignment_0_pair_valid": torch.ones(2, 4, dtype=torch.bool),
            "alignment_0_pair_is_correct": torch.ones(2, 4, dtype=torch.bool),
            "student_semantic_regions": [[("student", 0)], [("student", 1)]],
            "teacher_0_semantic_regions": [[("teacher", 0)], [("teacher", 1)]],
            "teacher_0_full_logits_ipc": [{"row": 0}, {"row": 1}],
        }
    )
    original_tensors = {
        key: value.clone()
        for key, value in data.items()
        if isinstance(value, torch.Tensor)
    }
    packed = torch.tensor(
        [
            [
                [1.0],
                [2.0],
                [3.0],
                [99.0],
                [10.0],
                [11.0],
                [12.0],
                [13.0],
                [14.0],
                [98.0],
                [97.0],
                [96.0],
            ]
        ],
        requires_grad=True,
    )
    observed: list[dict] = []

    def prepare_fn(*, logits, data, **_kwargs):
        observed.append(
            {
                "logits": logits.detach().clone(),
                "shapes": {
                    key: tuple(value.shape)
                    for key, value in data.items()
                    if isinstance(value, torch.Tensor)
                },
                "input_lengths": data["input_lengths"].clone(),
                "input_ids": data["input_ids"].clone(),
                "token_mask": data["token_mask"].clone(),
                "kd_token_mask": data["kd_token_mask"].clone(),
                "student_regions": data["student_semantic_regions"],
                "teacher_regions": data["teacher_0_semantic_regions"],
                "teacher_ipc": data["teacher_0_full_logits_ipc"],
            }
        )
        return {"logical_logits": logits}, data

    class RecordingLoss:
        def __call__(self, *, logical_logits, **_kwargs):
            return logical_logits.sum(), {"logical_calls": 1}

    wrapper = XTokenSequencePackingLossWrapper(
        loss_fn=RecordingLoss(),
        prepare_fn=prepare_fn,
        cu_seqlens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 12], dtype=torch.int32),
    )
    loss, metrics = wrapper(
        packed,
        data,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=torch.tensor(5.0),
    )
    loss.backward()

    assert len(observed) == 2
    assert observed[0]["logits"][:, :, 0].tolist() == [[1.0, 2.0, 3.0, 0.0]]
    assert observed[1]["logits"][:, :, 0].tolist() == [
        [10.0, 11.0, 12.0, 13.0, 14.0, 0.0, 0.0, 0.0]
    ]
    restored_positions = sum(item["logits"].shape[1] for item in observed)
    padded_cu = wrapper.cu_seqlens_q_padded
    expected_effective_positions = int((padded_cu[1:] - padded_cu[:-1]).sum())
    assert restored_positions == expected_effective_positions == 12
    assert restored_positions != data.size * data["input_ids"].shape[1]
    assert [item["shapes"]["input_ids"] for item in observed] == [(1, 4), (1, 8)]
    assert [item["shapes"]["token_mask"] for item in observed] == [(1, 4), (1, 8)]
    assert [item["shapes"]["kd_token_mask"] for item in observed] == [
        (1, 4),
        (1, 8),
    ]
    assert [item["shapes"]["alignment_0_student_chunk_id"] for item in observed] == [
        (1, 4),
        (1, 8),
    ]
    assert [
        item["shapes"]["alignment_0_student_exact_partition_mask"] for item in observed
    ] == [(1, 4), (1, 8)]
    assert [item["input_lengths"].tolist() for item in observed] == [[3], [5]]
    assert observed[0]["input_ids"].tolist() == [[0, 1, 2, 3]]
    assert observed[1]["input_ids"].tolist() == [[8, 9, 10, 11, 12, 13, 14, 15]]
    assert observed[0]["token_mask"].tolist() == [[1, 1, 1, 0]]
    assert observed[0]["kd_token_mask"].tolist() == [[0, 1, 1, 0]]
    for item in observed:
        shapes = item["shapes"]
        assert shapes["teacher_0_input_ids"] == (1, 9)
        assert shapes["teacher_0_token_mask"] == (1, 9)
        assert shapes["alignment_0_teacher_chunk_id"] == (1, 9)
        assert shapes["alignment_0_pair_valid"] == (1, 4)
        assert shapes["alignment_0_pair_is_correct"] == (1, 4)
    assert [item["student_regions"] for item in observed] == [
        [[("student", 0)]],
        [[("student", 1)]],
    ]
    assert [item["teacher_regions"] for item in observed] == [
        [[("teacher", 0)]],
        [[("teacher", 1)]],
    ]
    assert [item["teacher_ipc"] for item in observed] == [
        [{"row": 0}],
        [{"row": 1}],
    ]
    for key, original in original_tensors.items():
        torch.testing.assert_close(data[key], original)
    assert data["student_semantic_regions"] == [
        [("student", 0)],
        [("student", 1)],
    ]
    assert data["teacher_0_semantic_regions"] == [
        [("teacher", 0)],
        [("teacher", 1)],
    ]
    assert data["teacher_0_full_logits_ipc"] == [{"row": 0}, {"row": 1}]
    assert metrics == {
        "logical_calls": 2,
        XTOKEN_LOGICAL_METRICS_KEY: [
            {"logical_calls": 1},
            {"logical_calls": 1},
        ],
    }
    # Only real tokens receive gradient. Physical padding on either side of a
    # sample boundary cannot act as another sample's next-token predictor.
    assert packed.grad[:, :, 0].tolist() == [
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    ]


def test_xtoken_packed_effective_rows_match_unpacked_masked_loss_and_gradients():
    class MaskedLoss:
        def __call__(self, *, logical_logits, data, **_kwargs):
            mask = (data["token_mask"] * data["kd_token_mask"]).to(logical_logits.dtype)
            return (logical_logits.squeeze(-1) * mask).sum(), {
                "masked_tokens": int(mask.sum().item())
            }

    def prepare_fn(*, logits, data, **_kwargs):
        return {"logical_logits": logits}, data

    data = BatchedDataDict(
        {
            "input_ids": torch.arange(16).reshape(2, 8),
            "input_lengths": torch.tensor([3, 5]),
            "token_mask": torch.tensor(
                [
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0],
                ]
            ),
            "kd_token_mask": torch.tensor(
                [
                    [1, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 1, 0, 0, 0],
                ]
            ),
            "sample_mask": torch.ones(2),
        }
    )
    packed_logits = torch.tensor(
        [
            [
                [1.0],
                [2.0],
                [3.0],
                [99.0],
                [10.0],
                [11.0],
                [12.0],
                [13.0],
                [14.0],
                [98.0],
                [97.0],
                [96.0],
            ]
        ],
        requires_grad=True,
    )
    wrapper = XTokenSequencePackingLossWrapper(
        loss_fn=MaskedLoss(),
        prepare_fn=prepare_fn,
        cu_seqlens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 12], dtype=torch.int32),
    )
    packed_loss, packed_metrics = wrapper(
        packed_logits,
        data,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=torch.tensor(8.0),
        global_valid_kd_toks=torch.tensor(5.0),
    )
    packed_loss.backward()

    unpacked_logits = [
        torch.tensor(
            [[[1.0], [2.0], [3.0], [0.0], [0.0], [0.0], [0.0], [0.0]]],
            requires_grad=True,
        ),
        torch.tensor(
            [[[10.0], [11.0], [12.0], [13.0], [14.0], [0.0], [0.0], [0.0]]],
            requires_grad=True,
        ),
    ]
    unpacked_loss = torch.zeros(())
    for row_index, row_logits in enumerate(unpacked_logits):
        row_data = data.slice(row_index, row_index + 1)
        row_loss, _ = MaskedLoss()(
            logical_logits=row_logits,
            data=row_data,
        )
        unpacked_loss = unpacked_loss + row_loss
    unpacked_loss.backward()

    torch.testing.assert_close(packed_loss, unpacked_loss)
    expected_packed_grad = torch.cat(
        [
            unpacked_logits[0].grad[:, :4],
            unpacked_logits[1].grad[:, :8],
        ],
        dim=1,
    )
    torch.testing.assert_close(packed_logits.grad, expected_packed_grad)
    assert packed_metrics["masked_tokens"] == 4


def test_packed_metrics_preserve_unpacked_mbs1_cohorts_for_uneven_bins():
    class PerSampleLoss:
        def __call__(self, *, logical_logits, data, **_kwargs):
            sample_id = int(data["batch_item_id"][0].item())
            valid = int(data["sample_mask"][0].item())
            return logical_logits.sum() * valid, {
                "loss": float(sample_id + 1),
                "accuracy": float(sample_id) / 10,
                "ipc_reconstruction_fallbacks": 1,
                "ipc_reconstruction_fallbacks_t0": 1,
                "num_valid_samples": valid,
            }

    def prepare_fn(*, logits, data, **_kwargs):
        return {"logical_logits": logits}, data

    def run_bin(sample_ids, sample_masks):
        count = len(sample_ids)
        data = BatchedDataDict(
            {
                "input_ids": torch.zeros(count, 2, dtype=torch.long),
                "input_lengths": torch.full((count,), 2),
                "sample_mask": torch.tensor(sample_masks),
                "batch_item_id": torch.tensor(sample_ids),
            }
        )
        wrapper = XTokenSequencePackingLossWrapper(
            loss_fn=PerSampleLoss(),
            prepare_fn=prepare_fn,
            cu_seqlens_q=torch.arange(0, 2 * count + 1, 2, dtype=torch.int32),
        )
        logits = torch.ones(1, 2 * count, 1)
        _, metrics = wrapper(logits, data, torch.tensor(2.0), torch.tensor(4.0))
        return metrics[XTOKEN_LOGICAL_METRICS_KEY]

    # Physical bins have cardinalities two and one; row 1 is masked but still
    # executes above.  Flattened records match three unpacked MBS1 calls with
    # the same masked-row filtering, rather than two physical-bin means.
    packed_records = run_bin([0, 1], [1, 0]) + run_bin([2], [1])
    unpacked_records = run_bin([0], [1]) + run_bin([1], [0]) + run_bin([2], [1])
    assert packed_records == unpacked_records
    assert [record["accuracy"] for record in packed_records] == [0.0, 0.2]
    # Reconstruction runs before the masked-row check, but the worker's public
    # metric stream deliberately contains valid logical MBS1 cohorts only.
    assert sum(record["ipc_reconstruction_fallbacks"] for record in packed_records) == 2
    assert (
        sum(record["ipc_reconstruction_fallbacks_t0"] for record in packed_records) == 2
    )
