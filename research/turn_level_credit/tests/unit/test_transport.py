"""Tests for turn tensors crossing NeMo-RL batch transport operations."""

import torch
from turn_level_credit.trace import (
    TurnBatch,
    attach_turn_batch,
    scatter_turn_credit,
    turn_batch_from_mapping,
)

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _batch(
    rewards: list[float],
    spans: list[list[int]],
) -> BatchedDataDict:
    num_turns = len(rewards)
    batch = BatchedDataDict()
    attach_turn_batch(
        batch,
        TurnBatch(
            rewards=torch.tensor([rewards], dtype=torch.float32),
            mask=torch.ones((1, num_turns), dtype=torch.bool),
            trainable_mask=torch.ones((1, num_turns), dtype=torch.bool),
            assistant_spans=torch.tensor([spans], dtype=torch.int64),
            terminateds=torch.tensor(
                [[False] * (num_turns - 1) + [True]],
                dtype=torch.bool,
            ),
        ),
    )
    return batch


def test_turn_fields_survive_padding_reordering_and_slicing():
    short = _batch([0.25], [[1, 2]])
    long = _batch([0.1, -0.2, 1.0], [[1, 2], [3, 5], [6, 7]])

    combined = BatchedDataDict.from_batches([short, long])
    reordered = combined.select_indices([1, 0])
    sliced = reordered.slice(0, 1)
    restored = turn_batch_from_mapping(sliced)

    assert torch.allclose(
        restored.rewards,
        torch.tensor([[0.1, -0.2, 1.0]]),
    )
    assert restored.mask.tolist() == [[True, True, True]]
    assert restored.trainable_mask.tolist() == [[True, True, True]]
    assert restored.assistant_spans.tolist() == [[[1, 2], [3, 5], [6, 7]]]
    assert restored.terminateds.tolist() == [[False, False, True]]

    padded = turn_batch_from_mapping(combined)
    assert torch.allclose(
        padded.rewards,
        torch.tensor([[0.25, 0.0, 0.0], [0.1, -0.2, 1.0]]),
    )
    assert padded.mask.tolist() == [[True, False, False], [True, True, True]]
    assert padded.assistant_spans.tolist()[0] == [[1, 2], [0, 0], [0, 0]]


def test_filtered_row_receives_no_auxiliary_credit():
    batch = _batch([0.25], [[1, 2]])
    turn_batch = turn_batch_from_mapping(batch)

    scattered = scatter_turn_credit(
        torch.tensor([[0.25]]),
        turn_batch,
        torch.zeros((1, 3), dtype=torch.bool),
    )

    assert torch.equal(scattered, torch.zeros((1, 3)))
