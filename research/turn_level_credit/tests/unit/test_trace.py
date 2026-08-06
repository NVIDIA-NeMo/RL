"""Tests for native turn trace capture and token alignment."""

from typing import NamedTuple

import pytest
import torch
from turn_level_credit.trace import (
    TURN_REWARD_KEY,
    attach_turn_batch,
    compute_environment_credit,
    record_environment_turn,
    remove_turn_annotations,
    scatter_turn_credit,
    tensorize_turn_traces,
    turn_batch_from_mapping,
    validate_raw_reward_sums,
)


class _EnvironmentReturn(NamedTuple):
    observations: list
    metadata: list
    next_stop_strings: list
    rewards: torch.Tensor | dict[str, torch.Tensor]
    terminateds: torch.Tensor
    answers: list


def _message(role, token_ids, *, generated=False):
    message = {
        "role": role,
        "content": role,
        "token_ids": torch.tensor(token_ids, dtype=torch.int64),
    }
    if generated:
        message["generation_logprobs"] = torch.zeros(len(token_ids))
    return message


def _environment_return(rewards, terminateds):
    batch_size = len(terminateds)
    return _EnvironmentReturn(
        observations=[{"role": "user", "content": "observation"}] * batch_size,
        metadata=[None] * batch_size,
        next_stop_strings=[None] * batch_size,
        rewards=rewards,
        terminateds=torch.tensor(terminateds),
        answers=[None] * batch_size,
    )


def test_capture_tensorize_and_remove_annotations_with_assistant_history():
    logs = [
        [
            _message("user", [1, 2]),
            _message("assistant", [3]),
            _message("user", [4]),
            _message("assistant", [5, 6], generated=True),
        ],
        [
            _message("user", [7]),
            _message("assistant", [8, 9, 10], generated=True),
        ],
    ]
    record_environment_turn(
        logs,
        _environment_return(torch.tensor([0.2, -0.1]), [False, True]),
    )

    turn_batch = tensorize_turn_traces(logs)

    assert torch.allclose(
        turn_batch.rewards,
        torch.tensor([[0.2], [-0.1]]),
    )
    assert turn_batch.assistant_spans.tolist() == [[[4, 6]], [[1, 4]]]
    assert turn_batch.terminateds.tolist() == [[False], [True]]
    assert TURN_REWARD_KEY not in logs[0][1]
    assert TURN_REWARD_KEY in logs[0][3]

    remove_turn_annotations(logs)
    assert TURN_REWARD_KEY not in logs[0][3]
    assert TURN_REWARD_KEY not in logs[1][1]


def test_capture_named_components_and_validate_raw_sum():
    logs = [[_message("assistant", [1], generated=True)]]
    record_environment_turn(
        logs,
        _environment_return(
            {
                "correctness": torch.tensor([0.75]),
                "format": torch.tensor([0.25]),
            },
            [True],
        ),
    )
    turn_batch = tensorize_turn_traces(logs)

    assert turn_batch.rewards.tolist() == [[1.0]]
    validate_raw_reward_sums(
        turn_batch,
        torch.tensor([1.0]),
        atol=1.0e-6,
    )


def test_uneven_horizons_immediate_and_return_to_go():
    logs = [
        [_message("assistant", [1, 2], generated=True)],
        [_message("assistant", [5], generated=True)],
    ]
    record_environment_turn(
        [logs[0]],
        _environment_return(torch.tensor([0.2]), [False]),
    )
    logs[0].extend(
        [
            _message("user", [3]),
            _message("assistant", [4], generated=True),
        ]
    )
    record_environment_turn(
        [logs[0]],
        _environment_return(torch.tensor([1.0]), [True]),
    )
    record_environment_turn(
        [logs[1]],
        _environment_return(torch.tensor([-0.5]), [True]),
    )
    turn_batch = tensorize_turn_traces(logs)

    assert turn_batch.mask.tolist() == [[True, True], [True, False]]
    immediate = compute_environment_credit(
        turn_batch,
        mode="immediate",
        discount=1.0,
    )
    return_to_go = compute_environment_credit(
        turn_batch,
        mode="return_to_go",
        discount=0.5,
    )

    assert torch.allclose(
        immediate,
        torch.tensor([[0.2, 1.0], [-0.5, 0.0]]),
    )
    assert torch.allclose(
        return_to_go,
        torch.tensor([[0.7, 1.0], [-0.5, 0.0]]),
    )


def test_scatter_targets_only_generated_turn_spans():
    logs = [
        [
            _message("user", [1, 2]),
            _message("assistant", [3]),
            _message("assistant", [4, 5], generated=True),
        ]
    ]
    record_environment_turn(
        logs,
        _environment_return(torch.tensor([0.25]), [False]),
    )
    logs[0].extend(
        [
            _message("user", [6]),
            _message("assistant", [7, 8, 9], generated=True),
        ]
    )
    record_environment_turn(
        logs,
        _environment_return(torch.tensor([-0.5]), [True]),
    )
    turn_batch = tensorize_turn_traces(logs)
    token_mask = torch.tensor(
        [[0, 0, 0, 1, 1, 0, 1, 1, 1]],
        dtype=torch.bool,
    )
    credit = compute_environment_credit(
        turn_batch,
        mode="immediate",
        discount=1.0,
    )

    scattered = scatter_turn_credit(credit, turn_batch, token_mask)

    assert torch.allclose(
        scattered,
        torch.tensor([[0.0, 0.0, 0.0, 0.25, 0.25, 0.0, -0.5, -0.5, -0.5]]),
    )


def test_empty_generated_turn_is_observed_but_not_trainable():
    logs = [[_message("assistant", [], generated=True)]]
    record_environment_turn(
        logs,
        _environment_return(torch.tensor([1.0]), [True]),
    )

    turn_batch = tensorize_turn_traces(logs)

    assert turn_batch.mask.tolist() == [[True]]
    assert turn_batch.trainable_mask.tolist() == [[False]]
    assert turn_batch.assistant_spans.tolist() == [[[0, 0]]]


def test_turn_batch_round_trip_and_raw_sum_mismatch():
    logs = [[_message("assistant", [1], generated=True)]]
    record_environment_turn(
        logs,
        _environment_return(torch.tensor([0.5]), [True]),
    )
    turn_batch = tensorize_turn_traces(logs)
    mapping = {}
    attach_turn_batch(mapping, turn_batch)

    restored = turn_batch_from_mapping(mapping)
    assert torch.equal(restored.rewards, turn_batch.rewards)
    with pytest.raises(ValueError, match="do not sum"):
        validate_raw_reward_sums(
            restored,
            torch.tensor([1.0]),
            atol=1.0e-6,
        )


def test_scatter_rejects_span_touching_non_trainable_token():
    logs = [[_message("assistant", [1, 2], generated=True)]]
    record_environment_turn(
        logs,
        _environment_return(torch.tensor([1.0]), [True]),
    )
    turn_batch = tensorize_turn_traces(logs)

    with pytest.raises(ValueError, match="non-trainable token"):
        scatter_turn_credit(
            torch.tensor([[1.0]]),
            turn_batch,
            torch.tensor([[True, False]]),
        )


def test_turn_batch_rejects_fractional_spans():
    logs = [[_message("assistant", [1, 2], generated=True)]]
    record_environment_turn(
        logs,
        _environment_return(torch.tensor([1.0]), [True]),
    )
    turn_batch = tensorize_turn_traces(logs)
    mapping = {}
    attach_turn_batch(mapping, turn_batch)
    mapping["assistant_turn_spans"] = torch.tensor([[[0.5, 2.0]]])

    with pytest.raises(ValueError, match="torch.int64"):
        turn_batch_from_mapping(mapping)


def test_turn_batch_rejects_non_boolean_masks():
    logs = [[_message("assistant", [1], generated=True)]]
    record_environment_turn(
        logs,
        _environment_return(torch.tensor([1.0]), [True]),
    )
    turn_batch = tensorize_turn_traces(logs)
    mapping = {}
    attach_turn_batch(mapping, turn_batch)
    mapping["turn_mask"] = mapping["turn_mask"].float()

    with pytest.raises(ValueError, match="turn_mask must use torch.bool"):
        turn_batch_from_mapping(mapping)
