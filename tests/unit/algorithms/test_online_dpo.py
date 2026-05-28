from unittest.mock import MagicMock

import torch

from nemo_rl.algorithms.online_dpo import (
    add_reference_logprobs_to_preference_batch,
    build_preference_datums_from_rollouts,
    collate_preference_datums,
    strip_trailing_environment_messages,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _message_log(response_tokens: list[int], response_content: str):
    return [
        {
            "role": "user",
            "content": "question",
            "token_ids": torch.tensor([1, 2]),
        },
        {
            "role": "assistant",
            "content": response_content,
            "token_ids": torch.tensor(response_tokens),
        },
        {
            "role": "user",
            "content": "environment observation",
            "token_ids": torch.tensor([99]),
        },
    ]


def _rollout_batch(
    rewards: torch.Tensor,
    *,
    truncated: torch.Tensor | None = None,
) -> BatchedDataDict:
    batch = BatchedDataDict(
        {
            "message_log": [
                _message_log([3, 4], "first"),
                _message_log([5, 6, 7], "second"),
            ],
            "total_reward": rewards,
            "loss_multiplier": torch.tensor([1.0, 1.0]),
            "idx": [123, 123],
            "task_name": ["math", "math"],
        }
    )
    if truncated is not None:
        batch["truncated"] = truncated
    return batch


def _two_pair_rollout_batch() -> BatchedDataDict:
    return BatchedDataDict(
        {
            "message_log": [
                _message_log([3], "a"),
                _message_log([4], "b"),
                _message_log([5], "c"),
                _message_log([6], "d"),
            ],
            "total_reward": torch.tensor([0.0, 1.0, 0.2, 0.9]),
            "loss_multiplier": torch.ones(4),
            "idx": [0, 0, 1, 1],
            "task_name": ["math", "math", "math", "math"],
        }
    )


def test_strip_trailing_environment_messages_keeps_final_assistant():
    stripped = strip_trailing_environment_messages(_message_log([3, 4], "answer"))

    assert [message["role"] for message in stripped] == ["user", "assistant"]
    assert stripped[-1]["content"] == "answer"


def test_build_preference_datums_orders_by_reward_and_strips_env_observation():
    datums, metrics = build_preference_datums_from_rollouts(
        _rollout_batch(torch.tensor([0.25, 0.75])),
        min_reward_margin=0.0,
        drop_truncated_pairs=True,
    )

    assert len(datums) == 1
    assert datums[0]["message_log_chosen"][-1]["content"] == "second"
    assert datums[0]["message_log_rejected"][-1]["content"] == "first"
    assert datums[0]["length_chosen"] == 5
    assert datums[0]["length_rejected"] == 4
    assert metrics["usable_pairs"] == 1.0
    assert metrics["reward_margin"] == 0.5


def test_build_preference_datums_respects_max_pairs_and_counts_discards():
    datums, metrics = build_preference_datums_from_rollouts(
        _two_pair_rollout_batch(),
        min_reward_margin=0.0,
        drop_truncated_pairs=True,
        max_pairs=1,
    )

    assert len(datums) == 1
    assert datums[0]["idx"] == 0
    assert metrics["generated_pairs"] == 2.0
    assert metrics["usable_pairs"] == 1.0
    assert metrics["discarded_pairs"] == 1.0


def test_build_preference_datums_drops_ties_within_margin():
    datums, metrics = build_preference_datums_from_rollouts(
        _rollout_batch(torch.tensor([0.25, 0.30])),
        min_reward_margin=0.1,
        drop_truncated_pairs=True,
    )

    assert datums == []
    assert metrics["tie_pairs"] == 1.0
    assert metrics["dropped_pairs"] == 1.0


def test_build_preference_datums_drops_truncated_pairs_when_enabled():
    datums, metrics = build_preference_datums_from_rollouts(
        _rollout_batch(
            torch.tensor([0.25, 0.75]),
            truncated=torch.tensor([False, True]),
        ),
        min_reward_margin=0.0,
        drop_truncated_pairs=True,
    )

    assert datums == []
    assert metrics["truncated_pairs"] == 1.0
    assert metrics["dropped_pairs"] == 1.0


def test_collate_preference_datums_uses_dpo_interleaving_and_final_mask():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    datums, _ = build_preference_datums_from_rollouts(
        _rollout_batch(torch.tensor([0.25, 0.75])),
        min_reward_margin=0.0,
        drop_truncated_pairs=True,
    )

    batch = collate_preference_datums(
        datums,
        tokenizer=tokenizer,
        make_sequence_length_divisible_by=1,
    )

    assert batch["input_ids"].shape[0] == 2
    assert batch["sample_mask"].tolist() == [1.0, 1.0]
    assert batch["token_mask"][0].sum().item() == 3
    assert batch["token_mask"][1].sum().item() == 2


def test_add_reference_logprobs_rolls_reference_output():
    preference_batch = BatchedDataDict({"input_ids": torch.ones(2, 3)})
    policy = MagicMock()
    policy.get_reference_policy_logprobs.return_value = {
        "reference_logprobs": torch.tensor([[1, 2, 3], [4, 5, 6]])
    }

    add_reference_logprobs_to_preference_batch(
        preference_batch,
        policy,
        micro_batch_size=2,
    )

    assert preference_batch["reference_policy_logprobs"].tolist() == [
        [2, 3, 1],
        [5, 6, 4],
    ]
    policy.get_reference_policy_logprobs.assert_called_once_with(
        preference_batch,
        micro_batch_size=2,
        timer=None,
    )
