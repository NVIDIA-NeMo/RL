from unittest.mock import MagicMock

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.advantage_estimator import OfflineGRPOAdvantageEstimator
from nemo_rl.algorithms.loss import OfflineGRPOLossConfig, OfflineGRPOLossFn
from nemo_rl.algorithms.offline_grpo import (
    MasterConfig,
    OfflineGRPOConfig,
    _validate_batch_configuration,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.offline_grpo import (
    OfflineGRPODataset,
    OfflineGRPODatasetConfig,
    OfflineGRPOGroup,
    prepare_offline_grpo_batch,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class _Rows:
    column_names = ["prompt", "responses", "rewards"]

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def _tokenizer() -> MagicMock:
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    tokenizer.bos_token = "<bos>"
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token_id = 0
    tokenizer.apply_chat_template.side_effect = lambda messages, **_: "".join(
        f"<{message['role']}>{message['content']}" for message in messages
    )
    tokenizer.side_effect = lambda text, **_: {
        "input_ids": torch.arange(len(text), dtype=torch.long).unsqueeze(0)
    }
    return tokenizer


def _raw_group(dataset_index: int, rewards: list[float]) -> OfflineGRPOGroup:
    return OfflineGRPOGroup(
        prompt_messages=[{"role": "user", "content": "prompt"}],
        responses=[
            [{"role": "assistant", "content": f"response {index}"}]
            for index in range(len(rewards))
        ],
        rewards=torch.tensor(rewards, dtype=torch.float32),
        dataset_index=dataset_index,
    )


def _prepare_dataset() -> MagicMock:
    dataset = MagicMock(spec=OfflineGRPODataset)
    dataset.task_spec = TaskDataSpec(task_name="test")
    dataset.add_bos = True
    dataset.add_eos = True
    dataset.max_seq_length = 512
    return dataset


def test_offline_dataset_keeps_grouped_teacher_trajectories_raw(monkeypatch):
    rows = _Rows(
        [
            {
                "prompt": "What is 2 + 2?",
                "responses": ["It is 4.", "It is 5."],
                "rewards": [1.0, 0.0],
            }
        ]
    )
    monkeypatch.setattr(
        "nemo_rl.data.offline_grpo.load_dataset_from_path",
        lambda *_: rows,
    )
    tokenizer = _tokenizer()
    dataset = OfflineGRPODataset(
        OfflineGRPODatasetConfig(data_path="unused.jsonl"),
        tokenizer,
        max_seq_length=512,
        add_bos=True,
        add_eos=True,
    )

    group = dataset[0]

    assert group.prompt_messages == [{"role": "user", "content": "What is 2 + 2?"}]
    assert group.responses == ["It is 4.", "It is 5."]
    torch.testing.assert_close(group.rewards, torch.tensor([1.0, 0.0]))
    tokenizer.apply_chat_template.assert_not_called()


def test_offline_advantages_mixed_positive_and_non_positive_groups():
    estimator = OfflineGRPOAdvantageEstimator(
        use_leave_one_out_baseline=False,
        normalize_rewards=False,
        all_positive_bias=0.1,
        positive_reward_threshold=0.0,
    )
    prompt_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2])
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    advantages = estimator.compute_advantage(
        prompt_ids,
        rewards,
        torch.ones(9, 2),
    )

    torch.testing.assert_close(advantages[:4, 0], torch.tensor([0.5, 0.5, -0.5, -0.5]))
    torch.testing.assert_close(advantages[4:7], torch.full((3, 2), 0.1))
    torch.testing.assert_close(advantages[7:], torch.zeros(2, 2))


def test_offline_loss_is_probability_weighted_and_pushes_both_directions():
    loss_fn = OfflineGRPOLossFn(OfflineGRPOLossConfig(reference_policy_kl_penalty=0.0))
    next_token_logprobs = torch.log(
        torch.tensor([[0.25, 0.5], [0.4, 0.2]], requires_grad=True)
    )
    next_token_logprobs.retain_grad()
    data = BatchedDataDict(
        {
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
            "sample_mask": torch.ones(2),
            "advantages": torch.tensor([[0.0, 1.0, 1.0], [0.0, -1.0, -1.0]]),
        }
    )

    loss, metrics = loss_fn(
        next_token_logprobs,
        data,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=torch.tensor(4.0),
    )
    expected = torch.tensor((-0.25 - 0.5 + 0.4 + 0.2) / 4)
    torch.testing.assert_close(loss, expected)
    assert metrics["actor_loss"] == pytest.approx(expected.item())

    loss.backward()
    assert torch.all(next_token_logprobs.grad[0] < 0)
    assert torch.all(next_token_logprobs.grad[1] > 0)


def test_offline_loss_requires_reference_logprobs_when_kl_is_enabled():
    loss_fn = OfflineGRPOLossFn(OfflineGRPOLossConfig(reference_policy_kl_penalty=0.1))
    data = BatchedDataDict(
        {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "token_mask": torch.tensor([[0.0, 1.0]]),
            "sample_mask": torch.ones(1),
            "advantages": torch.tensor([[0.0, 1.0]]),
        }
    )

    with pytest.raises(ValueError, match="reference_policy_logprobs"):
        loss_fn(
            torch.tensor([[-1.0]]),
            data,
            global_valid_seqs=torch.tensor(1.0),
            global_valid_toks=torch.tensor(1.0),
        )


def test_prepare_offline_batch_preserves_groups_and_masks_only_responses():
    groups = [
        _raw_group(3, [1.0, 0.0, 0.5]),
        _raw_group(4, [1.0, 1.0, 0.0]),
    ]
    tokenizer = _tokenizer()

    prepared = prepare_offline_grpo_batch(
        groups,
        _prepare_dataset(),
        tokenizer,
        num_responses_per_prompt=2,
        response_selection="first",
        seed=42,
        step=0,
        positive_reward_threshold=0.0,
        make_sequence_length_divisible_by=1,
    )

    torch.testing.assert_close(prepared.prompt_ids, torch.tensor([0, 0, 1, 1]))
    torch.testing.assert_close(prepared.rewards, torch.tensor([1.0, 0.0, 1.0, 1.0]))
    assert torch.all(prepared.data["token_mask"].sum(dim=-1) > 0)
    assert prepared.metrics.mean_reward == pytest.approx(0.75)
    assert prepared.metrics.all_positive_group_fraction == pytest.approx(0.5)


def test_random_response_selection_is_step_deterministic():
    group = _raw_group(17, list(range(8)))
    tokenizer = _tokenizer()
    kwargs = {
        "groups": [group],
        "dataset": _prepare_dataset(),
        "tokenizer": tokenizer,
        "num_responses_per_prompt": 4,
        "response_selection": "random",
        "seed": 7,
        "positive_reward_threshold": 0.0,
        "make_sequence_length_divisible_by": 1,
    }

    first = prepare_offline_grpo_batch(**kwargs, step=5)
    repeated = prepare_offline_grpo_batch(**kwargs, step=5)
    next_step = prepare_offline_grpo_batch(**kwargs, step=6)

    torch.testing.assert_close(first.rewards, repeated.rewards)
    assert not torch.equal(first.rewards, next_step.rewards)


def test_only_selected_responses_are_tokenized():
    group = _raw_group(9, list(range(20)))
    tokenizer = _tokenizer()

    prepare_offline_grpo_batch(
        [group],
        _prepare_dataset(),
        tokenizer,
        num_responses_per_prompt=4,
        response_selection="first",
        seed=42,
        step=0,
        positive_reward_threshold=0.0,
        make_sequence_length_divisible_by=1,
    )

    # get_formatted_message_log formats the prompt and response once each.
    assert tokenizer.apply_chat_template.call_count == 8


def test_offline_batch_size_must_match_prompt_group_product():
    config = MasterConfig.model_construct(
        policy={"train_global_batch_size": 31},
        offline_grpo=OfflineGRPOConfig(
            num_prompts_per_step=4,
            num_responses_per_prompt=8,
        ),
    )

    with pytest.raises(ValueError, match="train_global_batch_size"):
        _validate_batch_configuration(config)
