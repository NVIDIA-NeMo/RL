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
"""GRPO group identity survives rewritten prompts and dynamic sampling."""

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms import grpo as grpo_mod
from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.grpo import (
    PROMPT_GROUP_IDS_KEY,
    _initial_grpo_save_state,
    _prompt_group_ids,
    _trajectory_group_ids,
    async_grpo_train,
    grpo_train,
)
from nemo_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    calculate_trivial_reward_distributions,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from tests.unit.algorithms import test_grpo as grpo_tests
from tests.unit.algorithms.test_grpo import (
    _mock_policy_generation,
    _run_single_grpo_train_step,
)

# Reuse the driver fixture without importing its unrelated environment autouse fixture.
mock_grpo_components = grpo_tests.mock_grpo_components


class TestPromptGroupIds:
    def test_rows_of_one_prompt_share_a_column_shaped_id(self):
        group_ids = _prompt_group_ids(6, 3, group_offset=0)

        assert group_ids.shape == (6, 1)
        assert group_ids.dtype == torch.long
        assert group_ids.squeeze(1).tolist() == [0, 0, 0, 1, 1, 1]

    def test_generation_batches_of_one_step_do_not_collide(self):
        first = _prompt_group_ids(4, 2, group_offset=0).squeeze(1).tolist()
        second = _prompt_group_ids(4, 2, group_offset=2).squeeze(1).tolist()

        assert not set(first) & set(second)
        assert len(set(second)) == 2

    def test_rejects_partial_prompt_groups(self):
        with pytest.raises(ValueError, match="whole number of prompt groups"):
            _prompt_group_ids(5, 2, group_offset=0)

    def test_trajectory_group_ids_follow_buffer_entries(self):
        def entry(size):
            return {"batch": BatchedDataDict({"total_reward": torch.zeros(size)})}

        group_ids = _trajectory_group_ids([entry(2), entry(3)])

        assert group_ids.shape == (5, 1)
        assert group_ids.dtype == torch.long
        assert group_ids.squeeze(1).tolist() == [0, 0, 1, 1, 1]
        assert _trajectory_group_ids([]).shape == (0, 1)

    def test_rewritten_first_message_stays_in_its_group(self):
        # Prompt 0's last rollout came back with a different first message (an
        # agent harness rewrote the history); grouping by prompt tokens made it
        # a singleton group with advantage 0.
        prompt_rows = torch.tensor([[1, 2, 0]] * 3 + [[1, 2, 9]] + [[5, 6, 0]] * 4)
        rewards = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        valid = torch.ones_like(rewards)

        token_baseline, token_std, _ = calculate_baseline_and_std_per_prompt(
            prompt_rows, rewards, valid, leave_one_out_baseline=True
        )
        # Singleton: baseline is its own reward (advantage 0), and its siblings
        # look uniform (zero std), so dynamic sampling would drop them.
        assert token_baseline[3].item() == rewards[3].item()
        assert torch.equal(token_std[:4], torch.zeros(4))

        baseline, std, _ = calculate_baseline_and_std_per_prompt(
            _prompt_group_ids(8, 4, group_offset=0),
            rewards,
            valid,
            leave_one_out_baseline=True,
        )
        assert baseline[3].item() == 0.0
        assert torch.allclose(baseline[:3], torch.full((3,), 1.0 / 3.0))
        assert torch.all(std[:3] > 0)


def test_grpo_train_groups_by_prompt_not_by_returned_prompt_tokens(
    monkeypatch: pytest.MonkeyPatch,
    mock_grpo_components: dict[str, Any],
) -> None:
    """A rollout whose first message was rewritten still belongs to its prompt group."""
    rollout_metrics = {"mean_gen_tokens_per_sample": 1.0}

    def fake_rollout(*_args: Any, **kwargs: Any) -> tuple[BatchedDataDict, dict]:
        rollout_batch = kwargs["input_batch"]
        for message_log in rollout_batch["message_log"]:
            message_log.append(
                {
                    "role": "assistant",
                    "content": "answer",
                    "token_ids": torch.tensor([4]),
                }
            )
        rewritten = dict(rollout_batch["message_log"][3][0])
        rewritten["token_ids"] = torch.cat(
            [rewritten["token_ids"], torch.tensor([7, 8])]
        )
        rollout_batch["message_log"][3][0] = rewritten
        rollout_batch["total_reward"] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        return rollout_batch, rollout_metrics

    captured: dict[str, torch.Tensor] = {}

    def capture_trivial(prompts, rewards, valid_mask):
        captured["prompts"] = prompts.clone()
        captured["is_trivial"] = calculate_trivial_reward_distributions(
            prompts, rewards, valid_mask
        )
        raise RuntimeError("captured dynamic-sampling inputs")

    monkeypatch.setattr(
        grpo_mod, "should_use_async_rollouts", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(grpo_mod, "run_multi_turn_rollout", fake_rollout)
    monkeypatch.setattr(
        grpo_mod, "refit_policy_generation", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        grpo_mod, "calculate_trivial_reward_distributions", capture_trivial
    )
    monkeypatch.setattr(grpo_mod, "MemoryTracker", MagicMock)

    master_config = mock_grpo_components["master_config"]
    master_config.grpo.max_num_steps = 1
    master_config.grpo.max_num_epochs = 1
    master_config.grpo.val_period = 0
    master_config.grpo.val_at_start = False
    master_config.grpo.val_at_end = False
    master_config.grpo.num_prompts_per_step = 1
    master_config.grpo.num_generations_per_prompt = 4
    master_config.grpo.dynamic_sampling_max_gen_batches = 2
    master_config.grpo.use_dynamic_sampling = True

    with pytest.raises(RuntimeError, match="captured dynamic-sampling inputs"):
        grpo_mod.grpo_train(
            mock_grpo_components["policy"],
            _mock_policy_generation(),
            mock_grpo_components["train_dataloader"],
            mock_grpo_components["val_dataloader"],
            mock_grpo_components["tokenizer"],
            mock_grpo_components["loss_fn"],
            mock_grpo_components["task_to_env"],
            mock_grpo_components["val_task_to_env"],
            mock_grpo_components["logger"],
            mock_grpo_components["checkpointer"],
            _initial_grpo_save_state(),
            master_config,
        )

    assert captured["prompts"].squeeze(1).tolist() == [0, 0, 0, 0]
    assert not captured["is_trivial"].any()


@pytest.mark.parametrize("train_func", [grpo_train, async_grpo_train])
def test_grpo_train_passes_prompt_group_ids_to_advantage_estimator(
    mock_grpo_components, train_func, monkeypatch
):
    """compute_advantage groups rows by prompt group id, not prompt tokens."""
    mock_adv_estimator = MagicMock()
    mock_adv_estimator.compute_advantage.return_value = torch.zeros(1, 2)
    monkeypatch.setattr(grpo_mod, "MemoryTracker", MagicMock)
    monkeypatch.setattr(
        "nemo_rl.algorithms.grpo._create_advantage_estimator",
        lambda _cfg: mock_adv_estimator,
    )

    _run_single_grpo_train_step(mock_grpo_components, train_func, monkeypatch)

    mock_adv_estimator.compute_advantage.assert_called_once()
    prompt_ids = mock_adv_estimator.compute_advantage.call_args.kwargs["prompt_ids"]
    assert torch.equal(prompt_ids, torch.zeros(1, 1, dtype=torch.long))


def test_duplicate_prompt_groups_have_independent_advantages():
    """Equal tokenized prompts sampled twice must not share a baseline."""
    rewards = torch.tensor([0.0, 2.0, 10.0, 14.0])
    estimator = GRPOAdvantageEstimator(
        AdvEstimatorConfig(normalize_rewards=False, use_leave_one_out_baseline=False),
        loss_config=grpo_tests.ClippedPGLossConfig(),
    )
    advantages = estimator.compute_advantage(
        prompt_ids=_prompt_group_ids(4, 2),
        rewards=rewards,
        mask=torch.ones(4, 1),
    )
    torch.testing.assert_close(advantages[:, 0], torch.tensor([-1.0, 1.0, -2.0, 2.0]))


def test_grpo_train_preserves_group_ids_across_dynamic_sampling_batches(
    mock_grpo_components, monkeypatch
):
    """Exercise the driver mint/filter/cache/concat/slice path with equal prompts."""
    original_batch = next(iter(mock_grpo_components["train_dataloader"]))
    input_batch = original_batch.repeat_interleave(2)
    mock_grpo_components["train_dataloader"].__iter__ = lambda self: iter(
        [input_batch, input_batch]
    )
    mock_grpo_components["train_dataloader"].__len__.return_value = 2
    reward_batches = iter(([0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]))
    rollout_calls = []

    def rollout(**kwargs):
        batch = kwargs["input_batch"]
        batch["total_reward"] = torch.tensor(next(reward_batches))
        for messages in batch["message_log"]:
            messages.append(
                {
                    "role": "assistant",
                    "content": "answer",
                    "token_ids": torch.tensor([4]),
                }
            )
        # A rewritten returned prompt must still belong to its original group.
        batch["message_log"][3][0] = {
            "role": "user",
            "content": "rewritten",
            "token_ids": torch.tensor([8, 9]),
        }
        rollout_calls.append(batch.size)
        return batch, {"mean_gen_tokens_per_sample": 1.0}

    def capture(batch):
        assert batch[PROMPT_GROUP_IDS_KEY].squeeze(1).tolist() == [1, 1, 2, 2]
        torch.testing.assert_close(
            batch["filtered_reward"], torch.tensor([0.0, 1.0, 0.0, 1.0])
        )
        raise RuntimeError("captured combined training batch")

    monkeypatch.setattr(grpo_mod, "should_use_async_rollouts", lambda _: False)
    monkeypatch.setattr(grpo_mod, "run_multi_turn_rollout", rollout)
    monkeypatch.setattr(grpo_mod, "refit_policy_generation", lambda *a, **k: {})
    monkeypatch.setattr(grpo_mod, "_apply_mask_sample_filter", capture)
    monkeypatch.setattr(grpo_mod, "MemoryTracker", MagicMock)
    cfg = mock_grpo_components["master_config"]
    cfg.grpo.max_num_steps = cfg.grpo.max_num_epochs = 1
    cfg.grpo.val_period = 0
    cfg.grpo.val_at_start = cfg.grpo.val_at_end = False
    cfg.grpo.num_prompts_per_step = cfg.grpo.num_generations_per_prompt = 2
    cfg.grpo.use_dynamic_sampling = True
    cfg.grpo.dynamic_sampling_max_gen_batches = 2
    with pytest.raises(RuntimeError, match="captured combined training batch"):
        grpo_train(
            mock_grpo_components["policy"],
            _mock_policy_generation(),
            mock_grpo_components["train_dataloader"],
            mock_grpo_components["val_dataloader"],
            mock_grpo_components["tokenizer"],
            mock_grpo_components["loss_fn"],
            mock_grpo_components["task_to_env"],
            mock_grpo_components["val_task_to_env"],
            mock_grpo_components["logger"],
            mock_grpo_components["checkpointer"],
            _initial_grpo_save_state(),
            cfg,
        )
    assert rollout_calls == [4, 4]
