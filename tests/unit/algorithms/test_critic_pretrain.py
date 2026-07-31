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

import math
from pathlib import Path
from types import SimpleNamespace

import torch

from nemo_rl.algorithms.critic_pretrain import (
    build_value_train_data,
    split_heldout,
    terminal_value_reward_auc,
)


def _paths(indices):
    return [Path(f"shard_000/group_{i:08d}.pt") for i in indices]


def test_split_heldout_partition():
    """idx % mod == 0 goes to held-out; mod<=0 disables the split."""
    files = _paths(range(33))
    train, heldout = split_heldout(files, 16)
    assert {int(p.name[6:14]) for p in heldout} == {0, 16, 32}
    assert len(train) + len(heldout) == len(files)
    assert set(train).isdisjoint(heldout)

    train_all, heldout_none = split_heldout(files, 0)
    assert train_all == files and heldout_none == []


def test_terminal_auc_perfect_and_inverted():
    """Terminal-token value ranking success perfectly => AUC 1; inverted => 0."""
    B, S = 4, 5
    mask = torch.ones(B, S)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    values = torch.zeros(B, S)
    values[:, -1] = torch.tensor([0.9, 0.1, 0.8, 0.2])  # separates classes
    assert terminal_value_reward_auc(values, rewards, mask) == 1.0
    values[:, -1] = torch.tensor([0.1, 0.9, 0.2, 0.8])  # inverted
    assert terminal_value_reward_auc(values, rewards, mask) == 0.0


def test_terminal_auc_single_class_and_ties():
    """Single outcome class => nan; fully tied scores => 0.5."""
    B, S = 3, 4
    mask = torch.ones(B, S)
    assert math.isnan(
        terminal_value_reward_auc(torch.rand(B, S), torch.ones(B), mask)
    )
    values = torch.full((4, S), 0.5)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    auc = terminal_value_reward_auc(values, rewards, torch.ones(4, S))
    assert abs(auc - 0.5) < 1e-6


def test_terminal_auc_respects_mask():
    """The scored token is the LAST masked position, not the last column."""
    B, S = 2, 6
    mask = torch.zeros(B, S)
    mask[0, 1:3] = 1  # response ends at position 2
    mask[1, 1:5] = 1  # response ends at position 4
    values = torch.zeros(B, S)
    values[0, 2] = 0.9  # positive sample's terminal value
    values[1, 4] = 0.1
    values[:, -1] = torch.tensor([0.0, 1.0])  # decoys at the last column
    rewards = torch.tensor([1.0, 0.0])
    assert terminal_value_reward_auc(values, rewards, mask) == 1.0


def _make_group(rewards, truncated, mask_sample, prompt_len=3, resp_len=4):
    """Build a group payload shaped like a stored stage-A shard."""
    n = len(rewards)
    message_log = []
    for _ in range(n):
        message_log.append(
            [
                {"role": "user", "token_ids": torch.arange(prompt_len)},
                {"role": "assistant", "token_ids": torch.arange(resp_len)},
            ]
        )
    batch = {
        "message_log": message_log,
        "length": torch.full((n,), prompt_len),
        "total_reward": torch.tensor(rewards, dtype=torch.float32),
        "loss_multiplier": torch.ones(n),
        "truncated": torch.tensor(truncated, dtype=torch.bool),
        "mask_sample": torch.tensor(mask_sample, dtype=torch.bool),
        "idx": [0] * n,
        "task_name": ["nemo_gym"] * n,
        "extra_env_info": [{} for _ in range(n)],
    }
    return {"format_version": 1, "dataset_idx": 0, "batch": batch}


def _master_config(overlong_filtering=True):
    return SimpleNamespace(
        ppo={"overlong_filtering": overlong_filtering},
        policy={"make_sequence_length_divisible_by": 1},
    )


def test_build_value_train_data_masks_and_shapes():
    """Assistant-only token mask; truncated/env-flagged samples get
    sample_mask=0 (matching the async PPO loop's processing)."""
    tokenizer = SimpleNamespace(pad_token_id=0)
    g1 = _make_group([1.0, 0.0], truncated=[False, True], mask_sample=[False, False])
    g2 = _make_group([0.0, 1.0], truncated=[False, False], mask_sample=[True, False])
    train_data, repeated_batch = build_value_train_data(
        [g1, g2], tokenizer, _master_config()
    )

    assert train_data["input_ids"].shape[0] == 4
    assert train_data["input_ids"].shape == train_data["token_mask"].shape
    # 3 prompt tokens masked out, 4 assistant tokens unmasked, per sample
    assert torch.equal(
        train_data["token_mask"].sum(dim=1), torch.full((4,), 4.0)
    )
    # sample 1 truncated (overlong filtering), sample 2 env-flagged
    torch.testing.assert_close(
        train_data["sample_mask"], torch.tensor([1.0, 0.0, 0.0, 1.0])
    )
    torch.testing.assert_close(
        train_data["rewards"], torch.tensor([1.0, 0.0, 0.0, 1.0])
    )
    assert repeated_batch["message_log"][0][1]["token_loss_mask"].sum() == 4


def test_build_value_train_data_no_overlong_filtering():
    """With overlong_filtering off, truncated samples keep sample_mask=1."""
    tokenizer = SimpleNamespace(pad_token_id=0)
    g = _make_group([1.0, 0.0], truncated=[True, True], mask_sample=[False, False])
    train_data, _ = build_value_train_data(
        [g], tokenizer, _master_config(overlong_filtering=False)
    )
    torch.testing.assert_close(train_data["sample_mask"], torch.ones(2))


def test_offline_returns_are_reward_to_go():
    """The stage-B invariant: with gae_lambda_value=1, gae_gamma=1, KL off, the
    critic's regression targets equal the terminal reward broadcast over
    response tokens — independent of the values fed in (so offline pretraining
    on stored rollouts is exactly the online warmup's optimization)."""
    from nemo_rl.algorithms.advantage_estimator import (
        GeneralizedAdvantageEstimator,
    )

    estimator = GeneralizedAdvantageEstimator(
        {
            "name": "gae",
            "gae_lambda": 1.0,
            "gae_gamma": 1,
            "normalize_advantages": True,
            "gae_lambda_value": 1.0,
            "gae_lambda_policy": 1,
            "length_adaptive_alpha": 1.5,
        },
        SimpleNamespace(
            use_kl_in_reward=False,
            reference_policy_kl_penalty=0.0,
            reference_policy_kl_type="low_var_kl",
        ),
    )
    B, S = 3, 6
    mask = torch.zeros(B, S)
    mask[0, 2:6] = 1
    mask[1, 1:4] = 1
    mask[2, 3:5] = 1
    rewards = torch.tensor([1.0, 0.0, 0.5])
    values = torch.randn(B, S)  # returns must NOT depend on these

    _, returns = estimator.compute_advantage(
        prompt_ids=torch.zeros(B, 2, dtype=torch.long),
        rewards=rewards,
        mask=mask,
        values=values,
        reference_logprobs=None,
        logprobs=None,  # the critic-pretrain calling pattern (no policy)
    )

    expected = rewards.unsqueeze(1) * mask
    torch.testing.assert_close(returns * mask, expected)
