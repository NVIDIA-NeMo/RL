import ast
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from nemo_rl.algorithms.critic_diagnostics import (
    critic_position_metrics,
    critic_reward_auc_metrics,
)
from nemo_rl.algorithms.critic_replay import CriticReplayBuffer, CriticReplayConfig


def test_quartile_auc_and_masking() -> None:
    values = torch.tensor([[0.1] * 6 + [0.9] * 2, [0.9] * 6 + [0.1] * 2, [1.0] * 8])
    rewards = torch.tensor([0.0, 1.0, 0.0])
    mask = torch.ones_like(values)
    mask[2] = 0
    metrics = critic_reward_auc_metrics(values, rewards, mask)
    assert metrics["critic/global_auc_q1"] == 1
    assert metrics["critic/global_auc_q4"] == 0
    assert metrics["critic/terminal_auc"] == 0
    assert metrics["critic/auc_positive_samples"] == 1
    assert metrics["critic/auc_negative_samples"] == 1
    tied = critic_reward_auc_metrics(torch.ones_like(values), rewards, mask)
    assert tied["critic/global_auc_q2"] == 0.5
    single = critic_reward_auc_metrics(values, torch.zeros(3), mask)
    assert single["critic/global_auc_q2"] != single["critic/global_auc_q2"]


@pytest.mark.parametrize("period", [0, 1, 2])
def test_production_probe_keeps_actor_data_unchanged(period: int) -> None:
    path = Path(__file__).parents[3] / "nemo_rl/algorithms/ppo.py"
    tree = ast.parse(path.read_text())
    names = {
        "_probe_critic_replay_update",
        "_compute_rollout_critic_metrics",
        "_masked_distribution_metrics",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    namespace = {
        "torch": torch,
        "BatchedDataDict": dict,
        "critic_position_metrics": critic_position_metrics,
        "critic_reward_auc_metrics": critic_reward_auc_metrics,
    }
    exec(compile(module, str(path), "exec"), namespace)
    data = {
        "values": torch.full((2, 8), 0.5),
        "returns": torch.tensor([[0.0] * 8, [1.0] * 8]),
        "rewards": torch.tensor([0.0, 1.0]),
        "token_mask": torch.ones(2, 8),
        "sample_mask": torch.ones(2),
        "advantages": torch.randn(2, 8),
    }
    before = {key: value.clone() for key, value in data.items()}
    model = Mock()
    model.get_values.return_value = {
        "values": torch.tensor([[0.25] * 8, [0.75] * 8]).unsqueeze(-1)
    }
    replay = CriticReplayBuffer(CriticReplayConfig(probe_every_n_steps=period))
    metrics = namespace["_probe_critic_replay_update"](
        model, data, replay, step=0, update=0
    )
    if period == 1:
        prefix = "critic_replay/fresh_probe/after_update_1/"
        assert metrics[prefix + "value_shift_rmse"] == 0.25
        assert metrics[prefix + "mse_improvement_from_pre"] == 0.1875
        assert metrics[prefix + "global_auc_q4"] == 1
        assert prefix + "ev_early" in metrics
        model.finish_inference.assert_called_once()
    else:
        assert metrics == {}
        model.get_values.assert_not_called()
    for key in data:
        torch.testing.assert_close(data[key], before[key])
