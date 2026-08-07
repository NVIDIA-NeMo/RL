"""Tests for scoped integration with NeMo-RL's synchronous GRPO modules."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("ray", reason="NeMo-RL integration dependencies are unavailable")

from run_grpo_turn_credit import load_master_and_turn_credit_config
from turn_level_credit.config import TurnCreditConfig
from turn_level_credit.integration import install_turn_credit_runtime

from nemo_rl.algorithms import grpo as grpo_module
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentReturn
from nemo_rl.experience import rollouts as rollout_module


def _master_config():
    return SimpleNamespace(
        grpo=grpo_module.GRPOConfig(),
        data_plane={"enabled": False},
        env={"should_use_nemo_gym": False},
        policy={
            "generation": {
                "backend": "vllm",
                "vllm_cfg": {"async_engine": False},
            }
        },
    )


def test_checked_in_config_loads_current_master_schema():
    config_path = (
        Path(__file__).parents[2] / "configs" / "grpo_math_0.5b_turn_credit.yaml"
    )

    master_config, turn_credit_config = load_master_and_turn_credit_config(
        str(config_path),
        [],
    )

    assert isinstance(master_config, grpo_module.MasterConfig)
    assert master_config.grpo.adv_estimator.name == "grpo"
    assert turn_credit_config == TurnCreditConfig(enabled=True, turn_weight=0.2)


def test_disabled_runtime_does_not_install_hooks():
    original_calculate_rewards = rollout_module.calculate_rewards
    original_rollout = grpo_module.run_multi_turn_rollout
    original_estimator_factory = grpo_module._create_advantage_estimator

    with install_turn_credit_runtime(TurnCreditConfig(enabled=False)):
        assert rollout_module.calculate_rewards is original_calculate_rewards
        assert grpo_module.run_multi_turn_rollout is original_rollout
        assert grpo_module._create_advantage_estimator is original_estimator_factory

    assert rollout_module.calculate_rewards is original_calculate_rewards
    assert grpo_module.run_multi_turn_rollout is original_rollout
    assert grpo_module._create_advantage_estimator is original_estimator_factory


def test_runtime_hooks_capture_metrics_and_restore_after_error(monkeypatch, capsys):
    batch = BatchedDataDict(
        {
            "message_log": [
                [
                    {
                        "role": "assistant",
                        "content": "answer",
                        "token_ids": torch.tensor([1, 2]),
                        "generation_logprobs": torch.zeros(2),
                    }
                ]
            ]
        }
    )

    def fake_calculate_rewards(_batch, _task_to_env):
        return EnvironmentReturn(
            observations=[{"role": "user", "content": "done"}],
            metadata=[None],
            next_stop_strings=[None],
            rewards=torch.tensor([0.75]),
            terminateds=torch.tensor([True]),
            answers=[None],
        )

    def fake_rollout(**kwargs):
        rollout_batch = kwargs["input_batch"]
        environment_return = rollout_module.calculate_rewards(rollout_batch, {})
        rollout_batch["total_reward"] = environment_return.rewards
        return rollout_batch, {"total_turns": 1}

    class _BaseEstimator:
        def compute_advantage(self, **_kwargs):
            return torch.ones((1, 2))

    def fake_estimator_factory(_master_config):
        return _BaseEstimator()

    monkeypatch.setattr(rollout_module, "calculate_rewards", fake_calculate_rewards)
    monkeypatch.setattr(grpo_module, "run_multi_turn_rollout", fake_rollout)
    monkeypatch.setattr(
        grpo_module,
        "_create_advantage_estimator",
        fake_estimator_factory,
    )

    with pytest.raises(RuntimeError, match="sentinel"):
        with install_turn_credit_runtime(
            TurnCreditConfig(enabled=True, turn_weight=0.2)
        ):
            final_batch, metrics = grpo_module.run_multi_turn_rollout(
                policy_generation=None,
                input_batch=batch,
                tokenizer=None,
                task_to_env={},
                max_seq_len=8,
            )
            estimator = grpo_module._create_advantage_estimator(_master_config())

            assert final_batch["turn_rewards"].tolist() == [[0.75]]
            assert final_batch["assistant_turn_spans"].tolist() == [[[0, 2]]]
            assert metrics["turn_credit/environment_reward/mean"] == 0.75
            assert metrics["turn_credit/credit/mean"] == 0.75
            metric_line = capsys.readouterr().out
            assert "TURN_CREDIT_ROLLOUT_METRICS" in metric_line
            assert "turn_credit/turns_per_sample/mean=1.0" in metric_line
            assert "turn_credit/environment_reward/mean=0.75" in metric_line
            assert torch.allclose(
                estimator.compute_advantage(
                    prompt_ids=torch.tensor([0]),
                    rewards=torch.tensor([0.75]),
                    mask=torch.tensor([[True, True]]),
                    repeated_batch=final_batch,
                ),
                torch.tensor([[1.15, 1.15]]),
            )
            raise RuntimeError("sentinel")

    assert rollout_module.calculate_rewards is fake_calculate_rewards
    assert grpo_module.run_multi_turn_rollout is fake_rollout
    assert grpo_module._create_advantage_estimator is fake_estimator_factory
