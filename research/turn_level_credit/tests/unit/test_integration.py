"""Tests for scoped integration with NeMo-RL's synchronous GRPO modules."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("ray", reason="NeMo-RL integration dependencies are unavailable")

from turn_level_credit.config import TurnCreditConfig
from turn_level_credit.integration import install_turn_credit_runtime

from nemo_rl.algorithms import grpo as grpo_module
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentReturn
from nemo_rl.experience import rollouts as rollout_module


def _master_config():
    return SimpleNamespace(
        grpo={
            "adv_estimator": {"name": "grpo"},
            "async_grpo": {"enabled": False},
        },
        data_plane={"enabled": False},
        env={"should_use_nemo_gym": False},
        policy={
            "generation": {
                "backend": "vllm",
                "vllm_cfg": {"async_engine": False},
            }
        },
    )


def test_runtime_hooks_capture_metrics_and_restore_after_error(monkeypatch):
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
        return rollout_batch, {}

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
