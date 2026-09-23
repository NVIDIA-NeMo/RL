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
"""Contracts between research adapters and upstream GRPO rollouts."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from just_grpo import train
from just_grpo.generation.megatron_generation import MegatronDiffusionGeneration
from omegaconf import OmegaConf
from test_sudoku_and_config import load, load_reference_vllm

from nemo_rl.models.generation.megatron.megatron_generation import MegatronGeneration
from nemo_rl.weight_sync.factory import create_weight_synchronizer


@pytest.mark.parametrize("mode", ["vllm", "megatron-sync", "megatron-async"])
def test_driver_uses_unmodified_setup_and_selects_upstream_trainer(
    monkeypatch, tmp_path, mode
):
    """Execute the research driver, mocking infrastructure and the training loops.

    Use the real native generation wrapper and synchronizer to check its
    constructor/refit requirements and that setup never starts the AR engine.
    The full upstream controller has optional dependencies in the CPU env.
    """
    config = load_reference_vllm() if mode == "vllm" else load()
    config.logger.log_dir = str(tmp_path)
    if mode == "megatron-async":
        config.grpo.async_grpo.enabled = True
        config.grpo.async_grpo.max_trajectory_age_steps = 2
    policy = Mock()
    logger = Mock(wandb_logger=None)
    checkpointer = MagicMock()
    returned_native = []
    trained = []

    def master_config(**raw):
        return SimpleNamespace(policy=raw["policy"], grpo=OmegaConf.create(raw["grpo"]))

    def setup(master, tokenizer, train_dataset, val_dataset):
        # Deliberately the upstream signature: no generation factory argument.
        if mode == "vllm":
            generation = Mock()
        else:
            generation = MegatronGeneration(
                config=master.policy, tokenizer=tokenizer, policy=policy
            )
            generation.weight_synchronizer = create_weight_synchronizer(
                policy=policy,
                generation=generation,
                generation_backend="megatron",
                colocated=True,
            )
            generation.weight_synchronizer.init_communicator()
            assert (
                policy.mock_calls == []
            )  # No engine initialization or refit in setup.
        returned_native.append(generation)
        return (
            policy,
            generation,
            None,
            None,
            [],
            [],
            Mock(),
            logger,
            checkpointer,
            Mock(),
            master,
            None,
            None,
        )

    def trainer(*args, **kwargs):
        generation = args[1]
        trained.append(generation)
        if mode == "vllm":
            assert generation is returned_native[0]
        else:
            assert isinstance(generation, MegatronDiffusionGeneration)
            assert generation is not returned_native[0]
            generation.weight_synchronizer.sync_weights()
            policy.offload_before_refit.assert_called_once()
            policy.prepare_for_lp_inference.assert_called_once()
            assert generation.blocks_training()
        assert kwargs == (
            {"max_trajectory_age_steps": 2} if mode == "megatron-async" else {}
        )

    grpo = ModuleType("nemo_rl.algorithms.grpo")
    grpo.MasterConfig = master_config
    grpo.setup = Mock(side_effect=setup)
    grpo.grpo_train = Mock(side_effect=trainer)
    grpo.async_grpo_train = Mock(side_effect=trainer)
    monkeypatch.setitem(sys.modules, grpo.__name__, grpo)
    utils = ModuleType("nemo_rl.algorithms.utils")
    utils.get_tokenizer = lambda config: SimpleNamespace(pad_token_id=0, eos_token_id=2)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    # The native colocated wrapper only imports Policy for its dedicated-model path.
    lm_policy = ModuleType("nemo_rl.models.policy.lm_policy")
    lm_policy.Policy = Mock()
    monkeypatch.setitem(sys.modules, lm_policy.__name__, lm_policy)
    from just_grpo.environments import sudoku

    from nemo_rl.distributed import ray_actor_environment_registry as registry
    from nemo_rl.distributed import virtual_cluster

    monkeypatch.setattr(sudoku, "SudokuResponseDataset", Mock(return_value=[]))
    monkeypatch.setattr(sudoku, "SudokuEnvironment", Mock())
    monkeypatch.setattr(registry, "ACTOR_ENVIRONMENT_REGISTRY", {})
    monkeypatch.setattr(registry, "get_actor_python_env", lambda _: sys.executable)
    monkeypatch.setattr(virtual_cluster, "init_ray", Mock())
    monkeypatch.setattr(train.ray, "shutdown", Mock())
    train.run(config)
    grpo.setup.assert_called_once()
    (
        grpo.async_grpo_train if mode == "megatron-async" else grpo.grpo_train
    ).assert_called_once()
    (
        grpo.grpo_train if mode == "megatron-async" else grpo.async_grpo_train
    ).assert_not_called()
    assert len(trained) == 1
    logger.finish.assert_called_once()
    train.ray.shutdown.assert_called_once()
    lm_policy.Policy.assert_not_called()
