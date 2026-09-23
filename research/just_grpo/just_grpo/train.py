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
"""Run Block JustGRPO through NeMo-RL's upstream GRPO controller."""

from pathlib import Path

import ray
from omegaconf import DictConfig, OmegaConf

from just_grpo.config import validate_config


def run(config: DictConfig) -> None:
    """Configure research adapters, then delegate setup and training upstream."""
    validate_config(config)
    # The full controller imports optional training and tracking dependencies.
    from just_grpo.environments.sudoku import SudokuEnvironment, SudokuResponseDataset
    from just_grpo.generation.megatron_generation import MegatronDiffusionGeneration
    from nemo_rl.algorithms.grpo import (
        MasterConfig,
        async_grpo_train,
        grpo_train,
        setup,
    )
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
        get_actor_python_env,
    )
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.weight_sync.factory import create_weight_synchronizer

    output = Path(config.logger.log_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "config.yaml")
    raw = OmegaConf.to_container(config, resolve=True)
    master = MasterConfig(**raw)
    worker = master.policy["worker_extension_cls_fqn"]
    ACTOR_ENVIRONMENT_REGISTRY[worker] = get_actor_python_env(
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
    )
    # Copy before attaching the full research config to avoid a recursive dict.
    master.policy = {**master.policy, "diffusion_config": raw}
    tokenizer = get_tokenizer(master.policy["tokenizer"])
    master.policy["generation"] = configure_generation_config(
        master.policy["generation"], tokenizer
    )
    master.policy["megatron_cfg"]["train_iters"] = master.grpo.max_num_steps
    if config.just_grpo.runtime == "reference_vllm":
        generation_worker = master.policy["generation"]["worker_extension_cls_fqn"]
        ACTOR_ENVIRONMENT_REGISTRY[generation_worker] = (
            config.just_grpo.generation_python
            or get_actor_python_env(
                "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
                if master.grpo.async_grpo.enabled
                else "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
            )
        )
    init_ray()
    train_data = SudokuResponseDataset(config, tokenizer)
    val_data = SudokuResponseDataset(config, tokenizer, validation=True)
    env = SudokuEnvironment.remote()
    environments = {config.data.default.env_name: env}
    logger = generation = None
    try:
        (
            policy,
            generation,
            _,
            _,
            dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            state,
            master,
            _,
            _,
        ) = setup(master, tokenizer, train_data, val_data)
        if config.just_grpo.runtime == "megatron":
            # Colocated setup has not started an engine (HTTP serving is disabled).
            generation.shutdown()
            generation = MegatronDiffusionGeneration(policy=policy, config=master)
            generation.weight_synchronizer = create_weight_synchronizer(
                policy=policy,
                generation=generation,
                generation_backend="megatron",
                colocated=True,
            )
            generation.weight_synchronizer.init_communicator()
        trainer = async_grpo_train if master.grpo.async_grpo.enabled else grpo_train
        trainer_kwargs = (
            {
                "max_trajectory_age_steps": master.grpo.async_grpo.max_trajectory_age_steps
            }
            if master.grpo.async_grpo.enabled
            else {}
        )
        if logger.wandb_logger is not None:
            (output / "wandb_url.txt").write_text(logger.wandb_logger.run.url + "\n")
        with checkpointer:
            trainer(
                policy,
                generation,
                dataloader,
                val_dataloader,
                tokenizer,
                loss_fn,
                environments,
                environments,
                logger,
                checkpointer,
                state,
                master,
                **trainer_kwargs,
            )
    finally:
        if generation is not None:
            generation.shutdown()
        if logger is not None:
            logger.finish()
        ray.shutdown()
