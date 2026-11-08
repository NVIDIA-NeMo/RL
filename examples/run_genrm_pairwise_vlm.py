# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import os
import re
import pprint
import time
from typing import Any, Optional
from collections import defaultdict

from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig, setup, shutdown_environments, grpo_train, async_grpo_train
from nemo_rl.algorithms.loss import DWRLPairwiseLossFn
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.genrm_multimodal_utils import VIDEO_DEFAULTS, setup_vlm_tokenizer
from nemo_rl.data.genrm_processors import dwrl_pairwise_data_processor
from nemo_rl.data.interfaces import TaskDataSpec, TaskDataProcessFnCallable
#from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir, log_container_init_timing
from nemo_rl.utils.timer import Timer
from nemo_rl.data.datasets.preference_datasets.reward_benchmarks import HelpSteer3LocalDataset
#from nemo_rl.environments.binary_genrm_environment import (
#    GenRMEnvironment,
#    format_scoring_stage_prompt,
#)
#from nemo_rl.environments.genrm_environment import GenRMEnvironment
from nemo_rl.environments.vanilla_genrm_environment import VanillaGenRMEnvironment

TokenizerType = PreTrainedTokenizerBase


def _select_trainer(master_config: MasterConfig):
    """Pick the synchronous trainer based on ``data_plane.enabled``.

    Factored out so test_architecture_invariants can verify dispatch
    without the full setup() path.
    """

    print("ðŸš€ Running synchronous DWRL training (legacy)")
    return grpo_train

def setup_data(
    tokenizer: TokenizerType,
    data_config: DataConfig,
    env_configs: dict[str, Any],
    seed: int,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\n▶ Setting up GRPO GenRM data...")

    # Create task spec for GenRM RLHF
    # Video settings (num_frames, video_temporal_patch_size, video_target_num_patches, ...) come from `data.default`;
    # anything missing falls back to VIDEO_DEFAULTS inside the data processor.
    default_data_config = data_config.get("default") or {}
    genrm_dwrl_task_spec = TaskDataSpec(
        task_name="genrm_dwrl",
        prompt_file=data_config.get("prompt_file"),
        system_prompt_file=data_config.get("system_prompt_file"),
        **{key: default_data_config[key] for key in VIDEO_DEFAULTS if default_data_config.get(key) is not None},
    )

    # Load dataset
    train_data_path = data_config["train"].get("data_path")
    val_data_path = data_config["validation"].get("data_path")
    train_dataset = HelpSteer3LocalDataset(train_data_path, media_path=data_config["train"].get("media_path"), task_name="genrm_dwrl", shuffle_seed=seed, split='train', double_swap=True)
    val_dataset = HelpSteer3LocalDataset(val_data_path, media_path=data_config["validation"].get("media_path"), task_name="genrm_dwrl", shuffle_seed=seed, split='validation', double_swap=True) if val_data_path else None

    # Set up data processor
    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (genrm_dwrl_task_spec, dwrl_pairwise_data_processor))
    )
    task_data_processors["genrm_dwrl"] = (
        genrm_dwrl_task_spec,
        dwrl_pairwise_data_processor,
    )
    # Also register under alternate name for backward compatibility
    task_data_processors["dwrl_genrm"] = (
        genrm_dwrl_task_spec,
        dwrl_pairwise_data_processor,
    )
    
    # Initialize GenRM environment
    '''
    genrm_env = GenRMEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.genrm_environment.GenRMEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs.get("genrm_dwrl", {}))
    '''
    genrm_env = VanillaGenRMEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.vanilla_genrm_environment.VanillaGenRMEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs.get("genrm_dwrl", {}))
    

    # Create training dataset
    processed_train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        genrm_dwrl_task_spec,  # default_task_data_spec
        task_data_processors,  # task_data_processors dict
        max_seq_length=data_config["max_input_seq_length"],
    )

    # Create validation dataset if available
    processed_val_dataset: Optional[AllTaskProcessedDataset] = None
    if val_dataset:
        processed_val_dataset = AllTaskProcessedDataset(
            val_dataset,
            tokenizer,
            genrm_dwrl_task_spec,  # default_task_data_spec
            task_data_processors,  # task_data_processors dict
            max_seq_length=data_config["max_input_seq_length"],
        )

    # Map task to environment
    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: genrm_env)
    task_to_env["genrm_dwrl"] = genrm_env
    # Also register under alternate name for backward compatibility
    task_to_env["dwrl_genrm"] = genrm_env

    return processed_train_dataset, processed_val_dataset, task_to_env, task_to_env


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def main() -> None:
    """Main entry point."""
    main_start = time.perf_counter()
    log_container_init_timing()
    rl_init_timer = Timer(context={"worker": "driver"})
    
    # Parse arguments
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_genrm_dwrl_pairwise_1B.yaml"
        )

    with rl_init_timer.time("config"):
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    
        if overrides:
            print(f"Overrides: {overrides}")
            config = parse_hydra_overrides(config, overrides)
    
        config = OmegaConf.to_container(config, resolve=True)
        config = MasterConfig(**config)
        print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"ðŸ“Š Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(
            f"ðŸ“Š Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = setup_vlm_tokenizer(config, __file__)
        
    assert config.policy["generation"] is not None, (
        "A generation config is required for GenRM GRPO"
    )
    has_refit_draft_weights = bool(config.policy["draft"]["enabled"])
    megatron_cfg = config.policy.get("megatron_cfg") or {}
    trains_mtp = bool(megatron_cfg.get("mtp_num_layers"))
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
        has_refit_draft_weights=has_refit_draft_weights,
        trains_mtp=trains_mtp,
    )

    # setup data
    with rl_init_timer.time("data"):
        (
            dataset,
            val_dataset,
            task_to_env,
            val_task_to_env,
        ) = setup_data(tokenizer, config.data, config.env, config.grpo.seed)
    
    with rl_init_timer.time("setup"):
        (
            policy,
            policy_generation,
            _nemo_gym,
            cluster,
            dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            grpo_state,
            master_config,
            teacher_worker_groups,
            alias_to_group_alias,

        ) = setup(
            config,
            tokenizer,
            dataset,
            val_dataset,
            policy_factory=None,
        )
    
    rl_init_timer.record("total", time.perf_counter() - main_start)

    rl_init_metrics = rl_init_timer.get_timing_metrics(reduction_op="sum")
    print("\n" + "=" * 60)
    print(" " * 14 + "RL INIT TIMING BREAKDOWN")
    for label, value in sorted(rl_init_metrics.items()):
        if isinstance(value, (int, float)):
            print(f"  {label}: {value:.1f}s")
    print("=" * 60 + "\n", flush=True)


    try:
        # Check if async mode is enabled
        if config.grpo.async_grpo.enabled:
            # Async GRPO does not support dynamic sampling, reward scaling, or reward shaping (DAPO features)
            if config.grpo.use_dynamic_sampling:
                raise NotImplementedError(
                    "use_dynamic_sampling is not supported with async GRPO"
                )
            if config.grpo.reward_scaling.enabled:
                raise NotImplementedError(
                    "reward_scaling is not supported with async GRPO"
                )
            if config.grpo.reward_shaping.enabled:
                raise NotImplementedError(
                    "reward_shaping is not supported with async GRPO"
                )

            # Async GRPO does not support multiple dataloaders
            if config.data["use_multiple_dataloader"]:
                raise NotImplementedError(
                    "use_multiple_dataloader is not supported with async GRPO"
                )

            print("ðŸš€ Running async GenRM GRPO training")

            # Run async GRPO training
            async_grpo_train(
                policy=policy,
                policy_generation=policy_generation,
                dataloader=dataloader,
                val_dataloader=val_dataloader,
                tokenizer=tokenizer,
                loss_fn=loss_fn,
                task_to_env=task_to_env,
                val_task_to_env=val_task_to_env,
                logger=logger,
                checkpointer=checkpointer,
                grpo_save_state=grpo_state,
                master_config=master_config,
                max_trajectory_age_steps=config.grpo.async_grpo.max_trajectory_age_steps,
                teacher_worker_groups=teacher_worker_groups,
                alias_to_group_alias=alias_to_group_alias,
            )
        else:
            # Two parallel synchronous trainers (verl-style â€” main_ppo.py vs
            # main_ppo_sync.py). data_plane.enabled selects which one runs.
            trainer = _select_trainer(master_config)
            # grpo_train_sync defers checkpoint finalization to the checkpointer's
            # background threads; the context manager guarantees they are flushed on
            # exit. (grpo_train also flushes internally; shutdown() is idempotent.)
            with checkpointer:
                trainer(
                    policy,
                    policy_generation,
                    dataloader,
                    val_dataloader,
                    tokenizer,
                    loss_fn,
                    task_to_env,
                    val_task_to_env,
                    logger,
                    checkpointer,
                    grpo_state,
                    master_config,
                )
    finally:
        shutdown_environments(task_to_env, val_task_to_env)
        try:
            policy_generation.shutdown()
        except Exception as error:
            print(f"Error shutting down generation: {error}", flush=True)


if __name__ == "__main__":
    main()
