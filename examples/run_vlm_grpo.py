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
import pprint
import time

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir, log_container_init_timing
from nemo_rl.utils.timer import Timer


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

    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "vlm_grpo_3B.yaml"
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
    print(f"📊 Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    with rl_init_timer.time("ray_connect"):
        init_ray()

    with rl_init_timer.time("tokenizer"):
        processor = get_tokenizer(config.policy["tokenizer"], get_processor=True)
    tokenizer = processor.tokenizer

    assert config.policy["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], processor.tokenizer
    )
    if "vllm_cfg" in config.policy["generation"]:
        assert (
            config.policy["generation"]["vllm_cfg"]["skip_tokenizer_init"] == False
        ), (
            "VLMs require tokenizer to be initialized before generation, so skip_tokenizer_init must be set to False."
        )

    with rl_init_timer.time("data"):
        dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(
            processor, config.data, config.env, is_vlm=True
        )

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
            _teacher_worker_groups,
            _alias_to_group_alias,
        ) = setup(config, tokenizer, dataset, val_dataset, processor=processor)

    rl_init_timer.record("total", time.perf_counter() - main_start)
    rl_init_metrics = rl_init_timer.get_timing_metrics(reduction_op="sum")
    print("\n" + "=" * 60)
    print(" " * 14 + "RL INIT TIMING BREAKDOWN")
    for label, value in sorted(rl_init_metrics.items()):
        if isinstance(value, (int, float)):
            print(f"  {label}: {value:.1f}s")
    print("=" * 60 + "\n", flush=True)

    grpo_train(
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


if __name__ == "__main__":
    main()
