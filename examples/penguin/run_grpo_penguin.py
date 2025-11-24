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
import json
import os
import pprint
from itertools import chain, repeat
from typing import Optional

# Increase the W&B single object size warning threshold. Initially 100_000 (100 KB) -> 10_000_000 (10 MB)
import wandb.util

wandb.util.VALUE_BYTES_LIMIT = 10_000_000

import ray
from omegaconf import OmegaConf
from wandb import Table

from nemo_rl.algorithms.grpo import (
    ColocatablePolicyInterface,
    EnvironmentInterface,
    GenerationInterface,
    Logger,
    MasterConfig,
    StatefulDataLoader,
    TokenizerType,
    _should_use_penguin,
    grpo_train,
    refit_policy_generation,
    setup,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.penguin import (
    Penguin,
    PenguinConfig,
    penguin_example_to_nemo_rl_datum_spec,
    setup_penguin_config,
)
from nemo_rl.experience.rollouts import run_async_penguin_rollout
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def setup_single_penguin_dataset(
    jsonl_fpath: str, tokenizer, num_repeats: Optional[int] = None
):
    with open(jsonl_fpath) as f:
        penguin_examples = list(map(json.loads, f))

    print(f"Loaded data at {jsonl_fpath}. Found {len(penguin_examples)} examples")

    if num_repeats:
        previous_length = len(penguin_examples)
        penguin_examples = list(
            chain.from_iterable(
                repeat(penguin_example, num_repeats)
                for penguin_example in penguin_examples
            )
        )
        print(
            f"Repeating examples (in a pattern of abc to aabbcc) for {jsonl_fpath} from {previous_length} to {len(penguin_examples)}!"
        )

    nemo_rl_compatible_examples: list[DatumSpec] = [
        penguin_example_to_nemo_rl_datum_spec(penguin_example, idx)
        for idx, penguin_example in enumerate(penguin_examples)
    ]

    passthrough_task_processor = lambda datum_dict, *args, **kwargs: datum_dict
    return AllTaskProcessedDataset(
        nemo_rl_compatible_examples,
        tokenizer,
        None,
        passthrough_task_processor,
    )


# These types are directly imported from grpo_train since if something about the architecture changes we want to immediately fail.
def collect_trajectories(
    policy: ColocatablePolicyInterface,
    policy_generation: GenerationInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer: TokenizerType,
    val_task_to_env: dict[str, EnvironmentInterface],
    logger: Logger,
    master_config: MasterConfig,
) -> None:
    """Run trajectory collection."""
    # common config/state items
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    refit_policy_generation(policy, policy_generation, colocated_inference)

    log_filename = "trajectory_collection.jsonl"

    print("\n🔍 Running trajectory collection...", flush=True)
    generation_config = master_config["policy"]["generation"]
    for val_batch in val_dataloader:
        penguin_rollout_result = run_async_penguin_rollout(
            policy_generation=policy_generation,
            input_batch=val_batch,
            tokenizer=tokenizer,
            task_to_env=val_task_to_env,
            max_seq_len=None,
            generation_config=generation_config,
            max_rollout_turns=None,
            greedy=False,
        )

        rows_to_log: list[str] = []
        for key, value in penguin_rollout_result.rollout_metrics.items():
            if "full_result" not in key:
                continue

            value: Table
            data: list[list[str]] = value.data  # (n, 1)
            rows_to_log.extend(v[0] for v in data)

        logger.log_string_list_as_jsonl(rows_to_log, log_filename)

        # TODO: eventually as trajectory collection use cases exceed 4 hours, we can leverage the dataloader save functionality to resume
        # And also leverage the TimeoutChecker functionality as well

    policy_generation.finish_generation()


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "grpo_dapo17k_bytedtsinghua_qwen3_4binstruct_nf.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # Penguin specific config setup.
    setup_penguin_config(config, tokenizer)

    # We assert here since this is right after the final config has been materialized.
    assert _should_use_penguin(config)

    print("\n▶ Setting up data...")
    train_dataset = setup_single_penguin_dataset(
        jsonl_fpath=config["data"]["train_jsonl_fpath"],
        tokenizer=tokenizer,
    )
    val_dataset = setup_single_penguin_dataset(
        jsonl_fpath=config["data"]["validation_jsonl_fpath"],
        tokenizer=tokenizer,
    )

    # Validation dataset config setup.
    if config["grpo"]["max_val_samples"] is not None:
        raise ValueError(
            """A non-null `grpo.max_val_samples` parameter is not supported.

Gym principle is that there is no hidden data pre or post processing from you. What you see is what you get.

The validation set you pass in will directly be used for validation with no additional preprocessing. If you want to have some number of repetitions, please include that in your dataset, via ``num_repeats``, in your dataset config and `ng_prepare_data` will prepare it accordingly."""
        )

    print(
        f"Setting `grpo.max_val_samples` and `grpo.val_batch_size` to the length of the validation dataset, which is {len(val_dataset)}"
    )
    config["grpo"]["max_val_samples"] = len(val_dataset)
    config["grpo"]["val_batch_size"] = config["grpo"]["max_val_samples"]

    # Print config
    print("Final config:")
    pprint.pprint(config)

    init_ray()

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    is_trajectory_collection = (
        config["env"]["penguin"].pop("is_trajectory_collection", False) or False
    )
    penguin_config = PenguinConfig(
        model_name=policy_generation.cfg["model_name"],
        base_urls=policy_generation.dp_openai_server_base_urls,
        initial_global_config_dict=config["env"]["penguin"],
    )
    penguin = Penguin.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.penguin.Penguin"
            ),
        }
    ).remote(penguin_config)
    # Blocking wait for penguin to spin up
    ray.get(penguin.health_check.remote())
    task_to_env = {"penguin": penguin}
    val_task_to_env = task_to_env

    if is_trajectory_collection:
        collect_trajectories(
            policy=policy,
            policy_generation=policy_generation,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            val_task_to_env=val_task_to_env,
            logger=logger,
            master_config=master_config,
        )
    else:
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
