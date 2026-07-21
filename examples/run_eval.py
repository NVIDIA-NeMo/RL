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
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import (
    AllTaskProcessedDataset,
    load_eval_dataset,
    load_response_dataset,
)
from nemo_rl.data.datasets.eval_datasets import _is_multimodal_dataset
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import setup_nemo_gym_generation_config
from nemo_rl.environments.utils import create_env
from nemo_rl.evals.eval import (
    EvalRunResult,
    MasterConfig,
    NemoGymEvalDataConfig,
    run_env_eval,
    setup,
    setup_nemo_gym_environment,
    should_use_nemo_gym,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config
from nemo_rl.utils.logger import Logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Evaluation with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, remaining = parser.parse_known_args()

    # Convert remaining args to OmegaConf format
    overrides = OmegaConf.from_dotlist(remaining)

    return args, overrides


def setup_data(tokenizer, data_config, env_configs, is_multimodal=False):
    # is_multimodal only affects non-Gym eval, where the multimodal processor is
    # already carried by `tokenizer` (built via get_tokenizer(get_processor=...)).
    # The NeMo Gym branch below returns before it is ever consulted; accept the
    # kwarg so the run_eval() call site stays consistent.
    print("Setting up data...")

    if isinstance(data_config, NemoGymEvalDataConfig):
        base_dataset = load_response_dataset(data_config.model_dump())
        dataset = AllTaskProcessedDataset(
            dataset=base_dataset.dataset,
            tokenizer=tokenizer,
            default_task_data_spec=base_dataset.task_spec,
            task_data_processors=base_dataset.processor,
            task_data_preprocessors=base_dataset.preprocessor,
            max_seq_length=data_config.max_input_seq_length,
        )
        return dataset, None, tokenizer

    # load dataset
    base_dataset = load_eval_dataset(data_config)
    rekeyed_ds = base_dataset.rekeyed_ds

    # Determine env from config: use explicit env_name if provided,
    # otherwise fall back to the single key in env_configs.
    env_key = next(iter(env_configs))
    env_name = data_config.get("env_name", env_key)
    env = create_env(env_name=env_name, env_config=env_configs[env_key])

    dataset = AllTaskProcessedDataset(
        dataset=rekeyed_ds,
        tokenizer=tokenizer,
        default_task_data_spec=base_dataset.task_spec,
        task_data_processors=base_dataset.processor,
        task_data_preprocessors=getattr(base_dataset, "preprocessor", None),
        max_seq_length=data_config["max_input_seq_length"],
    )

    return dataset, env, tokenizer


def run_eval(config: MasterConfig) -> EvalRunResult:
    """Run evaluation from an already validated eval configuration."""
    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Init ray
    init_ray()

    # Setup tokenizer — get_tokenizer handles both text-only and multimodal
    is_multimodal = not should_use_nemo_gym(config) and _is_multimodal_dataset(
        config.data["dataset_name"]
    )
    tokenizer = get_tokenizer(config.tokenizer, get_processor=is_multimodal)
    config.generation = configure_generation_config(
        config.generation, tokenizer, is_eval=True
    )
    if should_use_nemo_gym(config):
        setup_nemo_gym_generation_config(config.generation)

    # Setup data
    (
        dataset,
        env,
        tokenizer,
    ) = setup_data(tokenizer, config.data, config.env, is_multimodal=is_multimodal)

    # Setup
    (
        vllm_generation,
        dataloader,
        master_config,
    ) = setup(config, tokenizer, dataset)

    if should_use_nemo_gym(master_config):
        env = setup_nemo_gym_environment(vllm_generation, master_config)

    logger = Logger(master_config.logger) if master_config.logger is not None else None
    if logger is not None:
        logger.log_hyperparams(master_config.model_dump())

    # Run evaluation
    try:
        result = run_env_eval(
            vllm_generation,
            dataloader,
            env,
            master_config,
            tokenizer=tokenizer,
            logger=logger,
        )
    finally:
        if logger is not None:
            logger.close()
    return result


def main() -> None:
    """Load a standard eval YAML and run evaluation."""
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "evals", "eval.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        override_conf = OmegaConf.from_cli()
        print(f"Overrides: {override_conf}")
        config = OmegaConf.merge(config, override_conf)

    config = OmegaConf.to_container(config, resolve=True)
    master_config = MasterConfig(**config)
    print("Applied CLI overrides")
    run_eval(master_config)


if __name__ == "__main__":
    main()
