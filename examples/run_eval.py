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
from transformers import AutoProcessor, PreTrainedTokenizerBase

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_eval_dataset
from nemo_rl.data.datasets.eval_datasets import _is_multimodal_dataset
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.utils import create_env
from nemo_rl.evals.eval import MasterConfig, run_env_eval, setup
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config

TokenizerType = PreTrainedTokenizerBase


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


def setup_data(tokenizer, data_config, env_configs):
    print("Setting up data...")

    # load dataset
    base_dataset = load_eval_dataset(data_config)
    rekeyed_ds = base_dataset.rekeyed_ds

    is_multimodal = _is_multimodal_dataset(data_config["dataset_name"])

    if is_multimodal:
        # Use VLMEnvironment for multimodal datasets
        env_key = next((k for k in env_configs if k in ("mmau",)), None)
        if env_key is None:
            raise ValueError(
                f"No environment config found for multimodal dataset. "
                f"Available env configs: {list(env_configs.keys())}"
            )
        env = create_env(env_name="vlm", env_config=env_configs[env_key])
    else:
        # Original text-only path
        env = create_env(env_name="math", env_config=env_configs["math"])

    dataset = AllTaskProcessedDataset(
        dataset=rekeyed_ds,
        tokenizer=tokenizer,
        default_task_data_spec=base_dataset.task_spec,
        task_data_processors=base_dataset.processor,
        task_data_preprocessors=getattr(base_dataset, "preprocessor", None),
        max_seq_length=data_config["max_input_seq_length"],
    )

    return dataset, env, tokenizer


def main():
    """Main entry point."""
    # Parse arguments
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

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Init ray
    init_ray()

    # Determine if this is a multimodal dataset
    is_multimodal = _is_multimodal_dataset(config["data"]["dataset_name"])

    # Setup tokenizer — use AutoProcessor for multimodal, AutoTokenizer for text
    if is_multimodal:
        tokenizer = AutoProcessor.from_pretrained(
            config["tokenizer"]["name"], trust_remote_code=True
        )
        # configure_generation_config expects tokenizer with pad_token_id/eos_token_id
        config["generation"] = configure_generation_config(
            config["generation"], tokenizer.tokenizer, is_eval=True
        )
    else:
        tokenizer = get_tokenizer(config["tokenizer"])
        config["generation"] = configure_generation_config(
            config["generation"], tokenizer, is_eval=True
        )

    # Setup data
    (
        dataset,
        env,
        tokenizer,
    ) = setup_data(tokenizer, config["data"], config["env"])

    # Setup
    (
        vllm_generation,
        dataloader,
        master_config,
    ) = setup(config, tokenizer, dataset)

    # Run evaluation
    run_env_eval(
        vllm_generation,
        dataloader,
        env,
        master_config,
    )


if __name__ == "__main__":
    main()
