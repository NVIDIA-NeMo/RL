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

import jsonlines

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from omegaconf import OmegaConf

from examples.run_grpo import setup_data
from nemo_rl.algorithms.grpo import validate
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.evals.eval import MasterConfig, setup
from nemo_rl.models.generation.interfaces import configure_generation_config


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


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "eval.yaml")

    config = OmegaConf.load(args.config)
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

    # Setup tokenizer
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(
        config["generation"], tokenizer, is_eval=True
    )

    # setup data
    (
        dataset,
        _,
        task_to_env,
        _,
    ) = setup_data(tokenizer, config["data"], config["env"])

    # Setup
    (
        vllm_generation,
        dataloader,
        _,
    ) = setup(config, tokenizer, dataset)

    val_metrics, _, data_for_saving = validate(
        vllm_generation,
        dataloader,
        tokenizer,
        task_to_env,
        0,
        config["master_config"],
        num_repeats=config["eval"]["num_repeats"],
        return_data_for_saving=True,
    )

    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Atomic save using jsonlines
    base_name = Path(config["data"]["train"]["jsonl_path"]).stem
    output_file = os.path.join(save_dir, f"{base_name}_sampled.jsonl")
    temp_file = output_file + ".tmp"

    with jsonlines.open(temp_file, "w") as f:
        for item in data_for_saving:
            f.write(item)

    # Atomic rename
    os.rename(temp_file, output_file)
    print(f"Saved data to {output_file}")


if __name__ == "__main__":
    main()
