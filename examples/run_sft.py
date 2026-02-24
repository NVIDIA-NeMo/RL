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
from functools import partial

from datasets import concatenate_datasets
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.sft import MasterConfig, setup, sft_train
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import (
    AllTaskProcessedDataset,
    load_response_dataset,
    update_single_dataset_config,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SFT training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# =======================================================
# Data Processing
# =======================================================


# TODO @yukih: move to nemo_rl/data/utils.py after data processor refactored
def setup_data(
    tokenizer: AutoTokenizer, data_config: DataConfig
) -> tuple[AllTaskProcessedDataset, dict[str, AllTaskProcessedDataset]]:
    """Setup data for SFT training.

    Args:
        tokenizer: Tokenizer or processor.
        data_config: Data config.

    Returns:
        A tuple of (train dataset, validation dataset).
    """
    assert "train" in data_config, (
        "The dataset config structure is updated. Please refer to https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/sft.md#datasets "
        "and the Migrate Guide in https://github.com/NVIDIA-NeMo/RL/pull/1649 to update the dataset config."
    )

    # ==========================
    # Setup Train Dataset
    # ==========================
    print("\n▶ Setting up data...")
    task_data_processors = {}
    task_data_preprocessors = {}
    data_list = []

    # wrap dataset config in a list if it is a dictionary
    if isinstance(data_config["train"], dict):
        data_config["train"] = [data_config["train"]]

    for cfg in data_config["train"]:
        # update dataset config with default config
        if "default" in data_config and data_config["default"] is not None:
            update_single_dataset_config(cfg, data_config["default"])

        # load dataset
        data = load_response_dataset(cfg)
        task_name = data.task_name
        data_list.append(data)
        print(
            f"  - Loaded training dataset {task_name} with {len(data.dataset)} samples."
        )

        # bind task specific stuffs
        data_processor = partial(
            data.processor,
            add_bos=data_config["add_bos"],
            add_eos=data_config["add_eos"],
            add_generation_prompt=data_config["add_generation_prompt"],
        )
        task_data_processors[task_name] = (data.task_spec, data_processor)
        if hasattr(data, "preprocessor") and data.preprocessor is not None:
            task_data_preprocessors[task_name] = data.preprocessor

    merged_data = concatenate_datasets([data.dataset for data in data_list])
    dataset = AllTaskProcessedDataset(
        merged_data,
        tokenizer,
        None,
        task_data_processors,
        task_data_preprocessors=task_data_preprocessors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    print(f"  ✓ Training dataset loaded with {len(dataset)} samples.")

    # ==========================
    # Setup Validation Dataset
    # ==========================
    val_task_data_processors = {}
    val_task_data_preprocessors = {}
    val_data_dict = {}

    # validation dataset from train dataset (when train dataset's split_validation_size > 0)
    for data in data_list:
        if hasattr(data, "val_dataset") and data.val_dataset is not None:
            # extract val dataset from train dataset
            task_name = data.task_name
            val_data_dict[task_name] = data.val_dataset
            print(
                f"  - Loaded validation dataset {task_name} with {len(data.val_dataset)} samples."
            )

            # bind task specific stuffs
            val_task_data_processors[task_name] = task_data_processors[task_name]
            if task_name in task_data_preprocessors:
                val_task_data_preprocessors[task_name] = task_data_preprocessors[
                    task_name
                ]

    # validation dataset from config
    if "validation" in data_config and data_config["validation"] is not None:
        # wrap dataset config in a list if it is a dictionary
        if isinstance(data_config["validation"], dict):
            data_config["validation"] = [data_config["validation"]]

        for cfg in data_config["validation"]:
            # update dataset config with default config
            if "default" in data_config and data_config["default"] is not None:
                update_single_dataset_config(cfg, data_config["default"])

            # load dataset
            val_data = load_response_dataset(cfg)
            task_name = val_data.task_name
            val_data_dict[task_name] = val_data.dataset
            print(
                f"  - Loaded validation dataset {task_name} with {len(val_data.dataset)} samples."
            )

            # bind task specific stuffs
            val_data_processor = partial(
                val_data.processor,
                add_bos=data_config["add_bos"],
                add_eos=data_config["add_eos"],
                add_generation_prompt=data_config["add_generation_prompt"],
            )
            val_task_data_processors[task_name] = (
                val_data.task_spec,
                val_data_processor,
            )
            if hasattr(val_data, "preprocessor") and val_data.preprocessor is not None:
                val_task_data_preprocessors[task_name] = val_data.preprocessor

    val_dataset = {}
    if len(val_data_dict) > 0:
        val_dataset = {
            task_name: AllTaskProcessedDataset(
                val_data,
                tokenizer,
                None,
                val_task_data_processors,
                task_data_preprocessors=val_task_data_preprocessors,
                max_seq_length=data_config["max_input_seq_length"],
            )
            for task_name, val_data in val_data_dict.items()
        }
        val_sample_count = sum(len(val_data) for val_data in val_data_dict.values())
        print(f"  ✓ Validation dataset loaded with {val_sample_count} samples.")

    return dataset, val_dataset


def main(is_vlm: bool = False):
    """Main entry point."""
    # Parse arguments
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "sft.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer (or processor)
    tokenizer = get_tokenizer(config["policy"]["tokenizer"], get_processor=is_vlm)

    # setup data
    dataset, val_dataset = setup_data(tokenizer, config["data"])

    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        sft_save_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    sft_train(
        policy,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        checkpointer,
        sft_save_state,
    )


if __name__ == "__main__":
    main()
