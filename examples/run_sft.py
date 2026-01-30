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
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
OmegaConf.register_new_resolver("max", lambda a, b: max(a, b))


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
def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig):
    assert "train" in data_config, (
        "The dataset config structure is updated. Please refer to https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/sft.md#datasets "
        "and the Migrate Guide in https://github.com/NVIDIA-NeMo/RL/pull/1649 to update the dataset config."
    )

    print("\n▶ Setting up data...")
    # setup train dataset
    task_data_processors = {}
    data_list = []

    if isinstance(data_config["train"], dict):
        data_config["train"] = [data_config["train"]]

    for cfg in data_config["train"]:
        # load dataset
        if "default" in data_config and data_config["default"] is not None:
            update_single_dataset_config(cfg, data_config["default"])
        data = load_response_dataset(cfg)
        data_list.append(data)
        # bind task_name to task_data_processors
        data_processor = partial(
            data.processor,
            add_bos=data_config["add_bos"],
            add_eos=data_config["add_eos"],
            add_generation_prompt=data_config["add_generation_prompt"],
        )
        task_data_processors[data.task_name] = (data.task_spec, data_processor)

    merged_data = concatenate_datasets([data.dataset for data in data_list])
    dataset = AllTaskProcessedDataset(
        merged_data,
        tokenizer,
        None,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    print(f"  ✓ Training dataset loaded with {len(dataset)} samples.")

    # setup validation dataset
    val_task_data_processors = {}
    val_data_list = []

    # validation dataset from train dataset (when train dataset's split_validation_size > 0)
    for data in data_list:
        if hasattr(data, "val_dataset") and data.val_dataset is not None:
            val_data_list.append(data.val_dataset)
            # bind task_name to task_data_processors
            task_name = data.task_name
            val_task_data_processors[task_name] = task_data_processors[task_name]

    # validation dataset from config
    if "validation" in data_config and data_config["validation"] is not None:
        if isinstance(data_config["validation"], dict):
            data_config["validation"] = [data_config["validation"]]

        for cfg in data_config["validation"]:
            # load dataset
            if "default" in data_config and data_config["default"] is not None:
                update_single_dataset_config(cfg, data_config["default"])
            val_data = load_response_dataset(cfg)
            val_data_list.append(val_data.dataset)
            # bind task_name to task_data_processors
            val_data_processor = partial(
                val_data.processor,
                add_bos=data_config["add_bos"],
                add_eos=data_config["add_eos"],
                add_generation_prompt=data_config["add_generation_prompt"],
            )
            val_task_data_processors[val_data.task_name] = (
                val_data.task_spec,
                val_data_processor,
            )

    val_dataset = None
    if len(val_data_list) > 0:
        merged_val_data = concatenate_datasets(val_data_list)
        val_dataset = AllTaskProcessedDataset(
            merged_val_data,
            tokenizer,
            None,
            val_task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
        print(f"  ✓ Validation dataset loaded with {len(val_dataset)} samples.")

    return dataset, val_dataset


def main(is_vlm: bool = False):
    """Main entry point."""
    # Parse arguments
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
