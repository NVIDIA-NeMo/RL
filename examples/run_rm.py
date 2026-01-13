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

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.rm import MasterConfig, rm_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import (
    AllTaskProcessedDataset,
    load_preference_dataset,
    update_single_dataset_config,
)
from nemo_rl.data.datasets.preference_datasets import PreferenceDataset
from nemo_rl.data.processors import preference_preprocessor
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run RM training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# =======================================================
# Data Processing
# =======================================================
def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig):
    print("\n▶ Setting up data...")
    # setup train dataset
    if "default" in data_config:
        update_single_dataset_config(data_config["train"], data_config["default"])
    data = load_preference_dataset(data_config["train"])
    task_data_processors = {data.task_name: (data.task_spec, preference_preprocessor)}

    dataset = AllTaskProcessedDataset(
        data.dataset,
        tokenizer,
        None,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    print(f"  ✓ Training dataset loaded with {len(dataset)} samples.")

    # setup validation dataset
    # TODO @yukih: unify the code when support multiple datasets for other algorithms
    val_dataset = {}
    if "val_data_paths" in data_config and data_config["val_data_paths"]:
        assert isinstance(data_config["val_data_paths"], dict), (
            f"Invalid type for val_data_paths: {type(data_config['val_data_paths'])}. val_data_paths must be a dictionary."
        )
        val_data_paths = data_config["val_data_paths"]

        for val_dataset_name, val_dataset_path in val_data_paths.items():
            assert val_dataset_name not in val_dataset
            val_data = PreferenceDataset(val_dataset_path)
            print(
                f"  ✓ Validation dataset '{val_dataset_name}' loaded with {len(val_data.dataset)} samples."
            )
            val_dataset[val_dataset_name] = AllTaskProcessedDataset(
                val_data.dataset,
                tokenizer,
                val_data.task_spec,
                preference_preprocessor,
                max_seq_length=data_config["max_input_seq_length"],
            )
    elif "validation" in data_config and data_config["validation"] is not None:
        if "default" in data_config:
            update_single_dataset_config(
                data_config["validation"], data_config["default"]
            )
        val_data = load_preference_dataset(data_config["validation"])
        val_task_data_processors = {
            val_data.task_name: (val_data.task_spec, preference_preprocessor)
        }

        val_dataset["default"] = AllTaskProcessedDataset(
            val_data.dataset,
            tokenizer,
            None,
            val_task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
        print(
            f"  ✓ Validation dataset loaded with {len(val_dataset['default'])} samples."
        )

    return dataset, val_dataset


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "rm.yaml")

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

    assert config["policy"]["reward_model_cfg"]["enabled"]

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

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
        rm_save_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    rm_train(
        policy,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        checkpointer,
        rm_save_state,
    )


if __name__ == "__main__":
    main()
