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

from typing import Any, Optional

from datasets import concatenate_datasets
from transformers import AutoProcessor, AutoTokenizer

from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import (
    AllTaskProcessedDataset,
    extract_necessary_env_names,
    load_response_dataset,
    update_single_dataset_config,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.utils import create_env


def setup_data_with_envs(
    tokenizer: AutoProcessor | AutoTokenizer,
    data_config: DataConfig,
    env_configs: dict[str, Any],
    is_vlm: bool = False,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    """Setup data with environments.

    This function is used to setup the data and environments for the training and validation datasets.

    Args:
        tokenizer: Tokenizer or processor.
        data_config: Data config.
        env_configs: Environment configs.
        is_vlm: Whether to use VLM training or not.

    Returns:
        A tuple of (train dataset, validation dataset, task to environment, task to validation environment).
    """
    assert "train" in data_config, (
        "The dataset config structure is updated. Please refer to https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/grpo.md#dataset "
        "and the Migrate Guide in https://github.com/NVIDIA-NeMo/RL/pull/1649 to update the dataset config."
    )

    print("\n▶ Setting up envs...")
    env_name_list = extract_necessary_env_names(data_config)
    envs = {}
    for env_name in env_name_list:
        registered_env_name = "vlm" if is_vlm else env_name
        envs[env_name] = create_env(
            env_name=registered_env_name, env_config=env_configs[env_name]
        )

    print("\n▶ Setting up data...")
    # setup train dataset
    task_data_processors = {}
    task_to_env = {}
    data_list = []

    if isinstance(data_config["train"], dict):
        data_config["train"] = [data_config["train"]]

    for cfg in data_config["train"]:
        # load dataset
        if "default" in data_config and data_config["default"] is not None:
            update_single_dataset_config(cfg, data_config["default"])
        data = load_response_dataset(cfg)
        data_list.append(data)
        # bind task_name to task_data_processors and task_to_env
        task_name = data.task_name
        task_data_processors[task_name] = (data.task_spec, data.processor)
        task_to_env[task_name] = envs[cfg["env_name"]]

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
    val_task_to_env = {}
    val_data_list = []

    # validation dataset from train dataset (when train dataset's split_validation_size > 0)
    for data in data_list:
        if hasattr(data, "val_dataset") and data.val_dataset is not None:
            val_data_list.append(data.val_dataset)
            # bind task_name to task_data_processors and task_to_env
            task_name = data.task_name
            val_task_data_processors[task_name] = task_data_processors[task_name]
            val_task_to_env[task_name] = task_to_env[task_name]

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
            # bind task_name to task_data_processors and task_to_env
            task_name = val_data.task_name
            val_task_data_processors[task_name] = (
                val_data.task_spec,
                val_data.processor,
            )
            val_task_to_env[task_name] = envs[cfg["env_name"]]

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

    return dataset, val_dataset, task_to_env, val_task_to_env
