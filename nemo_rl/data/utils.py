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
    load_preference_dataset,
    load_response_dataset,
    update_single_dataset_config,
)
from nemo_rl.data.datasets.preference_datasets import PreferenceDataset
from nemo_rl.data.processors import preference_preprocessor
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.utils import create_env


# TODO: @yukih: unify to setup_data after dataset refactored
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
    if "default" in data_config:
        update_single_dataset_config(data_config["train"], data_config["default"])
    data = load_response_dataset(data_config["train"])
    task_data_processors = {data.task_name: (data.task_spec, data.processor)}
    task_to_env = {data.task_name: envs[data_config["train"]["env_name"]]}

    dataset = AllTaskProcessedDataset(
        data.dataset,
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
    if hasattr(data, "val_dataset") and data.val_dataset is not None:
        val_data_list.append(data.val_dataset)
        val_task_data_processors = task_data_processors.copy()
        val_task_to_env = task_to_env.copy()

    # validation dataset from config
    if "validation" in data_config and data_config["validation"] is not None:
        if "default" in data_config:
            update_single_dataset_config(
                data_config["validation"], data_config["default"]
            )
        val_data = load_response_dataset(data_config["validation"])
        val_data_list.append(val_data.dataset)
        val_task_data_processors[val_data.task_name] = (
            val_data.task_spec,
            val_data.processor,
        )
        val_task_to_env[val_data.task_name] = envs[
            data_config["validation"]["env_name"]
        ]

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


# TODO: @yukih: unify to setup_data after dataset refactored
def setup_preference_data(tokenizer: AutoTokenizer, data_config: DataConfig):
    """Setup preference data.

    This function is used to setup the preference data for the training and validation datasets.

    Args:
        tokenizer: Tokenizer.
        data_config: Data config for preference dataset.

    Returns:
        A tuple of (train dataset, validation dataset).
    """
    assert "train" in data_config, (
        "The dataset config structure is updated. Please refer to https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/dpo.md#datasets "
        "and the Migrate Guide in https://github.com/NVIDIA-NeMo/RL/pull/1763 to update the dataset config."
    )

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
    # TODO @yukih: unify the code when support multiple datasets for preference dataset
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
