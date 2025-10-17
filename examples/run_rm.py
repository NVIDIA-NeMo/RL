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
import logging
import os
import pprint
from typing import Any

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.rm import MasterConfig, rm_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_preference_dataset
from nemo_rl.data.datasets.preference_datasets import PreferenceDataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log
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
def rm_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary for RM training."""
    assert len(datum_dict["completions"]) == 2, (
        "RM training supports only two completions"
    )
    # Lower rank is preferred
    if datum_dict["completions"][0]["rank"] < datum_dict["completions"][1]["rank"]:
        chosen_completion = datum_dict["completions"][0]
        rejected_completion = datum_dict["completions"][1]
    elif datum_dict["completions"][0]["rank"] > datum_dict["completions"][1]["rank"]:
        chosen_completion = datum_dict["completions"][1]
        rejected_completion = datum_dict["completions"][0]
    else:
        raise NotImplementedError(
            "Ties are not supported yet. You can use the following command to filter out ties: `cat <PathToPreferenceDataset> | jq 'select(.completions[0].rank != .completions[1].rank)'`."
        )

    messages_chosen = datum_dict["context"] + chosen_completion["completion"]
    messages_rejected = datum_dict["context"] + rejected_completion["completion"]

    message_log_chosen = get_formatted_message_log(
        messages_chosen, tokenizer, task_data_spec
    )
    message_log_rejected = get_formatted_message_log(
        messages_rejected, tokenizer, task_data_spec
    )

    length_chosen = sum(len(m["token_ids"]) for m in message_log_chosen)
    length_rejected = sum(len(m["token_ids"]) for m in message_log_rejected)

    loss_multiplier = 1.0
    if max(length_chosen, length_rejected) > max_seq_length:
        # make smaller and mask out
        logging.warning(
            f"Truncating chosen and rejected messages to {max_seq_length} tokens"
        )
        for message in message_log_chosen:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log_chosen))
            ]
        for message in message_log_rejected:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log_rejected))
            ]
        loss_multiplier = 0.0

        length_chosen = sum(len(m["token_ids"]) for m in message_log_chosen)
        length_rejected = sum(len(m["token_ids"]) for m in message_log_rejected)

        # safeguard against edge case where there are too many turns to fit within the max length
        assert max(length_chosen, length_rejected) <= max_seq_length

    output = {
        "list_message_log": [message_log_chosen, message_log_rejected],
        "list_length": [length_chosen, length_rejected],
        "list_values": [],
        "list_weights": [],
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output


def rm_preprocessor_preferred_among_n(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary for PreferredAmongNAccuracy."""
    completions = datum_dict["completions"]
    assert len(completions) >= 2, "Needs at least two completions"
    # Lowest rank is preferred to all others
    completions = sorted(completions, key=lambda x: x["rank"])
    for i in range(len(completions) - 1):
        assert completions[0]["rank"] < completions[i + 1]["rank"], (
            "One completion must be preferred to all others"
        )

    list_messages = [
        datum_dict["context"] + completion["completion"] for completion in completions
    ]
    list_message_log = [
        get_formatted_message_log(messages, tokenizer, task_data_spec)
        for messages in list_messages
    ]
    list_length = [
        sum(len(m["token_ids"]) for m in message_log)
        for message_log in list_message_log
    ]

    loss_multiplier = 1.0
    if max(list_length) > max_seq_length:
        # make smaller and mask out

        logging.warning(
            f"Truncating chosen and rejected messages to {max_seq_length} tokens"
        )
        for message_log in list_message_log:
            for message in message_log:
                message["token_ids"] = message["token_ids"][
                    : min(4, max_seq_length // len(message_log))
                ]

        loss_multiplier = 0.0

        list_length = [
            sum(len(m["token_ids"]) for m in message_log)
            for message_log in list_message_log
        ]
        # safeguard against edge case where there are too many turns to fit within the max length
        assert max(list_length) <= max_seq_length

    output = {
        "list_message_log": list_message_log,
        "list_length": list_length,
        "list_values": [],
        "list_weights": [],
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output


def rm_preprocessor_best_of_n(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary for RewardBestofNValue."""
    completions = datum_dict["completions"]
    assert len(completions) >= 2, "Needs at least two completions"
    list_values = [completion["value"] for completion in completions]
    list_weights = [completion.get("weight", 1.0) for completion in completions]

    list_messages = [
        datum_dict["context"] + completion["completion"] for completion in completions
    ]
    list_message_log = [
        get_formatted_message_log(messages, tokenizer, task_data_spec)
        for messages in list_messages
    ]
    list_length = [
        sum(len(m["token_ids"]) for m in message_log)
        for message_log in list_message_log
    ]

    loss_multiplier = 1.0
    if max(list_length) > max_seq_length:
        # make smaller and mask out

        logging.warning(
            f"Truncating chosen and rejected messages to {max_seq_length} tokens"
        )
        for message_log in list_message_log:
            for message in message_log:
                message["token_ids"] = message["token_ids"][
                    : min(4, max_seq_length // len(message_log))
                ]

        loss_multiplier = 0.0

        list_length = [
            sum(len(m["token_ids"]) for m in message_log)
            for message_log in list_message_log
        ]
        # safeguard against edge case where there are too many turns to fit within the max length
        assert max(list_length) <= max_seq_length

    output = {
        "list_message_log": list_message_log,
        "list_length": list_length,
        "list_values": list_values,
        "list_weights": list_weights,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output


def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig):
    print("\n▶ Setting up data...")

    # load dataset
    data = load_preference_dataset(data_config)
    train_dataset = data.formatted_ds["train"]
    val_dataset = data.formatted_ds["validation"]

    print(f"  ✓ Training dataset loaded with {len(train_dataset)} samples.")
    if val_dataset:
        print(f"  ✓ Validation dataset loaded with {len(val_dataset)} samples.")

    rm_task_spec = data.task_spec

    train_dataset = {
        "dataset": AllTaskProcessedDataset(
            train_dataset,
            tokenizer,
            rm_task_spec,
            rm_preprocessor,
            max_seq_length=data_config["max_input_seq_length"],
        ),
        "loss_fn": "PreferenceLoss",
    }

    # TODO @yukih: unify the code when support multiple datasets for other algorithms
    if "val_data_paths" in data_config and data_config["val_data_paths"]:
        val_loss_fns = {}

        if "val_loss_fns" in data_config and data_config["val_loss_fns"]:
            assert isinstance(data_config["val_loss_fns"], dict), (
                f"Invalid type for val_loss_fns: {type(data_config['val_loss_fns'])}. val_loss_fns must be a dictionary."
            )
            val_loss_fns = data_config["val_loss_fns"]

        val_dataset = {}

        assert isinstance(data_config["val_data_paths"], dict), (
            f"Invalid type for val_data_paths: {type(data_config['val_data_paths'])}. val_data_paths must be a dictionary."
        )
        val_data_paths = data_config["val_data_paths"]

        for val_dataset_name, val_dataset_path in val_data_paths.items():
            assert val_dataset_name not in val_dataset
            val_data = PreferenceDataset(val_dataset_path)
            print(
                f"  ✓ Validation dataset '{val_dataset_name}' loaded with {len(val_data.formatted_ds['train'])} samples."
            )
            loss_fn_name = val_loss_fns.get(val_dataset_name, "PreferenceLoss")
            if loss_fn_name == "PreferenceLoss":
                preprocessor = rm_preprocessor
            elif loss_fn_name == "RewardBestofNValue":
                preprocessor = rm_preprocessor_best_of_n
            elif loss_fn_name == "PreferredAmongNAccuracy":
                preprocessor = rm_preprocessor_preferred_among_n
            else:
                raise ValueError(f"Invalid loss function name: {loss_fn_name}")
            val_dataset[val_dataset_name] = {
                "dataset": AllTaskProcessedDataset(
                    val_data.formatted_ds["train"],
                    tokenizer,
                    val_data.task_spec,
                    preprocessor,
                    max_seq_length=data_config["max_input_seq_length"],
                ),
                "loss_fn": loss_fn_name,
            }
    else:
        val_dataset = (
            {
                "default": {
                    "dataset": AllTaskProcessedDataset(
                        val_dataset,
                        tokenizer,
                        rm_task_spec,
                        rm_preprocessor,
                        max_seq_length=data_config["max_input_seq_length"],
                    ),
                    "loss_fn": "PreferenceLoss",
                }
            }
            if val_dataset
            else {}
        )

    return train_dataset, val_dataset, rm_task_spec


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
    (
        dataset,
        val_dataset,
        rm_task_spec,
    ) = setup_data(tokenizer, config["data"])

    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        train_loss_fn,
        val_loss_fn,
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
        train_loss_fn,
        val_loss_fn,
        master_config,
        logger,
        rm_task_spec,
        checkpointer,
        rm_save_state,
    )


if __name__ == "__main__":
    main()
