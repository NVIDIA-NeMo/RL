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
from nemo_rl.data import DataConfig, hf_datasets
from nemo_rl.data.datasets import AllTaskProcessedDataset
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
    assert len(datum_dict["completions"]) == 2  # Currently only supporting 2 completions
    # Lower rank is preferred
    if datum_dict["completions"][0]["rank"] < datum_dict["completions"][1]["rank"]:
        chosen_completion = datum_dict["completions"][0]
        rejected_completion = datum_dict["completions"][1]
    elif datum_dict["completions"][0]["rank"] > datum_dict["completions"][1]["rank"]:
        chosen_completion = datum_dict["completions"][1]
        rejected_completion = datum_dict["completions"][0]
    else:
        raise NotImplementedError("Ties are not supported yet.")
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
        "message_log_chosen": message_log_chosen,
        "length_chosen": length_chosen,
        "message_log_rejected": message_log_rejected,
        "length_rejected": length_rejected,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output


def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig):
    print("\n▶ Setting up data...")
    data_cls = data_config["dataset_name"]

    if data_cls == "PreferenceDataset":
        data_path = data_config["train_data_path"]
        data = hf_datasets.PreferenceDataset(data_path, split="train")
        train_dataset = data.formatted_ds["train"]
        val_dataset = None
        print(
            f"  ✓ Training dataset loaded with {len(data.formatted_ds['train'])} samples."
        )
    elif data_cls == "HelpSteer3":
        data = hf_datasets.HelpSteer3Dataset()
        train_dataset = data.formatted_ds["train"]
        val_dataset = data.formatted_ds["validation"]
        print(
            f"  ✓ Training and validation datasets loaded with {len(data.formatted_ds['train'])} and {len(data.formatted_ds['validation'])} samples, respectively."
        )
    else:
        raise ValueError(f"Unknown dataset class: {data_cls}")

    rm_task_spec = data.task_spec

    train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        rm_task_spec,
        rm_preprocessor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset = {
        "validation": AllTaskProcessedDataset(
            val_dataset,
            tokenizer,
            rm_task_spec,
            rm_preprocessor,
            max_seq_length=data_config["max_input_seq_length"],
        )
    } if val_dataset else {}

    if data_cls == "PreferenceDataset":
        if data_config.get("val_data_path"):
            assert data_config.get("val_data_paths") is None, "val_data_path and val_data_paths cannot be used together"
            val_data_paths = [{"validation": data_config.get("val_data_path")}]

        elif data_config.get("val_data_paths"):
            assert isinstance(data_config["val_data_paths"], list), f"Invalid type for val_data_paths: {type(data_config['val_data_paths'])}"
            val_data_paths = data_config.get("val_data_paths")

        else:
            raise ValueError("Either val_data_path or val_data_paths must be provided")

        for d in val_data_paths:
            assert len(d) == 1, "val_data_paths must be a list of <val_dataset_name: val_dataset_path> pairs."
            val_dataset_name = list(d.keys())[0]
            val_dataset_path = list(d.values())[0]
            assert val_dataset_name not in val_dataset or val_dataset_name == "validation" # Users can override the default "validation" set
            if val_dataset_name == "validation" and "validation" in val_dataset:
                print(f"  ✓ Overriding the default validation dataset")
            val_data = hf_datasets.PreferenceDataset(val_dataset_path, split="validation")
            print(
                f"  ✓ Validation dataset '{val_dataset_name}' loaded with {len(val_data.formatted_ds["validation"])} samples."
            )
            val_dataset[val_dataset_name] = AllTaskProcessedDataset(
                val_data.formatted_ds["validation"],
                tokenizer,
                val_data.task_spec,
                rm_preprocessor,
                max_seq_length=data_config["max_input_seq_length"],
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
        rm_task_spec,
        checkpointer,
        rm_save_state,
    )


if __name__ == "__main__":
    main()
