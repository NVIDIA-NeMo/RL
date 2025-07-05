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
from functools import partial
from typing import Any

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.sft import MasterConfig, setup, sft_train
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
def sft_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
    add_bos: bool = True,
    add_eos: bool = True,
    add_generation_prompt: bool = False,
) -> DatumSpec:
    """Process a datum dictionary for SFT training."""
    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_bos_token=add_bos,
        add_eos_token=add_eos,
        add_generation_prompt=add_generation_prompt,
    )

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output


def rm_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary for RM training."""
    messages_chosen = datum_dict["prompt"] + [
        {"role": "assistant", "content": datum_dict["chosen_response"]}
    ]
    messages_rejected = datum_dict["prompt"] + [
        {"role": "assistant", "content": datum_dict["rejected_response"]}
    ]

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


def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig, model_type: str):
    print("\n▶ Setting up data...")
    data_cls = data_config["dataset_name"]

    if model_type == "lm":
        data_preprocessor = partial(
            sft_preprocessor,
            add_bos=data_config["add_bos"],
            add_eos=data_config["add_eos"],
            add_generation_prompt=data_config["add_generation_prompt"],
        )

        if data_cls == "open_assistant":
            data = hf_datasets.OasstDataset(output_dir="/tmp/open_assistant")
        elif data_cls == "squad":
            data = hf_datasets.SquadDataset()
        elif data_cls == "prompt_response_dataset":
            data = hf_datasets.PromptResponseDataset(
                data_config["train_data_path"],
                data_config["val_data_path"],
                data_config["input_key"],
                data_config["output_key"],
            )
        elif data_cls == "openmathinstruct2":
            data = hf_datasets.OpenMathInstruct2Dataset(
                split=data_config["split"],
                output_key=data_config["output_key"],
                prompt_file=data_config["prompt_file"],
            )
        elif data_cls == "openai_format":
            data = hf_datasets.OpenAIFormatDataset(
                data_config["train_data_path"],
                data_config["val_data_path"],
                data_config["chat_key"],
                data_config["system_key"],
                data_config["system_prompt"],
            )
        else:
            raise ValueError(
                f"Unknown dataset class: {data_cls} for model_type: {model_type}"
            )
    elif model_type == "reward":
        data_preprocessor = rm_preprocessor

        if data_cls == "HelpSteer3":
            data = hf_datasets.HelpSteer3Dataset()
        else:
            raise ValueError(
                f"Unknown dataset class: {data_cls} for model_type: {model_type}"
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(
        f"  ✓ Training and validation datasets loaded with {len(data.formatted_ds['train'])} and {len(data.formatted_ds['validation'])} samples, respectively."
    )

    train_dataset = data.formatted_ds["train"]
    val_dataset = data.formatted_ds["validation"]
    sft_task_spec = data.task_spec

    train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        sft_task_spec,
        data_preprocessor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset = AllTaskProcessedDataset(
        val_dataset,
        tokenizer,
        sft_task_spec,
        data_preprocessor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    return train_dataset, val_dataset, sft_task_spec


def main():
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

    model_type = "reward" if "reward_model_type" in config["policy"] else "lm"

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
        sft_task_spec,
    ) = setup_data(tokenizer, config["data"], model_type)

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
        sft_task_spec,
        checkpointer,
        sft_save_state,
    )


if __name__ == "__main__":
    main()
