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
from typing import Any, Callable, Optional

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.sft import MasterConfig, setup, sft_train
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log
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
def sft_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
    add_bos: bool = True,
    add_eos: bool = True,
    add_generation_prompt: bool = False,
    datum_preprocessor: Optional[Callable] = None,
) -> DatumSpec:
    """Process a datum dictionary for SFT training."""
    # optional preprocessor
    if datum_preprocessor is not None:
        datum_dict = datum_preprocessor(datum_dict)

    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_bos_token=add_bos,
        add_eos_token=add_eos,
        add_generation_prompt=add_generation_prompt,
        tools=datum_dict.get("tools", None),  # Pass tools from data if present
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


def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig, seed: int):
    print("\n▶ Setting up data...")

    # load dataset
    data = load_response_dataset(data_config, seed)
    train_dataset = data.formatted_ds["train"]
    val_dataset = data.formatted_ds["validation"]
    sft_task_spec = data.task_spec
    print(
        f"  ✓ Training and validation datasets loaded with {len(train_dataset)} and {len(val_dataset) if val_dataset else 0} samples, respectively."
    )

    # add preprocessor if needed
    datum_preprocessor = None
    if "dataset_name" in data_config and data_config["dataset_name"] == "clevr-cogent":
        from nemo_rl.data.datasets.response_datasets.clevr import (
            format_clevr_cogent_dataset,
        )

        datum_preprocessor = partial(format_clevr_cogent_dataset, return_pil=True)

    train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        sft_task_spec,
        partial(
            sft_preprocessor,
            add_bos=data_config["add_bos"],
            add_eos=data_config["add_eos"],
            add_generation_prompt=data_config["add_generation_prompt"],
            datum_preprocessor=datum_preprocessor,
        ),
        max_seq_length=data_config["max_input_seq_length"],
    )

    if val_dataset is not None:
        val_dataset = AllTaskProcessedDataset(
            val_dataset,
            tokenizer,
            sft_task_spec,
            partial(
                sft_preprocessor,
                add_bos=data_config.get("add_bos", True),
                add_eos=data_config.get("add_eos", True),
                add_generation_prompt=data_config["add_generation_prompt"],
                datum_preprocessor=datum_preprocessor,
            ),
            max_seq_length=data_config["max_input_seq_length"],
        )

    return train_dataset, val_dataset, sft_task_spec


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
    (
        dataset,
        val_dataset,
        sft_task_spec,
    ) = setup_data(tokenizer, config["data"], config["sft"]["seed"])

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
