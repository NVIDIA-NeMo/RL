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
import warnings
from typing import Any

from omegaconf import OmegaConf

from nemo_rl.algorithms.dpo import MasterConfig, dpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig, hf_datasets
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run DPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# =======================================================
# Data Processing
# =======================================================
def dpo_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary for DPO training.

    Examples:
        ```{doctest}
        >>> from transformers import AutoTokenizer
        >>> from nemo_rl.data.interfaces import TaskDataSpec
        >>>
        >>> # Initialize tokenizer and task spec
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        >>> ## set a passthrough chat template for simplicity
        >>> tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        >>> task_spec = TaskDataSpec(task_name="test_dpo")
        >>>
        >>> datum = {
        ...     "prompt": "What is 2+2?",
        ...     "chosen_response": "4",
        ...     "rejected_response": "5"
        ... }
        >>>
        >>> processed = dpo_preprocessor(datum, task_spec, tokenizer, max_seq_length=128, idx=0)
        >>> len(processed["message_log_chosen"])
        2
        >>> processed["message_log_chosen"][0]["content"]
        '<|begin_of_text|>What is 2+2?'
        >>> processed["message_log_chosen"][-1]["content"]
        '4<|eot_id|>'
        >>> processed["message_log_rejected"][-1]["content"]
        '5<|eot_id|>'
        >>>
        >>> # prompt can also be a list with multiple messages
        >>> datum = {
        ...     "prompt": [{"role": "user", "content": "I have a question."}, {"role": "assistant", "content": "Sure!"}, {"role": "user", "content": "What is 2+2?"}],
        ...     "chosen_response": "4",
        ...     "rejected_response": "5"
        ... }
        >>> processed = dpo_preprocessor(datum, task_spec, tokenizer, max_seq_length=128, idx=0)
        >>> len(processed["message_log_chosen"])
        4
        >>> processed["message_log_chosen"][1]["content"]
        'Sure!'
        >>> processed["message_log_chosen"][-1]["content"]
        '4<|eot_id|>'
        >>> processed["message_log_rejected"][-1]["content"]
        '5<|eot_id|>'

        ```
    """
    if isinstance(datum_dict["prompt"], list):
        messages_chosen = datum_dict["prompt"].copy()
        messages_rejected = datum_dict["prompt"].copy()
    else:
        messages_chosen = [
            {
                "role": "user",
                "content": datum_dict["prompt"],
            },
        ]
        messages_rejected = [
            {
                "role": "user",
                "content": datum_dict["prompt"],
            },
        ]

    messages_chosen.append(
        {
            "role": "assistant",
            "content": datum_dict["chosen_response"],
        },
    )

    messages_rejected.append(
        {
            "role": "assistant",
            "content": datum_dict["rejected_response"],
        },
    )

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
        warnings.warn(
            f"Sequence length {max(length_chosen, length_rejected)} exceeds max_seq_length {max_seq_length}. Ignoring example."
        )
        # make smaller and mask out
        for message in message_log_chosen:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log_chosen))
            ]
        for message in message_log_rejected:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log_rejected))
            ]
        loss_multiplier = 0.0

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


def setup_data(data_config: DataConfig, policy_config: PolicyConfig):
    print("\n▶ Setting up data...")

    if data_config["dataset_name"] == "HelpSteer3":
        data = hf_datasets.HelpSteer3Dataset()
    else:
        data = hf_datasets.DPODataset(
            train_data_path=data_config["train_data_path"],
            val_data_path=data_config["val_data_path"],
        )
    train_dataset = data.formatted_ds["train"]
    val_dataset = data.formatted_ds["validation"]

    dpo_task_spec = data.task_spec

    tokenizer = get_tokenizer(policy_config["tokenizer"])
    train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        dpo_task_spec,
        dpo_preprocessor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset = AllTaskProcessedDataset(
        val_dataset,
        tokenizer,
        dpo_task_spec,
        dpo_preprocessor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    return train_dataset, val_dataset, tokenizer, dpo_task_spec


def main():
    """Main entry point."""
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "dpo.yaml")

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

    # setup data
    train_dataset, val_dataset, tokenizer, dpo_task_spec = setup_data(
        config["data"], config["policy"]
    )
    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        dpo_save_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)
    dpo_train(
        policy,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        checkpointer,
        dpo_save_state,
    )


if __name__ == "__main__":
    main()
