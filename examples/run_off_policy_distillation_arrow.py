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

"""
Off-Policy Distillation with Arrow Dataset Support

This script runs off-policy distillation using the same data loading
pattern as run_sft.py (load_response_dataset + AllTaskProcessedDataset).

Arrow files are loaded via the existing ArrowTextDataset class by setting
dataset_name: "arrow_text" in the config. ArrowTextDataset handles:
  - Loading .arrow files via glob patterns
  - Wrapping text as {"messages": [{"role": "assistant", "content": <text>}]}
  - Splitting into train/validation

Off-policy: no student generation, teacher provides logits for fixed responses.
"""

import argparse
import os
import pprint
from functools import partial
from typing import Any, Optional, Callable

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.off_policy_distillation import (
    OffPolicyMasterConfig,
    off_policy_distillation_train,
    setup,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
from nemo_rl.data.datasets.utils import load_dataset_from_path
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
OmegaConf.register_new_resolver("max", lambda a, b: max(a, b))


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run off-policy distillation with Arrow dataset support"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    args, overrides = parser.parse_known_args()
    return args, overrides


# =======================================================
# Data Processing (following run_sft.py pattern)
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
    """Process a datum dictionary for off-policy distillation.

    Same as run_sft.py's sft_preprocessor. ArrowTextDataset already wraps
    plain text into messages format, so we only need to handle messages here.
    """
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
    """Setup data for off-policy distillation.

    Uses load_response_dataset exactly like run_sft.py. For arrow files,
    set dataset_name: "arrow_text" in the config and ArrowTextDataset
    handles the rest.
    """
    print("\n▶ Setting up data...")

    if "train" not in data_config:
        raise ValueError(
            "The dataset config structure is updated. Please use data.train/default style."
        )

    train_cfg = data_config["train"]
    if isinstance(train_cfg, list):
        if len(train_cfg) != 1:
            raise ValueError(
                "Off-policy distillation currently supports exactly one train dataset config."
            )
        train_cfg = train_cfg[0]
    train_data_config: dict[str, Any] = dict(train_cfg)
    default_cfg = data_config.get("default")
    if isinstance(default_cfg, dict):
        merged_cfg = dict(default_cfg)
        merged_cfg.update(train_data_config)
        train_data_config = merged_cfg
    train_data_config["max_input_seq_length"] = data_config["max_input_seq_length"]

    arrow_files = train_data_config.get("arrow_files")
    if arrow_files:
        data = load_response_dataset(train_data_config, seed)
        train_dataset = data.formatted_ds["train"]
        val_dataset = data.formatted_ds["validation"]
        task_spec = data.task_spec
        print(f"  ✓ Using Arrow dataset input: {arrow_files}")
    else:
        # Fallback for user-provided non-arrow path or HF dataset id.
        dataset_path = train_data_config.get(
            "dataset_path", train_data_config.get("hf_dataset_name", "allenai/c4")
        )
        hf_dataset_subset_raw = train_data_config.get("hf_dataset_subset")
        hf_dataset_subset = (
            None
            if hf_dataset_subset_raw in (None, "", "null")
            else hf_dataset_subset_raw
        )
        hf_split = train_data_config.get("hf_split", "train")
        text_key = train_data_config.get("text_key", "text")
        val_split = float(train_data_config.get("val_split", 0.05))

        print(
            "  ↪ No data.arrow_files provided. "
            f"Loading dataset_path='{dataset_path}', subset='{hf_dataset_subset}', split='{hf_split}'"
        )
        hf_dataset = load_dataset_from_path(
            dataset_path, data_subset=hf_dataset_subset, data_split=hf_split
        )
        if text_key not in hf_dataset.column_names:
            raise ValueError(
                f"text_key='{text_key}' not found in HF dataset columns: {hf_dataset.column_names}"
            )

        def _to_messages(entry: dict[str, Any]) -> dict[str, Any]:
            text = entry[text_key]
            if not isinstance(text, str):
                text = str(text)
            return {
                "messages": [{"role": "assistant", "content": text}],
                "task_name": "off_policy_distillation",
            }

        formatted_ds = hf_dataset.map(
            _to_messages, remove_columns=hf_dataset.column_names
        )
        split_ds = formatted_ds.train_test_split(test_size=val_split, seed=seed)
        train_dataset = split_ds["train"]
        val_dataset = split_ds["test"]
        task_spec = TaskDataSpec(
            task_name="off_policy_distillation",
            prompt_file=train_data_config.get("prompt_file"),
            system_prompt_file=train_data_config.get("system_prompt_file"),
        )

    print(
        f"  ✓ Training and validation datasets loaded with {len(train_dataset)} and {len(val_dataset)} samples, respectively."
    )

    train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        task_spec,
        partial(
            sft_preprocessor,
            add_bos=train_data_config.get("add_bos", True),
            add_eos=train_data_config.get("add_eos", True),
            add_generation_prompt=train_data_config.get(
                "add_generation_prompt", False
            ),
        ),
        max_seq_length=train_data_config["max_input_seq_length"],
    )

    val_dataset = AllTaskProcessedDataset(
        val_dataset,
        tokenizer,
        task_spec,
        partial(
            sft_preprocessor,
            add_bos=train_data_config.get("add_bos", True),
            add_eos=train_data_config.get("add_eos", True),
            add_generation_prompt=train_data_config.get(
                "add_generation_prompt", False
            ),
        ),
        max_seq_length=train_data_config["max_input_seq_length"],
    )

    return train_dataset, val_dataset, task_spec


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "off_policy_distillation_math.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: OffPolicyMasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    # Setup data
    dataset, val_dataset, task_spec = setup_data(
        tokenizer, config["data"], config["distillation"]["seed"]
    )

    # ---------- quick dataloader sanity check ----------
    # from torchdata.stateful_dataloader import StatefulDataLoader
    # from nemo_rl.data.collate_fn import rl_collate_fn
    # batch_size = config["distillation"]["num_prompts_per_step"]
    # _dl = StatefulDataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=rl_collate_fn, drop_last=True)
    # print(f"\nDataloader: {len(dataset)} samples, {len(_dl)} batches (bs={batch_size})")
    # for _i, _b in enumerate(_dl):
    #     if _i >= 3:
    #         break
    #     print(f"  batch {_i}: lengths={_b['length'].tolist()}, loss_mult={_b['loss_multiplier'].tolist()}")
    #     _m = _b['message_log'][0][0]
    #     print(f"    sample[0]: role={_m['role']}, tok[:10]={_m['token_ids'][:10].tolist()}")
    #     print(f"    text[:100]: {tokenizer.decode(_m['token_ids'][:30], skip_special_tokens=False)[:100]}")
    # print("\nDataloader OK")
    # import sys; sys.exit(0)
    # ---------------------------------------------------

    # Setup off-policy distillation (no student generation needed)
    (
        student_policy,
        teacher_policy,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        distillation_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    # Run off-policy distillation training
    off_policy_distillation_train(
        student_policy,
        teacher_policy,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        logger,
        checkpointer,
        distillation_state,
        master_config,
    )


if __name__ == "__main__":
    main()
