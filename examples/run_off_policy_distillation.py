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
Off-Policy Distillation Training Script

This script runs off-policy distillation where:
- A fixed dataset of prompt-response pairs is used (no student generation)
- Teacher provides logits for the fixed responses
- Student aligns with teacher using KL divergence loss

Usage:
    python run_off_policy_distillation.py --config configs/off_policy_distillation.yaml

For your arrow dataset:
    python run_off_policy_distillation.py --config configs/off_policy_distillation.yaml \
        data.arrow_files="/path/to/your/*.arrow"

Reference: https://github.com/NVIDIA-NeMo/RL/discussions/1445
"""

import argparse
import glob
import os
from functools import partial
from typing import Any, Optional

import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.off_policy_distillation import (
    OffPolicyMasterConfig,
    off_policy_distillation_train,
    setup,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run off-policy distillation training"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


# ===============================================================================
# Data Processing for Off-Policy Distillation
# ===============================================================================


def off_policy_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """
    Process a datum dictionary for off-policy distillation.
    
    This processor handles datasets with prompt-response pairs where the response
    is already provided. It creates message_log with token_ids and loss masks.
    
    Supports multiple input formats:
    1. {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    2. {"conversations": [{"from": "human/gpt", "value": "..."}]}  # ShareGPT
    3. {"prompt": "...", "response": "..."}
    4. {"input": "...", "output": "..."}
    5. {"instruction": "...", "output": "..."}  # Alpaca
    6. {"text": "..."}  # Full text - train on all tokens (language modeling style)
    """
    
    # Special handling for raw text format (no chat structure)
    if "text" in datum_dict and len(datum_dict.keys()) == 1:
        # Raw text format - tokenize directly without chat template
        # Train on all tokens (language modeling / SFT style)
        text = datum_dict["text"]
        
        # Add BOS token if tokenizer has one
        if tokenizer.bos_token:
            text = tokenizer.bos_token + text
        
        token_ids = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )["input_ids"][0]
        
        # Train on all tokens
        token_loss_mask = torch.ones_like(token_ids)
        
        length = len(token_ids)
        loss_multiplier = 1.0 if length <= max_seq_length else 0.0
        
        message_log: LLMMessageLogType = [{
            "role": "assistant",  # Mark as assistant so loss is computed
            "content": text[:500] + "..." if len(text) > 500 else text,
            "token_ids": token_ids,
            "token_loss_mask": token_loss_mask,
        }]
        
        return {
            "message_log": message_log,
            "length": length,
            "extra_env_info": {},
            "loss_multiplier": loss_multiplier,
            "idx": idx,
            "task_name": "off_policy_distillation",
        }
    
    # Handle chat-structured formats
    messages = None
    
    if "messages" in datum_dict:
        messages = datum_dict["messages"]
    elif "conversations" in datum_dict:
        # ShareGPT format
        messages = []
        for conv in datum_dict["conversations"]:
            role_from = conv.get("from", conv.get("role", ""))
            if role_from in ["gpt", "assistant", "model", "chatbot"]:
                role = "assistant"
            elif role_from in ["system"]:
                role = "system"
            else:
                role = "user"
            content = conv.get("value", conv.get("content", ""))
            messages.append({"role": role, "content": content})
    elif "prompt" in datum_dict and "response" in datum_dict:
        messages = [
            {"role": "user", "content": datum_dict["prompt"]},
            {"role": "assistant", "content": datum_dict["response"]},
        ]
    elif "input" in datum_dict and "output" in datum_dict:
        messages = [
            {"role": "user", "content": datum_dict["input"]},
            {"role": "assistant", "content": datum_dict["output"]},
        ]
    elif "instruction" in datum_dict:
        user_content = datum_dict["instruction"]
        if "input" in datum_dict and datum_dict["input"]:
            user_content = f"{user_content}\n\n{datum_dict['input']}"
        response = datum_dict.get("output", datum_dict.get("response", ""))
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
    elif "text" in datum_dict:
        # Text with other keys - treat as assistant response
        messages = [{"role": "assistant", "content": datum_dict["text"]}]
    else:
        raise ValueError(
            f"Unsupported datum format. Expected: messages, conversations, "
            f"prompt/response, input/output, instruction/output, or text. "
            f"Got keys: {list(datum_dict.keys())}"
        )
    
    # Add system prompt if specified
    if task_data_spec.system_prompt:
        messages = [{"role": "system", "content": task_data_spec.system_prompt}] + messages
    
    # Build message_log with tokenization
    message_log: LLMMessageLogType = []
    
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        
        # Apply prompt template for user messages
        if role == "user" and task_data_spec.prompt:
            content = task_data_spec.prompt.format(content)
        
        # Add generation prompt only for last user message before assistant
        add_gen_prompt = (
            role == "user"
            and i + 1 < len(messages)
            and messages[i + 1]["role"] == "assistant"
        )
        
        # Tokenize
        chat_msg = [{"role": role, "content": content}]
        formatted = tokenizer.apply_chat_template(
            chat_msg,
            tokenize=False,
            add_generation_prompt=add_gen_prompt,
            add_special_tokens=(i == 0),
        )
        
        token_ids = tokenizer(
            formatted, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        
        # Loss mask: 1 for assistant, 0 for others
        if role == "assistant":
            token_loss_mask = torch.ones_like(token_ids)
        else:
            token_loss_mask = torch.zeros_like(token_ids)
        
        message_log.append({
            "role": role,
            "content": formatted,
            "token_ids": token_ids,
            "token_loss_mask": token_loss_mask,
        })
    
    length = sum(len(m["token_ids"]) for m in message_log)
    
    loss_multiplier = 1.0
    if length > max_seq_length:
        for message in message_log:
            max_per_msg = max(4, max_seq_length // len(message_log))
            message["token_ids"] = message["token_ids"][:max_per_msg]
            message["token_loss_mask"] = message["token_loss_mask"][:max_per_msg]
        loss_multiplier = 0.0
    
    return {
        "message_log": message_log,
        "length": length,
        "extra_env_info": datum_dict.get("extra_env_info", {}),
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict.get("task_name", "off_policy_distillation"),
    }


def load_arrow_dataset(
    data_files: list[str],
    val_split: float = 0.0,
    seed: int = 42,
) -> tuple[Dataset, Optional[Dataset]]:
    """Load dataset from arrow files."""
    print(f"Loading {len(data_files)} arrow files...")
    dataset = load_dataset("arrow", data_files=data_files, split="train")
    print(f"  ✓ Loaded {len(dataset)} samples")
    
    if val_split > 0:
        split = dataset.train_test_split(test_size=val_split, seed=seed)
        return split["train"], split["test"]
    return dataset, None


def setup_data(
    tokenizer: PreTrainedTokenizerBase,
    data_config: DataConfig,
    seed: int = 42,
) -> AllTaskProcessedDataset:
    """Setup data for off-policy distillation."""
    print("\n▶ Setting up data for off-policy distillation...")
    
    task_spec = TaskDataSpec(
        task_name="off_policy_distillation",
        prompt_file=data_config.get("prompt_file"),
        system_prompt_file=data_config.get("system_prompt_file"),
    )
    
    # Load dataset based on format
    if "arrow_files" in data_config:
        arrow_files = data_config["arrow_files"]
        if isinstance(arrow_files, str):
            arrow_files = glob.glob(arrow_files)
        train_ds, _ = load_arrow_dataset(arrow_files, seed=seed)
    elif "train_data_path" in data_config:
        train_ds = load_dataset(
            "json", data_files=data_config["train_data_path"], split="train"
        )
    elif "hf_dataset" in data_config:
        hf_config = data_config["hf_dataset"]
        train_ds = load_dataset(
            hf_config["name"],
            hf_config.get("subset"),
            split=hf_config.get("split", "train"),
        )
    else:
        raise ValueError(
            "Data config must have: 'arrow_files', 'train_data_path', or 'hf_dataset'"
        )
    
    # Create processed dataset
    train_dataset = AllTaskProcessedDataset(
        train_ds,
        tokenizer,
        task_spec,
        off_policy_data_processor,
        max_seq_length=data_config["max_input_seq_length"],
    )
    
    return train_dataset


def main() -> None:
    """Main entry point."""
    args, overrides = parse_args()
    
    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "off_policy_distillation.yaml"
        )
    
    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    
    config: OffPolicyMasterConfig = OmegaConf.to_container(config, resolve=True)
    
    # Get experiment directory
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    
    init_ray()
    
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    
    # Setup data
    train_dataset = setup_data(
        tokenizer, config["data"], config["distillation"]["seed"]
    )
    
    # Setup and run training
    (
        student_policy,
        teacher_policy,
        dataloader,
        loss_fn,
        logger,
        checkpointer,
        save_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset)
    
    off_policy_distillation_train(
        student_policy,
        teacher_policy,
        dataloader,
        tokenizer,
        loss_fn,
        logger,
        checkpointer,
        save_state,
        master_config,
    )


if __name__ == "__main__":
    main()
