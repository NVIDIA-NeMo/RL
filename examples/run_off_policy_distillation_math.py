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
Off-Policy Distillation Training Script (Math)

This script runs off-policy distillation where:
- A fixed dataset of prompt-response pairs is used (no student generation)
- Teacher provides logits for the fixed responses
- Student aligns with teacher using KL divergence loss

Key difference from on-policy distillation:
- No student generation step - uses pre-existing responses from dataset
- No environment needed for reward computation
"""

import argparse
import os
from functools import partial
from typing import Any

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.off_policy_distillation import (
    OffPolicyMasterConfig,
    off_policy_distillation_train,
    setup,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run off-policy distillation training with configuration"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# ===============================================================================
#                      Off-Policy Data Processor
# ===============================================================================
TokenizerType = PreTrainedTokenizerBase


def off_policy_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    idx: int,
    add_bos: bool = True,
    add_eos: bool = True,
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
        
        # Add BOS token if tokenizer has one and add_bos is True
        if add_bos and tokenizer.bos_token:
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


def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig, seed: int):
    """
    Setup data for off-policy distillation using load_response_dataset (like run_sft.py).
    
    This uses the same data loading infrastructure as SFT training.
    """
    print("\n▶ Setting up data for off-policy distillation...")

    # Load dataset using the same approach as run_sft.py
    data = load_response_dataset(data_config, seed)
    train_dataset = data.formatted_ds["train"]
    val_dataset = data.formatted_ds["validation"]
    task_spec = data.task_spec
    print(
        f"  ✓ Training and validation datasets loaded with {len(train_dataset)} and {len(val_dataset)} samples, respectively."
    )

    # Use the off-policy data processor (includes token_loss_mask for distillation)
    train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        task_spec,
        partial(
            off_policy_data_processor,
            add_bos=data_config.get("add_bos", True),
            add_eos=data_config.get("add_eos", True),
        ),
        max_seq_length=data_config["max_input_seq_length"],
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
    if overrides:
        config = parse_hydra_overrides(config, overrides)

    config: OffPolicyMasterConfig = OmegaConf.to_container(config, resolve=True)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    # Note: No generation config needed for off-policy distillation
    # since we don't generate responses from the student

    # Setup data using load_response_dataset (like run_sft.py)
    dataset, val_dataset, task_spec = setup_data(
        tokenizer, config["data"], config["distillation"]["seed"]
    )

    # Setup returns fewer items than on-policy (no student_generation, no val_dataloader)
    (
        student_policy,
        teacher_policy,
        dataloader,
        loss_fn,
        logger,
        checkpointer,
        distillation_state,
        master_config,
    ) = setup(config, tokenizer, dataset)

    # Off-policy training: no student_generation, no environments
    off_policy_distillation_train(
        student_policy,
        teacher_policy,
        dataloader,
        tokenizer,
        loss_fn,
        logger,
        checkpointer,
        distillation_state,
        master_config,
    )


if __name__ == "__main__":
    main()
