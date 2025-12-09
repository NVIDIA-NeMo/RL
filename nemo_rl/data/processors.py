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

"""Contains data processors for evaluation."""

from typing import Any, Dict, cast

import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    TaskDataProcessFnCallable,
    TaskDataSpec,
)

TokenizerType = PreTrainedTokenizerBase


def helpsteer3_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a HelpSteer3 preference datum into a DatumSpec for GRPO training.

    This function converts HelpSteer3 preference data to work with GRPO by:
    1. Using the context as the prompt
    2. Using the preferred completion as the target response
    3. Creating a reward signal based on preference scores
    """
    # Extract context and completions from HelpSteer3 format
    context = datum_dict["context"]
    preferred_completion = datum_dict["response"]

    # Build the conversation from context
    message_log: LLMMessageLogType = []

    # Add context messages
    if isinstance(context, list):
        for msg in context:
            message_log.append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                }
            )
    else:
        # If context is a string, treat it as a user message
        message_log.append(
            {
                "role": "user",
                "content": context,
            }
        )

    # Add the preferred completion as the target
    for completion_msg in preferred_completion:
        message_log.append(
            {
                "role": completion_msg["role"],
                "content": completion_msg["content"],
            }
        )

    # Apply chat template and tokenize
    formatted_conversation = tokenizer.apply_chat_template(
        message_log,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=True,
    )

    # Tokenize the entire conversation
    full_tokens = tokenizer(
        formatted_conversation,
        return_tensors="pt",
        add_special_tokens=False,  # Already added by chat template
    )["input_ids"][0]

    # For simplicity, assign all tokens to the first message
    # In a more sophisticated implementation, you might want to split tokens properly
    message_log[0]["token_ids"] = full_tokens
    message_log[0]["content"] = formatted_conversation

    # Clear token_ids for other messages to avoid double counting
    for i in range(1, len(message_log)):
        message_log[i]["token_ids"] = tokenizer("", return_tensors="pt")["input_ids"][
            0
        ]  # Empty tensor

    length = sum(len(m["token_ids"]) for m in message_log)

    # Create ground truth from the preferred completion for environment evaluation
    ground_truth = " ".join([msg["content"] for msg in preferred_completion])
    extra_env_info = {"ground_truth": ground_truth}

    loss_multiplier = 1.0
    if length > max_seq_length:
        # Truncate if too long
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(
                    max_seq_length // len(message_log), len(chat_message["token_ids"])
                )
            ]
        loss_multiplier = 0.0  # Reduce loss for truncated sequences

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output


# Example of a generic math data processor
def math_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from dataset) into a DatumSpec for the Math Environment."""
    problem = datum_dict["problem"]
    solution = str(datum_dict["expected_answer"])
    extra_env_info = {"ground_truth": solution}

    message_log: LLMMessageLogType = []

    # system prompt
    if task_data_spec.system_prompt:
        sys_prompt: dict[str, str | torch.Tensor] = {
            "role": "system",
            "content": task_data_spec.system_prompt,
        }
        sys = tokenizer.apply_chat_template(
            [cast(dict[str, str], sys_prompt)],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        sys_prompt["token_ids"] = tokenizer(
            sys, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        message_log.append(sys_prompt)

    # user prompt
    if task_data_spec.prompt:
        problem = task_data_spec.prompt.format(problem)
    user_message = {"role": "user", "content": problem}
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(
        message, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for indiv_message in message_log:
            indiv_message["token_ids"] = indiv_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output


def math_hf_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from data/hf_datasets/openmathinstruct2.py) into a DatumSpec for the Reward Model Environment."""
    user_message = datum_dict["messages"]
    problem = user_message[0]["content"]
    extra_env_info = {"ground_truth": user_message[1]["content"]}

    message_log: LLMMessageLogType = []
    formatted_content = (
        task_data_spec.prompt.format(problem) if task_data_spec.prompt else problem
    )
    user_message = {
        "role": "user",
        "content": formatted_content,
    }
    message: list[str] = tokenizer.apply_chat_template(  # type: ignore
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    user_message["token_ids"] = tokenizer(
        message,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }
    return output


def _construct_multichoice_prompt(
    prompt: str, question: str, options: dict[str, str]
) -> str:
    """Construct prompt from question and options."""
    output = prompt
    output += f"\n\nQuestion: {question}\nOptions:\n"
    output += "\n".join(
        [
            f"{letter}) {option}"
            for letter, option in options.items()
            if option is not None
        ]
    )
    return output


def multichoice_qa_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from dataset) into a DatumSpec for multiple-choice problems."""
    question = datum_dict["question"]
    answer = str(datum_dict["answer"])
    options = datum_dict["options"]
    extra_env_info = {"ground_truth": answer}
    if "subject" in datum_dict:
        extra_env_info.update({"subject": datum_dict["subject"]})

    message_log = []

    # system prompt
    if task_data_spec.system_prompt:
        sys_prompt: dict[str, str | torch.Tensor] = {
            "role": "system",
            "content": task_data_spec.system_prompt,
        }
        sys = tokenizer.apply_chat_template(
            [cast(dict[str, str], sys_prompt)],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        sys_prompt["token_ids"] = tokenizer(
            sys, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        message_log.append(sys_prompt)

    # user prompt
    if task_data_spec.prompt:
        question = _construct_multichoice_prompt(
            task_data_spec.prompt, question, options
        )
    user_message = {"role": "user", "content": question}
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(
        message, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)
    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": 1.0,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output


# Processor registry. Key is the processor name, value is the processor function.
# Note: We cast the literal dict to Dict[str, TaskDataProcessFnCallable] because
# type checkers see each concrete function's signature as a distinct callable type.
# Without the cast, the registry's inferred type becomes a union of those specific
# callables, which is not assignable to the uniform TaskDataProcessFnCallable.
# The cast asserts our intent that all entries conform to the common callable protocol.
PROCESSOR_REGISTRY: Dict[str, TaskDataProcessFnCallable] = cast(
    Dict[str, TaskDataProcessFnCallable],
    {
        "default": math_hf_data_processor,
        "math_hf_data_processor": math_hf_data_processor,
        "multichoice_qa_processor": multichoice_qa_processor,
        "math_data_processor": math_data_processor,
        "helpsteer3_data_processor": helpsteer3_data_processor,
    },
)


def register_processor(
    processor_name: str, processor_function: TaskDataProcessFnCallable
) -> None:
    if processor_name in PROCESSOR_REGISTRY:
        raise ValueError(f"Processor name {processor_name} already registered")
    PROCESSOR_REGISTRY[processor_name] = processor_function

    print(f"[INFO] Dataset processor {processor_name} registered")
