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

import logging
from typing import Any, Dict, cast

import torch
from transformers import AutoProcessor, PreTrainedTokenizerBase

from nemo_rl.data.interfaces import (
    DatumSpec,
    LLMMessageLogType,
    PreferenceDatumSpec,
    TaskDataProcessFnCallable,
    TaskDataSpec,
    VLMMessageLogType,
)
from nemo_rl.data.llm_message_utils import get_formatted_message_log

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


def sft_processor(
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
    # optional preprocessor
    if datum_dict["task_name"] == "clevr-cogent":
        from nemo_rl.data.datasets.response_datasets.clevr import (
            format_clevr_cogent_dataset,
        )

        datum_dict = format_clevr_cogent_dataset(datum_dict)

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

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output


def preference_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> PreferenceDatumSpec:
    """Process a datum dictionary for RM/DPO training.

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
        ...     "context": [{"role": "user", "content": "What is 2+2?"}],
        ...     "completions": [
        ...         {"rank": 0, "completion": [{"role": "assistant", "content": "4"}]},
        ...         {"rank": 1, "completion": [{"role": "assistant", "content": "5"}]}
        ...     ]
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
        >>> # context can also contain multiple turns
        >>> datum = {
        ...     "context": [{"role": "user", "content": "I have a question."}, {"role": "assistant", "content": "Sure!"}, {"role": "user", "content": "What is 2+2?"}],
        ...     "completions": [
        ...         {"rank": 0, "completion": [{"role": "assistant", "content": "4"}]},
        ...         {"rank": 1, "completion": [{"role": "assistant", "content": "5"}]}
        ...     ]
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
    assert len(datum_dict["completions"]) == 2, (
        "RM/DPO training supports only two completions"
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
        logging.warning(
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

        length_chosen = sum(len(m["token_ids"]) for m in message_log_chosen)
        length_rejected = sum(len(m["token_ids"]) for m in message_log_rejected)

        # safeguard against edge case where there are too many turns to fit within the max length
        assert max(length_chosen, length_rejected) <= max_seq_length

    output: PreferenceDatumSpec = {
        "message_log_chosen": message_log_chosen,
        "message_log_rejected": message_log_rejected,
        "length_chosen": length_chosen,
        "length_rejected": length_rejected,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
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


def vlm_hf_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    processor: AutoProcessor,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from response_datasets/<dataset_name>.py) into a DatumSpec for the VLM Environment."""
    from nemo_rl.data.datasets.response_datasets.clevr import (
        format_clevr_cogent_dataset,
    )
    from nemo_rl.data.datasets.response_datasets.geometry3k import (
        format_geometry3k_dataset,
    )
    from nemo_rl.data.datasets.response_datasets.refcoco import format_refcoco_dataset
    from nemo_rl.data.multimodal_utils import (
        PackedTensor,
        get_dim_to_pack_along,
        get_multimodal_keys_from_processor,
        resolve_to_image,
    )

    # depending on the task, format the data differently
    if datum_dict["task_name"] == "clevr-cogent":
        datum_dict = format_clevr_cogent_dataset(datum_dict)
    elif datum_dict["task_name"] == "refcoco":
        datum_dict = format_refcoco_dataset(datum_dict)
    elif datum_dict["task_name"] == "geometry3k":
        datum_dict = format_geometry3k_dataset(datum_dict)
    else:
        raise ValueError(f"No data processor for task {datum_dict['task_name']}")

    user_message = datum_dict["messages"]
    problem = user_message[0]["content"]
    extra_env_info = {"ground_truth": user_message[1]["content"]}

    message_log: VLMMessageLogType = []
    ### only one round of interaction is assumed, this can easily be extended to a conversational setting
    user_message: dict[str, Any] = {"role": "user", "content": []}
    #
    images = []
    if isinstance(problem, list):
        for content in problem:
            # for image, video, just append it
            # for text, format the prompt to the problem
            if content["type"] != "text":
                user_message["content"].append(content)
                if content["type"] == "image":
                    images.append(content["image"])
                else:
                    raise ValueError(f"Unsupported content type: {content['type']}")
            elif content["type"] == "text":
                user_message["content"].append(
                    {
                        "type": "text",
                        "text": task_data_spec.prompt.format(content["text"])
                        if task_data_spec.prompt
                        else content["text"],
                    }
                )
    else:
        # conversation consists of a text-only message
        user_message["content"] = task_data_spec.prompt.format(problem)

    images = [resolve_to_image(image) for image in images]

    # get formatted user message
    if hasattr(processor, "conversation_preprocessor"):
        user_message_for_chat_template = processor.conversation_preprocessor(
            user_message
        )
    else:
        user_message_for_chat_template = user_message

    # this is the string-tokenized conversation template for the generation policy (for vllm)
    string_formatted_dialog = processor.apply_chat_template(
        [user_message_for_chat_template],
        tokenize=False,
        add_generation_prompt=True,
    )

    # this is the id-tokenized and image processed conversation template for the policy
    message: dict = processor.apply_chat_template(
        [user_message],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # add this for backward compatibility
    user_message["token_ids"] = message["input_ids"][0]
    # add all keys and values to the user message, and the list of keys
    multimodal_keys = get_multimodal_keys_from_processor(processor)
    for key in multimodal_keys:
        if key in message:
            user_message[key] = PackedTensor(
                message[key], dim_to_pack=get_dim_to_pack_along(processor, key)
            )

    # specifically for gemma, we need to add token_type_ids to the user message as a sequence-type value
    if "token_type_ids" in message:
        user_message["token_type_ids"] = message["token_type_ids"][0]

    ### append to user message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0
    if length >= max_seq_length:
        # Treat truncated messages as text only
        vllm_kwargs = {
            "vllm_content": None,
            "vllm_images": [],
        }

        # make smaller and mask out
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
            for key, value in chat_message.items():
                if isinstance(value, PackedTensor):
                    chat_message[key] = PackedTensor.empty_like(value)
        loss_multiplier = 0.0
    else:
        # get the prompt content! (use this for vllm-backend that needs formatted dialog and list of images) for the entire conversation
        # add images for vllm serving
        vllm_kwargs = {
            "vllm_content": string_formatted_dialog,
            "vllm_images": images,
        }

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
        **vllm_kwargs,  # pyrefly: ignore[bad-unpacking]
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
        "helpsteer3_data_processor": helpsteer3_data_processor,
        "math_data_processor": math_data_processor,
        "math_hf_data_processor": math_hf_data_processor,
        "multichoice_qa_processor": multichoice_qa_processor,
        "sft_processor": sft_processor,
        "vlm_hf_data_processor": vlm_hf_data_processor,
    },
)


def register_processor(
    processor_name: str, processor_function: TaskDataProcessFnCallable
) -> None:
    if processor_name in PROCESSOR_REGISTRY:
        raise ValueError(f"Processor name {processor_name} already registered")
    PROCESSOR_REGISTRY[processor_name] = processor_function

    print(f"[INFO] Dataset processor {processor_name} registered")
