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
from typing import Any, Union

import torch
from transformers import AutoProcessor, PreTrainedTokenizerBase

from nemo_rl.data.interfaces import DatumSpec, PreferenceDatumSpec
from nemo_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

TokenizerType = Union[PreTrainedTokenizerBase, AutoProcessor]


def rl_collate_fn(data_batch: list[DatumSpec]) -> BatchedDataDict[Any]:
    """Collate function for RL training."""
    message_log = [datum_spec["message_log"] for datum_spec in data_batch]
    length = torch.tensor([datum_spec["length"] for datum_spec in data_batch])
    loss_multiplier = torch.tensor(
        [datum_spec["loss_multiplier"] for datum_spec in data_batch]
    )
    extra_env_info = [datum_spec["extra_env_info"] for datum_spec in data_batch]

    task_names = []
    for datum_spec in data_batch:
        task_names.append(datum_spec.get("task_name", None))

    idx = [datum_spec["idx"] for datum_spec in data_batch]
    batch_max_length = torch.ones_like(length) * length.max()

    # Extract stop_strings if present
    stop_strings = [datum.get("stop_strings", None) for datum in data_batch]

    # check if any of the data batch has vllm content and images
    extra_args = {}
    if any(
        [datum_spec.get("vllm_content", None) is not None for datum_spec in data_batch]
    ):
        vllm_content = [
            datum_spec.get("vllm_content", None) for datum_spec in data_batch
        ]
        vllm_images = [datum_spec.get("vllm_images", []) for datum_spec in data_batch]
        vllm_videos = [datum_spec.get("vllm_videos", []) for datum_spec in data_batch]
        extra_args["vllm_content"] = vllm_content
        extra_args["vllm_images"] = vllm_images
        extra_args["vllm_videos"] = vllm_videos

    output: BatchedDataDict[Any] = BatchedDataDict(
        message_log=message_log,
        length=length,
        loss_multiplier=loss_multiplier,
        extra_env_info=extra_env_info,
        task_name=task_names,
        idx=idx,
        batch_max_length=batch_max_length,
        stop_strings=stop_strings,
        **extra_args,
    )

    standard_keys = {
        "message_log",
        "length",
        "loss_multiplier",
        "extra_env_info",
        "task_name",
        "idx",
        "batch_max_length",
        "stop_strings",
        "vllm_content",
        "vllm_images",
        "vllm_videos",
    }
    extra_keys: set[str] = set()
    for datum_spec in data_batch:
        extra_keys.update(k for k in datum_spec.keys() if k not in standard_keys)
    for key in extra_keys:
        output[key] = [datum_spec.get(key, None) for datum_spec in data_batch]

        
    return output


def eval_collate_fn(data_batch: list[DatumSpec]) -> BatchedDataDict[Any]:
    """Collate function for evaluation.

    Takes a list of data samples and combines them into a single batched dictionary
    for model evaluation.

    Args:
        data_batch: List of data samples with message_log, extra_env_info, and idx fields.

    Returns:
        BatchedDataDict with message_log, extra_env_info, and idx fields.

    Examples:
    ```{doctest}
    >>> import torch
    >>> from nemo_rl.data.collate_fn import eval_collate_fn
    >>> from nemo_rl.data.interfaces import DatumSpec
    >>> data_batch = [
    ...     DatumSpec(
    ...         message_log=[{"role": "user", "content": "Hello", "token_ids": torch.tensor([1, 2, 3])}],
    ...         extra_env_info={'ground_truth': '1'},
    ...         idx=0,
    ...     ),
    ...     DatumSpec(
    ...         message_log=[{"role": "assistant", "content": "Hi there", "token_ids": torch.tensor([4, 5, 6, 7])}],
    ...         extra_env_info={'ground_truth': '2'},
    ...         idx=1,
    ...     ),
    ... ]
    >>> output = eval_collate_fn(data_batch)
    >>> output['message_log'][0]
    [{'role': 'user', 'content': 'Hello', 'token_ids': tensor([1, 2, 3])}]
    >>> output['message_log'][1]
    [{'role': 'assistant', 'content': 'Hi there', 'token_ids': tensor([4, 5, 6, 7])}]
    >>> output['extra_env_info']
    [{'ground_truth': '1'}, {'ground_truth': '2'}]
    >>> output['idx']
    [0, 1]
    """
    message_log = [datum_spec["message_log"] for datum_spec in data_batch]
    extra_env_info = [datum_spec["extra_env_info"] for datum_spec in data_batch]
    idx = [datum_spec["idx"] for datum_spec in data_batch]

    output: BatchedDataDict[Any] = BatchedDataDict(
        message_log=message_log,
        extra_env_info=extra_env_info,
        idx=idx,
    )
    return output


def preference_collate_fn(
    data_batch: list[PreferenceDatumSpec],
    tokenizer: TokenizerType,
    make_sequence_length_divisible_by: int,
    add_loss_mask: bool,
) -> BatchedDataDict[Any]:
    """Collate function for preference data training.

    This function separates the chosen and rejected responses to create
    two examples per prompt. The chosen and rejected examples are interleaved
    along the batch dimension, resulting in a batch size of 2 * len(data_batch).

    Args:
        data_batch: List of data samples with message_log_chosen, message_log_rejected, length_chosen, length_rejected, loss_multiplier, idx, and task_name fields.
        tokenizer: Tokenizer for text processing
        make_sequence_length_divisible_by: Make the sequence length divisible by this value
        add_loss_mask: Whether to add a token_mask to the returned data
    Returns:
        BatchedDataDict with input_ids, input_lengths, token_mask (optional), and sample_mask fields.
    """
    message_log = []
    length = []
    loss_multiplier = []
    idx = []
    task_names = []
    for datum_spec in data_batch:
        ## interleave chosen and rejected examples
        message_log.append(datum_spec["message_log_chosen"])
        message_log.append(datum_spec["message_log_rejected"])
        length.append(datum_spec["length_chosen"])
        length.append(datum_spec["length_rejected"])
        loss_multiplier.extend([datum_spec["loss_multiplier"]] * 2)
        idx.extend([datum_spec["idx"]] * 2)
        task_names.extend([datum_spec.get("task_name", None)] * 2)
    length_batch: torch.Tensor = torch.tensor(length)
    loss_multiplier_batch: torch.Tensor = torch.tensor(loss_multiplier)

    batch_max_length = torch.ones_like(length_batch) * length_batch.max()

    batch: BatchedDataDict[Any] = BatchedDataDict(
        message_log=message_log,
        length=length_batch,
        loss_multiplier=loss_multiplier_batch,
        task_name=task_names,
        idx=idx,
        batch_max_length=batch_max_length,
    )

    if add_loss_mask:
        add_loss_mask_to_message_log(
            batch["message_log"],
            only_unmask_final=True,
        )

    cat_and_padded, input_lengths = batched_message_log_to_flat_message(
        batch["message_log"],
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )

    data: BatchedDataDict[Any] = BatchedDataDict(
        {
            "input_ids": cat_and_padded["token_ids"],
            "input_lengths": input_lengths,
            "sample_mask": batch["loss_multiplier"],
        }
    )
    if add_loss_mask:
        data["token_mask"] = cat_and_padded["token_loss_mask"]

    return data
