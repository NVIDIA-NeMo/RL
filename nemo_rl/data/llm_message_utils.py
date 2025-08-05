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
import warnings
from typing import Any, Optional, cast, Union
from nemo_rl.data.interfaces import DatumSpec

import torch
from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.data.interfaces import (
    FlatMessagesType,
    LLMMessageLogType,
    TaskDataSpec,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.data.multimodal_utils import PackedGenericDataItem, get_multimodal_keys_from_processor

Tensor = torch.Tensor
TokenizerType = PreTrainedTokenizerBase


def message_log_to_flat_messages(
    message_log: LLMMessageLogType,
) -> FlatMessagesType:
    """Converts a message log (sequence of message turns) into a flattened representation.

    This function takes a message log (list of dict messages with 'role', 'content', 'token_ids', etc.)
    and converts it to a flat dictionary where all tensors of the same key are concatenated and
    all strings of the same key are put into lists.

    Args:
        message_log: List of message dictionaries with 'role', 'content', and potentially 'token_ids'

    Returns:
        FlatMessagesType: Dictionary mapping keys to concatenated tensors and string lists

    Examples:
    ```{doctest}
    >>> import torch
    >>> from nemo_rl.data.llm_message_utils import message_log_to_flat_messages
    >>> # Create a simple message log with two messages
    >>> message_log = [
    ...     {'role': 'user', 'content': 'Hello', 'token_ids': torch.tensor([1, 2, 3])},
    ...     {'role': 'assistant', 'content': 'Hi there', 'token_ids': torch.tensor([4, 5, 6, 7])}
    ... ]
    >>> flat_msgs = message_log_to_flat_messages(message_log)
    >>> flat_msgs['role']
    ['user', 'assistant']
    >>> flat_msgs['content']
    ['Hello', 'Hi there']
    >>> flat_msgs['token_ids']
    tensor([1, 2, 3, 4, 5, 6, 7])
    ```
    """
    result: dict[str, list[Any]] = {}

    if len(message_log) == 0:
        return cast(FlatMessagesType, result)

    # Get all unique keys across all messages
    all_keys: set[str] = set()
    for msg in message_log:
        all_keys.update(msg.keys())

    # Initialize result with empty lists for each key
    for key in all_keys:
        result[key] = []

    # Collect values for each key
    for msg in message_log:
        for key in all_keys:
            if key in msg:
                result[key].append(msg[key])

    # Concatenate tensors for each key
    concat: FlatMessagesType = {}
    for key in result:
        if result[key] and isinstance(result[key][0], Tensor):
            try:
                concat[key] = torch.cat(result[key])
            except RuntimeError as e:
                if "same number of dimensions" in str(e):
                    raise RuntimeError(
                        f"tensors for {key=} must have same number of dimensions: {[t.shape for t in result[key]]}"
                    ) from e
                raise
        elif result[key] and isinstance(result[key][0], PackedGenericDataItem):
            try:
                # returning as item because we are probably not going to shard this batch into individual items
                concat[key] = result[key][0].__class__.concat_packed_items(result[key], return_as_item=True)  
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Error concatenating packed multimodal data for {key=}") from e

    output: FlatMessagesType = {**result, **concat}
    return output


def get_keys_from_message_log(
    message_log: LLMMessageLogType, keys: list[str]
) -> LLMMessageLogType:
    """Return a new LLMMessageLogType containing only the specified keys from each message.

    Args:
        message_log: Original message log to extract keys from
        keys: List of keys to keep in each message

    Returns:
        LLMMessageLogType: New list with only specified keys
    """
    return [{k: msg[k] for k in keys if k in msg} for msg in message_log]


def add_loss_mask_to_message_log(
    batch_message_log: list[LLMMessageLogType],
    roles_to_train_on: list[str] = ["assistant"],
    only_unmask_final: bool = False,
) -> None:
    """Add token-level loss masks to each message in a message log.

    Args:
        message_log (LLMMessageLogType): List of message dictionaries containing token IDs and metadata
        roles_to_train_on (list[str]): List of strings indicating which speakers to unmask. Default: ["assistant"]
        only_unmask_final (bool): If True, only unmask the final message in the log. Default: False
    """
    for i, role in enumerate(roles_to_train_on):
        roles_to_train_on[i] = role.lower()

    for message_log in batch_message_log:
        for i, message in enumerate(message_log):
            if only_unmask_final:
                if i == len(message_log) - 1:
                    message["token_loss_mask"] = torch.ones_like(
                        cast(Tensor, message["token_ids"])
                    )
                else:
                    message["token_loss_mask"] = torch.zeros_like(
                        cast(Tensor, message["token_ids"])
                    )
            else:
                if message["role"] in roles_to_train_on:
                    message["token_loss_mask"] = torch.ones_like(
                        cast(Tensor, message["token_ids"])
                    )
                else:
                    message["token_loss_mask"] = torch.zeros_like(
                        cast(Tensor, message["token_ids"])
                    )


def _pad_tensor(
    tensor: Tensor,
    max_len: int,
    pad_side: str,
    pad_value: int = 0,
) -> Tensor:
    """Pad a tensor to the specified length.

    Args:
        tensor: Tensor to pad
        max_len: Length to pad to
        pad_side: Whether to pad on the 'left' or 'right'
        pad_value: Value to use for padding

    Returns:
        torch.Tensor: Padded tensor
    """
    pad_len = max_len - tensor.size(0)
    if pad_len <= 0:
        return tensor

    padding = torch.full(
        (pad_len, *tensor.shape[1:]),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat(
        [padding, tensor] if pad_side == "left" else [tensor, padding], dim=0
    )


def _validate_tensor_consistency(tensors: list[Tensor]) -> None:
    """Validate that all tensors have consistent dtypes and devices.

    Args:
        tensors: List of tensors to validate

    Raises:
        RuntimeError: If tensors have different dtypes or devices
    """
    if not tensors:
        return

    first = tensors[0]
    if not all(t is None or t.dtype == first.dtype for t in tensors):
        raise RuntimeError(
            f"expected consistent types but got: {[t.dtype for t in tensors]}"
        )
    if not all(t is None or t.device == first.device for t in tensors):
        raise RuntimeError(
            f"expected tensors on the same device but got: {[t.device for t in tensors]}"
        )


def batched_message_log_to_flat_message(
    message_log_batch: list[LLMMessageLogType],
    pad_value_dict: Optional[dict[str, int]] = None,
    make_sequence_length_divisible_by: int = 1,
) -> tuple[BatchedDataDict[FlatMessagesType], Tensor]:
    """Process and pad a batch of message logs for model input.

    For each message log in the batch:
    1. Converts it to a flat representation using message_log_to_flat_messages
    2. Pads all resulting tensors to the same length for batching
    3. Returns a BatchedDataDict and sequence lengths tensor

    Padding is always applied to the right side of sequences.

    Args:
        message_log_batch: List of LLMMessageLogType (each a conversation with multiple turns)
        pad_value_dict: Dictionary mapping keys to padding values (default is 0)
        make_sequence_length_divisible_by: forces the data to be divisible by this value

    Returns:
        BatchedDataDict[FlatMessagesType]: Dictionary containing padded stacked tensors
        torch.Tensor: Input lengths tensor with shape [batch_size] (pre-padding lengths)

    Raises:
        RuntimeError: If tensors have different dtypes or devices

    Examples:
    ```{doctest}
    >>> import torch
    >>> from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
    >>> from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    >>> # Create a batch of two message logs with different lengths
    >>> message_log_batch = [
    ...     # First conversation
    ...     [
    ...         {'role': 'user', 'content': 'What is 2+2?', 'token_ids': torch.tensor([1, 2, 3, 4, 5])},
    ...         {'role': 'assistant', 'content': '4', 'token_ids': torch.tensor([6, 7])}
    ...     ],
    ...     # Second conversation
    ...     [
    ...         {'role': 'user', 'content': 'Solve x+10=15', 'token_ids': torch.tensor([1, 8, 9, 10, 11, 12])},
    ...         {'role': 'assistant', 'content': 'x=5', 'token_ids': torch.tensor([13, 14, 15])}
    ...     ]
    ... ]
    >>> pad_value_dict = {'token_ids': 0}
    >>> batched_flat, input_lengths = batched_message_log_to_flat_message(message_log_batch, pad_value_dict)
    >>> batched_flat['token_ids'][0].tolist()
    [1, 2, 3, 4, 5, 6, 7, 0, 0]
    >>> batched_flat['token_ids'][1].tolist()
    [1, 8, 9, 10, 11, 12, 13, 14, 15]
    >>> batched_flat['content'][0]
    ['What is 2+2?', '4']
    >>> batched_flat['content'][1]
    ['Solve x+10=15', 'x=5']
    >>> batched_flat['role']
    [['user', 'assistant'], ['user', 'assistant']]
    >>> input_lengths
    tensor([7, 9], dtype=torch.int32)
    >>>
    ```
    """
    if not message_log_batch:
        return BatchedDataDict(), torch.empty(0)

    # Process each message log into a flat representation
    sequenced_lists = [message_log_to_flat_messages(ml) for ml in message_log_batch]
    all_keys = {k for seq in sequenced_lists for k in seq}

    # Find max length and identify tensor keys
    max_len = 0
    tensor_keys = []
    multimodal_keys = []
    for seq in sequenced_lists:
        for key, value in seq.items():
            if isinstance(value, Tensor):
                tensor_keys.append(key)
                max_len = max(max_len, value.size(0))

    if max_len % make_sequence_length_divisible_by != 0:
        max_len = (
            (max_len // make_sequence_length_divisible_by) + 1
        ) * make_sequence_length_divisible_by

    # Handle non-tensor case
    if not tensor_keys:
        result: BatchedDataDict[FlatMessagesType] = BatchedDataDict(
            {
                k: [seq[k][0] if k in seq else None for seq in sequenced_lists]
                for k in all_keys
            }
        )
        return result, torch.empty(0)

    # Create input_lengths tensor
    input_lengths = []
    for seq in sequenced_lists:
        # Find the maximum length among all tensors in the dictionary, default to 0 if none exist
        # Use maximum here since there may be keys that aren't populated for all messages yet.
        # For example, logprobs don't get populated for non-generated tokens until post-processing.
        seq_len = max(
            (v.size(0) for v in seq.values() if isinstance(v, Tensor)), default=0
        )
        input_lengths.append(seq_len)
    input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.int32)

    # Process each key
    result = BatchedDataDict()
    for key in all_keys:
        values = [seq.get(key) for seq in sequenced_lists]

        # if the values are packed multimodal data, then concatenate them
        if values and isinstance(values[0], PackedGenericDataItem):
            result[key] = values[0].__class__.concat_packed_items(values, return_as_item=False)
            continue
        # if not a tensor or DNE, then return the list of values
        if not values or not isinstance(values[0], Tensor):
            result[key] = values
            continue

        # Filter out None values and validate consistency
        values: list[Tensor | None] = cast(list[Tensor | None], values)
        tensors = cast(list[Tensor], [t for t in values if t is not None])
        _validate_tensor_consistency(tensors)

        # Create zero tensors for None values
        filled_values: list[Tensor] = [
            (
                torch.zeros(0, dtype=tensors[0].dtype, device=tensors[0].device)  # type: ignore
                if v is None
                else v
            )
            for v in values
        ]

        # Pad and stack tensors (always right padding)
        pad_value = pad_value_dict.get(key, 0) if pad_value_dict else 0

        padded = [_pad_tensor(t, max_len, "right", pad_value) for t in filled_values]

        result[key] = torch.stack(padded)

    return result, input_lengths_tensor


def message_log_shape(message_log: LLMMessageLogType) -> list[dict[str, torch.Size]]:
    """Get the shape of the tensors in the message log.

    This utility function examines each message in the message log and reports
    the shape of tensor values or recursively processes list values.

    Args:
        message_log: The message log to analyze

    Returns:
        List of dictionaries containing tensor shapes for each key in messages
    """
    shapes = []
    for message in message_log:
        shape = {}
        for k in message.keys():
            if isinstance(message[k], Tensor):
                shape[k] = message[k].shape  # type: ignore # we know it's a tensor
            elif isinstance(message[k], list):
                shape[k] = [message_log_shape(v) for v in message[k]]  # type: ignore
        shapes.append(shape)
    return shapes


def get_first_index_that_differs(str1: str, str2: str) -> int:
    """Get the first index that differs between two strings."""
    for i, (c1, c2) in enumerate(zip(str1, str2)):
        if c1 != c2:
            return i
    return min(len(str1), len(str2))

def get_images_from_message(message: dict[str, Any]) -> list[Any]:
    """Get all images from a message log item."""
    if isinstance(message["content"], str):
        return []
    # iterate over the content list
    images = []
    for item in message["content"]:
        if item["type"] == "image":
            images.extend(list(item["image"])) if isinstance(item["image"], (list, tuple)) else images.append(item["image"])
    return images
    

def get_formatted_message_log(
    message_log: LLMMessageLogType,
    tokenizer: TokenizerType,
    task_data_spec: TaskDataSpec,
    add_bos_token: bool = True,
    add_eos_token: bool = True,
    add_generation_prompt: bool = False,
) -> LLMMessageLogType:
    """Format and tokenize chat messages using the specified template.

    Args:
        message_log: List of message dicts with 'role' and 'content' keys
        tokenizer: Tokenizer for converting text to token IDs
        task_data_spec: Task spec for this dataset.
        add_bos_token: Whether to add bos token to first message if it is not already present. Default: True
        add_eos_token: Whether to add eos token to last message if it is not already present. Default: True
        add_generation_prompt: Whether to include assistant's generation prompt in user messages. Default: False

    Returns:
        The message log with updated 'token_ids' and 'content' fields.
    """
    new_message_log: LLMMessageLogType = []
    prev_formatted_message = ""
    message_log_strs: list[dict[str, str]] = cast(
        list[dict[str, str]], message_log
    )  # we just use the str:str parts here

    multimodal_keys = get_multimodal_keys_from_processor(tokenizer)

    def _format_content_helper(content: Union[str, list[dict[str, Any]]]) -> Union[str, list[dict[str, Any]]]:
        if isinstance(content, str):
            return task_data_spec.prompt.format(content)
        # this is a list of dicts, format only the text ones
        for item in content:
            if item["type"] == "text":
                item["text"] = task_data_spec.prompt.format(item["text"])
        return content

    if task_data_spec.prompt:
        message_log_strs = [
            {
                "role": "user",
                "content": _format_content_helper(message_log_strs[0]["content"]),
            }
        ] + message_log_strs[1:]
    
    for i, message in enumerate(message_log_strs):
        # If enabled, add_generation_prompt is only used on user messages to include
        # the assistant's generation prompt as part of the user message.
        formatted_message: str = tokenizer.apply_chat_template(  # type: ignore
            message_log_strs[: i + 1],
            add_generation_prompt=add_generation_prompt and message["role"] == "user",
            tokenize=False,
            add_special_tokens=False,
        )

        ## get the length of the previous message, excluding the eos token (if present)
        prev_message_len_no_eos: int = get_first_index_that_differs(
            prev_formatted_message,
            formatted_message,
        )

        ## pull out the chunk corresponding to the current message
        message_chunk = formatted_message[prev_message_len_no_eos:]

        if i == 0:
            if add_bos_token:
                if tokenizer.bos_token is None:
                    warnings.warn(
                        "add_bos_token is True but the tokenizer does not have a BOS token. Skipping BOS token addition."
                    )
                elif not message_chunk.startswith(tokenizer.bos_token):
                    message_chunk = tokenizer.bos_token + message_chunk

        if i == len(message_log_strs) - 1:
            message_chunk = message_chunk.rstrip("\n")
            if add_eos_token:
                if tokenizer.eos_token is None:
                    warnings.warn(
                        "add_eos_token is True but the tokenizer does not have an EOS token. Skipping EOS token addition."
                    )
                elif not message_chunk.endswith(tokenizer.eos_token):
                    message_chunk += tokenizer.eos_token
        
        # get images too (extend this for other modalities)
        images_cur_message = get_images_from_message(message)

        new_message = message.copy()
        # extend this if statement to check for all(len(modality)) == 0 when adding other modalities
        if len(images_cur_message) == 0:
            new_message["token_ids"] = tokenizer(
                text=message_chunk, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
        else:
            # extend the else statement to add other modalities (in this case, tokenizer will be a processor)
            processed_chunk = tokenizer(
                text=[message_chunk], images=images_cur_message, return_tensors="pt", add_special_tokens=True
            )
            new_message["token_ids"] = processed_chunk["input_ids"][0]

            # add all vlm keys to the message
            for key in multimodal_keys:
                if key in processed_chunk:
                    new_message[key] = PackedGenericDataItem(processed_chunk[key], dim_to_pack=0)

        if len(new_message["token_ids"]) == 0:
            # if there is an empty message, the empty `token_ids` tensor ends up being in fp32,
            # which causes `_validate_tensor_consistency` to fail. To fix this, we convert the
            # empty tensor to int64.
            new_message["token_ids"] = new_message["token_ids"].to(torch.int64)  # type: ignore

        # format content correctly
        if isinstance(message['content'], str):
            new_message["content"] = message_chunk
        else:
            # format the content list of new message the same way as the original message but replace the text with the new message chunk
            new_message["content"] = []
            for item in message['content']:
                if item['type'] == 'text':
                    new_message["content"].append({
                        "type": "text",
                        "text": message_chunk
                    })
                else:
                    new_message["content"].append(item)

        new_message_log.append(new_message)
        prev_formatted_message = formatted_message

    return new_message_log


def remap_dataset_keys(
    dataset: Dataset,
    mapping_dict: dict[str, str],
) -> Dataset:
    """Remap dataset keys as per mapping.

    Args:
        dataset: The input dataset to remap keys in
        mapping_dict: A dictionary mapping input keys to output keys

    Returns:
        Dataset: A new dataset with remapped keys
    """
    # no need to remap if the keys are already correct
    if all(k == v for k, v in mapping_dict.items()):
        return dataset

    # return the remapped dataset
    return dataset.map(
        lambda x: {v: x[k] for k, v in mapping_dict.items()},
        remove_columns=list(mapping_dict.keys()),
    )

