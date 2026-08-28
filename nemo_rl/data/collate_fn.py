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
from typing import Any, Union, cast

import torch
from transformers import AutoProcessor, PreTrainedTokenizerBase

from nemo_rl.data.interfaces import DatumSpec, PreferenceDatumSpec
from nemo_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
)
from nemo_rl.data.megatron_sft_packed import MegatronSFTPackedDatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

TokenizerType = Union[PreTrainedTokenizerBase, AutoProcessor]

_MEGATRON_SFT_PACKED_FIELDS = {
    "input_ids",
    "target_ids",
    "token_mask",
    "position_ids",
    "packed_cu_seqlens",
    "packed_max_seqlen",
    "packed_context_parallel_size",
}
_MEGATRON_SFT_PACKED_REQUIRED_FIELDS = _MEGATRON_SFT_PACKED_FIELDS | {
    "length",
    "loss_multiplier",
    "idx",
}


def _collate_megatron_sft_packed(
    data_batch: list[DatumSpec],
    megatron_sft_context_parallel_size: int | None,
) -> BatchedDataDict[Any] | None:
    packed_rows = ["packed_cu_seqlens" in datum_spec for datum_spec in data_batch]
    if not any(packed_rows):
        return None
    if not all(packed_rows):
        raise ValueError(
            "Megatron SFT collate rows must be all packed or all non-packed"
        )

    expected_pack_length: int | None = None
    expected_context_parallel_size: int | None = None
    for row_index, datum_spec in enumerate(data_batch):
        missing_fields = _MEGATRON_SFT_PACKED_REQUIRED_FIELDS.difference(datum_spec)
        if missing_fields:
            raise ValueError(
                f"Megatron SFT row {row_index} has partial Megatron SFT packed "
                f"fields; missing {sorted(missing_fields)}"
            )

        packed_datum = cast(MegatronSFTPackedDatumSpec, datum_spec)
        tensor_fields = (
            ("input_ids", packed_datum["input_ids"]),
            ("target_ids", packed_datum["target_ids"]),
            ("token_mask", packed_datum["token_mask"]),
            ("position_ids", packed_datum["position_ids"]),
            ("packed_cu_seqlens", packed_datum["packed_cu_seqlens"]),
        )
        for field, value in tensor_fields:
            if not torch.is_tensor(value):
                raise ValueError(
                    f"Megatron SFT packed field {field} must be a torch.Tensor"
                )

        input_ids = packed_datum["input_ids"]
        target_ids = packed_datum["target_ids"]
        token_mask = packed_datum["token_mask"]
        position_ids = packed_datum["position_ids"]
        if input_ids.ndim != 1:
            raise ValueError("Megatron SFT packed input_ids must be a 1D tensor")
        pack_length = len(input_ids)
        if pack_length < 1:
            raise ValueError("Megatron SFT packed input_ids must not be empty")
        if any(
            tensor.ndim != 1 or len(tensor) != pack_length
            for tensor in (target_ids, token_mask, position_ids)
        ):
            raise ValueError(
                "Megatron SFT packed input_ids, target_ids, token_mask, and "
                "position_ids must be 1D tensors of equal length"
            )
        if int(packed_datum["length"]) != pack_length:
            raise ValueError(
                "Megatron SFT packed datum length must match input_ids length"
            )
        if expected_pack_length is None:
            expected_pack_length = pack_length
        elif pack_length != expected_pack_length:
            raise ValueError(
                "All Megatron SFT packed rows in a batch must have equal length"
            )

        cu_seqlens = packed_datum["packed_cu_seqlens"]
        if cu_seqlens.ndim != 1 or len(cu_seqlens) < 2:
            raise ValueError("packed_cu_seqlens must be a 1D tensor with >=2 items")
        if int(cu_seqlens[0].item()) != 0:
            raise ValueError("packed_cu_seqlens must start at 0")
        if bool((cu_seqlens[1:] < cu_seqlens[:-1]).any().item()):
            raise ValueError("packed_cu_seqlens must be monotonically non-decreasing")
        if int(cu_seqlens[-1].item()) != pack_length:
            raise ValueError("packed_cu_seqlens final value must equal input length")

        segment_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        expected_max_seqlen = int(segment_lengths.max().item())
        if int(packed_datum["packed_max_seqlen"]) != expected_max_seqlen:
            raise ValueError(
                "packed_max_seqlen must equal max(diff(packed_cu_seqlens))"
            )

        context_parallel_size = int(packed_datum["packed_context_parallel_size"])
        if context_parallel_size < 1:
            raise ValueError("packed_context_parallel_size must be >= 1")
        if (
            megatron_sft_context_parallel_size is not None
            and context_parallel_size != megatron_sft_context_parallel_size
        ):
            raise ValueError(
                "Packed Megatron SFT datum was prepared for "
                f"context_parallel_size={context_parallel_size}, but policy "
                "context_parallel_size="
                f"{megatron_sft_context_parallel_size}"
            )
        if expected_context_parallel_size is None:
            expected_context_parallel_size = context_parallel_size
        elif context_parallel_size != expected_context_parallel_size:
            raise ValueError(
                "All Megatron SFT packed rows in a batch must use the same "
                "context_parallel_size"
            )
        if context_parallel_size > 1:
            cp_granularity = 2 * context_parallel_size
            if bool((segment_lengths % cp_granularity != 0).any().item()):
                raise ValueError(
                    "Packed Megatron SFT cu_seqlens segment lengths must be "
                    f"divisible by {cp_granularity} for "
                    f"context_parallel_size={context_parallel_size}"
                )

    packed_data_batch = [
        cast(MegatronSFTPackedDatumSpec, datum_spec) for datum_spec in data_batch
    ]
    max_cu_seqlens_length = max(
        len(datum_spec["packed_cu_seqlens"]) for datum_spec in packed_data_batch
    )
    padded_cu_seqlens = []
    cu_seqlens_lengths = []
    for datum_spec in packed_data_batch:
        cu_seqlens = datum_spec["packed_cu_seqlens"]
        cu_seqlens_lengths.append(len(cu_seqlens))
        if len(cu_seqlens) < max_cu_seqlens_length:
            cu_seqlens = torch.nn.functional.pad(
                cu_seqlens,
                (0, max_cu_seqlens_length - len(cu_seqlens)),
                value=-1,
            )
        padded_cu_seqlens.append(cu_seqlens)

    return BatchedDataDict(
        input_ids=torch.stack(
            [datum_spec["input_ids"] for datum_spec in packed_data_batch]
        ),
        target_ids=torch.stack(
            [datum_spec["target_ids"] for datum_spec in packed_data_batch]
        ),
        token_mask=torch.stack(
            [datum_spec["token_mask"] for datum_spec in packed_data_batch]
        ),
        position_ids=torch.stack(
            [datum_spec["position_ids"] for datum_spec in packed_data_batch]
        ),
        input_lengths=torch.tensor(
            [datum_spec["length"] for datum_spec in packed_data_batch],
            dtype=torch.int64,
        ),
        sample_mask=torch.tensor(
            [datum_spec["loss_multiplier"] for datum_spec in packed_data_batch],
            dtype=torch.float32,
        ),
        packed_cu_seqlens=torch.stack(padded_cu_seqlens),
        packed_cu_seqlens_lengths=torch.tensor(cu_seqlens_lengths, dtype=torch.int64),
        packed_max_seqlen=torch.tensor(
            [datum_spec["packed_max_seqlen"] for datum_spec in packed_data_batch],
            dtype=torch.int64,
        ),
        idx=[datum_spec["idx"] for datum_spec in packed_data_batch],
        task_name=[
            datum_spec.get("task_name", None) for datum_spec in packed_data_batch
        ],
    )


def rl_collate_fn(
    data_batch: list[DatumSpec],
    megatron_sft_context_parallel_size: int | None = None,
) -> BatchedDataDict[Any]:
    """Collate function for RL training."""
    packed_batch = _collate_megatron_sft_packed(
        data_batch,
        megatron_sft_context_parallel_size,
    )
    if packed_batch is not None:
        return packed_batch

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
        vllm_audios = [datum_spec.get("vllm_audios", []) for datum_spec in data_batch]
        extra_args["vllm_content"] = vllm_content
        extra_args["vllm_images"] = vllm_images
        extra_args["vllm_videos"] = vllm_videos
        extra_args["vllm_audios"] = vllm_audios

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
    task_names = [datum_spec.get("task_name", None) for datum_spec in data_batch]

    # Check if any of the data batch has vllm content (multimodal data)
    extra_args = {}
    if any(
        datum_spec.get("vllm_content", None) is not None for datum_spec in data_batch
    ):
        extra_args["vllm_content"] = [
            datum_spec.get("vllm_content", None) for datum_spec in data_batch
        ]
        extra_args["vllm_images"] = [
            datum_spec.get("vllm_images", []) for datum_spec in data_batch
        ]
        extra_args["vllm_audios"] = [
            datum_spec.get("vllm_audios", []) for datum_spec in data_batch
        ]
        extra_args["vllm_videos"] = [
            datum_spec.get("vllm_videos", []) for datum_spec in data_batch
        ]

    output: BatchedDataDict[Any] = BatchedDataDict(
        message_log=message_log,
        extra_env_info=extra_env_info,
        idx=idx,
        task_name=task_names,
        **extra_args,
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
