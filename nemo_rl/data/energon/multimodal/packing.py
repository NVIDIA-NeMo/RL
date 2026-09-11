# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Energon-owned selection and materialization of multimodal SFT packs."""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.data.energon.multimodal.types import EncodedSFTSample, PackedSFTSample
from nemo_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
    message_log_to_flat_messages,
)
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data.packing import SequencePacker
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _cost(sample: EncodedSFTSample, multiple: int) -> int:
    if sample.packing_cost < sample.length:
        raise ValueError(f"Invalid packing cost for sample {sample.sample_key!r}.")
    return ((sample.packing_cost + multiple - 1) // multiple) * multiple


def select_samples_to_pack(
    samples: list[EncodedSFTSample],
    *,
    packer: SequencePacker,
    sequence_length_pad_multiple: int,
) -> list[list[EncodedSFTSample]]:
    """Group compatible sources and run the configured packer."""
    if sequence_length_pad_multiple <= 0:
        raise ValueError("Packing alignment must be positive.")
    groups: dict[tuple[Any, ...], list[EncodedSFTSample]] = {}
    for sample in samples:
        groups.setdefault(sample.group_key, []).append(sample)
    result: list[list[EncodedSFTSample]] = []
    for group in groups.values():
        bins = packer.pack(
            [_cost(sample, sequence_length_pad_multiple) for sample in group]
        )
        indexes = [index for bin_indexes in bins for index in bin_indexes]
        if sorted(indexes) != list(range(len(group))):
            raise RuntimeError("Packing must preserve every source exactly once.")
        result.extend([[group[index] for index in bin_indexes] for bin_indexes in bins])
    return result


def pack_selected_samples(
    samples: list[EncodedSFTSample],
    *,
    pack_capacity: int,
    sequence_length_pad_multiple: int,
) -> PackedSFTSample:
    """Turn one selected source group into a physical pack."""
    if not samples or any(
        sample.group_key != samples[0].group_key for sample in samples
    ):
        raise ValueError("A physical pack needs compatible sources.")
    padded_lengths = [_cost(sample, sequence_length_pad_multiple) for sample in samples]
    if sum(padded_lengths) > pack_capacity:
        raise ValueError("Selected sources exceed the pack capacity.")
    return PackedSFTSample.derive_from(
        samples[0],
        __key__=",".join(sample.sample_key for sample in samples),
        samples=list(samples),
        source_padded_lengths=padded_lengths,
        group_key=samples[0].group_key,
        pack_capacity=pack_capacity,
    )


def prepare_packed_sft_batch(
    packs: list[PackedSFTSample],
    *,
    tokenizer: PreTrainedTokenizerBase,
    only_unmask_final: bool,
    loss_mask_mode: str | None = None,
) -> BatchedDataDict[Any]:
    """Create model tensors for a batch of physical Energon packs."""
    if not packs or tokenizer.pad_token_id is None:
        raise ValueError("Packed SFT requires packs and a tokenizer pad token.")
    if loss_mask_mode not in (None, "precomputed"):
        raise ValueError(f"Unsupported packed SFT loss_mask_mode={loss_mask_mode!r}.")
    if loss_mask_mode == "precomputed" and only_unmask_final:
        raise ValueError(
            "only_unmask_final cannot override precomputed packed SFT loss masks."
        )
    capacities = {pack.pack_capacity for pack in packs}
    if len(capacities) != 1:
        raise ValueError("All physical packs in a batch need one capacity.")
    capacity = capacities.pop()
    packed_logs: list[list[dict[str, Any]]] = []
    boundaries: list[torch.Tensor | None] = []
    padded_boundaries: list[torch.Tensor | None] = []
    source_ids: list[list[str]] = []

    for pack in packs:
        logs = [
            [dict(message) for message in sample.message_log] for sample in pack.samples
        ]
        if loss_mask_mode == "precomputed":
            for log in logs:
                for message in log:
                    tokens = message.get("token_ids")
                    mask = message.get("token_loss_mask")
                    if (
                        not isinstance(tokens, torch.Tensor)
                        or not isinstance(mask, torch.Tensor)
                        or tokens.ndim != 1
                        or mask.ndim != 1
                        or tokens.shape != mask.shape
                        or bool(((mask != 0) & (mask != 1)).any())
                    ):
                        raise ValueError(
                            "Precomputed packed SFT masks must be binary vectors "
                            "matching each token vector."
                        )
        else:
            add_loss_mask_to_message_log(
                logs,
                roles_to_train_on=["assistant"],
                only_unmask_final=only_unmask_final,
            )
        templates = {
            key: value
            for log in logs
            for message in log
            for key, value in message.items()
            if key not in {"token_ids", "token_loss_mask"}
            and isinstance(value, torch.Tensor)
        }
        lengths: list[int] = []
        combined: list[dict[str, Any]] = []
        token_dtype = torch.long
        for log, sample, padded_length in zip(
            logs, pack.samples, pack.source_padded_lengths
        ):
            tokens = message_log_to_flat_messages(log).get("token_ids")
            if not isinstance(tokens, torch.Tensor) or tokens.numel() == 0:
                raise TypeError("Packed SFT sources require token tensors.")
            token_dtype = tokens.dtype
            length = tokens.shape[0]
            lengths.append(length)
            for message in log:
                message["token_loss_mask"] = (
                    message["token_loss_mask"] * sample.loss_multiplier
                )
                for key, template in templates.items():
                    message.setdefault(
                        key,
                        torch.zeros(
                            (message["token_ids"].shape[0], *template.shape[1:]),
                            dtype=template.dtype,
                        ),
                    )
            log[0]["token_loss_mask"][0] = 0
            padding = padded_length - length
            if padding < 0:
                raise ValueError("A source exceeds its padded length.")
            if padding:
                pad_message = {
                    "role": "padding",
                    "token_ids": torch.full(
                        (padding,), tokenizer.pad_token_id, dtype=token_dtype
                    ),
                    "token_loss_mask": torch.zeros(padding, dtype=torch.float32),
                }
                pad_message.update(
                    {
                        key: torch.zeros((padding, *value.shape[1:]), dtype=value.dtype)
                        for key, value in templates.items()
                    }
                )
                log.append(pad_message)
            combined.extend(log)
        tail = capacity - sum(pack.source_padded_lengths)
        if tail:
            tail_message = {
                "role": "padding",
                "token_ids": torch.full(
                    (tail,), tokenizer.pad_token_id, dtype=token_dtype
                ),
                "token_loss_mask": torch.zeros(tail, dtype=torch.float32),
            }
            tail_message.update(
                {
                    key: torch.zeros((tail, *value.shape[1:]), dtype=value.dtype)
                    for key, value in templates.items()
                }
            )
            combined.append(tail_message)
        packed_logs.append(combined)
        boundaries.append(
            torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
            )
        )
        padded = [0, *torch.tensor(pack.source_padded_lengths).cumsum(0).tolist()]
        padded[-1] = capacity
        padded_boundaries.append(torch.tensor(padded, dtype=torch.int32))
        source_ids.append([sample.sample_key for sample in pack.samples])

    flat, input_lengths = batched_message_log_to_flat_message(
        packed_logs, pad_value_dict={"token_ids": tokenizer.pad_token_id}
    )
    prepared = BatchedDataDict(
        {
            "input_ids": flat["token_ids"],
            "input_lengths": input_lengths,
            "token_mask": flat["token_loss_mask"],
            "sample_mask": flat["token_loss_mask"].bool().any(1).float(),
            # TP replica broadcast rejects tensor-bearing Python lists;
            # PackedTensor carries these jagged per-pack boundaries over NCCL.
            "cu_seqlens": PackedTensor(boundaries, dim_to_pack=0),
            "cu_seqlens_padded": PackedTensor(padded_boundaries, dim_to_pack=0),
            "source_ids": source_ids,
        }
    )
    prepared.update(flat.get_multimodal_dict(as_tensors=False))
    return prepared


__all__ = [
    "pack_selected_samples",
    "prepare_packed_sft_batch",
    "select_samples_to_pack",
]
