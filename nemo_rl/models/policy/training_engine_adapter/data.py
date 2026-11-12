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

"""Conversion between NeMo RL batches and TrainingEngine Datums.

A NeMo RL batch is right-padded ``[B, S]`` with the value for token ``i`` at position ``i``.
A Datum is unpadded and target-aligned: position ``i`` holds input token ``i`` and the token
it must predict, token ``i + 1``. Sample ``b`` of length ``L`` therefore becomes a Datum of
``L - 1`` positions, and Datum position ``i`` maps back to batch position ``i + 1``.
"""

from typing import Any

import torch
from megatron.bridge.engine import Datum

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# Datum field for each per-token batch field the clipped policy-gradient loss reads.
_LOSS_FIELDS = {
    "advantages": "advantages",
    "prev_logprobs": "old_logprobs",
    "reference_policy_logprobs": "reference_logprobs",
    "generation_logprobs": "rollout_logprobs",
}


def _token_rows(data: BatchedDataDict[Any]) -> list[tuple[torch.Tensor, int]]:
    """Each sample's real tokens and length, rejecting samples with no target."""
    rows = []
    lengths = data["input_lengths"].tolist()
    for index, (row, length) in enumerate(zip(data["input_ids"], lengths)):
        if length < 2:
            raise ValueError(
                f"Sample {index} has {length} token(s); the engine needs at least 2."
            )
        rows.append((row[:length].to(torch.int64), length))
    return rows


def scoring_datums(data: BatchedDataDict[Any]) -> list[Datum]:
    """One Datum per sample, in batch order, with every position selected."""
    return [
        Datum(
            input_ids=tokens[:-1],
            target_ids=tokens[1:],
            loss_mask=torch.ones(length - 1, dtype=torch.float32),
        )
        for tokens, length in _token_rows(data)
    ]


def training_datums(data: BatchedDataDict[Any]) -> list[Datum]:
    """One Datum per sample with the loss mask and the per-token loss inputs."""
    loss_mask = data["token_mask"] * data["sample_mask"].unsqueeze(-1)
    if not torch.all((loss_mask == 0) | (loss_mask == 1)):
        raise ValueError(
            "The TrainingEngine policy needs binary token and sample masks; "
            "weighted masks are not supported."
        )
    datums = []
    for index, (tokens, length) in enumerate(_token_rows(data)):
        fields = {
            datum_field: data[batch_field][index, 1:length].to(torch.float32)
            for batch_field, datum_field in _LOSS_FIELDS.items()
        }
        datums.append(
            Datum(
                input_ids=tokens[:-1],
                target_ids=tokens[1:],
                loss_mask=loss_mask[index, 1:length].to(torch.float32),
                **fields,
            )
        )
    return datums


def padded_scores(scores: list[torch.Tensor], width: int) -> torch.Tensor:
    """Per-Datum scores as a ``[B, width]`` batch tensor; the first token and padding are 0."""
    padded = torch.zeros(len(scores), width, dtype=torch.float32)
    for row, values in zip(padded, scores):
        row[1 : values.numel() + 1] = values.to(device="cpu", dtype=torch.float32)
    return padded
