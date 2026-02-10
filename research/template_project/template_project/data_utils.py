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
"""Data utilities for template project."""

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def create_batch_from(tokenizer, sentences: list[str]) -> BatchedDataDict:
    """Create a tiny batch from raw sentences (no chat templates)."""
    assert len(sentences) > 0, "sentences list must not be empty"

    enc = tokenizer(
        sentences,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"].to(torch.float32)
    input_lengths = attention_mask.sum(dim=1).to(torch.int32)
    sample_mask = torch.ones(input_ids.size(0), dtype=torch.float32)

    # For simple NLL training, use the attention mask as token_mask
    # (loss will be applied to positions 1..len-1 via NLLLoss)
    token_mask = torch.ones_like(input_ids)

    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
        }
    )
