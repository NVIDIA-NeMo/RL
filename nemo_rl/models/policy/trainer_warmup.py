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
"""Synthetic batch for the trainer warm-up (NRL_TRAINER_WARMUP=1).

Every rank builds the same batch from a fixed seed, so all ranks pack the same
microbatches and issue the same collectives.
"""

from typing import Any, Mapping, Optional, Sequence

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

SEQLENS_ENV = "NRL_TRAINER_WARMUP_SEQLENS"
_TOKEN_ID_LOW, _TOKEN_ID_HIGH = 100, 1000
_SEED = 1234


def warmup_seqlens(cfg: Mapping[str, Any], override: Optional[str] = None) -> list[int]:
    """Warm-up sequence lengths: ``override``'s comma-separated list if set.

    Otherwise three halves and a quarter of a microbatch's token budget, which pack
    into two microbatches of different sizes near the budget: the autotuners see the
    real size, and torch.compile meets its second shape during the warm-up.
    """
    max_len = int(cfg["max_total_sequence_length"])
    if override:
        try:
            lengths = [int(part) for part in override.split(",") if part.strip()]
        except ValueError:
            lengths = []
        if not lengths or any(not 2 <= n <= max_len for n in lengths):
            raise ValueError(
                f"{SEQLENS_ENV}={override!r}: need lengths in [2, {max_len}]"
            )
        return lengths

    pad = int(cfg["make_sequence_length_divisible_by"])
    if "sequence_packing" in cfg and cfg["sequence_packing"]["enabled"]:
        budget = int(cfg["sequence_packing"]["train_mb_tokens"])
    elif cfg["dynamic_batching"]["enabled"]:
        budget = int(cfg["dynamic_batching"]["train_mb_tokens"])
    else:
        budget = max_len

    def fit(n: int) -> int:
        return max(pad, min(n, max_len) // pad * pad)

    half, quarter = fit(budget // 2), fit(budget // 4)
    return [half, half, half, quarter]


def may_draw_token_id(token_id: int) -> bool:
    """Whether ``build_warmup_batch`` can put ``token_id`` into a sequence."""
    return _TOKEN_ID_LOW <= token_id < _TOKEN_ID_HIGH


def build_warmup_batch(
    seqlens: Sequence[int],
    *,
    train_fields: Sequence[str],
    pad_to_multiple: int = 1,
) -> BatchedDataDict[Any]:
    """Right-padded CPU batch holding exactly ``train_fields``, in the rollout dtypes.

    The first eighth of each sequence is prompt. Advantages alternate +1 / -1 per
    sequence; the logprob fields are 0 until the caller fills them.
    """
    if not seqlens:
        raise ValueError("seqlens must not be empty")
    rows = len(seqlens)
    width = -(-max(seqlens) // pad_to_multiple) * pad_to_multiple
    generator = torch.Generator().manual_seed(_SEED)
    input_ids = torch.zeros((rows, width), dtype=torch.int64)
    token_mask = torch.zeros((rows, width), dtype=torch.int64)
    for row, length in enumerate(seqlens):
        input_ids[row, :length] = torch.randint(
            _TOKEN_ID_LOW, _TOKEN_ID_HIGH, (length,), generator=generator
        )
        token_mask[row, max(1, length // 8) : length] = 1
    sign = torch.where(torch.arange(rows) % 2 == 0, 1.0, -1.0)
    zeros = torch.zeros((rows, width), dtype=torch.float32)
    values: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "input_lengths": torch.tensor(list(seqlens), dtype=torch.int32),
        "generation_logprobs": zeros.clone(),
        "prev_logprobs": zeros.clone(),
        "reference_policy_logprobs": zeros.clone(),
        "advantages": sign[:, None] * token_mask.to(torch.float32),
        "token_mask": token_mask,
        "sample_mask": torch.ones(rows, dtype=torch.float32),
    }
    missing = [name for name in train_fields if name not in values]
    if missing:
        raise ValueError(f"no synthetic value for train fields {missing}")
    return BatchedDataDict[Any]({name: values[name] for name in train_fields})
