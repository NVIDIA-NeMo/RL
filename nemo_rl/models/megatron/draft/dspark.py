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

"""Megatron DSpark draft model for online co-training.

Training-side counterpart of vLLM 0.26's
``vllm/model_executor/models/qwen3_dspark.py``. DSpark blocks have no bonus
slot: ``W = gamma`` and every slot (the anchor included) predicts the *next*
token (``sample_pos = query_pos + 1``) — the same labels
``x_{p+1} .. x_{p+gamma}`` as DFlash. On top of the shared machinery it adds a
low-rank Markov head: ``markov_w2(markov_w1(prev_token))`` is a logit bias
added to each slot's base logits, teacher-forced with the ground-truth
previous token during training.

Everything else (trunk K/V projection, block attention, mask embedding,
LM head) is inherited from
:class:`~nemo_rl.models.megatron.draft.dflash.DFlashDraftModel`.
"""

from __future__ import annotations

from typing import Optional

import torch
from megatron.core import tensor_parallel
from megatron.core.transformer import TransformerConfig
from torch import Tensor

from nemo_rl.models.megatron.draft.dflash import DFlashDraftModel


class DSparkDraftModel(DFlashDraftModel):
    """DSpark block draft: ``W = gamma``, plus the low-rank Markov head."""

    method = "dspark"

    def __init__(
        self,
        config: TransformerConfig,
        *,
        gamma: int,
        mask_token_id: int,
        num_aux_hidden_states: int,
        target_hidden_size: Optional[int] = None,
        markov_rank: int = 64,
        trunk_chunk: int = 1024,
    ):
        super().__init__(
            config,
            gamma=gamma,
            mask_token_id=mask_token_id,
            num_aux_hidden_states=num_aux_hidden_states,
            target_hidden_size=target_hidden_size,
            trunk_chunk=trunk_chunk,
            block_width=gamma,  # no bonus anchor slot
        )
        if markov_rank < 1:
            raise ValueError(f"markov_rank must be >= 1, got {markov_rank}.")
        self.markov_rank = int(markov_rank)
        self.markov_w1 = torch.nn.Embedding(
            config.vocab_size, self.markov_rank, dtype=config.params_dtype
        )
        self.markov_w2 = tensor_parallel.ColumnParallelLinear(
            self.markov_rank,
            config.draft_vocab_size,
            config=config,
            init_method=torch.nn.init.zeros_,
            bias=False,
            skip_bias_add=False,
            gather_output=False,
            skip_weight_param_allocation=False,
        )

    def markov_bias(self, prev_tokens: Tensor) -> Tensor:
        """DSpark transition bias for ground-truth previous tokens.

        Args:
            prev_tokens: ``[B, N, gamma]`` the token preceding each predicted
                position (teacher forcing; slot 0's prev is the anchor token).

        Returns:
            ``[B, N, gamma, draft_vocab_local]`` logit bias (vocab-parallel).
        """
        embedded = self.markov_w1(prev_tokens)
        # Megatron's ColumnParallelLinear expects the [s, b, h] layout.
        bias, _ = self.markov_w2(embedded.reshape(-1, 1, self.markov_rank))
        return bias.reshape(*prev_tokens.shape, -1)
