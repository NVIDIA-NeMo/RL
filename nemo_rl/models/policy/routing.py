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

from typing import Optional

import torch
from torch import nn


class SequenceRouter(nn.Module):
    """Small Bernoulli routing policy over transformer layer groups.

    The router is intentionally independent from the frozen LM weights. For the
    first routed MVP this avoids tensor-parallel embedding concerns and gives a
    compact policy that can be saved and trained separately from the base model.
    """

    def __init__(
        self,
        vocab_size: int,
        num_routes: int,
        hidden_size: int = 128,
        init_keep_bias: float = 3.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_routes = num_routes
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_routes),
        )
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, init_keep_bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got shape={input_ids.shape}")

        input_ids = input_ids.clamp(min=0, max=self.vocab_size - 1)
        embeds = self.token_embedding(input_ids)

        if input_lengths is None:
            mask = torch.ones(
                input_ids.shape,
                dtype=embeds.dtype,
                device=input_ids.device,
            )
        else:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            mask = (positions.unsqueeze(0) < input_lengths.unsqueeze(1)).to(
                embeds.dtype
            )

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (embeds * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.net(pooled)
