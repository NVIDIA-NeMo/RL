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
from types import SimpleNamespace

import pytest
import torch


class _MoELayer(torch.nn.Module):
    def __init__(self, dispatcher: object | None) -> None:
        super().__init__()
        self.token_dispatcher = dispatcher


@pytest.mark.mcore
def test_clear_transient_megatron_caches_nulls_dispatcher_tensors():
    """Routing tensors are released; non-tensor attributes and absent dispatchers are kept."""
    from nemo_rl.models.megatron.common import clear_transient_megatron_caches

    dispatcher = SimpleNamespace(
        probs=torch.ones(2, requires_grad=True) * 1.0,
        routing_map=torch.ones(2, dtype=torch.bool),
        reversed_local_input_permutation_mapping=torch.arange(2),
        local_probs=torch.ones(2),
        local_map=torch.ones(2, dtype=torch.bool),
        num_local_experts=4,
    )
    model = torch.nn.Sequential(_MoELayer(dispatcher), _MoELayer(None))

    clear_transient_megatron_caches(model)

    for attr in (
        "probs",
        "routing_map",
        "reversed_local_input_permutation_mapping",
        "local_probs",
        "local_map",
    ):
        assert getattr(dispatcher, attr) is None
    assert dispatcher.num_local_experts == 4


@pytest.mark.mcore
def test_clear_transient_megatron_caches_clears_rope_cache(monkeypatch):
    """The RotaryEmbedding lru_cache is emptied."""
    from megatron.core.models.common.embeddings import rotary_pos_embedding

    from nemo_rl.models.megatron.common import clear_transient_megatron_caches

    cleared: list[bool] = []
    monkeypatch.setattr(
        rotary_pos_embedding.RotaryEmbedding.forward,
        "cache_clear",
        lambda: cleared.append(True),
        raising=False,
    )

    clear_transient_megatron_caches(torch.nn.Linear(1, 1))

    assert cleared == [True]
