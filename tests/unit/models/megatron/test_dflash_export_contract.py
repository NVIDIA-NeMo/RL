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

import pytest
import torch
from torch import nn

from nemo_rl.models.megatron.draft.utils import (
    export_dflash_weights,
    validate_dflash_export_state_dict,
)


pytestmark = pytest.mark.mcore


class _DraftBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(4, 4, bias=False)
        self.branch_norm = nn.LayerNorm(4)


def test_allowed_dflash_export_round_trips_exactly() -> None:
    """Body-only export preserves every trainable DFlash tensor exactly."""
    torch.manual_seed(123)
    source = _DraftBody()
    exported = dict(export_dflash_weights(source))
    restored = _DraftBody()
    restored.load_state_dict(exported)

    assert exported.keys() == source.state_dict().keys()
    for name, tensor in source.state_dict().items():
        assert torch.equal(restored.state_dict()[name], tensor), name


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "lm_head.weight",
        "module.draft_model.output_layer.weight",
        "draft.mask_embedding.weight",
        "module.mask_token",
    ],
)
def test_dflash_export_rejects_target_owned_components(forbidden_key: str) -> None:
    """Target-head and mask-token ownership violations fail before export."""
    with pytest.raises(ValueError, match=forbidden_key):
        validate_dflash_export_state_dict({forbidden_key: torch.ones(1)})


def test_dflash_export_checks_components_instead_of_substrings() -> None:
    """Related body parameter names are not rejected by substring matching."""
    allowed = {
        "head_projection.weight": torch.ones(1),
        "output_layernorm.weight": torch.ones(1),
        "mask_tokenizer_projection.weight": torch.ones(1),
    }

    validate_dflash_export_state_dict(allowed)
