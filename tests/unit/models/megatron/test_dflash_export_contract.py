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

import nemo_rl.models.megatron.draft as draft_api
import nemo_rl.models.megatron.draft.utils as draft_utils
from nemo_rl.models.megatron.draft.utils import (
    validate_dflash_export_state_dict,
)

pytestmark = pytest.mark.mcore


def test_raw_dflash_export_is_not_a_public_api() -> None:
    """Raw TP-local state must not masquerade as an interoperable export."""
    assert not hasattr(draft_api, "export_dflash_weights")
    assert not hasattr(draft_utils, "export_dflash_weights")


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
