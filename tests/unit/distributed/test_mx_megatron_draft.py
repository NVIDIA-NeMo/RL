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

import sys
import types
from unittest.mock import patch

import torch

from nemo_rl.distributed.mx_megatron_helpers import (
    ROLE_REPLICATED,
    publish_eagle_draft_weights,
)


class FakePublisher:
    def __init__(self):
        self.calls = []

    def add_tensor(self, **kwargs):
        self.calls.append(kwargs)


def test_publish_eagle_draft_weights_uses_draft_prefix_and_replicated_role():
    fake_draft_module = types.ModuleType("nemo_rl.models.megatron.draft")
    source_tensor = torch.ones(2, 3, dtype=torch.float32).t()
    fake_draft_module.export_eagle_weights_to_hf = lambda _model: [
        ("eagle_module.fc.weight", source_tensor),
    ]
    publisher = FakePublisher()

    with patch.dict(
        sys.modules,
        {"nemo_rl.models.megatron.draft": fake_draft_module},
    ):
        count = publish_eagle_draft_weights(
            publisher=publisher,
            draft_model=object(),
            dtype=torch.bfloat16,
        )

    assert count == 1
    assert len(publisher.calls) == 1
    call = publisher.calls[0]
    assert call["name"] == "draft.eagle_module.fc.weight"
    assert call["megatron_role"] == ROLE_REPLICATED
    assert call["is_expert"] is False
    assert call["expert_axis"] == 0
    assert call["owned_expert_ids"] == set()
    assert call["megatron_extras"] == {}
    assert call["tensor"].dtype is torch.bfloat16
    assert call["tensor"].is_contiguous()


def test_publish_eagle_draft_weights_noops_without_draft_model():
    publisher = FakePublisher()

    count = publish_eagle_draft_weights(
        publisher=publisher,
        draft_model=None,
        dtype=torch.bfloat16,
    )

    assert count == 0
    assert publisher.calls == []
