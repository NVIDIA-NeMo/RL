# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

pytestmark = pytest.mark.vllm

MULTIPLIER = 0.25


class _FoldingMoE(torch.nn.Module):
    """Mimics vLLM Muse Spark's one-shot router output-multiplier fold."""

    def __init__(self):
        super().__init__()
        self.router = torch.nn.Linear(2, 2, bias=False)
        self.router_output_multiplier = MULTIPLIER
        self.moe_router = SimpleNamespace(router_output_multiplier=MULTIPLIER)

    def fold_router_output_multiplier(self):
        if self.router_output_multiplier is None:
            return
        with torch.no_grad():
            self.router.weight.mul_(self.router_output_multiplier)
        self.router_output_multiplier = None
        self.moe_router.router_output_multiplier = None


def _model_config(**text_config):
    return SimpleNamespace(hf_text_config=SimpleNamespace(**text_config))


def test_rearm_makes_refit_fold_again():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        _rearm_router_output_multiplier,
    )

    moe = _FoldingMoE()
    model = torch.nn.Sequential(moe)
    config = _model_config(output_multiplier=MULTIPLIER)
    raw = torch.ones(2, 2)

    for _ in range(2):  # initial load, then a refit with unfolded weights
        with torch.no_grad():
            moe.router.weight.copy_(raw)
        _rearm_router_output_multiplier(model, config)
        moe.fold_router_output_multiplier()
        torch.testing.assert_close(moe.router.weight, raw * MULTIPLIER)
    assert moe.moe_router.router_output_multiplier is None


def test_rearm_is_noop_before_first_fold_and_without_multiplier():
    from nemo_rl.models.generation.vllm.vllm_backend import (
        _rearm_router_output_multiplier,
    )

    moe = _FoldingMoE()
    _rearm_router_output_multiplier(
        torch.nn.Sequential(moe), _model_config(output_multiplier=0.5)
    )
    assert moe.router_output_multiplier == MULTIPLIER  # armed modules are untouched

    moe.fold_router_output_multiplier()
    _rearm_router_output_multiplier(torch.nn.Sequential(moe), _model_config())
    assert (
        moe.router_output_multiplier is None
    )  # no multiplier in config: nothing to re-arm
