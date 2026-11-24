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

from nemo_rl.models.generation.vllm.draft_utils import (
    draft_module_sharing_disable_required,
    load_draft_weights_with_stable_kv_buffers,
)


def _draft_config(
    method: str, load_format: str = "dummy", draft_full_refit: bool = False
) -> dict:
    return {
        "vllm_cfg": {"load_format": load_format},
        "vllm_kwargs": {"speculative_config": {"method": method}},
        "_draft_full_refit": draft_full_refit,
    }


@pytest.mark.parametrize(
    ("method", "load_format", "draft_full_refit", "expected"),
    [
        # Automodel full-stream co-training (all 3 methods): disable
        # required regardless of method name.
        ("dspark", "dummy", True, True),
        ("dflash", "dummy", True, True),
        ("eagle3", "dummy", True, True),
        # Megatron partial-stream co-training (relies on module sharing):
        # method name alone must NOT imply full-stream.
        ("dspark", "dummy", False, False),
        ("dflash", "dummy", False, False),
        ("eagle3", "dummy", False, False),
        # load_format != "dummy": weights are loaded from checkpoint, no
        # dummy-init alias to guard against.
        ("dspark", "auto", True, False),
        # Non-cotrained method: never applies.
        ("mtp", "dummy", True, False),
    ],
)
def test_draft_module_sharing_disable_required(
    method, load_format, draft_full_refit, expected
):
    config = _draft_config(
        method, load_format=load_format, draft_full_refit=draft_full_refit
    )
    assert draft_module_sharing_disable_required(config) == expected


def test_draft_module_sharing_disable_required_no_speculative_config():
    config = {"vllm_cfg": {"load_format": "dummy"}, "vllm_kwargs": {}}
    assert draft_module_sharing_disable_required(config) is False


def test_draft_refit_preserves_captured_context_kv_storage():
    class Draft:
        def __init__(self):
            with torch.inference_mode():
                self.model = SimpleNamespace(
                    _fused_kv_weight=torch.zeros(2, 3),
                    _fused_kv_bias=None,
                    _k_norm_weights=torch.zeros(2, 3),
                )

        def load_weights(self, weights):
            self.model._fused_kv_weight = dict(weights)["kv"].clone()
            self.model._k_norm_weights = dict(weights)["norm"].clone()

    draft = Draft()
    captured_kv = draft.model._fused_kv_weight
    captured_norm = draft.model._k_norm_weights
    for refit_value in (1, 2):
        kv = torch.full_like(captured_kv, refit_value)
        norm = torch.full_like(captured_norm, refit_value + 2)
        load_draft_weights_with_stable_kv_buffers(draft, [("kv", kv), ("norm", norm)])
        assert draft.model._fused_kv_weight is captured_kv
        assert draft.model._k_norm_weights is captured_norm
        torch.testing.assert_close(captured_kv, kv)
        torch.testing.assert_close(captured_norm, norm)
