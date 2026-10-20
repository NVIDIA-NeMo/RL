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

from nemo_rl.models.generation.vllm.draft_utils import (
    draft_module_sharing_disable_required,
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
