# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_rl.models.generation.vllm.vllm_worker import (
    BaseVllmGenerationWorker,
    _draft_module_sharing_disable_required,
)


@pytest.mark.parametrize(
    ("base_max_tokens", "input_len", "max_model_len", "spec_lookahead", "expected"),
    [
        (256, 100, 1024, 5, 256),  # clamp inactive: base wins
        (256, 900, 1024, 5, 118),  # clamp active: 1024 - 900 - 6
        (256, 1018, 1024, 5, 1),  # at boundary: floor at 1
        (256, 1050, 1024, 5, 1),  # past boundary: floor at 1
        (8, 100, 1024, 5, 8),  # base < headroom: base wins
    ],
)
def test_spec_decode_max_tokens_clamp(
    base_max_tokens, input_len, max_model_len, spec_lookahead, expected
):
    assert (
        BaseVllmGenerationWorker._spec_decode_max_tokens(
            base_max_tokens, input_len, max_model_len, spec_lookahead
        )
        == expected
    )


@pytest.mark.parametrize(
    ("cap_to_context", "spec_lookahead", "expected"),
    [
        (False, 0, 400),
        (True, 0, 300),
        (False, 5, 294),
        (True, 5, 294),
    ],
)
def test_request_max_new_tokens_combines_context_and_spec_limits(
    cap_to_context, spec_lookahead, expected
):
    assert (
        BaseVllmGenerationWorker._request_max_new_tokens(
            configured_max_new_tokens=400,
            input_length=700,
            max_model_len=1000,
            cap_to_context=cap_to_context,
            spec_lookahead=spec_lookahead,
        )
        == expected
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
        # DTensor-v2 full-stream co-training (all 3 methods): disable
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
    assert _draft_module_sharing_disable_required(config) == expected


def test_draft_module_sharing_disable_required_no_speculative_config():
    config = {"vllm_cfg": {"load_format": "dummy"}, "vllm_kwargs": {}}
    assert _draft_module_sharing_disable_required(config) is False
