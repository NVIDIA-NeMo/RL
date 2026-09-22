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
"""Resolution of ``policy.dtensor_cfg.fsdp_output_dtype``."""

import pytest
import torch

from nemo_rl.models.policy.utils import resolve_fsdp_output_dtype


def test_default_keeps_float32_outputs():
    assert resolve_fsdp_output_dtype(None, torch.bfloat16) is torch.float32
    assert resolve_fsdp_output_dtype("float32", torch.bfloat16) is torch.float32
    assert resolve_fsdp_output_dtype(" FLOAT32 ", torch.bfloat16) is torch.float32


def test_param_keeps_compute_dtype():
    assert resolve_fsdp_output_dtype("param", torch.bfloat16) is None
    assert resolve_fsdp_output_dtype("param", torch.float32) is None


def test_explicit_dtype_names():
    assert resolve_fsdp_output_dtype("float16", torch.bfloat16) is torch.float16
    assert resolve_fsdp_output_dtype("fp16", torch.bfloat16) is torch.float16
    assert resolve_fsdp_output_dtype("fp32", torch.bfloat16) is torch.float32
    assert resolve_fsdp_output_dtype("bf16", torch.float32) is torch.bfloat16
    # Explicit dtype equal to the param dtype collapses to "no cast".
    assert resolve_fsdp_output_dtype("bfloat16", torch.bfloat16) is None
    assert resolve_fsdp_output_dtype("float32", torch.float32) is None


def test_invalid_setting_raises():
    with pytest.raises(ValueError, match="fsdp_output_dtype"):
        resolve_fsdp_output_dtype("fp8", torch.bfloat16)


def test_parallelize_entry_points_require_output_dtype_by_keyword():
    """No function-parameter default: the caller must pass the resolved dtype."""
    import inspect

    from nemo_rl.models.dtensor import parallelize

    for fn in (parallelize._parallelize_model, parallelize._parallelize_nm5_h):
        param = inspect.signature(fn).parameters["output_dtype"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty
