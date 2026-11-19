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

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollouts import _attach_output_caps


def _attach(extra_env_info, backend="vllm"):
    generation_input_data = BatchedDataDict()
    batch = BatchedDataDict(extra_env_info=extra_env_info)
    _attach_output_caps(
        generation_input_data, batch, SimpleNamespace(cfg={"backend": backend})
    )
    return generation_input_data


def test_row_caps_are_forwarded_to_vllm():
    data = _attach([{"max_output_tokens": 4}, {}, None])
    assert data["max_output_tokens"] == [4, None, None]


@pytest.mark.parametrize("extra_env_info", [None, [{}, None, {"other": 1}]])
def test_no_caps_leave_generation_input_unchanged(extra_env_info):
    assert "max_output_tokens" not in _attach(extra_env_info, backend="megatron")


def test_caps_fail_loudly_on_backends_that_ignore_them():
    with pytest.raises(ValueError, match="only supported by the vLLM"):
        _attach([{"max_output_tokens": 4}], backend="megatron")
