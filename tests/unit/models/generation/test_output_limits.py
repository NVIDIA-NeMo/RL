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

from nemo_rl.models.generation.vllm.output_limits import per_request_output_limits


def test_mixed_caps_respect_global_and_remaining_context_without_changing_null_rows():
    assert per_request_output_limits(
        [None, 3, 200, 200, None],
        [98, 5, 5, 98, 99],
        configured_max_new_tokens=10,
        max_model_len=100,
    ) == [10, 3, 10, 2, 10]


@pytest.mark.parametrize(
    "caps", [[1], [0, None], [True, None], [1.5, None], [-2, None]]
)
def test_invalid_request_caps_are_rejected(caps):
    with pytest.raises(ValueError, match="max_output_tokens"):
        per_request_output_limits(
            caps, [5, 5], configured_max_new_tokens=10, max_model_len=100
        )


def test_explicit_cap_on_exhausted_context_is_rejected():
    with pytest.raises(ValueError, match="exhausts"):
        per_request_output_limits(
            [5], [100], configured_max_new_tokens=10, max_model_len=100
        )
