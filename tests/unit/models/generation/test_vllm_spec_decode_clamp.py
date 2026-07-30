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

from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker


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
