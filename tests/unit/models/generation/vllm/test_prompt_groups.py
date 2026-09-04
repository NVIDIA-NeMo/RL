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

from nemo_rl.models.generation.vllm.lfs.prompt_groups import make_request_unique_prompts


def test_make_request_unique_prompts_diverges_at_first_token() -> None:
    prompts = make_request_unique_prompts([3, 2, 4])
    token_ids = [prompt["prompt_token_ids"] for prompt in prompts]

    assert token_ids == [
        [1000, 1000, 1000],
        [1001, 1000],
        [1002, 1000, 1000, 1000],
    ]
    assert len({tokens[0] for tokens in token_ids}) == len(token_ids)


def test_make_request_unique_prompts_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_request_unique_prompts([1, 0])
