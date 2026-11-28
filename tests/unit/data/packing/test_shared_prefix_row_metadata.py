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
"""Tests for the prompt hash that the rollout reassembler stamps on every row.

The driver matches the prompts of two rows by this hash. The rows can come from
different processes, and from a data-plane checkpoint of an earlier process.
"""

from collections.abc import Sequence

import pytest

from nemo_rl.data.packing.shared_prefix.row_metadata import prompt_hash


def test_the_hash_of_a_prompt_is_the_same_in_every_process() -> None:
    # A value that changes with the process or the Python build fails this literal.
    assert prompt_hash([1, 2, 3]) == 6520558355739401531


def test_the_hash_depends_on_the_tokens_and_not_on_their_container() -> None:
    assert prompt_hash((1, 2, 3)) == prompt_hash([1, 2, 3])
    assert prompt_hash(range(1, 4)) == prompt_hash([1, 2, 3])


def test_different_prompts_have_different_hashes() -> None:
    prompts = [[1, 2, 3], [1, 2, 4], [1, 2], [3, 2, 1], []]
    assert len({prompt_hash(prompt) for prompt in prompts}) == len(prompts)


@pytest.mark.parametrize(
    "token_ids",
    [[], [0], range(4096), [2**31 - 1] * 64],
    ids=["empty", "one_token", "long", "large_ids"],
)
def test_the_hash_is_a_non_negative_int64(token_ids: Sequence[int]) -> None:
    assert 0 <= prompt_hash(token_ids) < 2**63
