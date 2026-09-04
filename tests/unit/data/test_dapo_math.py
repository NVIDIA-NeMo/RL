# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datasets import Dataset

from nemo_rl.data.datasets.response_datasets.dapo_math import (
    _select_unique_dapo_prompts,
)


def _row(prompt: str, answer: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "reward_model": {"ground_truth": answer, "style": "rule"},
    }


def test_select_unique_dapo_prompts_drops_reward_conflicts() -> None:
    dataset = Dataset.from_list(
        [
            _row("a", "1"),
            _row("a", "1"),
            _row("b", "2"),
            _row("c", "3"),
            _row("c", "different"),
        ]
    )

    cleaned = _select_unique_dapo_prompts(
        dataset, drop_conflicting_rewards=True
    )

    assert len(cleaned) == 2
    assert [row[0]["content"] for row in cleaned["prompt"]] == ["a", "b"]


def test_select_unique_dapo_prompts_can_keep_first_conflicting_row() -> None:
    dataset = Dataset.from_list([_row("a", "1"), _row("a", "different")])

    cleaned = _select_unique_dapo_prompts(
        dataset, drop_conflicting_rewards=False
    )

    assert len(cleaned) == 1
    assert cleaned[0]["reward_model"]["ground_truth"] == "1"
