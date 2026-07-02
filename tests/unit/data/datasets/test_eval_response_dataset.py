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

import random

from nemo_rl.data.datasets.response_datasets.daily_omni import DailyOmniDataset
from nemo_rl.data.datasets.response_datasets.gpqa import GPQADataset
from nemo_rl.data.datasets.response_datasets.math import MathDataset
from nemo_rl.data.datasets.response_datasets.mmlu import MMLUDataset
from nemo_rl.data.datasets.response_datasets.mmlu_pro import MMLUProDataset


def test_gpqa_rekey_preserves_correct_answer():
    dataset = GPQADataset.__new__(GPQADataset)
    dataset.task_name = "GPQA_main"
    dataset._rng = random.Random(42)

    result = dataset._rekey(
        {
            "Question": "question",
            "Correct Answer": "correct",
            "Incorrect Answer 1": "wrong 1",
            "Incorrect Answer 2": "wrong 2",
            "Incorrect Answer 3": "wrong 3",
        }
    )

    assert result["options"][result["answer"]] == "correct"
    assert result["task_name"] == "GPQA_main"


def test_math_rekey_adds_task_name():
    dataset = MathDataset.__new__(MathDataset)
    dataset.task_name = "math_500_test"

    assert dataset._rekey({"Question": "1 + 1", "Answer": "2"}) == {
        "problem": "1 + 1",
        "expected_answer": "2",
        "task_name": "math_500_test",
    }


def test_mmlu_rekey_adds_subject_and_task_name():
    dataset = MMLUDataset.__new__(MMLUDataset)
    dataset.task_name = "MMLU_EN-US"

    result = dataset._rekey(
        {
            "Question": "question",
            "A": "a",
            "B": "b",
            "C": "c",
            "D": "d",
            "Answer": "B",
            "Subject": "math",
        }
    )

    assert result["options"] == {"A": "a", "B": "b", "C": "c", "D": "d"}
    assert result["answer"] == "B"
    assert result["subject"] == "math"
    assert result["task_name"] == "MMLU_EN-US"


def test_mmlu_pro_rekey_supports_more_than_four_options():
    dataset = MMLUProDataset.__new__(MMLUProDataset)
    dataset.task_name = "MMLU-Pro"

    result = dataset._rekey(
        {
            "question": "question",
            "options": ["one", "two", "three", "four", "five"],
            "answer": "E",
            "category": "biology",
        }
    )

    assert result["options"]["E"] == "five"
    assert result["task_name"] == "MMLU-Pro"


def test_daily_omni_eval_can_disable_training_answer_instruction():
    data = {"Question": "question", "Choice": ["choice A", "choice B"]}

    training_prompt = DailyOmniDataset.get_prompt(data)
    eval_prompt = DailyOmniDataset.get_prompt(
        data, include_single_letter_instruction=False
    )

    assert "only a single letter" in training_prompt
    assert "only a single letter" not in eval_prompt
