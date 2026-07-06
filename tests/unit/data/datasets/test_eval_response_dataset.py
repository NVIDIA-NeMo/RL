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

import pytest
from datasets import Dataset

from nemo_rl.data.datasets.response_datasets import aime as aime_module
from nemo_rl.data.datasets.response_datasets import (
    is_multimodal_response_dataset,
    load_response_dataset,
)
from nemo_rl.data.datasets.response_datasets.aime import AIMEDataset
from nemo_rl.data.datasets.response_datasets.audiomcq import AudioMCQDataset
from nemo_rl.data.datasets.response_datasets.avqa import AVQADataset
from nemo_rl.data.datasets.response_datasets.clevr import CLEVRCoGenTDataset
from nemo_rl.data.datasets.response_datasets.daily_omni import DailyOmniDataset
from nemo_rl.data.datasets.response_datasets.geometry3k import Geometry3KDataset
from nemo_rl.data.datasets.response_datasets.gpqa import GPQADataset
from nemo_rl.data.datasets.response_datasets.intent import (
    IntentBenchDataset,
    IntentTrainDataset,
)
from nemo_rl.data.datasets.response_datasets.math import MathDataset
from nemo_rl.data.datasets.response_datasets.mmau import MMAUDataset
from nemo_rl.data.datasets.response_datasets.mmlu import MMLUDataset
from nemo_rl.data.datasets.response_datasets.mmlu_pro import MMLUProDataset
from nemo_rl.data.datasets.response_datasets.mmpr_tiny import MMPRTinyDataset
from nemo_rl.data.datasets.response_datasets.refcoco import RefCOCODataset
from nemo_rl.data.processors import PROCESSOR_REGISTRY


@pytest.mark.parametrize(
    ("dataset_cls", "dataset_name", "expected_processor"),
    [
        (AIMEDataset, "AIME2024", "math_hf_data_processor"),
        (GPQADataset, "gpqa", "multichoice_qa_processor"),
        (MathDataset, "math", "math_data_processor"),
        (MMLUDataset, "mmlu", "multichoice_qa_processor"),
        (MMLUProDataset, "mmlu_pro", "multichoice_qa_processor"),
        (AudioMCQDataset, "audiomcq", "vlm_hf_data_processor"),
        (AVQADataset, "avqa", "vlm_hf_data_processor"),
        (CLEVRCoGenTDataset, "clevr-cogent", "vlm_hf_data_processor"),
        (MMAUDataset, "mmau", "vlm_hf_data_processor"),
        (DailyOmniDataset, "daily-omni", "vlm_hf_data_processor"),
        (Geometry3KDataset, "geometry3k", "vlm_hf_data_processor"),
        (IntentTrainDataset, "intent-train", "vlm_hf_data_processor"),
        (IntentBenchDataset, "intent-bench", "vlm_hf_data_processor"),
        (MMPRTinyDataset, "mmpr-tiny", "vlm_hf_data_processor"),
        (RefCOCODataset, "refcoco", "vlm_hf_data_processor"),
    ],
)
def test_eval_dataset_selects_default_processor_without_config_duplication(
    dataset_cls, dataset_name, expected_processor
):
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.task_name = dataset_name
    dataset.set_task_spec({"dataset_name": dataset_name})
    dataset.set_processor()

    assert dataset.processor is PROCESSOR_REGISTRY[expected_processor]


def test_explicit_processor_still_overrides_dataset_default():
    dataset = MathDataset.__new__(MathDataset)
    dataset.task_name = "math_test"
    dataset.set_task_spec(
        {"dataset_name": "math", "processor": "math_hf_data_processor"}
    )
    dataset.set_processor()

    assert dataset.processor is PROCESSOR_REGISTRY["math_hf_data_processor"]


@pytest.mark.parametrize(
    "name",
    [
        "audiomcq",
        "avqa",
        "clevr-cogent",
        "daily-omni",
        "geometry3k",
        "intent-train",
        "intent-bench",
        "mmau",
        "TwinkStart/MMAU",
        "mmpr-tiny",
        "refcoco",
    ],
)
def test_multimodal_eval_dataset_capability(name):
    assert is_multimodal_response_dataset(name) is True


@pytest.mark.parametrize("name", ["AIME2024", "gpqa", "math", "mmlu_pro"])
def test_text_eval_dataset_capability(name):
    assert is_multimodal_response_dataset(name) is False


def test_aime_defaults_to_one_copy_and_supports_explicit_repeat(monkeypatch):
    source = Dataset.from_list(
        [
            {"problem": "problem 1", "answer": 1},
            {"problem": "problem 2", "answer": 2},
        ]
    )
    monkeypatch.setattr(aime_module, "load_dataset", lambda *args, **kwargs: source)

    default_dataset = load_response_dataset({"dataset_name": "AIME2024"})
    repeated_dataset = load_response_dataset({"dataset_name": "AIME2024", "repeat": 3})

    assert len(default_dataset.dataset) == 2
    assert len(repeated_dataset.dataset) == 6
    assert default_dataset.processor is PROCESSOR_REGISTRY["math_hf_data_processor"]


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
