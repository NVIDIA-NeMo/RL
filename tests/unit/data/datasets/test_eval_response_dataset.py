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

import random
from pathlib import Path

import pytest
from datasets import Dataset
from omegaconf import OmegaConf

from nemo_rl.data.datasets.response_datasets import aime as aime_module
from nemo_rl.data.datasets.response_datasets import daily_omni as daily_omni_module
from nemo_rl.data.datasets.response_datasets import gpqa as gpqa_module
from nemo_rl.data.datasets.response_datasets import (
    is_multimodal_response_dataset,
    load_response_dataset,
    validate_eval_data_config,
)
from nemo_rl.data.datasets.response_datasets import math as math_module
from nemo_rl.data.datasets.response_datasets import mmau as mmau_module
from nemo_rl.data.datasets.response_datasets import mmlu as mmlu_module
from nemo_rl.data.datasets.response_datasets import mmlu_pro as mmlu_pro_module
from nemo_rl.data.datasets.response_datasets import (
    response_dataset as response_dataset_module,
)
from nemo_rl.data.datasets.response_datasets.aime import AIMEDataset
from nemo_rl.data.datasets.response_datasets.daily_omni import DailyOmniDataset
from nemo_rl.data.datasets.response_datasets.gpqa import GPQADataset
from nemo_rl.data.datasets.response_datasets.math import MathDataset
from nemo_rl.data.datasets.response_datasets.mmau import MMAUDataset
from nemo_rl.data.datasets.response_datasets.mmlu import MMLUDataset
from nemo_rl.data.datasets.response_datasets.mmlu_pro import MMLUProDataset
from nemo_rl.data.datasets.response_datasets.response_dataset import ResponseDataset
from nemo_rl.data.processors import PROCESSOR_REGISTRY
from nemo_rl.evals.eval import MasterConfig
from nemo_rl.utils.config import load_config


@pytest.mark.parametrize(
    ("dataset_cls", "dataset_name", "expected_processor"),
    [
        (AIMEDataset, "AIME2024", "math_hf_data_processor"),
        (GPQADataset, "gpqa", "multichoice_qa_processor"),
        (MathDataset, "math", "math_data_processor"),
        (MMLUDataset, "mmlu", "multichoice_qa_processor"),
        (MMLUProDataset, "mmlu_pro", "multichoice_qa_processor"),
        (MMAUDataset, "mmau", "vlm_hf_data_processor"),
        (DailyOmniDataset, "daily-omni", "vlm_hf_data_processor"),
        (ResponseDataset, "ResponseDataset", "math_hf_data_processor"),
    ],
)
def test_eval_dataset_selects_explicitly_configured_processor(
    dataset_cls, dataset_name, expected_processor
):
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.task_name = dataset_name
    dataset.set_task_spec(
        {"dataset_name": dataset_name, "processor": expected_processor}
    )
    dataset.set_processor()

    assert dataset.processor is PROCESSOR_REGISTRY[expected_processor]


def test_null_processor_fails_with_clear_message():
    dataset = MathDataset.__new__(MathDataset)
    dataset.task_name = "math_test"
    dataset.set_task_spec({"dataset_name": "math", "processor": None})

    with pytest.raises(AssertionError, match="Processor None not found"):
        dataset.set_processor()


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

    default_dataset = load_response_dataset(
        {"dataset_name": "AIME2024", "processor": "math_hf_data_processor"}
    )
    repeated_dataset = load_response_dataset(
        {
            "dataset_name": "AIME2024",
            "processor": "math_hf_data_processor",
            "repeat": 3,
        }
    )

    assert len(default_dataset.dataset) == 2
    assert len(repeated_dataset.dataset) == 6
    assert default_dataset.processor is PROCESSOR_REGISTRY["math_hf_data_processor"]


@pytest.fixture
def mocked_eval_dataset_sources(monkeypatch, tmp_path):
    def load_aime_source(dataset_name, *args, **kwargs):
        if dataset_name == "HuggingFaceH4/aime_2024":
            return Dataset.from_list([{"problem": "problem", "answer": 1}])
        if dataset_name == "opencompass/AIME2025":
            return Dataset.from_list([{"question": "question", "answer": 2}])
        if dataset_name == "MathArena/aime_2026":
            return Dataset.from_list([{"problem": "problem", "answer": 3}])
        raise AssertionError(f"Unexpected AIME dataset: {dataset_name}")

    monkeypatch.setattr(aime_module, "load_dataset", load_aime_source)
    monkeypatch.setattr(
        gpqa_module,
        "load_dataset",
        lambda *args, **kwargs: Dataset.from_list(
            [
                {
                    "Question": "question",
                    "Correct Answer": "correct",
                    "Incorrect Answer 1": "wrong 1",
                    "Incorrect Answer 2": "wrong 2",
                    "Incorrect Answer 3": "wrong 3",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        math_module,
        "load_dataset",
        lambda *args, **kwargs: Dataset.from_list(
            [{"Question": "1 + 1", "Answer": "2"}]
        ),
    )
    monkeypatch.setattr(
        mmlu_module,
        "load_dataset",
        lambda *args, **kwargs: Dataset.from_list(
            [
                {
                    "Question": "question",
                    "A": "a",
                    "B": "b",
                    "C": "c",
                    "D": "d",
                    "Answer": "B",
                    "Subject": "math",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        mmlu_pro_module,
        "load_dataset",
        lambda *args, **kwargs: Dataset.from_list(
            [
                {
                    "question": "question",
                    "options": ["one", "two", "three", "four", "five"],
                    "answer": "E",
                    "category": "biology",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        mmau_module,
        "load_dataset",
        lambda *args, **kwargs: Dataset.from_list(
            [
                {
                    "audio": {"bytes": b"fake audio", "path": None},
                    "question": "question",
                    "choices": ["a", "b"],
                    "answer": "a",
                }
            ]
        ),
    )

    daily_omni_root = tmp_path / "daily_omni"
    daily_omni_root.mkdir()
    (daily_omni_root / "Videos").mkdir()
    (daily_omni_root / "qa.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        daily_omni_module,
        "get_huggingface_cache_path",
        lambda *args, **kwargs: str(daily_omni_root),
    )
    monkeypatch.setattr(
        daily_omni_module,
        "load_dataset_from_path",
        lambda *args, **kwargs: Dataset.from_list(
            [
                {
                    "video_id": "video",
                    "Question": "question",
                    "Choice": ["a", "b"],
                    "Answer": "A",
                }
            ]
        ),
    )


@pytest.mark.parametrize(
    ("data_config", "expected_task_name"),
    [
        (
            {"dataset_name": "AIME2024", "processor": "math_hf_data_processor"},
            "AIME2024",
        ),
        (
            {"dataset_name": "AIME2025", "processor": "math_hf_data_processor"},
            "AIME2025",
        ),
        (
            {"dataset_name": "AIME2026", "processor": "math_hf_data_processor"},
            "AIME2026",
        ),
        (
            {"dataset_name": "gpqa", "processor": "multichoice_qa_processor"},
            "GPQA_main",
        ),
        (
            {
                "dataset_name": "gpqa_diamond",
                "processor": "multichoice_qa_processor",
            },
            "GPQA_diamond",
        ),
        (
            {"dataset_name": "math", "processor": "math_data_processor"},
            "math_test",
        ),
        (
            {"dataset_name": "math500", "processor": "math_data_processor"},
            "math_500_test",
        ),
        (
            {
                "dataset_name": "mmlu",
                "language": "EN-US",
                "processor": "multichoice_qa_processor",
            },
            "MMLU_EN-US",
        ),
        (
            {
                "dataset_name": "mmlu",
                "language": "ZH-CN",
                "processor": "multichoice_qa_processor",
            },
            "MMLU_ZH-CN",
        ),
        (
            {
                "dataset_name": "mmlu_pro",
                "processor": "multichoice_qa_processor",
            },
            "MMLU-Pro",
        ),
        (
            {
                "dataset_name": "mmau",
                "processor": "vlm_hf_data_processor",
                "split": "v05.15.25",
            },
            "mmau",
        ),
        (
            {
                "dataset_name": "TwinkStart/MMAU",
                "processor": "vlm_hf_data_processor",
                "split": "v05.15.25",
            },
            "mmau",
        ),
        (
            {
                "dataset_name": "daily-omni",
                "processor": "vlm_hf_data_processor",
                "include_single_letter_instruction": False,
            },
            "daily-omni",
        ),
    ],
)
def test_migrated_eval_dataset_load_flow(
    mocked_eval_dataset_sources, data_config, expected_task_name
):
    dataset = load_response_dataset(data_config)

    assert len(dataset.dataset) > 0
    assert dataset.dataset[0]["task_name"] == expected_task_name
    assert dataset.task_spec.task_name == expected_task_name
    assert dataset.processor is PROCESSOR_REGISTRY[data_config["processor"]]


def test_local_eval_yaml_load_flow(monkeypatch):
    monkeypatch.setattr(
        response_dataset_module,
        "load_dataset_from_path",
        lambda *args, **kwargs: Dataset.from_list([{"Question": "1 + 1", "Answer": 2}]),
    )
    repo_root = Path(__file__).resolve().parents[4]
    config = load_config(repo_root / "examples/configs/evals/local_eval.yaml")
    config_dict = OmegaConf.to_container(config, resolve=True)
    master_config = MasterConfig(**config_dict)

    dataset = load_response_dataset(master_config.data)

    assert dataset.dataset[0]["messages"][1]["content"] == "2"
    assert dataset.processor is PROCESSOR_REGISTRY["math_hf_data_processor"]


@pytest.mark.parametrize("legacy_key", ["file_format", "problem_key", "solution_key"])
def test_legacy_local_eval_keys_link_to_migration_guide(legacy_key):
    with pytest.raises(AssertionError, match=r"NVIDIA-NeMo/RL/pull/3039"):
        validate_eval_data_config(
            {"dataset_name": "ResponseDataset", legacy_key: "legacy-value"}
        )


def test_gpqa_format_data_preserves_correct_answer():
    dataset = GPQADataset.__new__(GPQADataset)
    dataset.task_name = "GPQA_main"
    dataset._rng = random.Random(42)

    result = dataset.format_data(
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
