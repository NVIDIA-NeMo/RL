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

"""Unit tests for ``NuminaMath15Dataset`` filtering and formatting.

Registry dispatch, ``split_validation_size`` and the emitted schema are covered
for this dataset by the parametrized ``test_build_in_dataset_with_split_validation``
in ``test_response_dataset.py`` (real data). The tests here are separate because
the filter behavior needs controlled source rows — one row per exclusion reason —
which a real-data test cannot express.
"""

from __future__ import annotations

import pytest
from datasets import Dataset

from nemo_rl.data.datasets import load_response_dataset
from nemo_rl.data.datasets.response_datasets import numinamath as numinamath_module


def _row(
    problem,
    answer,
    question_type="math-word-problem",
    problem_valid="Yes",
    solution_valid="Yes",
):
    return {
        "problem": problem,
        "solution": "irrelevant",
        "answer": answer,
        "problem_type": "Algebra",
        "question_type": question_type,
        "problem_is_valid": problem_valid,
        "solution_is_valid": solution_valid,
        "source": "olympiads",
        "synthetic": False,
    }


# One row per exclusion reason, plus two that must survive.
_SOURCE_ROWS = [
    _row("keep me", "42"),
    _row("sentinel proof answer", "proof"),
    _row("sentinel notfound answer", "notfound"),
    _row("proof question type", "7", question_type="proof"),
    _row("invalid problem", "8", problem_valid="Incomplete"),
    _row("invalid solution", "9", solution_valid="Problem not solved"),
    _row("empty answer", ""),
    _row("keep me too", "1/2"),
]


@pytest.fixture
def patched_load_dataset(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        return Dataset.from_list(_SOURCE_ROWS)

    monkeypatch.setattr(numinamath_module, "load_dataset", fake_load_dataset)


def _problems(dataset):
    return [ex["messages"][0]["content"] for ex in dataset.dataset]


def test_defaults_keep_only_verifiable_valid_rows(patched_load_dataset):
    """Defaults drop sentinel answers, proof-typed rows, and invalid rows."""
    dataset = load_response_dataset({"dataset_name": "NuminaMath-1.5"})

    assert _problems(dataset) == ["keep me", "keep me too"]
    assert dataset.task_name == "NuminaMath-1.5"


def test_emitted_schema(patched_load_dataset):
    dataset = load_response_dataset({"dataset_name": "NuminaMath-1.5"})
    example = dataset.dataset[0]

    assert set(example.keys()) == {"messages", "task_name"}
    assert example["messages"] == [
        {"role": "user", "content": "keep me"},
        {"role": "assistant", "content": "42"},
    ]


def test_verifiable_only_disabled_keeps_sentinel_answers(patched_load_dataset):
    """``verifiable_only=False`` keeps proof/notfound rows (still validity-filtered)."""
    dataset = load_response_dataset(
        {"dataset_name": "NuminaMath-1.5", "verifiable_only": False}
    )

    kept = _problems(dataset)
    assert "sentinel proof answer" in kept
    assert "proof question type" in kept
    # still excluded by require_valid
    assert "invalid problem" not in kept


def test_require_valid_disabled_keeps_flagged_rows(patched_load_dataset):
    """``require_valid=False`` keeps rows the dataset flags as invalid."""
    dataset = load_response_dataset(
        {"dataset_name": "NuminaMath-1.5", "require_valid": False}
    )

    kept = _problems(dataset)
    assert "invalid problem" in kept
    assert "invalid solution" in kept
    # still excluded by verifiable_only
    assert "sentinel notfound answer" not in kept


def test_split_validation_size_carves_a_validation_set(patched_load_dataset):
    dataset = load_response_dataset(
        {
            "dataset_name": "NuminaMath-1.5",
            "require_valid": False,
            "verifiable_only": False,
            "split_validation_size": 0.5,
            "seed": 42,
        }
    )

    assert dataset.val_dataset is not None
    assert len(dataset.dataset) + len(dataset.val_dataset) == len(_SOURCE_ROWS)
