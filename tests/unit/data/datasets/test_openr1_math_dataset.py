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

"""Unit tests for ``OpenR1Math220KDataset`` loading and formatting behavior."""

from __future__ import annotations

from datasets import Dataset

from nemo_rl.data.datasets.response_datasets import load_response_dataset, openr1_math


def _fake_load_dataset(captured):
    def fake_load_dataset(path, name=None, split=None):
        captured["path"] = path
        captured["name"] = name
        captured["split"] = split
        return Dataset.from_list(
            [
                {
                    "problem": "What is 2 + 2?",
                    "answer": "4",
                    "correctness_math_verify": ["correct"],
                }
            ]
        )

    return fake_load_dataset


def test_openr1_math_220k_loads_and_formats(monkeypatch):
    captured = {}
    monkeypatch.setattr(openr1_math, "load_dataset", _fake_load_dataset(captured))

    dataset = load_response_dataset(
        {
            "dataset_name": "OpenR1-Math-220k",
            "subset": "all",
            "split": "train",
        }
    )

    assert captured == {
        "path": "open-r1/OpenR1-Math-220k",
        "name": "all",
        "split": "train",
    }
    assert dataset.task_name == "OpenR1-Math-220k"
    assert dataset.dataset[0] == {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ],
        "task_name": "OpenR1-Math-220k",
    }


def test_openr1_math_220k_defaults_to_default_subset(monkeypatch):
    """When no ``subset`` is provided, the default HF config is loaded."""
    captured = {}
    monkeypatch.setattr(openr1_math, "load_dataset", _fake_load_dataset(captured))

    load_response_dataset({"dataset_name": "OpenR1-Math-220k", "split": "train"})

    assert captured["name"] == "default"
