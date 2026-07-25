# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import json

from datasets import Dataset
from pydantic import TypeAdapter

from nemo_rl.data import DataConfig
from nemo_rl.data.datasets.preference_datasets.mmpr import (
    MMPRPreferenceDataset,
    format_mmpr_preference_dataset,
)


def test_format_mmpr_preference_dataset_preserves_image_and_pair_order():
    formatted = format_mmpr_preference_dataset(
        {
            "question": "Look at <image> and answer.",
            "image": ["/data/example.png"],
            "chosen_response": "correct",
            "rejected_response": "incorrect",
            "task_name": "mmpr",
        }
    )

    user_content = formatted["context"][0]["content"]
    assert user_content == [
        {"type": "text", "text": "Look at"},
        {"type": "image", "image": "/data/example.png"},
        {"type": "text", "text": "and answer."},
    ]
    assert formatted["completions"][0]["rank"] == 0
    assert formatted["completions"][0]["completion"][0]["content"] == "correct"
    assert formatted["completions"][1]["rank"] == 1


def test_mmpr_preference_dataset_loads_legacy_meta_recipe(tmp_path):
    recipe_dir = tmp_path / "recipes"
    recipe_dir.mkdir()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "example.png"
    image_path.write_bytes(b"image")
    annotation_path = tmp_path / "annotations.jsonl"
    annotation_path.write_text(
        json.dumps(
            {
                "question": "<image> What is shown?",
                "image": "example.png",
                "chosen_response": "A shape",
                "rejected_response": "Nothing",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recipe_path = recipe_dir / "meta.json"
    recipe_path.write_text(
        json.dumps(
            {
                "tiny": {
                    "root": "images",
                    "annotation": "annotations.jsonl",
                }
            }
        ),
        encoding="utf-8",
    )

    dataset = MMPRPreferenceDataset(
        data_path=str(recipe_path),
        split_validation_size=0,
        cache_dir=str(tmp_path / "cache"),
    )

    assert len(dataset.dataset) == 1
    assert dataset.dataset[0]["image"] == [str(image_path.resolve())]
    assert dataset.dataset[0]["chosen_response"] == "<think></think>\n\nA shape"
    assert dataset.dataset[0]["rejected_response"] == "<think></think>\n\nNothing"
    assert dataset.preprocessor is format_mmpr_preference_dataset
    formatted = dataset.preprocessor(dict(dataset.dataset[0]))
    assert formatted["context"][0] == {"role": "system", "content": ""}

    cached_dataset = MMPRPreferenceDataset(
        data_path=str(recipe_path),
        split_validation_size=0,
        cache_dir=str(tmp_path / "cache"),
    )
    assert len(cached_dataset.dataset) == 1


def test_mmpr_preference_dataset_reproduces_legacy_validation_slice(tmp_path):
    recipe_dir = tmp_path / "recipes"
    recipe_dir.mkdir()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "example.png").write_bytes(b"image")
    records = [
        {
            "question": f"<image> Question {index}",
            "image": "example.png",
            "chosen": f"chosen {index}",
            "rejected": f"rejected {index}",
        }
        for index in range(20)
    ]
    annotation_path = tmp_path / "annotations.jsonl"
    annotation_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    recipe_path = recipe_dir / "meta.json"
    recipe_path.write_text(
        json.dumps(
            {
                "tiny": {
                    "root": "images",
                    "annotation": "annotations.jsonl",
                }
            }
        ),
        encoding="utf-8",
    )

    dataset = MMPRPreferenceDataset(
        data_path=str(recipe_path),
        split_validation_size=2,
        legacy_validation_split=True,
        seed=42,
        cache_dir=str(tmp_path / "cache"),
    )
    expected_questions = Dataset.from_list(records).shuffle(seed=42)["question"]

    assert dataset.val_dataset is not None
    assert dataset.val_dataset["question"] == expected_questions[:2]
    assert dataset.dataset["question"] == expected_questions[2:]


def test_mmpr_preference_config_preserves_loader_options():
    config = TypeAdapter(DataConfig).validate_python(
        {
            "max_input_seq_length": 8192,
            "shuffle": True,
            "train": {
                "dataset_name": "MMPRPreference",
                "data_path": "/data/meta.json",
                "split_validation_size": 0.01,
                "legacy_validation_split": True,
                "seed": 42,
                "max_samples": 1024,
                "cache_dir": "/cache/mmpr",
            },
        }
    )

    assert config["train"]["max_samples"] == 1024
    assert config["train"]["cache_dir"] == "/cache/mmpr"
    assert config["train"]["legacy_validation_split"] is True
