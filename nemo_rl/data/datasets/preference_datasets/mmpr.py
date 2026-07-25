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

"""MMPR preference data for offline image MPO."""

import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_from_disk

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.interfaces import TaskDataPreProcessFnCallable

_IMAGE_PLACEHOLDER_RE = re.compile(r"<image(?:_\d+)?>")
_CACHE_VERSION = 1


def format_mmpr_preference_dataset(example: dict[str, Any]) -> dict[str, Any]:
    """Convert an MMPR row to the standard two-completion preference schema."""
    images = example["image"]
    if isinstance(images, str):
        images = [images]
    question = str(example["question"])

    segments = _IMAGE_PLACEHOLDER_RE.split(question)
    user_content: list[dict[str, Any]] = []
    image_index = 0
    for segment_index, segment in enumerate(segments):
        text = segment.strip()
        if text:
            user_content.append({"type": "text", "text": text})
        if segment_index < len(segments) - 1 and image_index < len(images):
            user_content.append({"type": "image", "image": images[image_index]})
            image_index += 1
    while image_index < len(images):
        user_content.insert(
            image_index, {"type": "image", "image": images[image_index]}
        )
        image_index += 1

    chosen = str(example.get("chosen_response", example.get("chosen", "")))
    rejected = str(example.get("rejected_response", example.get("rejected", "")))
    chosen = _IMAGE_PLACEHOLDER_RE.sub("", chosen)
    rejected = _IMAGE_PLACEHOLDER_RE.sub("", rejected)

    context: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    if example.get("system") is not None:
        context.insert(0, {"role": "system", "content": str(example["system"])})

    return {
        "context": context,
        "completions": [
            {
                "rank": 0,
                "completion": [{"role": "assistant", "content": chosen}],
            },
            {
                "rank": 1,
                "completion": [{"role": "assistant", "content": rejected}],
            },
        ],
        "task_name": example.get("task_name", "mmpr"),
    }


class MMPRPreferenceDataset(RawDataset):
    """Load the legacy MMPR meta-recipe as a canonical preference dataset."""

    def __init__(
        self,
        data_path: str,
        split_validation_size: float | int = 0.01,
        legacy_validation_split: bool = False,
        seed: int = 42,
        max_samples: int | None = None,
        cache_dir: str | None = None,
        **_: Any,
    ) -> None:
        self.task_name = "mmpr"
        self.preprocessor = cast(
            TaskDataPreProcessFnCallable, format_mmpr_preference_dataset
        )
        recipe_path = Path(data_path).expanduser().resolve()
        hf_home = Path(
            os.environ.get(
                "HF_HOME",
                str(Path.home() / ".cache" / "huggingface"),
            )
        )
        default_cache_root = Path(
            os.environ.get(
                "HF_DATASETS_CACHE",
                str(hf_home / "datasets"),
            )
        )
        cache_root = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else default_cache_root
        )
        fingerprint = hashlib.sha256(
            (
                f"{recipe_path}|{recipe_path.stat().st_mtime_ns}|"
                f"{max_samples}|{_CACHE_VERSION}"
            ).encode()
        ).hexdigest()[:16]
        prepared_cache = cache_root / f"mmpr_preference_{fingerprint}"

        dataset: Dataset | None = None
        if (prepared_cache / "dataset_info.json").is_file():
            try:
                dataset = load_from_disk(str(prepared_cache))
            except Exception as error:
                warnings.warn(
                    f"Could not load cached MMPR data from {prepared_cache}: {error}"
                )

        if dataset is None:
            with recipe_path.open(encoding="utf-8") as recipe_file:
                recipe = json.load(recipe_file)

            dataset_root = recipe_path.parent.parent
            records: list[dict[str, Any]] = []
            for dataset_info in recipe.values():
                image_root = dataset_root / dataset_info["root"]
                annotation_path = dataset_root / dataset_info["annotation"]
                with annotation_path.open(encoding="utf-8") as annotation_file:
                    for line in annotation_file:
                        record = json.loads(line)
                        chosen_key = (
                            "chosen_response"
                            if "chosen_response" in record
                            else "chosen"
                        )
                        rejected_key = (
                            "rejected_response"
                            if "rejected_response" in record
                            else "rejected"
                        )
                        if "<think>" not in str(record[chosen_key]):
                            # Preserve the legacy MPO data contract used for the
                            # parity baseline with the reasoning checkpoint.
                            record[chosen_key] = "<think></think>\n\n" + str(
                                record[chosen_key]
                            )
                            record[rejected_key] = "<think></think>\n\n" + str(
                                record[rejected_key]
                            )
                            record["system"] = ""
                        images = record["image"]
                        if isinstance(images, str):
                            images = [images]
                        resolved_images = [
                            str((image_root / image).resolve()) for image in images
                        ]
                        # Nano image MPO qualification starts with one valid image.
                        if (
                            len(resolved_images) != 1
                            or not Path(resolved_images[0]).is_file()
                        ):
                            continue
                        record["image"] = resolved_images
                        record["task_name"] = self.task_name
                        records.append(record)
                        if max_samples is not None and len(records) >= max_samples:
                            break
                if max_samples is not None and len(records) >= max_samples:
                    break

            if not records:
                raise ValueError(
                    f"No valid single-image MMPR rows found via {data_path}"
                )
            dataset = Dataset.from_list(records)
            try:
                prepared_cache.parent.mkdir(parents=True, exist_ok=True)
                dataset.save_to_disk(str(prepared_cache))
            except Exception as error:
                warnings.warn(
                    f"Could not cache prepared MMPR data at {prepared_cache}: {error}"
                )

        self.dataset = dataset.shuffle(seed=seed)
        self.val_dataset = None
        if legacy_validation_split:
            # The legacy Omni MPO loader shuffled once and then selected the
            # leading validation rows. Reproduce that ordering for curve parity
            # instead of performing RawDataset's second random split.
            requested_val_size = (
                int(split_validation_size)
                if split_validation_size >= 1
                else 2000
            )
            val_size = min(requested_val_size, len(self.dataset) // 10)
            if val_size > 0:
                self.val_dataset = self.dataset.select(range(val_size))
                self.dataset = self.dataset.select(
                    range(val_size, len(self.dataset))
                )
        else:
            self.split_train_validation(split_validation_size, seed)
