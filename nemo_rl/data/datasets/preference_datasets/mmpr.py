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
_CACHE_VERSION = 2


def _path_signature(path: Path) -> str:
    """Return a cheap source signature suitable for cache invalidation."""
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"


def format_mmpr_preference_dataset(example: dict[str, Any]) -> dict[str, Any]:
    """Convert an MMPR row to the standard two-completion preference schema."""
    images = example["image"]
    if isinstance(images, str):
        images = [images]
    question = str(example["question"])
    placeholder_count = len(_IMAGE_PLACEHOLDER_RE.findall(question))
    if placeholder_count > 0 and placeholder_count != len(images):
        raise ValueError(
            "MMPR rows with image placeholders must contain exactly one "
            "placeholder per image: "
            f"found {placeholder_count} placeholder(s) and {len(images)} image(s)"
        )

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
    # Legacy ``*_wo_image`` MMPR subsets carry image paths without inline
    # placeholders. Preserve their contract by placing media before the text.
    if placeholder_count == 0:
        user_content[0:0] = [{"type": "image", "image": image} for image in images]

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
        with recipe_path.open(encoding="utf-8") as recipe_file:
            recipe = json.load(recipe_file)
        dataset_root = recipe_path.parent.parent
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
        source_signatures = [_path_signature(recipe_path)]
        for dataset_name, dataset_info in sorted(recipe.items()):
            annotation_path = dataset_root / dataset_info["annotation"]
            image_root = dataset_root / dataset_info["root"]
            source_signatures.extend(
                [
                    dataset_name,
                    _path_signature(annotation_path),
                    _path_signature(image_root),
                ]
            )
        fingerprint = hashlib.sha256(
            "|".join(
                [*source_signatures, str(max_samples), str(_CACHE_VERSION)]
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
                        # Preserve the legacy MPO data contract used for the
                        # parity baseline with the reasoning checkpoint.
                        for response_key in (chosen_key, rejected_key):
                            response = str(record[response_key])
                            if "<think>" not in response:
                                record[response_key] = "<think></think>\n\n" + response
                        record.setdefault("system", "")
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
                int(split_validation_size) if split_validation_size >= 1 else 2000
            )
            val_size = min(requested_val_size, len(self.dataset) // 10)
            if val_size > 0:
                self.val_dataset = self.dataset.select(range(val_size))
                self.dataset = self.dataset.select(range(val_size, len(self.dataset)))
        else:
            self.split_train_validation(split_validation_size, seed)
