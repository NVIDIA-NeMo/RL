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

from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

# Sentinel values the dataset uses in `answer` when there is no closed-form
# answer to check against: proof-style problems and rows whose answer could
# not be recovered from the solution.
NON_VERIFIABLE_ANSWERS = frozenset({"proof", "notfound"})


class NuminaMath15Dataset(RawDataset):
    """Simple wrapper around the NuminaMath-1.5 dataset.

    ``AI-MO/NuminaMath-1.5`` is a large competition-math corpus (896,215 rows).
    A sizeable portion of it has no checkable answer, so by default the wrapper
    keeps only rows suitable for verifiable-answer training — otherwise the
    literal strings ``"proof"`` / ``"notfound"`` would be handed to the math
    verifier as ground truth.

    Args:
        split: Split name for the dataset, default is "train" (the only split).
        verifiable_only: Drop rows whose ``answer`` is a non-verifiable
            sentinel and rows whose ``question_type`` is ``"proof"``, default
            is True. Both checks are needed: the two sets only partially
            overlap, so a row can be tagged ``"math-word-problem"`` and still
            carry ``answer="proof"``.
        require_valid: Keep only rows the dataset marks as both
            ``problem_is_valid == "Yes"`` and ``solution_is_valid == "Yes"``,
            default is True.
        split_validation_size: Size of the validation data, default is 0
        seed: Seed for train/validation split when split_validation_size > 0, default is 42
    """

    def __init__(
        self,
        split: str = "train",
        verifiable_only: bool = True,
        require_valid: bool = True,
        split_validation_size: float = 0.0,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self.task_name = "NuminaMath-1.5"

        # load from huggingface
        self.dataset = load_dataset("AI-MO/NuminaMath-1.5", split=split)

        if verifiable_only:
            self.dataset = self.dataset.filter(self._has_verifiable_answer)
        if require_valid:
            self.dataset = self.dataset.filter(self._is_marked_valid)

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

        # `self.val_dataset` is used (not None) only when current dataset is used for both training and validation
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    @staticmethod
    def _has_verifiable_answer(data: dict[str, Any]) -> bool:
        answer = (data.get("answer") or "").strip()
        if not answer or answer.lower() in NON_VERIFIABLE_ANSWERS:
            return False
        return (data.get("question_type") or "").strip().lower() != "proof"

    @staticmethod
    def _is_marked_valid(data: dict[str, Any]) -> bool:
        return (
            data.get("problem_is_valid") == "Yes"
            and data.get("solution_is_valid") == "Yes"
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": data["problem"]},
                {"role": "assistant", "content": data["answer"]},
            ],
            "task_name": self.task_name,
        }
