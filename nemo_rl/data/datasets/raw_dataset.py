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

from datasets import Dataset

from nemo_rl.data.interfaces import TaskDataProcessFnCallable, TaskDataSpec
from nemo_rl.data.processors import PROCESSOR_REGISTRY


class RawDataset:
    """Base class for datasets with shared functionality."""

    dataset: Dataset
    # `val_dataset` is used only when current dataset is used for both training and validation
    val_dataset: Dataset | None
    task_name: str
    task_spec: TaskDataSpec
    processor: TaskDataProcessFnCallable

    def common_init(
        self,
        default_task_name: str,
        skip_set_processor: bool,
        task_name: str | None = None,
        prompt_file: str | None = None,
        system_prompt_file: str | None = None,
        processor: str | None = None,
        **kwargs,
    ):
        # 1. set task name, use task_name if provided, otherwise use default_task_name
        self.task_name = task_name if task_name is not None else default_task_name

        # 2. bind prompt and system prompt
        self.task_spec = TaskDataSpec(
            task_name=self.task_name,
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )

        # 3. bind processor (Remove after the data processor is refactored. https://github.com/NVIDIA-NeMo/RL/issues/1658)
        if not skip_set_processor:
            if processor is None:
                processor = "default"
            assert processor in PROCESSOR_REGISTRY, (
                f"Processor {processor} not found in PROCESSOR_REGISTRY. Please call nemo_rl.data.processors.register_processor() to register the processor."
            )
            self.processor = PROCESSOR_REGISTRY[processor]

    def split_train_validation(self, test_size: float, seed: int):
        if test_size > 0:
            split_dataset = self.dataset.train_test_split(
                test_size=test_size, seed=seed
            )
            self.dataset = split_dataset["train"]
            self.val_dataset = split_dataset["test"]

    def extract_task_name_from_path(self, data_path: str) -> str:
        task_name = "-".join(data_path.split("/")[-2:]).split(".")[0]
        if task_name[0] == "-":
            task_name = task_name[1:]
        return task_name
