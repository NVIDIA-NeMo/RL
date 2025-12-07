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

from nemo_rl.data.interfaces import TaskDataProcessFnCallable, TaskDataSpec
from nemo_rl.data.processors import PROCESSOR_REGISTRY


class RawDataset:
    def __init__(self, data_config: dict, seed: int = 42):
        self.data_config: dict = data_config
        self.seed: int = seed
        self.task_name: str | None = None
        self.processor: TaskDataProcessFnCallable | None = None
        self.task_spec: TaskDataSpec | None = None
        raise NotImplementedError("__init__ is not implemented")

    def set_processor(self):
        processor_name = (
            self.data_config["processor"]
            if "processor" in self.data_config
            else "default"
        )
        assert processor_name in PROCESSOR_REGISTRY, (
            f"Processor {processor_name} not found in PROCESSOR_REGISTRY. Please call nemo_rl.data.processors.register_processor() to register the processor."
        )
        self.processor = PROCESSOR_REGISTRY[processor_name]

    def set_task_spec(self, data_config: dict):
        self.data_config = data_config
        system_prompt_file = self.data_config.get("system_prompt_file", None)
        prompt_file = self.data_config.get("prompt_file", None)
        self.task_spec = TaskDataSpec(
            task_name=self.task_name,
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
