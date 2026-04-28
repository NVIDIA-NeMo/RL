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

"""HumanEval+ dataset for code-execution eval.

Loads the upstream `evalplus/humanevalplus` HuggingFace dataset and shapes each
example into the columns expected by
:func:`nemo_rl.data.processors.code_data_processor` and graded by
:class:`nemo_rl.environments.code_environment.CodeUnitTestEnvironment`.
"""

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class HumanEvalPlusDataset:
    """Code-execution eval over the HumanEval+ test split.

    The ``env_name`` attribute lets the runner pick the right environment
    (``code_unit_test`` instead of ``math``/``multichoice``).
    """

    env_name: str = "code_unit_test"

    def __init__(
        self,
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        dataset_path: str = "evalplus/humanevalplus",
        split: str = "test",
    ):
        ds = load_dataset(dataset_path, split=split)
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name="humaneval_plus",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.code_data_processor

    @staticmethod
    def _rekey(data: dict[str, Any]) -> dict[str, Any]:
        prompt = data.get("prompt") or ""
        test = data.get("test") or ""
        entry_point = data.get("entry_point") or ""
        return {
            "problem": str(prompt),
            # HumanEval+ ships the full check() harness in `test`; the grader
            # will exec it and call check(<entry_point>).
            "test_code": str(test),
            "test_list": [],
            "test_imports": [],
            "entry_point": str(entry_point),
            # HumanEval+ uses a continuation-style prompt: the model is
            # expected to emit only the *body* of `entry_point`. The grader
            # must prepend this stub before exec'ing so that `entry_point`
            # actually gets defined in the namespace.
            "code_prefix": str(prompt),
        }
