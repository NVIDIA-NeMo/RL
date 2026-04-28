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

"""MBPP+ (mostly-basic Python problems, plus) dataset for code-execution eval.

Loads the upstream `evalplus/mbppplus` HuggingFace dataset and shapes each
example into the columns expected by
:func:`nemo_rl.data.processors.code_data_processor` and graded by
:class:`nemo_rl.environments.code_environment.CodeUnitTestEnvironment`.
"""

import re
from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


_FUNCTION_DEF_RE = re.compile(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")


def _infer_entry_point(reference_code: str, test_list: list[str]) -> str:
    """Best-effort inference of the candidate function name."""
    if reference_code:
        match = _FUNCTION_DEF_RE.search(reference_code)
        if match:
            return match.group(1)
    for assertion in test_list:
        match = re.search(r"assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", str(assertion))
        if match:
            return match.group(1)
    return ""


class MBPPPlusDataset:
    """Code-execution eval over the MBPP+ test split.

    The ``env_name`` attribute lets the runner pick the right environment
    (``code_unit_test`` instead of ``math``/``multichoice``).
    """

    env_name: str = "code_unit_test"

    def __init__(
        self,
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        dataset_path: str = "evalplus/mbppplus",
        split: str = "test",
    ):
        ds = load_dataset(dataset_path, split=split)
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name="mbpp_plus",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.code_data_processor

    @staticmethod
    def _rekey(data: dict[str, Any]) -> dict[str, Any]:
        prompt = data.get("prompt") or data.get("text") or ""
        test_list = data.get("test_list") or []
        if not isinstance(test_list, list):
            test_list = [test_list]
        test_imports = data.get("test_imports") or []
        if not isinstance(test_imports, list):
            test_imports = [test_imports]
        # MBPP+ does not ship an explicit entry_point; infer it so that
        # HumanEval-style ``check(candidate)`` harnesses keep working when
        # the same processor is reused.
        entry_point = _infer_entry_point(
            str(data.get("code", "")), [str(x) for x in test_list]
        )
        return {
            "problem": str(prompt),
            # MBPP+'s ``test`` is the canonical-solution test script; the
            # grader uses ``test_list`` (assertions) instead.
            "test_code": "",
            "test_list": [str(x) for x in test_list],
            "test_imports": [str(x) for x in test_imports],
            "entry_point": entry_point,
        }
