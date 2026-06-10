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

"""Daily-Omni evaluation dataset wrapper."""

import re
from typing import Any, Optional

from nemo_rl.data.datasets.response_datasets.daily_omni import DailyOmniDataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import vlm_hf_data_processor

# The training-side ``DailyOmniDataset.get_prompt`` ends with a hard
# "must contain only a single letter" instruction that overrides any later
# ``<answer>`` formatting request. Strip it for eval so the prompt_file template
# can dictate output formatting without conflict.
_SINGLE_LETTER_LINE = re.compile(
    r"\n+Your replies must contain only a single letter[^\n]*"
)


class DailyOmniEvalDataset:
    """Daily-Omni evaluation dataset.

    Reuses the response-side ``DailyOmniDataset`` (HF snapshot, tar extraction,
    qa.json load) and exposes the attributes that ``run_eval.py`` needs:
    ``rekeyed_ds``, ``task_spec``, ``processor``, and ``preprocessor``.

    ``prompt_file`` / ``system_prompt_file`` are optional templates with a single
    ``{}`` placeholder for the question text — used by ``vlm_hf_data_processor``
    to wrap the user message (e.g. to require ``<answer> </answer>`` formatting).
    """

    def __init__(
        self,
        split: str = "train",
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        self._base = DailyOmniDataset(split=split)
        self.rekeyed_ds = self._base.dataset
        self.task_spec = TaskDataSpec(
            task_name=self._base.task_name,
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = vlm_hf_data_processor
        self.preprocessor = self._format_for_eval

    def _format_for_eval(self, data: dict[str, Any]) -> dict[str, Any]:
        out = self._base.format_data(data)
        text_item = out["messages"][0]["content"][1]
        text_item["text"] = _SINGLE_LETTER_LINE.sub("", text_item["text"])
        return out
