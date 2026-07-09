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

from dataclasses import dataclass
from typing import Any, Optional

from nemo_rl.data.interfaces import LLMMessageLogType, VLMMessageLogType


@dataclass
class Completion:
    """A single generated completion for one prompt."""

    message_log: LLMMessageLogType | VLMMessageLogType
    env_extras: Optional[dict[str, Any]]
    truncated: bool
    reward: float


@dataclass
class PromptGroupRecord:
    """All completions for a single prompt, with prompt-level metadata."""

    prompt_idx: int
    prompt: LLMMessageLogType | VLMMessageLogType
    extra_env_info: Optional[dict[str, Any]]
    metadata: dict[str, Any]
    completions: list["Completion"]
    rollout_metrics: dict[str, Any]
