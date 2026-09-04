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

NEMO_GYM_TASK_INDEX_KEY = "_ng_task_index"
# Gym-visible generation identity within a task, conventionally in [0, K).
# This is distinct from the private _rowidx ordering key even though both are
# assigned the same value when a single prompt's K rollout rows are expanded.
NEMO_GYM_ROLLOUT_INDEX_KEY = "_ng_rollout_index"
# Zero-based retry attempt for a NeMo-Gym row. The initial attempt is 0.
NEMO_GYM_ATTEMPT_INDEX_KEY = "_ng_attempt_index"
# RL-local marker for a successful Gym response with no output items. Such rows
# are retained for batch shape/accounting but must never contribute training loss.
NEMO_RL_EMPTY_RESPONSE_OUTPUT_KEY = "_nemo_rl_empty_response_output"
# Trainer version/step for which an async rollout was reserved. This is internal
# replay metadata and is deliberately not sent to Gym or the generation backend.
TARGET_WEIGHT_VERSION_KEY = "target_weight_version"
NEXT_NEMO_GYM_TASK_INDEX_KEY = "next_ng_task_index"
# Unconsumed suffix of a gap-fill dataloader batch, carried in the async
# collector's rollouts state so a checkpoint cannot strand yielded prompts.
PENDING_PROMPTS_KEY = "pending_prompt_batch"
# Frontier-aligned async checkpoint metadata (rollouts.pt). The frontier key
# holds the checkpoint cut — the resume filter threshold — and the base is
# the ordinal the saved dataloader snapshot resumes yielding from. Together
# they let a resume regenerate every unaccounted prompt instead of skipping
# it.
FRONTIER_ORDINAL_KEY = "frontier_ordinal"
RESUME_BASE_ORDINAL_KEY = "resume_base_ordinal"
# Post-restore replay-buffer metadata: ordinals of the retained prompt groups,
# reported after age/step filtering so the collector can regenerate the rest.
RETAINED_TASK_INDICES_KEY = "retained_task_indices"
# Ordinals already trained but at or above the checkpoint cut (rollouts.pt).
# The resume folds them into the covered set so the re-yielded window drops
# them instead of training them a second time.
TRAINED_TASK_INDICES_KEY = "trained_task_indices"


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
