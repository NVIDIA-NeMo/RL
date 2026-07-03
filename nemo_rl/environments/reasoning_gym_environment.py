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

import re
from typing import Any, Optional, TypedDict

import ray
import reasoning_gym
import torch
from reasoning_gym.utils import extract_answer

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


class ReasoningGymMetadata(TypedDict):
    rg_entry: dict[str, Any]
    task: str


@ray.remote  # pragma: no cover
class ReasoningGymEnvironment(EnvironmentInterface[ReasoningGymMetadata]):
    """Generic reasoning_gym-backed environment with binary correctness reward."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        # If False, return reasoning_gym's raw fractional score (dense signal, e.g.
        # sudoku cell-accuracy) instead of a 0/1 correctness reward. Defaults to binary.
        self.binary_reward = self.cfg.get("binary_reward", True)
        # "sudoku_blanks": SPG-style reward = fraction of originally-blank cells
        # filled correctly (copying givens earns nothing). Overrides binary_reward.
        self.reward_mode = self.cfg.get("reward_mode", None)
        self._score_fns: dict[str, Any] = {}

    def _get_score_fn(self, task: str) -> Any:
        if task not in self._score_fns:
            self._score_fns[task] = reasoning_gym.get_score_answer_fn(task)
        return self._score_fns[task]

    def _sudoku_blanks_score(self, ans: str, rg_entry: dict) -> float:
        """SPG-style reward: fraction of originally-blank cells filled correctly."""
        meta = rg_entry.get("metadata", {}) if isinstance(rg_entry, dict) else {}
        puzzle = meta.get("puzzle")
        solution = meta.get("solution")
        if not puzzle or not solution:
            return 0.0
        # Parse per line (row) so a missing/extra digit in one row does not
        # shift alignment for every later cell.
        ans_rows = [
            [int(ch) for ch in line if ch.isdigit()]
            for line in ans.splitlines()
            if any(ch.isdigit() for ch in line)
        ]
        blanks = 0
        correct = 0
        for r, prow in enumerate(puzzle):
            for c, val in enumerate(prow):
                if val == 0:
                    blanks += 1
                    if (
                        r < len(ans_rows)
                        and c < len(ans_rows[r])
                        and ans_rows[r][c] == solution[r][c]
                    ):
                        correct += 1
        if blanks == 0:
            return 0.0
        return correct / blanks

    def _countdown_agrpo_score(self, ans: str, rg_entry: dict) -> float:
        """AGRPO-style countdown reward: 1/3 valid ASCII expression + 1/3 exact
        number usage + 1/3 evaluates to target. Answer format: "expr = target"."""
        meta = rg_entry.get("metadata", {}) if isinstance(rg_entry, dict) else {}
        nums = meta.get("numbers")
        target = meta.get("target")
        if nums is None or target is None:
            return 0.0
        expr = ans.split("=")[0].strip() if "=" in ans else ans.strip()
        reward = 0.0
        ascii_ok = (
            bool(expr)
            and "**" not in expr
            and re.match(r"^[\d\s+\-*/()]+$", expr) is not None
        )
        if ascii_ok:
            reward += 1.0 / 3.0
        used = [int(n) for n in re.findall(r"\d+", expr)]
        if sorted(used) == sorted(nums):
            reward += 1.0 / 3.0
        if ascii_ok:
            try:
                if abs(eval(expr, {"__builtins__": {}}, {}) - target) < 1e-9:
                    reward += 1.0 / 3.0
            except Exception:
                pass
        return reward

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[ReasoningGymMetadata],
    ) -> EnvironmentReturn[ReasoningGymMetadata]:
        """Scores a batch of reasoning_gym interactions inline (scoring is cheap)."""
        rewards: list[float] = []
        extracted_answers: list[str | None] = []
        for conversation, meta in zip(message_log_batch, metadata):
            assistant_text = "".join(
                str(interaction["content"])
                for interaction in conversation
                if interaction["role"] == "assistant"
            )
            ans = extract_answer(assistant_text, tag_name="answer")
            if ans is None:
                ans = assistant_text
            extracted_answers.append(ans)

            try:
                if self.reward_mode == "sudoku_blanks":
                    reward = self._sudoku_blanks_score(ans, meta["rg_entry"])
                elif self.reward_mode == "countdown_agrpo":
                    reward = self._countdown_agrpo_score(ans, meta["rg_entry"])
                else:
                    score = float(
                        self._get_score_fn(meta["task"])(
                            answer=ans, entry=meta["rg_entry"]
                        )
                    )
                    reward = (
                        (1.0 if score >= 1.0 else 0.0)
                        if self.binary_reward
                        else score
                    )
            except Exception:
                # Malformed model output must not crash the batch.
                reward = 0.0
            rewards.append(reward)

        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if reward >= 1.0
                else "Environment: incorrect",
            }
            for reward in rewards
        ]

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=torch.tensor(rewards, dtype=torch.float32).cpu(),
            terminateds=torch.ones(len(message_log_batch), dtype=torch.bool).cpu(),
            answers=extracted_answers,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        # Mirror MathEnvironment: index 0 holds correctness for multi-reward batches.
        rewards = (
            batch["rewards"] if batch["rewards"].ndim == 1 else batch["rewards"][:, 0]
        )
        # Do not count sequences that did not terminate properly as correct.
        if "is_end" in batch:
            rewards = rewards * batch["is_end"]
        accuracy = (
            (rewards == 1.0).float().mean().item() if len(rewards) > 0 else 0.0
        )
        return batch, {"accuracy": accuracy}
