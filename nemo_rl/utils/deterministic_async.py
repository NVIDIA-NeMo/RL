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

"""Logical rollout plans and completion notifications for deterministic async GRPO."""

from dataclasses import dataclass
from threading import Condition


@dataclass(frozen=True)
class AsyncRolloutPlan:
    """Fix batch membership and policy versions independently of worker timing.

    Step zero and the first lookahead use W0. Every subsequent batch Bk
    uses W(k-1). Prompt ordinals refer to occurrences in the input stream,
    so repeating a dataset across epochs does not reuse a sampling stream.
    """

    num_steps: int
    prompts_per_step: int
    generations_per_prompt: int

    def __post_init__(self) -> None:
        if min(self.num_steps, self.prompts_per_step, self.generations_per_prompt) <= 0:
            raise ValueError(
                "A rollout plan requires positive steps, prompts, and generations"
            )

    def targets(self, version: int) -> list[int]:
        if version < 0:
            raise ValueError("Policy version must be nonnegative")
        return [
            target
            for target in ([0, 1] if version == 0 else [version + 1])
            if target < self.num_steps
        ]

    def ordinals(self, target: int) -> list[int]:
        if not 0 <= target < self.num_steps:
            raise ValueError(f"Target {target} is outside the rollout plan")
        start = target * self.prompts_per_step
        return list(range(start, start + self.prompts_per_step))

    def validate_batch(self, *, target: int, version: int, ordinals: list[int]) -> None:
        if ordinals != self.ordinals(target) or version != max(0, target - 1):
            raise ValueError(
                f"Batch for target {target} differs from its fixed prompt/policy manifest"
            )


class AsyncRolloutProgress:
    """Track generation and reward completion separately, without polling.

    Called directly by collector threads, never by a self-RPC to the waiting
    Ray actor. Only unfinished targets and out-of-order completions are retained.
    A failure wakes both types of waiter and never converts missing work into
    an accepted batch. No cancellation is treated as generation completion.
    """

    def __init__(self, plan: AsyncRolloutPlan) -> None:
        self.plan = plan
        self._condition = Condition()
        self._generating: dict[int, set[int]] = {}
        self._buffering: dict[int, set[int]] = {}
        self._generated: set[int] = set()
        self._buffered: set[int] = set()
        self._generated_through = -1
        self._buffered_through = -1
        self._error: str | None = None

    def register(self, *, target: int, version: int, ordinals: list[int]) -> None:
        self.plan.validate_batch(target=target, version=version, ordinals=ordinals)
        with self._condition:
            if (
                target <= self._generated_through
                or target in self._generated
                or target in self._generating
            ):
                raise ValueError(f"Target {target} was already dispatched")
            self._generating[target] = set(
                range(self.plan.prompts_per_step * self.plan.generations_per_prompt)
            )
            self._buffering[target] = set(ordinals)

    def generation_complete(self, target: int, sample_index: int) -> None:
        """Release a sample only after it can no longer issue generation calls."""
        with self._condition:
            self._generating[target].remove(sample_index)
            if not self._generating[target]:
                del self._generating[target]
                self._generated.add(target)
                while self._generated_through + 1 in self._generated:
                    self._generated_through += 1
                    self._generated.remove(self._generated_through)
            self._condition.notify_all()

    def group_buffered(self, target: int, ordinal: int) -> None:
        """Mark a group ready only after replay acknowledges storing its rewards."""
        with self._condition:
            self._buffering[target].remove(ordinal)
            if not self._buffering[target]:
                del self._buffering[target]
                self._buffered.add(target)
                while self._buffered_through + 1 in self._buffered:
                    self._buffered_through += 1
                    self._buffered.remove(self._buffered_through)
            self._condition.notify_all()

    def fail(self, message: str) -> None:
        with self._condition:
            if self._error is None:
                self._error = message
            self._condition.notify_all()

    def wait_for_batch(self, target: int) -> None:
        self.plan.ordinals(target)
        with self._condition:
            self._condition.wait_for(
                lambda: self._error is not None or self._buffered_through >= target
            )
            if self._error is not None:
                raise RuntimeError(self._error)

    def wait_for_refit(self, version: int) -> None:
        # Includes not-yet-admitted lookahead, but never requests B(num_steps).
        last_target = min(version + 1, self.plan.num_steps - 1)
        with self._condition:
            self._condition.wait_for(
                lambda: self._error is not None
                or self._generated_through >= last_target
            )
            if self._error is not None:
                raise RuntimeError(self._error)
