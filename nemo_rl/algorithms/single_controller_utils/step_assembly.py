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

"""Step-assembly accounting for the SingleController train pump.

Owns the cursor over the step the train pump is assembling and charges a
permanently lost prompt group to the step that will close short for it. Kept out
of ``SingleControllerActor`` so the assembling cursor, its step-assembly record
and the shortfall-crediting rules live in one place; the controller drives it
through ``begin_step``/``finish_step`` and the per-loss ``credit_shortfall``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nemo_rl.algorithms.async_utils.staleness_sampler import PromptGroupSampler
    from nemo_rl.algorithms.rollout_checkpointing.state import RolloutRuntimeState
    from nemo_rl.algorithms.single_controller import SingleControllerActor


@dataclass(frozen=True)
class ShortfallCredit:
    target_step: Optional[int]
    charged_step: int
    allow_short: bool


@dataclass
class StepAssembly:
    version: int
    selected_groups: int
    selection_credits: Optional[list[ShortfallCredit]]


class StepAssemblyAccountant:
    """Track the step being assembled and charge lost groups to it.

    The controller injects itself as the backref; the sampler, the shared
    ``RolloutRuntimeState`` (read for its current ``batch_shortfall`` /
    ``steps_allowed_short`` / ``shortfall_reconciliation_pending``) and the
    per-step prompt-group target are read live off it rather than cached, so
    the accountant tracks the controller's current state. The cursor starts
    from the restored trainer version so step ``0`` stays valid after a restore.
    """

    def __init__(
        self,
        *,
        controller: "SingleControllerActor",
        initial_version: int,
    ) -> None:
        self._controller = controller
        self._assembling_version = initial_version
        self._step_assembly: Optional[StepAssembly] = None

    @property
    def _sampler(self) -> "PromptGroupSampler":
        return self._controller._sampler

    @property
    def _runtime(self) -> "RolloutRuntimeState":
        return self._controller._rollout_runtime

    @property
    def _num_prompts_per_step(self) -> int:
        return self._controller._algo_cfg.num_prompts_per_step

    @property
    def assembling_version(self) -> int:
        """The step the train pump is selecting groups for, or will select next."""
        return self._assembling_version

    @property
    def step_assembly(self) -> Optional[StepAssembly]:
        """The open step's assembly record, or ``None`` between steps."""
        return self._step_assembly

    def begin_step(self, version: int) -> None:
        """Open a fresh step's assembly record at ``version``."""
        self._step_assembly = StepAssembly(version, 0, None)

    def finish_step(self, version_during_step: int) -> None:
        """Close the step: a group lost from here on counts toward the next."""
        self._assembling_version = max(
            self._assembling_version, version_during_step + 1
        )
        self._step_assembly = None

    def accounting_step(self, target_step: Optional[int]) -> int:
        """The step a group counts toward: its own, or the one being assembled."""
        return self._assembling_version if target_step is None else target_step

    def _charge_step_for_lost_group(self, target_step: Optional[int]) -> Optional[int]:
        """The step a permanently lost group is charged to, or None for no step."""
        return self._sampler.charge_step_for_lost_group(
            target_step=target_step, assembling_version=self._assembling_version
        )

    def advance_assembling_version(self) -> None:
        """Move past steps whose selected groups and losses fill their batch."""
        assembly = self._step_assembly
        while True:
            selected = (
                assembly.selected_groups
                if assembly is not None and assembly.version == self._assembling_version
                else 0
            )
            lost = self._runtime.batch_shortfall.get(self._assembling_version, 0)
            if selected + lost < self._num_prompts_per_step:
                return
            self._assembling_version += 1

    def reconcile_selection_credits(
        self, assembling_version: int, credits: list[ShortfallCredit]
    ) -> None:
        """Reapply only losses that raced an awaited selection, using its result."""
        for credit in credits:
            remaining = self._runtime.batch_shortfall[credit.charged_step] - 1
            if remaining:
                self._runtime.batch_shortfall[credit.charged_step] = remaining
            else:
                del self._runtime.batch_shortfall[credit.charged_step]
        self._assembling_version = assembling_version
        self.advance_assembling_version()
        for credit in credits:
            self.credit_shortfall(credit.target_step, allow_short=credit.allow_short)
        self._runtime.shortfall_reconciliation_pending = False

    def credit_shortfall(
        self, target_step: Optional[int], *, allow_short: bool = False
    ) -> None:
        """Record that a step will never receive a group it is waiting for."""
        self.advance_assembling_version()
        charged_step = self._charge_step_for_lost_group(target_step)
        if charged_step is None:
            return
        self._runtime.batch_shortfall[charged_step] = (
            self._runtime.batch_shortfall.get(charged_step, 0) + 1
        )
        if allow_short:
            self._runtime.steps_allowed_short.add(charged_step)
        assembly = self._step_assembly
        if (
            self._sampler.requires_shortfall_reconciliation
            and assembly is not None
            and assembly.selection_credits is not None
        ):
            assembly.selection_credits.append(
                ShortfallCredit(target_step, charged_step, allow_short)
            )
            self._runtime.shortfall_reconciliation_pending = True
        self.advance_assembling_version()
        print(
            f"  target_step={charged_step} is one group short "
            f"({self._runtime.batch_shortfall[charged_step]} total); the train pump "
            "will close that step early",
            flush=True,
        )
