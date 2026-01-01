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

"""Fault injection utilities for testing fault tolerance mechanisms."""

import logging
import math
import random
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from nvidia_resiliency_ext.inprocess.tools.inject_fault import (
        Fault,
        clear_workload_exception,
        dispatch_fault_injection,
        maybe_raise_workload_exception,
    )

    NVRX_FAULT_INJECTION_AVAILABLE = True
except ImportError:
    Fault = None
    dispatch_fault_injection = None
    maybe_raise_workload_exception = None
    clear_workload_exception = None
    NVRX_FAULT_INJECTION_AVAILABLE = False


@dataclass
class FaultPlan:
    """Fault injection plan specifying when and where to inject a fault.

    Attributes:
        fault_type: Fault enum value (e.g., Fault.GPU_SLEEP.value)
        delay: Seconds until fault triggers after dispatch
        target_rank: Which worker rank to fault (0-indexed)
    """

    fault_type: int
    delay: float
    target_rank: int


class FaultInjector:
    """Controller-side fault injector that plans faults for testing FT mechanisms."""

    def __init__(self, config: dict):
        """Initialize fault injector from config.

        Args:
            config: Dictionary with keys:
                - enabled: bool
                - target_section: str (training|generation|environment|null)
                - target_rank: int
                - fault_type: str (GPU_SLEEP|WORKLOAD_EXC|SIGKILL|etc)
                - delay_seconds: float (fixed delay)
                - mtti_seconds: float (mean time to injection, exponential dist)
                - offset_seconds: float (minimum delay)
                - seed: int (for reproducibility)
        """
        self.enabled = config.get("enabled", False)
        self.target_section = config.get("target_section", None)
        self.target_rank = config.get("target_rank", 0)
        self.fault_type_str = config.get("fault_type", "GPU_SLEEP")
        self.fixed_delay = config.get("delay_seconds", None)
        self.mtti_seconds = config.get("mtti_seconds", None)
        self.offset_seconds = config.get("offset_seconds", 0.0)
        self.seed = config.get("seed", None)
        self._triggered = False

        self._rng = random.Random(self.seed) if self.seed is not None else random

        if self.enabled:
            if not NVRX_FAULT_INJECTION_AVAILABLE:
                logger.warning(
                    "Fault injection enabled but nvidia-resiliency-ext not available. Disabling."
                )
                self.enabled = False
            else:
                try:
                    self.fault_type = Fault[self.fault_type_str].value
                except KeyError:
                    logger.error(f"Unknown fault type: {self.fault_type_str}")
                    self.enabled = False

        if self.enabled and self.fixed_delay is None and self.mtti_seconds is None:
            logger.warning("No timing specified (delay_seconds or mtti_seconds). Disabling.")
            self.enabled = False

    def get_plan(self, section: str) -> Optional[FaultPlan]:
        """Get fault plan for a given section.

        Args:
            section: Name of the section (e.g., "training", "generation")

        Returns:
            FaultPlan if fault should be injected for this section, None otherwise.
        """
        if not self.enabled or self._triggered:
            return None
        if self.target_section and section != self.target_section:
            return None

        self._triggered = True
        delay = self._compute_delay()

        plan = FaultPlan(
            fault_type=self.fault_type,
            delay=delay,
            target_rank=self.target_rank,
        )

        logger.warning(
            f"FAULT INJECTION: Planning {Fault(self.fault_type).name} for rank {self.target_rank} "
            f"in section '{section}' with delay {delay:.2f}s"
        )
        return plan

    def _compute_delay(self) -> float:
        """Compute delay using fixed or MTTI-based exponential distribution."""
        if self.fixed_delay is not None:
            return self.fixed_delay
        if self.mtti_seconds is not None:
            lambda_inj = 1.0 / self.mtti_seconds
            return self.offset_seconds + (-math.log(1.0 - self._rng.random()) / lambda_inj)
        return 0.0


# --- Worker-side functions ---


def dispatch_fault_if_target(plan: Optional[FaultPlan], my_rank: int) -> None:
    """Worker-side: dispatch fault if this rank is the target.

    Call at start of section (e.g., train(), generate()) to schedule fault injection.

    Args:
        plan: FaultPlan from controller, or None if no fault injection
        my_rank: Rank of the current worker (0-indexed)
    """
    if plan is None:
        logger.debug(f"FAULT INJECTION: Rank {my_rank} received no fault plan")
        return
    if not NVRX_FAULT_INJECTION_AVAILABLE:
        logger.warning("Fault plan provided but nvidia-resiliency-ext not available.")
        return
    if plan.target_rank != my_rank:
        logger.info(
            f"FAULT INJECTION: Rank {my_rank} received plan for target_rank {plan.target_rank}, skipping (not the target)"
        )
        return

    clear_workload_exception()
    fault = Fault(plan.fault_type)

    logger.warning(
        f"FAULT INJECTION: Rank {my_rank} scheduling {fault.name} with delay {plan.delay:.2f}s"
    )
    dispatch_fault_injection(fault=fault, delay=plan.delay, callback=None)


def check_workload_exception() -> None:
    """Worker-side: check for pending WORKLOAD_EXC at safe points."""
    if not NVRX_FAULT_INJECTION_AVAILABLE:
        return
    maybe_raise_workload_exception()
