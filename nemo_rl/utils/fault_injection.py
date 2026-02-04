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
from enum import Enum
from functools import wraps
from typing import Any, Callable, Literal, Optional, TypedDict, TypeVar

from nvidia_resiliency_ext.inprocess.tools.inject_fault import (
    Fault,
    clear_workload_exception,
    dispatch_fault_injection,
    maybe_raise_workload_exception,
)

logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., Any])

FaultTypeLiteral = Literal[
    "GPU_ERROR",
    "GPU_SLEEP",
    "WORKLOAD_EXC",
    "ASYNC_EXC",
    "SIGNAL_EXC",
    "OS_ABORT",
    "LOCK_GIL",
    "SEGFAULT",
    "SIGINT",
    "SIGKILL",
    "SIGTERM",
    "SIGSTOP",
]

SectionLiteral = Literal["training", "generation", "environment"]


@dataclass
class FaultInjectionConfig:
    """Configuration for fault injection testing."""

    enabled: bool = False
    target_section: SectionLiteral | None = (
        None  # None = inject in first section that runs
    )
    target_rank: int = 0
    fault_type: FaultTypeLiteral = "GPU_SLEEP"
    delay_seconds: float | None = None  # Fixed delay before fault
    mtti_seconds: float | None = (
        None  # Mean time to injection (exponential distribution)
    )
    offset_seconds: float = 0.0  # Minimum delay when using mtti
    seed: int | None = None  # For reproducibility


class FaultToleranceConfig(TypedDict, total=False):
    """Configuration for fault tolerance features."""

    fault_injection: FaultInjectionConfig


class Section(str, Enum):
    """Target sections for fault injection.

    These correspond to different phases of the training loop where
    faults can be injected for testing fault tolerance mechanisms.
    """

    TRAINING = "training"
    GENERATION = "generation"
    ENVIRONMENT = "environment"


@dataclass
class FaultPlan:
    """Fault injection plan specifying when and where to inject a fault.

    Attributes:
        fault_type: Fault enum from nvidia_resiliency_ext (e.g., Fault.GPU_SLEEP)
        delay: Seconds until fault triggers after dispatch
        target_rank: Which worker rank to fault (0-indexed)
    """

    fault_type: Fault
    delay: float
    target_rank: int


class FaultInjector:
    """Controller-side fault injector that plans faults for testing FT mechanisms."""

    def __init__(self, config: FaultInjectionConfig):
        """Initialize fault injector from config.

        Args:
            config: FaultInjectionConfig dataclass with fault injection settings.
        """
        self.enabled = config.enabled
        self.target_rank = config.target_rank
        self.fixed_delay = config.delay_seconds
        self.mtti_seconds = config.mtti_seconds
        self.offset_seconds = config.offset_seconds
        self.seed = config.seed
        self._triggered = False

        self._rng = random.Random(self.seed) if self.seed is not None else random

        # Convert typed literals to enums (types guarantee valid values)
        self.target_section = (
            Section(config.target_section) if config.target_section else None
        )
        self.fault_type = Fault[config.fault_type] if self.enabled else None

        if self.enabled and self.fixed_delay is None and self.mtti_seconds is None:
            raise ValueError(
                "Fault injection enabled but no timing specified. "
                "Set either delay_seconds or mtti_seconds."
            )

    def get_plan(self, section: Section) -> Optional[FaultPlan]:
        """Get fault plan for a given section.

        Args:
            section: Section enum value (e.g., Section.TRAINING)

        Returns:
            FaultPlan if fault should be injected for this section, None otherwise.
        """
        if not self.enabled or self._triggered:
            return None
        if self.target_section is not None and section != self.target_section:
            return None

        self._triggered = True
        delay = self._compute_delay()

        assert self.fault_type is not None  # Guaranteed by enabled check
        plan = FaultPlan(
            fault_type=self.fault_type,
            delay=delay,
            target_rank=self.target_rank,
        )

        logger.warning(
            f"FAULT INJECTION: Planning {self.fault_type.name} for rank {self.target_rank} "
            f"in section '{section.value}' with delay {delay:.2f}s"
        )
        return plan

    def _compute_delay(self) -> float:
        """Compute delay using fixed or MTTI-based exponential distribution."""
        if self.fixed_delay is not None:
            return self.fixed_delay
        if self.mtti_seconds is not None:
            lambda_inj = 1.0 / self.mtti_seconds
            return self.offset_seconds + (
                -math.log(1.0 - self._rng.random()) / lambda_inj
            )
        return 0.0


# =============================================================================
# Global Fault Injector Registry
# =============================================================================

_fault_injector: Optional[FaultInjector] = None


def set_global_fault_injector(injector: Optional[FaultInjector]) -> None:
    """Set the global fault injector instance.

    Call this once at the start of training (e.g., in grpo_train) to enable
    fault injection across all workers.

    Args:
        injector: FaultInjector instance, or None to disable.
    """
    global _fault_injector
    _fault_injector = injector


def get_global_fault_injector() -> Optional[FaultInjector]:
    """Get the global fault injector instance."""
    return _fault_injector


# =============================================================================
# Decorator for Fault Injection
# =============================================================================


def with_fault_injection(section: Section) -> Callable[[F], F]:
    """Decorator that injects faults at the start of a method.

    Usage:
        @with_fault_injection(Section.TRAINING)
        def train(self, data, loss_fn):
            ...

    The decorated method's class must have a `rank` attribute (self.rank).

    Args:
        section: Section enum for fault targeting (e.g., Section.TRAINING)

    Returns:
        Decorated function that checks for fault injection before execution.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Get fault plan for this section from global injector
            if _fault_injector:
                plan = _fault_injector.get_plan(section)
                dispatch_fault_if_target(plan, self.rank)

            # Run the actual method
            result = func(self, *args, **kwargs)

            # Check for workload exceptions at safe point
            check_workload_exception()

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# =============================================================================
# Worker-side Helper Functions
# =============================================================================


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
    if plan.target_rank != my_rank:
        logger.info(
            f"FAULT INJECTION: Rank {my_rank} received plan for target_rank {plan.target_rank}, "
            "skipping (not the target)"
        )
        return

    clear_workload_exception()

    logger.warning(
        f"FAULT INJECTION: Rank {my_rank} scheduling {plan.fault_type.name} "
        f"with delay {plan.delay:.2f}s"
    )
    dispatch_fault_injection(fault=plan.fault_type, delay=plan.delay, callback=None)


def check_workload_exception() -> None:
    """Worker-side: check for pending WORKLOAD_EXC at safe points."""
    maybe_raise_workload_exception()
