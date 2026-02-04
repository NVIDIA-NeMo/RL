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

"""Tests for fault injection utilities."""

from unittest.mock import patch

import pytest
from nvidia_resiliency_ext.inprocess.tools.inject_fault import Fault

from nemo_rl.utils.fault_injection import (
    FaultInjectionConfig,
    FaultInjector,
    FaultPlan,
    Section,
    dispatch_fault_if_target,
    set_global_fault_injector,
    with_fault_injection,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_global_injector():
    """Clear global injector before and after each test."""
    set_global_fault_injector(None)
    yield
    set_global_fault_injector(None)


@pytest.fixture
def base_config():
    """Base config for enabled fault injector."""
    return FaultInjectionConfig(
        enabled=True,
        target_section="training",
        target_rank=0,
        fault_type="GPU_SLEEP",
        delay_seconds=5.0,
    )


@pytest.fixture
def mock_worker_class():
    """Reusable mock worker class for decorator tests."""

    class MockWorker:
        def __init__(self, rank=0):
            self.rank = rank

        @with_fault_injection(Section.TRAINING)
        def train(self, data):
            return {"result": "success"}

    return MockWorker


# =============================================================================
# FaultInjector Tests
# =============================================================================


def test_injector_disabled_when_not_enabled():
    """FaultInjector returns no plan when disabled."""
    config = FaultInjectionConfig(enabled=False)
    injector = FaultInjector(config)
    assert injector.enabled is False
    assert injector.get_plan(Section.TRAINING) is None


def test_injector_disabled_when_no_timing():
    """FaultInjector disables when no timing is specified."""
    config = FaultInjectionConfig(enabled=True, fault_type="GPU_SLEEP")
    injector = FaultInjector(config)
    assert injector.enabled is False
    assert injector.get_plan(Section.TRAINING) is None


def test_injector_section_targeting_and_oneshot(base_config):
    """get_plan respects section targeting, null-targeting, and one-shot behavior."""
    # Test section matching
    injector = FaultInjector(base_config)
    plan = injector.get_plan(Section.TRAINING)
    assert plan is not None
    assert plan.fault_type == Fault.GPU_SLEEP
    assert plan.target_rank == 0
    assert plan.delay == 5.0

    # Test non-matching section
    assert injector.get_plan(Section.GENERATION) is None

    # Test one-shot: plan already triggered above, should be None now
    injector2 = FaultInjector(base_config)
    injector2.get_plan(Section.TRAINING)
    assert injector2.get_plan(Section.TRAINING) is None

    # Test null targeting (any section)
    from dataclasses import replace

    config_any_section = replace(base_config, target_section=None)
    injector3 = FaultInjector(config_any_section)
    assert injector3.get_plan(Section.GENERATION) is not None


def test_injector_mtti_and_seed():
    """MTTI uses exponential distribution; seed provides determinism."""
    config = FaultInjectionConfig(
        enabled=True,
        fault_type="GPU_SLEEP",
        mtti_seconds=100.0,
        offset_seconds=5.0,
        seed=42,
    )

    plan1 = FaultInjector(config).get_plan(Section.TRAINING)
    plan2 = FaultInjector(config).get_plan(Section.TRAINING)

    assert plan1.delay >= 5.0  # Respects offset
    assert plan1.delay == plan2.delay  # Seed makes it deterministic


# =============================================================================
# Worker-Side Function Tests
# =============================================================================


@pytest.mark.parametrize("my_rank,should_dispatch", [(0, True), (1, False)])
def test_dispatch_fault_rank_targeting(my_rank, should_dispatch):
    """dispatch_fault_if_target only dispatches for target rank."""
    plan = FaultPlan(fault_type=Fault.GPU_SLEEP, delay=5.0, target_rank=0)

    with patch(
        "nemo_rl.utils.fault_injection.dispatch_fault_injection"
    ) as mock_dispatch:
        with patch("nemo_rl.utils.fault_injection.clear_workload_exception"):
            dispatch_fault_if_target(plan, my_rank=my_rank)

            if should_dispatch:
                mock_dispatch.assert_called_once_with(
                    fault=Fault.GPU_SLEEP, delay=5.0, callback=None
                )
            else:
                mock_dispatch.assert_not_called()


# =============================================================================
# Decorator Tests
# =============================================================================


def test_decorator_noop_and_exception_check(mock_worker_class):
    """Decorator is no-op when disabled and always checks workload exceptions."""
    worker = mock_worker_class()

    with patch(
        "nemo_rl.utils.fault_injection.maybe_raise_workload_exception"
    ) as mock_check:
        result = worker.train({"data": "test"})
        mock_check.assert_called_once()

    assert result == {"result": "success"}


def test_decorator_dispatch_when_matching(base_config, mock_worker_class):
    """Decorator dispatches fault when section and rank match."""
    set_global_fault_injector(FaultInjector(base_config))
    worker = mock_worker_class(rank=0)

    with patch(
        "nemo_rl.utils.fault_injection.dispatch_fault_injection"
    ) as mock_dispatch:
        with patch("nemo_rl.utils.fault_injection.clear_workload_exception"):
            with patch("nemo_rl.utils.fault_injection.maybe_raise_workload_exception"):
                result = worker.train({"data": "test"})
                mock_dispatch.assert_called_once()

    assert result == {"result": "success"}


@pytest.mark.parametrize(
    "target_section,my_rank",
    [
        ("generation", 0),  # Wrong section
        ("training", 1),  # Wrong rank
    ],
)
def test_decorator_skip_when_not_matching(target_section, my_rank, mock_worker_class):
    """Decorator skips dispatch when section or rank doesn't match."""
    config = FaultInjectionConfig(
        enabled=True,
        target_section=target_section,
        target_rank=0,
        fault_type="GPU_SLEEP",
        delay_seconds=5.0,
    )
    set_global_fault_injector(FaultInjector(config))
    worker = mock_worker_class(rank=my_rank)

    with patch(
        "nemo_rl.utils.fault_injection.dispatch_fault_injection"
    ) as mock_dispatch:
        with patch("nemo_rl.utils.fault_injection.maybe_raise_workload_exception"):
            result = worker.train({"data": "test"})
            mock_dispatch.assert_not_called()

    assert result == {"result": "success"}
