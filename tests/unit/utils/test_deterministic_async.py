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

"""Exercise logical scheduling with deliberately reordered worker completions."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from nemo_rl.utils.deterministic_async import AsyncRolloutPlan, AsyncRolloutProgress


def register(progress, target):
    progress.register(
        target=target,
        version=max(0, target - 1),
        ordinals=progress.plan.ordinals(target),
    )


def finish_generation(progress, target):
    for sample in reversed(
        range(progress.plan.prompts_per_step * progress.plan.generations_per_prompt)
    ):
        progress.generation_complete(target, sample)


def finish_rewards(progress, target):
    for ordinal in reversed(progress.plan.ordinals(target)):
        progress.group_buffered(target, ordinal)


def test_bootstrap_lookahead_and_final_target():
    plan = AsyncRolloutPlan(4, 2, 3)
    assert [plan.targets(v) for v in range(5)] == [[0, 1], [2], [3], [], []]
    assert plan.ordinals(2) == [4, 5]
    assert AsyncRolloutPlan(1, 2, 3).targets(0) == [0]
    with pytest.raises(ValueError):
        plan.ordinals(4)


@pytest.mark.parametrize(
    "ordinals,version", [([1, 0], 0), ([0, 0], 0), ([2, 3], 0), ([0, 1], 1), ([0], 0)]
)
def test_rejects_wrong_membership_or_version(ordinals, version):
    with pytest.raises(ValueError, match="manifest"):
        AsyncRolloutPlan(4, 2, 3).validate_batch(
            target=0, version=version, ordinals=ordinals
        )


def test_refit_waits_for_unadmitted_lookahead_but_not_terminal_rewards():
    progress = AsyncRolloutProgress(AsyncRolloutPlan(3, 2, 2))
    register(progress, 0)
    finish_generation(progress, 0)
    finish_rewards(progress, 0)
    started = Event()

    def refit():
        started.set()
        progress.wait_for_refit(0)

    with ThreadPoolExecutor(2) as executor:
        future = executor.submit(refit)
        assert started.wait(2)
        assert not future.done()
        register(progress, 1)
        # Three of four requests are insufficient to release the old weights.
        for sample in (2, 0, 3):
            progress.generation_complete(1, sample)
        assert not future.done()
        progress.generation_complete(1, 1)
        future.result(timeout=2)
        # Refit is safe, but training must still wait for the reward payloads.
        batch = executor.submit(progress.wait_for_batch, 1)
        progress.group_buffered(1, 3)
        assert not batch.done()
        progress.group_buffered(1, 2)
        batch.result(timeout=2)


def test_out_of_order_bootstrap_cannot_release_unfinished_current_batch():
    progress = AsyncRolloutProgress(AsyncRolloutPlan(2, 1, 1))
    register(progress, 0)
    register(progress, 1)
    finish_generation(progress, 1)
    finish_rewards(progress, 1)
    with ThreadPoolExecutor(2) as executor:
        refit = executor.submit(progress.wait_for_refit, 0)
        batch = executor.submit(progress.wait_for_batch, 0)
        assert not refit.done() and not batch.done()
        finish_generation(progress, 0)
        finish_rewards(progress, 0)
        refit.result(timeout=2)
        batch.result(timeout=2)


@pytest.mark.parametrize("wait_for", ["batch", "refit"])
def test_failure_wakes_waiter_without_accepting_partial_work(wait_for):
    progress = AsyncRolloutProgress(AsyncRolloutPlan(2, 1, 1))
    register(progress, 0)
    method = progress.wait_for_batch if wait_for == "batch" else progress.wait_for_refit
    with ThreadPoolExecutor(1) as executor:
        waiter = executor.submit(method, 0)
        progress.fail("backend lost; cleanup unconfirmed")
        with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
            waiter.result(timeout=2)


def test_long_run_keeps_only_bounded_progress_and_rejects_duplicate_dispatch():
    progress = AsyncRolloutProgress(AsyncRolloutPlan(100, 2, 3))
    for version in range(100):
        for target in progress.plan.targets(version):
            register(progress, target)
            finish_generation(progress, target)
            finish_rewards(progress, target)
        progress.wait_for_refit(version)
        progress.wait_for_batch(version)
        with pytest.raises(ValueError, match="already dispatched"):
            register(progress, version)
    assert progress._generating == progress._buffering == {}
    assert progress._generated == progress._buffered == set()


def test_single_step_refit_never_waits_for_unused_lookahead():
    progress = AsyncRolloutProgress(AsyncRolloutPlan(1, 1, 1))
    register(progress, 0)
    finish_generation(progress, 0)
    progress.wait_for_refit(0)
