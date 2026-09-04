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

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm.v1.core.sched.async_scheduler import AsyncScheduler  # noqa: E402

from nemo_rl.models.generation.vllm.lfs.engine_schedulers import (  # noqa: E402
    OracleLengthRequestQueue,
    OracleLengthScheduler,
    ProbeLfsRequestQueue,
    ProbeLfsScheduler,
)


def request(group_id: int, request_id: str, arrival_time: float) -> SimpleNamespace:
    return SimpleNamespace(
        priority=group_id,
        request_id=request_id,
        arrival_time=arrival_time,
    )


def test_scheduler_preserves_vllm_async_scheduling() -> None:
    assert issubclass(ProbeLfsScheduler, AsyncScheduler)
    assert issubclass(OracleLengthScheduler, AsyncScheduler)


def test_oracle_queue_uses_exact_length_priority() -> None:
    queue = OracleLengthRequestQueue()
    queue.add_request(request(10, "short", 0.0))
    queue.add_request(request(100, "long-first", 1.0))
    queue.add_request(request(100, "long-second", 2.0))
    queue.add_request(request(50, "medium", 3.0))

    assert [queue.pop_request().request_id for _ in range(4)] == [
        "long-first",
        "long-second",
        "medium",
        "short",
    ]


def test_queue_launches_one_probe_from_every_group_first() -> None:
    queue = ProbeLfsRequestQueue({})
    for group_id in range(3):
        queue.add_request(request(group_id, f"{group_id}-0", group_id * 2.0))
        queue.add_request(request(group_id, f"{group_id}-1", group_id * 2.0 + 1.0))

    probes = [queue.pop_request() for _ in range(3)]

    assert [probe.priority for probe in probes] == [0, 1, 2]
    assert [probe.request_id for probe in probes] == ["0-0", "1-0", "2-0"]


def test_queue_round_robins_when_probes_do_not_fill_first_batch() -> None:
    queue = ProbeLfsRequestQueue({})
    for group_id in range(3):
        for sample_id in range(3):
            queue.add_request(
                request(
                    group_id,
                    f"{group_id}-{sample_id}",
                    group_id * 3.0 + sample_id,
                )
            )

    first_batch = [queue.pop_request() for _ in range(8)]

    assert [item.priority for item in first_batch] == [0, 1, 2, 0, 1, 2, 0, 1]
    assert [item.request_id for item in first_batch] == [
        "0-0",
        "1-0",
        "2-0",
        "0-1",
        "1-1",
        "2-1",
        "0-2",
        "1-2",
    ]


def test_queue_uses_longest_group_first_after_probe_results() -> None:
    estimates: dict[int, int] = {}
    queue = ProbeLfsRequestQueue(estimates)
    for group_id in range(3):
        queue.add_request(request(group_id, f"{group_id}-0", group_id * 2.0))
        queue.add_request(request(group_id, f"{group_id}-1", group_id * 2.0 + 1.0))

    for _ in range(3):
        queue.pop_request()
    estimates.update({0: 10, 1: 100, 2: 50})

    assert [queue.pop_request().priority for _ in range(3)] == [1, 2, 0]


def test_prepend_restores_an_unscheduled_probe() -> None:
    queue = ProbeLfsRequestQueue({})
    first_probe = request(0, "0-0", 0.0)
    queue.add_request(first_probe)
    queue.add_request(request(0, "0-1", 1.0))
    queue.add_request(request(1, "1-0", 2.0))

    assert queue.pop_request() is first_probe
    queue.prepend_request(first_probe)

    assert queue.pop_request() is first_probe


def test_prepend_restores_round_robin_position() -> None:
    queue = ProbeLfsRequestQueue({})
    for group_id in range(2):
        queue.add_request(request(group_id, f"{group_id}-0", group_id * 2.0))
        queue.add_request(request(group_id, f"{group_id}-1", group_id * 2.0 + 1.0))

    queue.pop_request()
    queue.pop_request()
    extra_request = queue.pop_request()
    queue.prepend_request(extra_request)

    assert queue.pop_request() is extra_request
