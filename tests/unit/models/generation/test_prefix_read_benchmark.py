# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Verify read benchmark accounting without a TQ deployment."""

import asyncio

import pytest

from tools.benchmark_tq_prefix_reads import read_workload, summarize


pytestmark = pytest.mark.nemo_gym


@pytest.mark.parametrize("mode", ["unbatched", "coalesced"])
def test_read_workload_counts_and_verifies_all_keys(mode: str) -> None:
    chains = [[f"{i}-a", f"{i}-b"] for i in range(16)]

    def verify(keys: list[str], rows: list[str]) -> None:
        assert rows == keys

    result = asyncio.run(
        read_workload(
            lambda keys: keys,
            verify,
            chains,
            mode=mode,
            concurrency=16,
            arrival_interval_ms=0,
            model_max_len=16,
            batch_size=256,
            batch_max_tokens=1024,
            coalesce_ms=10,
        )
    )
    assert result["verified_requests"] == 16
    assert result["verified_keys"] == 32
    samples = result["samples"]
    assert sum(samples["keys_per_get"]) == 32
    assert len(samples["request_seconds"]) == 16
    assert len(samples["queue_seconds"]) == 32
    assert all(value >= 0 for value in samples["queue_seconds"])
    assert len(samples["get_seconds"]) == (16 if mode == "unbatched" else 1)


def test_read_workload_staggering_and_token_limit() -> None:
    result = asyncio.run(
        read_workload(
            lambda keys: keys,
            lambda keys, rows: None,
            [[str(i)] for i in range(4)],
            mode="coalesced",
            concurrency=2,
            arrival_interval_ms=5,
            model_max_len=16,
            batch_size=256,
            batch_max_tokens=16,
            coalesce_ms=1,
        )
    )
    assert result["samples"]["keys_per_get"] == [1, 1, 1, 1]
    assert result["wall_seconds"] >= 0.015


def test_summary_nearest_rank() -> None:
    assert summarize([]) == {}
    assert summarize([1, 2, 3, 4]) == {
        "mean": 2.5,
        "p50": 2,
        "p95": 4,
        "p99": 4,
        "max": 4,
    }
