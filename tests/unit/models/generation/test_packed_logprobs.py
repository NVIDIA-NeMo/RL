# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Packed capture value, ownership and admission-gating checks."""

import math
from array import array
from dataclasses import dataclass

import numpy as np
import pytest

from nemo_rl.models.generation.vllm.packed_logprobs import PackedSelectedLogprobs
from nemo_rl.models.generation.vllm.utils import extract_generation_logprobs


@dataclass
class Logprob:
    logprob: float
    rank: int
    decoded_token: str | None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("count", [0, 1, 8, 257])
def test_append_preserves_values_and_frozen_reads(dtype: object, count: int) -> None:
    history = PackedSelectedLogprobs(Logprob)
    expected_ids, expected_values, expected_ranks = [], [], []
    cumulative = 0.0
    for size in [3, count, 2]:
        ids = np.arange(size * 2, dtype=np.int64).reshape(size, 2)[:, :1]
        values = np.linspace(-100.0, -0.001, size * 2, dtype=dtype).reshape(size, 2)[
            :, :1
        ]
        ranks = np.arange(size * 2, dtype=np.int64)[::2] + 1
        before = len(history)
        history.append_arrays(ids, values, ranks)
        cumulative = history.cumulative_after(cumulative, start=before)
        expected_ids.extend(ids[:, 0].tolist())
        expected_values.extend(values[:, 0].tolist())
        expected_ranks.extend(ranks.tolist())
        reference = 0.0
        for value in expected_values:
            reference += value
        assert cumulative == reference
        assert (
            extract_generation_logprobs(expected_ids, history, start=0)
            == expected_values
        )
        assert (
            extract_generation_logprobs(expected_ids[:before], history, start=0)
            == expected_values[:before]
        )
        assert [next(iter(item)) for item in history] == expected_ids
        assert [next(iter(item.values())).rank for item in history] == expected_ranks
    for selection in [
        slice(None),
        slice(1, 4),
        slice(None, None, -1),
        slice(None, None, 2),
    ]:
        assert list(history[selection]) == list(history)[selection]
    assert history[-1] == list(history)[-1]
    with pytest.raises(IndexError):
        history[len(history)]


def test_owned_buffers_and_special_floats() -> None:
    history = PackedSelectedLogprobs(Logprob)
    ids = np.array([[1], [2], [3], [4]])
    values = np.array([[-0.0], [-np.inf], [np.inf], [np.nan]])
    ranks = np.arange(4)
    history.append_arrays(ids, values, ranks)
    ids[:] = 99
    values[:] = 99
    ranks[:] = 99
    assert math.copysign(1, history[0][1].logprob) == -1
    assert history[1][2].logprob == -math.inf
    assert history[2][3].logprob == math.inf
    assert math.isnan(history[3][4].logprob)
    assert math.isnan(history.cumulative_after(0.0, start=0))
    assert history[3][4].rank == 3
    with pytest.raises(RuntimeError, match="selected token"):
        extract_generation_logprobs([9], history, start=0)


@pytest.mark.parametrize("bad", ["shape", "ids", "ranks", "overflow"])
def test_bad_append_preserves_history(bad: str) -> None:
    history = PackedSelectedLogprobs(Logprob)
    history.append_arrays(np.array([[1]]), np.array([[-0.1]]), np.array([1]))
    before = list(history)
    ids, values, ranks = np.array([[2]]), np.array([[-0.2]]), np.array([2])
    if bad == "shape":
        values = np.array([[-0.2, -0.3]])
    elif bad == "ids":
        ids = np.array([[2.5]])
    elif bad == "ranks":
        ranks = np.array([2.5])
    else:
        ids = np.array([[2**63]], dtype=np.uint64)
    with pytest.raises((ValueError, OverflowError)):
        history.append_arrays(ids, values, ranks)
    assert list(history) == before


def test_allocation_failure_rolls_back() -> None:
    class FailingValues(array):
        def append(self, value: float) -> None:
            raise MemoryError("injected")

    history = PackedSelectedLogprobs(Logprob)
    history._values = FailingValues("d")
    with pytest.raises(MemoryError, match="injected"):
        history.append_arrays(np.array([[1]]), np.array([[-0.1]]), np.array([1]))
    assert (
        len(history)
        == len(history._ids)
        == len(history._values)
        == len(history._ranks)
        == 0
    )
