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

import pytest
import torch

from nemo_rl.models.policy.optimizer_offload import (
    copy_to_reusable_cpu_buffer,
    move_optimizer_state,
)


def test_reusable_optimizer_cpu_buffer_preserves_storage_and_values() -> None:
    destination = None
    destination_ptr = None
    for value in (1.0, 2.0, 3.0):
        source = torch.full((3, 4), value, dtype=torch.float32)
        destination = copy_to_reusable_cpu_buffer(source, destination)
        if destination_ptr is None:
            destination_ptr = destination.data_ptr()
        assert destination.data_ptr() == destination_ptr
        torch.testing.assert_close(destination, source)


@pytest.mark.parametrize(
    "source",
    [
        torch.ones(2, 3, dtype=torch.float32),
        torch.ones(3, 2, dtype=torch.float64),
        torch.ones(2, 3, dtype=torch.float32).t(),
    ],
)
def test_reusable_optimizer_cpu_buffer_reallocates_for_incompatible_layout(
    source: torch.Tensor,
) -> None:
    old_destination = torch.empty_strided(
        (3, 2), (2, 1), dtype=torch.float32, device="cpu"
    )

    destination = copy_to_reusable_cpu_buffer(source, old_destination)

    assert destination is not old_destination
    assert destination.shape == source.shape
    assert destination.dtype == source.dtype
    assert destination.stride() == source.stride()
    torch.testing.assert_close(destination, source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_move_optimizer_state_reuses_pageable_cpu_buffer() -> None:
    state_key = object()
    optimizer_state = {
        state_key: {"exp_avg": torch.arange(12, device="cuda", dtype=torch.float32)}
    }
    cache: dict[tuple[object, str], torch.Tensor] = {}

    move_optimizer_state(
        optimizer_state,
        device="cpu",
        reuse_cpu_buffers=True,
        cpu_buffer_cache=cache,
    )
    first_cpu = optimizer_state[state_key]["exp_avg"]
    first_cpu_ptr = first_cpu.data_ptr()
    assert not first_cpu.is_pinned()

    move_optimizer_state(
        optimizer_state,
        device="cuda",
        reuse_cpu_buffers=True,
        cpu_buffer_cache=cache,
    )
    optimizer_state[state_key]["exp_avg"].add_(10)
    expected = optimizer_state[state_key]["exp_avg"].cpu()
    move_optimizer_state(
        optimizer_state,
        device="cpu",
        reuse_cpu_buffers=True,
        cpu_buffer_cache=cache,
    )
    second_cpu = optimizer_state[state_key]["exp_avg"]

    assert second_cpu.data_ptr() == first_cpu_ptr
    torch.testing.assert_close(second_cpu, expected)
