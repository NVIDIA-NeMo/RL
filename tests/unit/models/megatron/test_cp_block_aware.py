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

"""Unit tests for the block-aware context-parallel layout helpers."""

import sys
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.megatron import cp_block_aware


@pytest.fixture
def block_aware_on(monkeypatch):
    monkeypatch.setattr(cp_block_aware, "_BLOCK_AWARE_CP", True)


def _fake_parallel_state(monkeypatch, cp_size: int) -> None:
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.parallel_state",
        SimpleNamespace(get_context_parallel_world_size=lambda: cp_size),
    )


@pytest.mark.parametrize(
    "value,multiple,expected",
    [(0, 4, 0), (1, 4, 4), (4, 4, 4), (5, 4, 8), (7, 1, 7), (7, 0, 7)],
)
def test_round_up(value: int, multiple: int, expected: int):
    assert cp_block_aware.round_up(value, multiple) == expected


def test_everything_is_off_without_the_flag(monkeypatch):
    monkeypatch.setattr(cp_block_aware, "_BLOCK_AWARE_CP", False)
    _fake_parallel_state(monkeypatch, cp_size=4)
    assert not cp_block_aware.block_aware_cp_enabled()
    assert cp_block_aware.block_aware_cp_padding(16) is None
    data = {
        "diffu_grpo_noisy_lengths": torch.tensor([8, 8]),
        "diffu_grpo_clean_padded_lengths": torch.tensor([4, 4]),
    }
    assert cp_block_aware.block_aware_segments(data) is None


def test_padding_multiples_follow_cp_and_block_size(monkeypatch, block_aware_on):
    _fake_parallel_state(monkeypatch, cp_size=2)
    # Noisy chunks must be whole blocks: 2 * cp * block. Clean needs only 2 * cp.
    assert cp_block_aware.block_aware_cp_padding(16) == (64, 4)
    assert cp_block_aware.block_aware_cp_padding(None) == (4, 4)


def test_padding_is_off_at_cp_one(monkeypatch, block_aware_on):
    _fake_parallel_state(monkeypatch, cp_size=1)
    assert cp_block_aware.block_aware_cp_padding(16) is None


def test_segments_read_constant_lengths(block_aware_on):
    data = {
        "diffu_grpo_noisy_lengths": torch.tensor([8, 8]),
        "diffu_grpo_clean_padded_lengths": torch.tensor([4, 4]),
    }
    assert cp_block_aware.block_aware_segments(data) == (8, 4)


def test_segments_ignore_non_diffusion_batches(block_aware_on):
    assert cp_block_aware.block_aware_segments({"input_ids": torch.zeros(2, 3)}) is None


def test_segments_reject_varying_lengths(block_aware_on):
    data = {
        "diffu_grpo_noisy_lengths": torch.tensor([8, 16]),
        "diffu_grpo_clean_padded_lengths": torch.tensor([4, 4]),
    }
    with pytest.raises(ValueError, match="constant noisy/clean lengths"):
        cp_block_aware.block_aware_segments(data)


def test_divisibility_check():
    cp_block_aware.assert_block_aware_divisible((8, 4), cp_size=2, total_length=12)
    with pytest.raises(AssertionError, match="noisy segment"):
        cp_block_aware.assert_block_aware_divisible((6, 4), cp_size=2, total_length=10)
    with pytest.raises(AssertionError, match="do not cover"):
        cp_block_aware.assert_block_aware_divisible((8, 4), cp_size=2, total_length=16)
