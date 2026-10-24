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

"""Behavioral tests for packed draft sequence coordinates under CP and SP."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import torch

pytestmark = pytest.mark.mcore

_LAYOUT_MODULE = "nemo_rl.models.megatron.draft.sequence_layout"
_LAYOUT_PATH = (
    Path(__file__).resolve().parents[4]
    / "nemo_rl/models/megatron/draft/sequence_layout.py"
)


def _load_layout_module() -> ModuleType:
    if not _LAYOUT_PATH.is_file():
        pytest.fail(
            f"Draft sequence-layout production contract is missing: {_LAYOUT_PATH}",
            pytrace=False,
        )
    spec = importlib.util.spec_from_file_location(_LAYOUT_MODULE, _LAYOUT_PATH)
    if spec is None:
        raise RuntimeError("Unable to create the draft sequence-layout module spec")
    if spec.loader is None:
        raise RuntimeError("Draft sequence-layout module spec has no loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_LAYOUT_MODULE] = module
    spec.loader.exec_module(module)
    return module


def _build_layout(
    *,
    sample_ids: list[int],
    cu_seqlens_q: list[int],
    cu_seqlens_q_padded: list[int],
    cp_rank: int,
    cp_size: int,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> Any:
    module = _load_layout_module()
    return module.build_draft_sequence_layout(
        logical_sample_ids=torch.tensor(sample_ids, dtype=torch.int64),
        cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int64),
        cu_seqlens_q_padded=torch.tensor(
            cu_seqlens_q_padded,
            dtype=torch.int64,
        ),
        cp_rank=cp_rank,
        cp_size=cp_size,
        tp_rank=tp_rank,
        tp_size=tp_size,
        device=torch.device("cpu"),
    )


def test_dense_layout_resets_logical_positions_and_marks_padding() -> None:
    """Catches treating padded packed offsets as one continuous sequence."""
    layout = _build_layout(
        sample_ids=[101, 303],
        cu_seqlens_q=[0, 3, 5],
        cu_seqlens_q_padded=[0, 4, 8],
        cp_rank=0,
        cp_size=1,
    )

    assert torch.equal(layout.logical_lengths, torch.tensor([3, 2]))
    assert torch.equal(
        layout.packed_to_logical_sample,
        torch.tensor([101, 101, 101, -1, 303, 303, -1, -1]),
    )
    assert torch.equal(
        layout.packed_logical_positions,
        torch.tensor([0, 1, 2, -1, 0, 1, -1, -1]),
    )
    assert torch.equal(
        layout.packed_valid_mask,
        torch.tensor([True, True, True, False, True, True, False, False]),
    )
    assert torch.equal(layout.cp_global_positions, torch.arange(8))
    assert torch.equal(layout.cp_global_to_local, torch.arange(8))
    assert torch.equal(layout.owner_cp_rank, torch.zeros(8, dtype=torch.int64))


@pytest.mark.parametrize("cp_size", [2, 4])
def test_context_parallel_layout_matches_per_sequence_zigzag(cp_size: int) -> None:
    """Catches global packing shards that violate MCore's per-sequence zigzag."""
    padded_per_sequence = 2 * cp_size
    cu_seqlens_q_padded = [0, padded_per_sequence, 2 * padded_per_sequence]
    valid_lengths = [padded_per_sequence - 1, cp_size + 1]
    cu_seqlens_q = [0, valid_lengths[0], sum(valid_lengths)]

    layouts = [
        _build_layout(
            sample_ids=[11, 22],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cp_rank=cp_rank,
            cp_size=cp_size,
        )
        for cp_rank in range(cp_size)
    ]

    expected_by_rank: list[list[int]] = []
    for cp_rank in range(cp_size):
        first = [cp_rank, 2 * cp_size - cp_rank - 1]
        second = [
            padded_per_sequence + cp_rank,
            2 * padded_per_sequence - cp_rank - 1,
        ]
        expected_by_rank.append(first + second)

    for layout, expected in zip(layouts, expected_by_rank, strict=True):
        assert layout.cp_global_positions.tolist() == expected
        for local_position, global_position in enumerate(expected):
            assert layout.cp_global_to_local[global_position] == local_position

    owned_positions = (
        torch.cat([layout.cp_global_positions for layout in layouts]).sort().values
    )
    assert torch.equal(owned_positions, torch.arange(2 * padded_per_sequence))
    assert torch.equal(
        layouts[0].packed_logical_positions[:padded_per_sequence],
        torch.tensor(list(range(valid_lengths[0])) + [-1]),
    )
    for cp_rank, layout in enumerate(layouts):
        assert torch.equal(
            layout.owner_cp_rank[layout.cp_global_positions],
            torch.full(
                (layout.cp_global_positions.numel(),),
                cp_rank,
                dtype=torch.int64,
            ),
        )


def test_sequence_parallel_shards_reconstruct_one_cp_local_stream() -> None:
    """Catches SP indexing that crosses CP lanes or changes CP-local order."""
    layouts = [
        _build_layout(
            sample_ids=[7, 9],
            cu_seqlens_q=[0, 5, 8],
            cu_seqlens_q_padded=[0, 8, 16],
            cp_rank=1,
            cp_size=2,
            tp_rank=tp_rank,
            tp_size=2,
        )
        for tp_rank in range(2)
    ]

    assert torch.equal(layouts[0].cp_global_positions, layouts[1].cp_global_positions)
    assert torch.equal(layouts[0].sp_local_positions, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(layouts[1].sp_local_positions, torch.tensor([4, 5, 6, 7]))
    reconstructed = torch.cat(
        [layout.cp_global_positions[layout.sp_local_positions] for layout in layouts]
    )
    assert torch.equal(reconstructed, layouts[0].cp_global_positions)
    assert torch.equal(layouts[0].descriptor, layouts[1].descriptor)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"cu_seqlens_q": [1, 3]}, "start at zero"),
        (
            {
                "sample_ids": [1, 2],
                "cu_seqlens_q": [0, 5, 4],
                "cu_seqlens_q_padded": [0, 5, 8],
            },
            "non-decreasing",
        ),
        ({"cu_seqlens_q_padded": [0, 2]}, "at least the unpadded"),
        (
            {
                "cu_seqlens_q": [0, 3],
                "cu_seqlens_q_padded": [0, 6],
                "cp_size": 2,
            },
            r"2 \* cp_size",
        ),
        ({"cp_rank": 2, "cp_size": 2}, "cp_rank"),
        ({"tp_rank": 2, "tp_size": 2}, "tp_rank"),
    ],
)
def test_layout_rejects_malformed_metadata_before_mapping(
    override: dict[str, object],
    message: str,
) -> None:
    """Catches ambiguous layouts that would make distributed ranks disagree."""
    values: dict[str, object] = {
        "sample_ids": [1],
        "cu_seqlens_q": [0, 4],
        "cu_seqlens_q_padded": [0, 4],
        "cp_rank": 0,
        "cp_size": 1,
        "tp_rank": 0,
        "tp_size": 1,
    }
    values.update(override)

    with pytest.raises((TypeError, ValueError), match=message):
        _build_layout(**cast(Any, values))
