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

"""CPU tests for the draft CP K/V exchange and target-SP reconstruction guards."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import Tensor

pytestmark = pytest.mark.mcore

_DRAFT_DIR = Path(__file__).resolve().parents[4] / "nemo_rl/models/megatron/draft"
_GROUP = object()


def _load(name: str) -> ModuleType:
    module_name = f"nemo_rl.models.megatron.draft.{name}"
    spec = importlib.util.spec_from_file_location(
        module_name, _DRAFT_DIR / f"{name}.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _layout(
    *,
    cp_rank: int = 0,
    cp_size: int = 1,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> Any:
    return _load("sequence_layout").build_draft_sequence_layout(
        logical_sample_ids=torch.tensor([7, 9], dtype=torch.int64),
        cu_seqlens_q=torch.tensor([0, 5, 8], dtype=torch.int64),
        cu_seqlens_q_padded=torch.tensor([0, 8, 16], dtype=torch.int64),
        cp_rank=cp_rank,
        cp_size=cp_size,
        tp_rank=tp_rank,
        tp_size=tp_size,
        device=torch.device("cpu"),
    )


def _fake_group(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    *,
    world_size: int,
    rank: int,
    gather: Callable[[Tensor], Tensor],
) -> None:
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: world_size)
    monkeypatch.setattr(module.dist, "get_rank", lambda group: rank)
    monkeypatch.setattr(
        module,
        "all_gather_tensor_autograd",
        lambda tensor, *, gather_dim, group: gather(tensor),
    )


def _gather_kv(projected: Tensor, layout: Any, group: object | None = _GROUP) -> Tensor:
    return _load("context_parallel").gather_projected_kv(
        projected,
        sequence_layout=layout,
        cp_group=group,
        sequence_dim=1,
    )


def test_cp_gather_restores_packed_order_from_zigzag_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches rank-major K/V reaching attention in the wrong packed order."""
    module = _load("context_parallel")
    layouts = [_layout(cp_rank=rank, cp_size=2) for rank in range(2)]
    packed = torch.arange(16, dtype=torch.float32)[None, :, None].expand(2, 16, 3)
    shards = [packed[:, layout.cp_global_positions] for layout in layouts]
    _fake_group(
        monkeypatch,
        module,
        world_size=2,
        rank=1,
        gather=lambda _local: torch.cat([s.movedim(1, 0) for s in shards]),
    )

    restored = module.gather_projected_kv(
        shards[1],
        sequence_layout=layouts[1],
        cp_group=_GROUP,
        sequence_dim=1,
    )

    assert torch.equal(restored, packed)


def test_cp1_gather_is_identity_without_a_group() -> None:
    """Catches CP1 paths that require a process group they never use."""
    projected = torch.randn(2, 16, 3)

    assert _gather_kv(projected, _layout(), group=None) is projected


@pytest.mark.parametrize(
    ("projected", "layout_override", "message"),
    [
        (torch.tensor(1.0), {}, "sequence dimension"),
        (torch.zeros(2, 15, 3), {}, "length does not match"),
        (
            torch.zeros(2, 16, 3),
            {"owner_cp_rank": torch.zeros(16, dtype=torch.int64, device="meta")},
            "share a device",
        ),
        (torch.zeros(2, 16, 3), {"cp_rank": 1}, "cp_rank zero"),
        (
            torch.zeros(2, 16, 3),
            {"owner_cp_rank": torch.zeros(17, dtype=torch.int64)},
            "does not cover",
        ),
    ],
)
def test_cp1_gather_rejects_inconsistent_inputs(
    projected: Tensor,
    layout_override: dict[str, Any],
    message: str,
) -> None:
    """Catches a K/V tensor or layout from another microbatch or rank."""
    layout = dataclasses.replace(_layout(), **layout_override)

    with pytest.raises(ValueError, match=message):
        _gather_kv(projected, layout)


@pytest.mark.parametrize(
    ("world_size", "rank", "gathered_length", "error", "message"),
    [
        (4, 0, 16, ValueError, "group size"),
        (2, 1, 16, ValueError, "group rank"),
        (2, 0, 12, RuntimeError, "did not cover"),
    ],
)
def test_cp_gather_rejects_group_that_disagrees_with_layout(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    rank: int,
    gathered_length: int,
    error: type[Exception],
    message: str,
) -> None:
    """Catches a K/V exchange on a group that does not match the layout."""
    module = _load("context_parallel")
    _fake_group(
        monkeypatch,
        module,
        world_size=world_size,
        rank=rank,
        gather=lambda _local: torch.zeros(gathered_length, 2, 3),
    )

    with pytest.raises(error, match=message):
        module.gather_projected_kv(
            torch.zeros(2, 8, 3),
            sequence_layout=_layout(cp_rank=0, cp_size=2),
            cp_group=_GROUP,
            sequence_dim=1,
        )


def test_cp_gather_requires_initialized_group() -> None:
    """Catches CP>1 silently skipping the exchange without a group."""
    with pytest.raises(RuntimeError, match="initialized CP group"):
        _gather_kv(torch.zeros(2, 8, 3), _layout(cp_size=2), group=None)


def test_cp_gather_rejects_unequal_rank_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a layout whose ranks own different packed lengths."""
    module = _load("context_parallel")
    layout = _layout(cp_rank=0, cp_size=2)
    skewed_owners = layout.owner_cp_rank.clone()
    skewed_owners[torch.where(layout.owner_cp_rank == 1)[0][0]] = 0
    _fake_group(
        monkeypatch,
        module,
        world_size=2,
        rank=0,
        gather=lambda _local: torch.zeros(16, 2, 3),
    )

    with pytest.raises(ValueError, match="equal packed sequence lengths"):
        module.gather_projected_kv(
            torch.zeros(2, 8, 3),
            sequence_layout=dataclasses.replace(layout, owner_cp_rank=skewed_owners),
            cp_group=_GROUP,
            sequence_dim=1,
        )


def _reconstruct(local: Tensor, layout: Any, group: object | None = _GROUP) -> Tensor:
    return _load("sequence_parallel").reconstruct_tp_sequence(
        local,
        sequence_layout=layout,
        tp_group=group,
        sequence_dim=0,
    )


def test_tp_reconstruction_concatenates_ordered_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches SP shards rejoined out of TP-rank order."""
    module = _load("sequence_parallel")
    layouts = [_layout(cp_rank=1, cp_size=2, tp_rank=r, tp_size=2) for r in range(2)]
    cp_local = torch.arange(8, dtype=torch.float32)[:, None]
    shards = [cp_local[layout.sp_local_positions] for layout in layouts]
    _fake_group(
        monkeypatch,
        module,
        world_size=2,
        rank=0,
        gather=lambda _local: torch.cat(shards),
    )

    reconstructed = module.reconstruct_tp_sequence(
        shards[0],
        sequence_layout=layouts[0],
        tp_group=_GROUP,
        sequence_dim=0,
    )

    assert torch.equal(reconstructed, cp_local)


def test_tp1_reconstruction_is_identity_without_a_group() -> None:
    """Catches TP1 paths that require a process group they never use."""
    local = torch.randn(16, 3)

    assert _reconstruct(local, _layout(), group=None) is local


@pytest.mark.parametrize(
    ("local", "layout_override", "message"),
    [
        (torch.tensor(1.0), {}, "sequence dimension"),
        (torch.zeros(15, 3), {}, "shard length"),
        (torch.zeros(16, 3), {"tp_rank": 1}, "tp_rank zero"),
        (
            torch.zeros(16, 3),
            {"cp_global_positions": torch.arange(17)},
            "does not cover",
        ),
    ],
)
def test_tp1_reconstruction_rejects_inconsistent_inputs(
    local: Tensor,
    layout_override: dict[str, Any],
    message: str,
) -> None:
    """Catches a target shard or layout from another rank or microbatch."""
    layout = dataclasses.replace(_layout(), **layout_override)

    with pytest.raises(ValueError, match=message):
        _reconstruct(local, layout)


@pytest.mark.parametrize(
    ("world_size", "rank", "gathered_length", "error", "message"),
    [
        (4, 0, 16, ValueError, "group size"),
        (2, 1, 16, ValueError, "group rank"),
        (2, 0, 12, RuntimeError, "did not reconstruct"),
    ],
)
def test_tp_reconstruction_rejects_group_that_disagrees_with_layout(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    rank: int,
    gathered_length: int,
    error: type[Exception],
    message: str,
) -> None:
    """Catches a reconstruction on a group that does not match the layout."""
    module = _load("sequence_parallel")
    _fake_group(
        monkeypatch,
        module,
        world_size=world_size,
        rank=rank,
        gather=lambda _local: torch.zeros(gathered_length, 3),
    )

    with pytest.raises(error, match=message):
        module.reconstruct_tp_sequence(
            torch.zeros(8, 3),
            sequence_layout=_layout(tp_rank=0, tp_size=2),
            tp_group=_GROUP,
            sequence_dim=0,
        )


def test_tp_reconstruction_requires_initialized_group() -> None:
    """Catches TP>1 silently skipping SP reconstruction without a group."""
    with pytest.raises(RuntimeError, match="initialized TP group"):
        _reconstruct(torch.zeros(8, 3), _layout(tp_size=2), group=None)
