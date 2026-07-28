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
"""Tests for the Python xferdtensor implementation.

Covers the ``xferdtensor_python_impl`` signature/exports (which must stay
drop-in compatible with the real ``nccl.m2n.reshard`` dispatch in
``xferdtensor.py``), public entry-point orchestration, and pure-CPU geometry
planning for non-trivial reshard layouts.
"""

import inspect
from contextlib import nullcontext

import pytest
import torch
from torch.distributed._tensor import Replicate, Shard

from nemo_rl.weight_sync import xferdtensor_python as impl


class _Mesh:
    def __init__(self, ranks, shape):
        self.mesh = torch.tensor(ranks, dtype=torch.int64).reshape(shape)
        self._mesh = self.mesh


class _TensorRef:
    def __init__(self, local_tensor, global_shape):
        self._local_tensor = local_tensor
        self.shape = torch.Size(global_shape)
        self.device = local_tensor.device
        self.dtype = local_tensor.dtype


class _ProcessGroup:
    def __init__(self, rank=0):
        self.rank = rank
        self.nccl_communicator = object()


class _Stream:
    cuda_stream = 0xC0FFEE


@pytest.fixture(autouse=True)
def _isolated_caches():
    impl.clear_xferdtensor_python_caches()
    yield
    impl.clear_xferdtensor_python_caches()


def test_public_api_and_exports_are_upstream_compatible():
    assert list(inspect.signature(impl.xferdtensor_python_impl).parameters) == [
        "src_tensor",
        "src_mesh",
        "src_placement",
        "dst_tensor",
        "dst_mesh",
        "dst_placement",
        "process_group",
        "stream",
    ]
    assert impl.__all__ == [
        "clear_xferdtensor_python_caches",
        "xferdtensor_python_impl",
    ]


def test_public_api_orchestrates_preflight_split_p2p_and_broadcast(monkeypatch):
    calls = []
    device = torch.device("cuda", 0)
    stream = _Stream()
    process_group = _ProcessGroup(rank=0)
    tensor = _TensorRef(torch.ones(1), (1,))
    geometry = (
        {0: ((0, 1),)},
        {0: ((0, 1),)},
        ((((0, 1),), (0,), 0),),
        ((0, 0, ((0, 1),)),),
    )

    monkeypatch.setattr(
        impl, "_tensor_metadata", lambda *_args: ((1,), device, torch.float32)
    )
    monkeypatch.setattr(impl, "_plan_geometry", lambda *_args: geometry)
    monkeypatch.setattr(
        impl,
        "_validate_local_inputs",
        lambda *_args: calls.append("validate"),
    )
    monkeypatch.setattr(
        impl,
        "_get_replica_subcommunicator",
        lambda *_args: calls.append("split") or "subcomm",
    )
    monkeypatch.setattr(
        impl,
        "_stage_rank_operations",
        lambda *_args: calls.append("stage") or ([], [], tensor._local_tensor),
    )
    monkeypatch.setattr(
        impl,
        "_exchange_exact_overlaps",
        lambda *_args: calls.append("exchange"),
    )
    monkeypatch.setattr(
        impl,
        "_broadcast_destination",
        lambda *_args: calls.append("broadcast"),
    )
    monkeypatch.setattr(impl.torch.cuda, "current_stream", lambda device=None: stream)
    monkeypatch.setattr(impl.torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(impl.torch.cuda, "stream", lambda _stream: nullcontext())

    assert (
        impl.xferdtensor_python_impl(
            tensor,
            _Mesh([0], (1,)),
            (Replicate(),),
            tensor,
            _Mesh([0], (1,)),
            (Replicate(),),
            process_group,
        )
        is None
    )
    assert calls == ["validate", "split", "stage", "exchange", "broadcast"]


def test_intersect_and_local_slices():
    assert impl._intersect(((0, 4),), ((2, 6),)) == ((2, 4),)
    assert impl._intersect(((0, 2),), ((2, 4),)) is None
    assert impl._local_slices(
        ((2, 4), (3, 6)),
        ((1, 5), (2, 8)),
    ) == (slice(1, 3), slice(1, 4))


def test_compute_shard_slices_uses_uneven_torch_chunk_semantics():
    slices = [
        impl._compute_shard_slices(
            global_shape=(10,),
            mesh_shape=(4,),
            coordinates=(coordinate,),
            placements=(Shard(0),),
        )[0]
        for coordinate in range(4)
    ]

    assert slices == [
        slice(0, 3),
        slice(3, 6),
        slice(6, 9),
        slice(9, 10),
    ]


def test_plan_geometry_tp2_to_tp1_gather():
    src_regions, dst_regions, destination_groups, transfers = impl._plan_geometry(
        _Mesh([0, 1], (2,)),
        (Shard(0),),
        _Mesh([2], (1,)),
        (Replicate(),),
        (8, 4),
    )

    assert src_regions == {
        0: ((0, 4), (0, 4)),
        1: ((4, 8), (0, 4)),
    }
    assert dst_regions == {2: ((0, 8), (0, 4))}
    assert destination_groups == ((((0, 8), (0, 4)), (2,), 2),)
    assert transfers == (
        (0, 2, ((0, 4), (0, 4))),
        (1, 2, ((4, 8), (0, 4))),
    )


def test_plan_geometry_tp1_to_tp2_split():
    src_regions, dst_regions, destination_groups, transfers = impl._plan_geometry(
        _Mesh([0], (1,)),
        (Replicate(),),
        _Mesh([1, 2], (2,)),
        (Shard(0),),
        (8, 4),
    )

    assert src_regions == {0: ((0, 8), (0, 4))}
    assert dst_regions == {
        1: ((0, 4), (0, 4)),
        2: ((4, 8), (0, 4)),
    }
    assert destination_groups == (
        (((0, 4), (0, 4)), (1,), 1),
        (((4, 8), (0, 4)), (2,), 2),
    )
    assert transfers == (
        (0, 1, ((0, 4), (0, 4))),
        (0, 2, ((4, 8), (0, 4))),
    )


def test_plan_geometry_deduplicates_dp_replicas():
    src_regions, dst_regions, destination_groups, transfers = impl._plan_geometry(
        _Mesh([0, 1], (2,)),
        (Shard(0),),
        _Mesh([2, 3, 4, 5], (2, 2)),
        (Replicate(), Shard(0)),
        (8, 4),
    )

    assert src_regions == {
        0: ((0, 4), (0, 4)),
        1: ((4, 8), (0, 4)),
    }
    assert dst_regions == {
        2: ((0, 4), (0, 4)),
        3: ((4, 8), (0, 4)),
        4: ((0, 4), (0, 4)),
        5: ((4, 8), (0, 4)),
    }
    assert destination_groups == (
        (((0, 4), (0, 4)), (2, 4), 2),
        (((4, 8), (0, 4)), (3, 5), 3),
    )
    assert transfers == (
        (0, 2, ((0, 4), (0, 4))),
        (1, 3, ((4, 8), (0, 4))),
    )


def test_striped_geometry_distributes_replica_receives():
    striped = impl._striped_geometry(
        _Mesh([0, 1], (2,)),
        (Shard(0),),
        _Mesh([2, 3, 4, 5], (2, 2)),
        (Replicate(), Shard(0)),
        (8, 4),
    )

    assert striped is not None
    _src_regions, _dst_regions, _groups, transfers, stripes_by_rank = striped
    assert transfers == (
        (0, 2, ((0, 2), (0, 4))),
        (0, 4, ((2, 4), (0, 4))),
        (1, 3, ((4, 6), (0, 4))),
        (1, 5, ((6, 8), (0, 4))),
    )
    assert stripes_by_rank == {
        2: ((0, 2), (0, 4)),
        4: ((2, 4), (0, 4)),
        3: ((4, 6), (0, 4)),
        5: ((6, 8), (0, 4)),
    }
