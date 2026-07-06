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
"""Extensive CPU-only tests for the upstream Python xferdtensor implementation.

The parameter matrices in this file intentionally collect as individual pytest
cases.  Geometry expectations are produced with actual ``torch.chunk`` calls,
independently of the implementation's ceil-division arithmetic.
"""

import gc
import inspect
import itertools
import math
import random
import sys
import types
from collections import OrderedDict
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


class _ToLocalRef:
    def __init__(self, local_tensor, global_shape):
        self._tensor = local_tensor
        self.shape = torch.Size(global_shape)
        self.device = local_tensor.device
        self.dtype = local_tensor.dtype

    def to_local(self):
        return self._tensor


class _SplitComm:
    def __init__(self):
        self.split_calls = []

    def split(self, color, key):
        child = object()
        self.split_calls.append((color, key, child))
        return child


class _ProcessGroup:
    def __init__(self, rank=0, communicator=None):
        self.rank = rank
        self.nccl_communicator = communicator or _SplitComm()


class _Stream:
    cuda_stream = 0xC0FFEE


@pytest.fixture(autouse=True)
def _isolated_caches():
    impl.clear_xferdtensor_python_caches()
    yield
    impl.clear_xferdtensor_python_caches()


def _chunk_sizes(size, chunks):
    pieces = torch.empty(size).chunk(chunks)
    sizes = [piece.numel() for piece in pieces]
    sizes.extend([0] * (chunks - len(sizes)))
    return sizes


def _oracle_region(global_shape, mesh_shape, coordinates, placements):
    """Independent sequential-sharding oracle based on torch.chunk outputs."""
    regions = [(0, int(size)) for size in global_shape]
    for mesh_dim, placement in enumerate(placements):
        if not isinstance(placement, Shard):
            continue
        tensor_dim = placement.dim % len(global_shape)
        owner_start, owner_stop = regions[tensor_dim]
        sizes = _chunk_sizes(owner_stop - owner_start, int(mesh_shape[mesh_dim]))
        coordinate = int(coordinates[mesh_dim])
        relative_start = sum(sizes[:coordinate])
        regions[tensor_dim] = (
            owner_start + relative_start,
            owner_start + relative_start + sizes[coordinate],
        )
    return tuple(regions)


def _slices_to_region(slices, global_shape):
    return tuple(
        (
            0 if item.start is None else int(item.start),
            int(global_shape[dim]) if item.stop is None else int(item.stop),
        )
        for dim, item in enumerate(slices)
    )


def _points(region):
    return set(itertools.product(*(range(start, stop) for start, stop in region)))


def _coords_for_rank(mesh, rank):
    location = (mesh.mesh == rank).nonzero(as_tuple=False)
    return tuple(int(value) for value in location[0].tolist())


def _random_placements(rng, mesh_ndim, tensor_ndim):
    result = []
    for _ in range(mesh_ndim):
        if rng.random() < 0.35:
            result.append(Replicate())
        else:
            dim = rng.randrange(tensor_ndim)
            result.append(Shard(dim if rng.random() < 0.5 else dim - tensor_ndim))
    return tuple(result)


SINGLE_AXIS_CASES = [
    pytest.param(size, chunks, coordinate, id=f"n{size}-p{chunks}-c{coordinate}")
    for size in range(18)
    for chunks in range(1, 7)
    for coordinate in range(chunks)
]


REPEATED_AXIS_CASES = [
    pytest.param(
        size, mesh_shape, coordinates, id=f"n{size}-m{mesh_shape}-c{coordinates}"
    )
    for size in (0, 1, 2, 3, 5, 7, 10, 13, 17)
    for mesh_shape in ((2, 2), (2, 3), (3, 2), (3, 3), (4, 2))
    for coordinates in itertools.product(*(range(part) for part in mesh_shape))
]


def _make_mixed_cases(count=128):
    rng = random.Random(0xC0DEC0DE)
    cases = []
    for case_id in range(count):
        tensor_ndim = rng.randint(1, 4)
        mesh_ndim = rng.randint(1, 3)
        global_shape = tuple(rng.randint(0, 15) for _ in range(tensor_ndim))
        mesh_shape = tuple(rng.randint(1, 4) for _ in range(mesh_ndim))
        coordinates = tuple(rng.randrange(size) for size in mesh_shape)
        placements = _random_placements(rng, mesh_ndim, tensor_ndim)
        cases.append(
            pytest.param(
                global_shape,
                mesh_shape,
                coordinates,
                placements,
                id=f"mixed-{case_id:03d}",
            )
        )
    return cases


MIXED_LAYOUT_CASES = _make_mixed_cases()


def _make_plan_cases(count=128):
    rng = random.Random(0x5EEDFACE)
    mesh_shapes = ((1,), (2,), (3,), (4,), (2, 2), (2, 3), (3, 2))
    cases = []
    for case_id in range(count):
        tensor_ndim = rng.randint(1, 3)
        global_shape = tuple(rng.randint(0, 7) for _ in range(tensor_ndim))
        src_shape = rng.choice(mesh_shapes)
        dst_shape = rng.choice(mesh_shapes)
        src_ranks = list(range(math.prod(src_shape)))
        dst_ranks = list(range(32, 32 + math.prod(dst_shape)))
        rng.shuffle(src_ranks)
        rng.shuffle(dst_ranks)
        src_placements = _random_placements(rng, len(src_shape), tensor_ndim)
        dst_placements = _random_placements(rng, len(dst_shape), tensor_ndim)
        cases.append(
            pytest.param(
                global_shape,
                src_shape,
                tuple(src_ranks),
                src_placements,
                dst_shape,
                tuple(dst_ranks),
                dst_placements,
                id=f"plan-{case_id:03d}",
            )
        )
    return cases


PLAN_CASES = _make_plan_cases()


def test_public_api_and_exports_are_upstream_compatible():
    assert list(inspect.signature(impl.xferdtensor_python_impl).parameters) == [
        "src_tensor",
        "src_mesh",
        "src_placement",
        "dst_tensor",
        "dst_mesh",
        "dst_placement",
        "process_group",
    ]
    assert impl.__all__ == [
        "clear_xferdtensor_python_caches",
        "xferdtensor_python_impl",
    ]
    assert not hasattr(impl, "xferdtensor_python_impl_batched")


@pytest.mark.parametrize("tensor_ndim", range(1, 6))
@pytest.mark.parametrize("dim", range(-5, 5))
def test_normalize_shard_dim_accepts_exact_valid_range(tensor_ndim, dim):
    if -tensor_ndim <= dim < tensor_ndim:
        assert impl._normalize_shard_dim(dim, tensor_ndim) == dim % tensor_ndim
    else:
        with pytest.raises(ValueError, match="Shard dim"):
            impl._normalize_shard_dim(dim, tensor_ndim)


@pytest.mark.parametrize("size,chunks,coordinate", SINGLE_AXIS_CASES)
def test_single_axis_sharding_matches_actual_torch_chunk(size, chunks, coordinate):
    actual = impl._compute_shard_slices((size,), (chunks,), (coordinate,), (Shard(0),))
    expected = _oracle_region((size,), (chunks,), (coordinate,), (Shard(0),))
    assert _slices_to_region(actual, (size,)) == expected


@pytest.mark.parametrize("size,mesh_shape,coordinates", REPEATED_AXIS_CASES)
def test_repeated_sharding_matches_nested_torch_chunk(size, mesh_shape, coordinates):
    placements = (Shard(-1), Shard(0))
    actual = impl._compute_shard_slices((size,), mesh_shape, coordinates, placements)
    expected = _oracle_region((size,), mesh_shape, coordinates, placements)
    assert _slices_to_region(actual, (size,)) == expected


@pytest.mark.parametrize(
    "global_shape,mesh_shape,coordinates,placements", MIXED_LAYOUT_CASES
)
def test_mixed_multidimensional_layouts_match_independent_oracle(
    global_shape, mesh_shape, coordinates, placements
):
    actual = impl._compute_shard_slices(
        global_shape, mesh_shape, coordinates, placements
    )
    expected = _oracle_region(global_shape, mesh_shape, coordinates, placements)
    assert _slices_to_region(actual, global_shape) == expected


@pytest.mark.parametrize(
    "global_shape,src_shape,src_ranks,src_placements,dst_shape,dst_ranks,dst_placements",
    PLAN_CASES,
)
def test_exact_plan_covers_each_unique_destination_element_once(
    global_shape,
    src_shape,
    src_ranks,
    src_placements,
    dst_shape,
    dst_ranks,
    dst_placements,
):
    src_mesh = _Mesh(src_ranks, src_shape)
    dst_mesh = _Mesh(dst_ranks, dst_shape)
    src_regions, dst_regions, destination_groups, transfers = impl._plan_geometry(
        src_mesh,
        src_placements,
        dst_mesh,
        dst_placements,
        global_shape,
    )

    for rank in src_ranks:
        expected = _oracle_region(
            global_shape, src_shape, _coords_for_rank(src_mesh, rank), src_placements
        )
        assert src_regions[rank] == expected
    for rank in dst_ranks:
        expected = _oracle_region(
            global_shape, dst_shape, _coords_for_rank(dst_mesh, rank), dst_placements
        )
        assert dst_regions[rank] == expected

    representatives = {group[2] for group in destination_groups}
    assert {target for _source, target, _overlap in transfers} <= representatives
    assert tuple(transfers) == tuple(sorted(transfers, key=lambda item: item))

    for destination, _members, representative in destination_groups:
        expected_points = _points(destination)
        seen = set()
        for source, target, overlap in transfers:
            if target != representative:
                continue
            overlap_points = _points(overlap)
            assert source in src_ranks
            assert overlap_points <= _points(src_regions[source])
            assert seen.isdisjoint(overlap_points)
            seen.update(overlap_points)
        assert seen == expected_points


def test_scalar_replica_plan_moves_the_single_value_once():
    geometry = impl._plan_geometry(
        _Mesh([0, 1], (2,)),
        (Replicate(),),
        _Mesh([2, 3, 4, 5], (2, 2)),
        (Replicate(), Replicate()),
        (),
    )
    assert geometry[3] == ((0, 2, ()),)
    assert impl._region_numel(()) == 1


def test_destination_representative_is_replica_coordinate_zero_not_lowest_rank():
    mesh = _Mesh([8, 2, 5, 9, 1, 6], (2, 3))
    regions = impl._rank_regions(mesh, (Shard(0), Replicate()), (12,))
    groups = impl._destination_groups(mesh, (Shard(0), Replicate()), regions)
    assert groups == (
        (((0, 6),), (8, 2, 5), 8),
        (((6, 12),), (9, 1, 6), 9),
    )


def test_replicated_sources_are_deduplicated_and_prefer_local_copy():
    src_regions = OrderedDict([(7, ((0, 4),)), (3, ((0, 4),)), (9, ((4, 8),))])
    groups = ((((0, 8),), (3, 11), 3),)
    transfers = impl._build_exact_plan(src_regions, (7, 3, 9), groups)
    assert transfers == (
        (3, 3, ((0, 4),)),
        (9, 3, ((4, 8),)),
    )


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (((0, 3),), ((3, 5),), None),
        (((0, 3),), ((2, 5),), ((2, 3),)),
        (((0, 4), (1, 8)), ((2, 5), (3, 6)), ((2, 4), (3, 6))),
        (((0, 0),), ((0, 0),), None),
    ],
)
def test_intersection_edge_cases(left, right, expected):
    assert impl._intersect(left, right) == expected


def test_local_slices_and_region_numel():
    overlap = ((4, 7), (8, 10), (2, 6))
    owner = ((1, 9), (5, 12), (2, 8))
    assert impl._local_slices(overlap, owner) == (
        slice(3, 6),
        slice(3, 5),
        slice(0, 4),
    )
    assert impl._region_numel(overlap) == 24


def test_mesh_helpers_preserve_permuted_rank_order_and_coordinates():
    mesh = _Mesh([8, 2, 5, 9, 1, 6], (2, 3))
    assert impl._mesh_ranks(mesh) == [8, 2, 5, 9, 1, 6]
    assert impl._mesh_signature(mesh) == ((2, 3), (8, 2, 5, 9, 1, 6))
    assert impl._mesh_coordinates(mesh) == {
        8: (0, 0),
        2: (0, 1),
        5: (0, 2),
        9: (1, 0),
        1: (1, 1),
        6: (1, 2),
    }


def test_mesh_without_rank_tensor_is_rejected():
    with pytest.raises(ValueError, match="does not expose"):
        impl._mesh_rank_tensor(object())


def test_placement_signature_canonicalizes_negative_dims():
    assert impl._placement_signature((Shard(-1), Replicate()), 3) == (
        ("Shard", 2),
        ("Replicate", None),
    )


class _Partial:
    pass


@pytest.mark.parametrize("dim", (-8, -4, 3, 7))
def test_invalid_shard_dimensions_fail_before_planning(dim):
    mesh = _Mesh([0], (1,))
    with pytest.raises(ValueError, match="invalid for a 3D tensor"):
        impl._plan_geometry(mesh, (Shard(dim),), mesh, (Replicate(),), (2, 3, 4))


def test_partial_and_placement_count_mismatch_fail_clearly():
    mesh = _Mesh([0], (1,))
    with pytest.raises(NotImplementedError, match="only Shard and Replicate"):
        impl._plan_geometry(mesh, (_Partial(),), mesh, (Replicate(),), (8,))
    with pytest.raises(ValueError, match="1D mesh"):
        impl._plan_geometry(mesh, (), mesh, (Replicate(),), (8,))


def test_plan_cache_returns_identity_and_obeys_lru_bound(monkeypatch):
    mesh = _Mesh([0], (1,))
    first = impl._plan_geometry(mesh, (Replicate(),), mesh, (Replicate(),), (1,))
    assert (
        impl._plan_geometry(mesh, (Replicate(),), mesh, (Replicate(),), (1,)) is first
    )

    monkeypatch.setattr(impl, "_PLAN_CACHE_MAX_SIZE", 3)
    for size in (2, 3, 4, 5):
        impl._plan_geometry(mesh, (Replicate(),), mesh, (Replicate(),), (size,))
    assert len(impl._PLAN_CACHE) == 3


def test_mesh_rank_permutation_changes_cache_key():
    placements = (Shard(0),)
    impl._plan_geometry(
        _Mesh([0, 1], (2,)), placements, _Mesh([2, 3], (2,)), placements, (8,)
    )
    impl._plan_geometry(
        _Mesh([1, 0], (2,)), placements, _Mesh([2, 3], (2,)), placements, (8,)
    )
    assert len(impl._PLAN_CACHE) == 2


def test_local_tensor_supports_attribute_and_to_local_forms():
    tensor = torch.arange(4)
    assert impl._local_tensor(_TensorRef(tensor, (4,))) is tensor
    assert impl._local_tensor(_ToLocalRef(tensor, (4,))) is tensor
    assert impl._local_tensor(None) is None
    with pytest.raises(ValueError, match="does not expose"):
        impl._local_tensor(object())


def test_tensor_metadata_validates_shape_dtype_and_missing_inputs():
    src = _TensorRef(torch.empty(2, dtype=torch.float32), (2,))
    dst = _TensorRef(torch.empty(2, dtype=torch.float32), (2,))
    assert impl._tensor_metadata(src, dst) == ((2,), src.device, src.dtype)
    with pytest.raises(ValueError, match="global shapes"):
        impl._tensor_metadata(src, _TensorRef(torch.empty(3), (3,)))
    with pytest.raises(ValueError, match="dtypes"):
        impl._tensor_metadata(
            src, _TensorRef(torch.empty(2, dtype=torch.float64), (2,))
        )
    with pytest.raises(ValueError, match="neither a source nor a destination"):
        impl._tensor_metadata(None, None)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "did not receive"),
        ("shape", "planned shape"),
        ("dtype", "does not match"),
        ("device", "does not match"),
        ("outside", "non-source rank"),
    ],
)
def test_local_input_preflight_rejects_malformed_rank_state(mutation, match):
    local = torch.empty((2, 3), dtype=torch.float32)
    tensor = _TensorRef(local, (2, 3))
    regions = {0: ((0, 2), (0, 3))}
    rank = 0
    src_tensor = tensor
    device = local.device
    dtype = local.dtype
    if mutation == "missing":
        src_tensor = None
    elif mutation == "shape":
        regions = {0: ((0, 3), (0, 3))}
    elif mutation == "dtype":
        dtype = torch.float64
    elif mutation == "device":
        device = torch.device("meta")
    elif mutation == "outside":
        regions = {}
    with pytest.raises(ValueError, match=match):
        impl._validate_local_inputs(rank, src_tensor, None, regions, {}, device, dtype)


def test_local_input_preflight_accepts_overlapping_source_destination_rank():
    src = _TensorRef(torch.empty((2, 3)), (4, 3))
    dst = _TensorRef(torch.empty((2, 3)), (4, 3))
    impl._validate_local_inputs(
        2,
        src,
        dst,
        {2: ((0, 2), (0, 3))},
        {2: ((2, 4), (0, 3))},
        src.device,
        src.dtype,
    )


def test_stage_operations_performs_local_copy_without_staging():
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    destination = torch.full_like(source, -1)
    sends, receives, returned = impl._stage_rank_operations(
        0,
        _TensorRef(source, (3, 4)),
        _TensorRef(destination, (3, 4)),
        {0: ((0, 3), (0, 4))},
        {0: ((0, 3), (0, 4))},
        ((0, 0, ((0, 3), (0, 4))),),
        source.device,
        source.dtype,
    )
    assert sends == []
    assert receives == []
    assert returned is destination
    assert torch.equal(destination, source)


def test_stage_operations_pack_only_noncontiguous_send_overlap():
    source = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    sends, receives, _dst = impl._stage_rank_operations(
        0,
        _TensorRef(source, (4, 7)),
        None,
        {0: ((0, 4), (0, 7))},
        {1: ((0, 4), (2, 5))},
        ((0, 1, ((0, 4), (2, 5))),),
        source.device,
        source.dtype,
    )
    assert receives == []
    assert len(sends) == 1
    assert sends[0][0] == 1
    assert sends[0][1].shape == (4, 3)
    assert sends[0][1].is_contiguous()
    assert torch.equal(sends[0][1], source[:, 2:5])


def test_stage_operations_receive_directly_or_with_exact_overlap_staging():
    contiguous = torch.empty((3, 5))
    receives = impl._stage_rank_operations(
        1,
        None,
        _TensorRef(contiguous, (3, 5)),
        {0: ((0, 3), (0, 5))},
        {1: ((0, 3), (0, 5))},
        ((0, 1, ((0, 3), (0, 5))),),
        contiguous.device,
        contiguous.dtype,
    )[1]
    assert receives[0][2] is None
    assert (
        receives[0][1].untyped_storage().data_ptr()
        == contiguous.untyped_storage().data_ptr()
    )

    strided_target = torch.empty((4, 7))
    receives = impl._stage_rank_operations(
        1,
        None,
        _TensorRef(strided_target, (4, 7)),
        {0: ((0, 4), (2, 5))},
        {1: ((0, 4), (0, 7))},
        ((0, 1, ((0, 4), (2, 5))),),
        strided_target.device,
        strided_target.dtype,
    )[1]
    assert receives[0][1].shape == (4, 3)
    assert receives[0][2].shape == (4, 3)
    assert (
        receives[0][1].untyped_storage().data_ptr()
        != strided_target.untyped_storage().data_ptr()
    )


class _P2PComm:
    def __init__(self, fail_send=False):
        self.calls = []
        self.fail_send = fail_send

    def send(self, buffer, peer, stream):
        self.calls.append(("send", peer, stream, buffer.clone()))
        if self.fail_send:
            raise RuntimeError("send failed")

    def recv(self, buffer, peer, stream):
        self.calls.append(("recv", peer, stream, buffer))
        buffer.fill_(peer + 1)


def _install_fake_nccl(monkeypatch, events):
    package = types.ModuleType("nccl")
    package.__path__ = []
    core = types.ModuleType("nccl.core")
    core.group_start = lambda: events.append("start")
    core.group_end = lambda: events.append("end")
    monkeypatch.setitem(sys.modules, "nccl", package)
    monkeypatch.setitem(sys.modules, "nccl.core", core)


def test_exchange_groups_p2p_in_order_and_copies_staged_receive(monkeypatch):
    events = []
    _install_fake_nccl(monkeypatch, events)
    communicator = _P2PComm()
    destination = torch.zeros((2, 3))
    staging = torch.empty((2, 2))
    sends = [(5, torch.ones(4)), (7, torch.ones(2))]
    receives = [(3, staging, destination[:, 1:3])]
    impl._exchange_exact_overlaps(communicator, sends, receives, _Stream())
    assert events == ["start", "end"]
    assert [(call[0], call[1], call[2]) for call in communicator.calls] == [
        ("send", 5, _Stream.cuda_stream),
        ("send", 7, _Stream.cuda_stream),
        ("recv", 3, _Stream.cuda_stream),
    ]
    assert torch.equal(destination[:, 1:3], torch.full((2, 2), 4.0))


def test_exchange_always_closes_nccl_group_on_exception(monkeypatch):
    events = []
    _install_fake_nccl(monkeypatch, events)
    with pytest.raises(RuntimeError, match="send failed"):
        impl._exchange_exact_overlaps(
            _P2PComm(fail_send=True), [(1, torch.ones(1))], [], _Stream()
        )
    assert events == ["start", "end"]


class _BroadcastComm:
    def __init__(self, value=7):
        self.value = value
        self.calls = []

    def broadcast(self, sendbuf, recvbuf, root, stream):
        self.calls.append((sendbuf, recvbuf, root, stream))
        recvbuf.fill_(self.value)


@pytest.mark.parametrize("strided", (False, True))
def test_destination_broadcast_is_in_place_and_handles_strided_local(strided):
    base = torch.zeros((4, 6))
    destination = base if not strided else base[:, ::2]
    communicator = _BroadcastComm(value=9)
    impl._broadcast_destination(communicator, destination, _Stream())
    assert len(communicator.calls) == 1
    sendbuf, recvbuf, root, stream = communicator.calls[0]
    assert sendbuf is recvbuf
    assert root == 0
    assert stream == _Stream.cuda_stream
    assert torch.equal(destination, torch.full_like(destination, 9))
    if not strided:
        assert sendbuf is destination
    else:
        assert sendbuf is not destination


def test_empty_destination_broadcast_is_noop():
    communicator = _BroadcastComm()
    impl._broadcast_destination(communicator, torch.empty(0), _Stream())
    impl._broadcast_destination(None, torch.ones(1), _Stream())
    impl._broadcast_destination(communicator, None, _Stream())
    assert communicator.calls == []


def _patch_cuda_contexts(monkeypatch, synchronize_calls):
    monkeypatch.setattr(impl.torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        impl.torch.cuda,
        "synchronize",
        lambda device=None: synchronize_calls.append(device),
    )


@pytest.mark.parametrize(
    "rank,expected_color,expected_key,active",
    [
        (8, 0, 0, True),
        (2, 0, 1, True),
        (5, 0, 2, True),
        (9, 1, 0, True),
        (6, 1, 2, True),
        (12, 2, 12, False),
    ],
)
def test_replica_subcommunicator_split_colors_and_keys(
    monkeypatch, rank, expected_color, expected_key, active
):
    groups = (
        (((0, 8),), (8, 2, 5), 8),
        (((8, 16),), (9, 1, 6), 9),
    )
    communicator = _SplitComm()
    process_group = _ProcessGroup(rank=rank, communicator=communicator)
    synchronize_calls = []
    _patch_cuda_contexts(monkeypatch, synchronize_calls)
    result = impl._get_replica_subcommunicator(
        process_group, groups, torch.device("cuda", 0)
    )
    assert communicator.split_calls[0][:2] == (expected_color, expected_key)
    assert (result is not None) is active
    assert synchronize_calls == [torch.device("cuda", 0)]


def test_replica_subcommunicator_is_cached_and_empty_signature_does_not_split(
    monkeypatch,
):
    communicator = _SplitComm()
    process_group = _ProcessGroup(rank=0, communicator=communicator)
    synchronize_calls = []
    _patch_cuda_contexts(monkeypatch, synchronize_calls)
    groups = ((((0, 8),), (0, 1), 0),)
    first = impl._get_replica_subcommunicator(
        process_group, groups, torch.device("cuda", 0)
    )
    second = impl._get_replica_subcommunicator(
        process_group, groups, torch.device("cuda", 0)
    )
    assert first is second
    assert len(communicator.split_calls) == 1
    assert len(synchronize_calls) == 1

    impl.clear_xferdtensor_python_caches(process_group)
    no_replicas = ((((0, 8),), (0,), 0),)
    assert (
        impl._get_replica_subcommunicator(
            process_group, no_replicas, torch.device("cuda", 0)
        )
        is None
    )
    assert len(communicator.split_calls) == 1


def test_active_replica_signature_ignores_singletons_and_empty_regions():
    groups = (
        (((0, 4),), (0, 1), 0),
        (((4, 8),), (2,), 2),
        (((8, 8),), (3, 4), 3),
    )
    assert impl._active_replica_signature(groups) == ((0, (0, 1)),)


def test_explicit_and_finalizer_cleanup_evict_only_owned_communicators():
    first = _ProcessGroup()
    second = _ProcessGroup()
    first_id = impl._parent_communicator_key(first)
    second_id = impl._parent_communicator_key(second)
    impl._SUBCOMM_CACHE[(first_id, "first")] = object()
    impl._SUBCOMM_CACHE[(second_id, "second")] = object()

    impl.clear_xferdtensor_python_caches(first)
    assert all(key[0] != first_id for key in impl._SUBCOMM_CACHE)
    assert (second_id, "second") in impl._SUBCOMM_CACHE

    del second
    gc.collect()
    assert all(key[0] != second_id for key in impl._SUBCOMM_CACHE)


def test_non_weakrefable_process_group_still_gets_cache_key():
    class Group:
        __slots__ = ("rank", "nccl_communicator")

        def __init__(self):
            self.rank = 0
            self.nccl_communicator = _SplitComm()

    group = Group()
    assert impl._parent_communicator_key(group) == id(group.nccl_communicator)
    impl.clear_xferdtensor_python_caches(group)


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
