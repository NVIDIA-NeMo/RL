"""Exact-transfer, Python-only DTensor resharding.

``xferdtensor_golden_v3`` is a drop-in replacement for the seven-argument
``xferdtensor_golden`` API.  It avoids the reference implementation's global
tensor materialization and full-tensor broadcast:

1. Compute exact intersections between unique source and destination shards.
2. Transfer each destination element once to one representative replica.
3. Broadcast the completed destination shard in place to its replicas on a
   cached NCCL split communicator.

The implementation is self-contained apart from PyTorch and ``nccl4py``.  It
supports arbitrary Shard/Replicate layouts, including uneven and empty shards,
negative shard dimensions, repeated sharding of one tensor dimension, permuted
mesh ranks, and overlapping source/destination meshes.  Unsupported placement
types (for example Partial) are rejected explicitly.

All work is enqueued on the caller's current CUDA stream.  Contiguous source
and destination overlap views are passed directly to NCCL.  A temporary is
created only for a noncontiguous overlap, and is released after its final
same-stream operation has been enqueued.
"""

import weakref
from collections import OrderedDict

import torch
from torch.distributed._tensor import Replicate, Shard


# ---------------------------------------------------------------------------
# Mesh and shard geometry
# ---------------------------------------------------------------------------


def _mesh_rank_tensor(mesh):
    ranks = getattr(mesh, "mesh", None)
    if ranks is None:
        ranks = getattr(mesh, "_mesh", None)
    if ranks is None:
        raise ValueError("mesh does not expose a `.mesh` / `._mesh` rank tensor.")
    return ranks


def _mesh_ranks(mesh):
    return [int(rank) for rank in _mesh_rank_tensor(mesh).flatten().tolist()]


def _mesh_signature(mesh):
    ranks = _mesh_rank_tensor(mesh)
    return (
        tuple(int(size) for size in ranks.shape),
        tuple(int(rank) for rank in ranks.flatten().tolist()),
    )


def _mesh_coordinates(mesh):
    ranks = _mesh_rank_tensor(mesh)
    result = {}
    for index in range(ranks.numel()):
        coordinate = []
        remainder = index
        for size in reversed(ranks.shape):
            coordinate.append(remainder % int(size))
            remainder //= int(size)
        result[int(ranks.flatten()[index])] = tuple(reversed(coordinate))
    return result


def _normalize_shard_dim(dim, tensor_ndim):
    normalized = dim + tensor_ndim if dim < 0 else dim
    if normalized < 0 or normalized >= tensor_ndim:
        raise ValueError(f"Shard dim {dim} is invalid for a {tensor_ndim}D tensor.")
    return normalized


def _placement_signature(placements, tensor_ndim):
    signature = []
    for placement in placements:
        dim = getattr(placement, "dim", None)
        if isinstance(placement, Shard):
            dim = _normalize_shard_dim(dim, tensor_ndim)
        signature.append((type(placement).__name__, dim))
    return tuple(signature)


def _validate_layout(mesh, placements, global_shape, name):
    mesh_ndim = _mesh_rank_tensor(mesh).ndim
    if len(placements) != mesh_ndim:
        raise ValueError(
            f"{name}_placement has {len(placements)} entries for a {mesh_ndim}D mesh."
        )
    for placement in placements:
        if not isinstance(placement, (Shard, Replicate)):
            raise NotImplementedError(
                "xferdtensor_golden_v3 supports only Shard and Replicate "
                f"placements, got {type(placement).__name__}."
            )
        if isinstance(placement, Shard):
            _normalize_shard_dim(placement.dim, len(global_shape))


def _compute_shard_slices(global_shape, mesh_shape, coordinates, placements):
    """Return DTensor's sequential sharding slices for one mesh coordinate.

    Sequential semantics matter when multiple mesh dimensions shard the same
    tensor dimension: each placement shards the local chunk produced by the
    preceding placement, including its uneven remainder.
    """
    result = [slice(None) for _ in global_shape]
    shards_by_tensor_dim = OrderedDict()
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            tensor_dim = _normalize_shard_dim(placement.dim, len(global_shape))
            shards_by_tensor_dim.setdefault(tensor_dim, []).append(mesh_dim)

    for tensor_dim, mesh_dims in shards_by_tensor_dim.items():
        start = 0
        local_size = int(global_shape[tensor_dim])
        for mesh_dim in mesh_dims:
            mesh_size = int(mesh_shape[mesh_dim])
            coordinate = int(coordinates[mesh_dim])
            if coordinate < 0 or coordinate >= mesh_size:
                raise ValueError(
                    f"Invalid mesh coordinate {coordinate} for dimension {mesh_dim}."
                )
            # DTensor Shard follows torch.chunk rather than balanced
            # tensor_split semantics.  For example, 10 elements over 4 ranks
            # are 3,3,3,1 (not 3,3,2,2).  Apply that rule at every sequential
            # sharding step and retain explicit trailing empty shards.
            chunk_size = (local_size + mesh_size - 1) // mesh_size if local_size else 0
            relative_start = min(local_size, coordinate * chunk_size)
            remaining = max(0, local_size - relative_start)
            local_size = min(chunk_size, remaining)
            start += relative_start
        result[tensor_dim] = slice(start, start + local_size)
    return result


def _rank_regions(mesh, placements, global_shape):
    ranks = _mesh_rank_tensor(mesh)
    mesh_shape = tuple(int(size) for size in ranks.shape)
    coordinates = _mesh_coordinates(mesh)
    result = OrderedDict()
    for rank in _mesh_ranks(mesh):
        slices = _compute_shard_slices(
            global_shape, mesh_shape, coordinates[rank], placements
        )
        region = []
        for dim, shard_slice in enumerate(slices):
            start = 0 if shard_slice.start is None else int(shard_slice.start)
            stop = (
                int(global_shape[dim])
                if shard_slice.stop is None
                else int(shard_slice.stop)
            )
            region.append((start, stop))
        result[rank] = tuple(region)
    return result


def _intersect(left, right):
    overlap = []
    for (left_start, left_stop), (right_start, right_stop) in zip(left, right):
        start = max(left_start, right_start)
        stop = min(left_stop, right_stop)
        if start >= stop:
            return None
        overlap.append((start, stop))
    return tuple(overlap)


def _local_slices(overlap, owner_region):
    return tuple(
        slice(start - owner_region[dim][0], stop - owner_region[dim][0])
        for dim, (start, stop) in enumerate(overlap)
    )


def _region_numel(region):
    result = 1
    for start, stop in region:
        result *= stop - start
    return result


def _local_tensor(tensor):
    if tensor is None:
        return None
    local = getattr(tensor, "_local_tensor", None)
    if local is not None:
        return local
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return to_local()
    raise ValueError("tensor does not expose `._local_tensor` or `.to_local()`.")


def _tensor_metadata(src_tensor, dst_tensor):
    tensor = src_tensor if src_tensor is not None else dst_tensor
    if tensor is None:
        raise ValueError(
            "Unable to infer tensor metadata: this process-group rank has "
            "neither a source nor a destination tensor."
        )
    global_shape = tuple(int(size) for size in tensor.shape)
    if src_tensor is not None and dst_tensor is not None:
        if tuple(int(size) for size in src_tensor.shape) != tuple(
            int(size) for size in dst_tensor.shape
        ):
            raise ValueError("Source and destination global shapes do not match.")
        if src_tensor.dtype != dst_tensor.dtype:
            raise ValueError("Source and destination dtypes do not match.")
    return global_shape, tensor.device, tensor.dtype


# ---------------------------------------------------------------------------
# Exact transfer planning
# ---------------------------------------------------------------------------


def _destination_groups(dst_mesh, dst_placements, dst_regions):
    """Group identical destination regions and select one DP0 representative."""
    groups_by_region = OrderedDict()
    for rank in _mesh_ranks(dst_mesh):
        groups_by_region.setdefault(dst_regions[rank], []).append(rank)

    coordinates = _mesh_coordinates(dst_mesh)
    replicate_dims = [
        dim
        for dim, placement in enumerate(dst_placements)
        if isinstance(placement, Replicate)
    ]
    groups = []
    for region, members in groups_by_region.items():
        preferred = [
            rank
            for rank in members
            if all(coordinates[rank][dim] == 0 for dim in replicate_dims)
        ]
        representative = (preferred or members)[0]
        groups.append((region, tuple(members), representative))
    return tuple(groups)


def _build_exact_plan(src_regions, src_ranks, destination_groups):
    """Return ``(source, representative, overlap)`` transfers.

    Replicated source regions are deduplicated.  If a destination
    representative also owns a valid source replica, use it for a local copy;
    otherwise use the first holder in source-mesh order.
    """
    source_groups = OrderedDict()
    for rank in src_ranks:
        source_groups.setdefault(src_regions[rank], []).append(rank)

    transfers = []
    for dst_region, _members, representative in destination_groups:
        for src_region, holders in source_groups.items():
            overlap = _intersect(src_region, dst_region)
            if overlap is None:
                continue
            source = representative if representative in holders else holders[0]
            transfers.append((source, representative, overlap))

    # A single deterministic ordering is used by every rank.  In particular,
    # multiple messages between one peer pair are issued in matching order.
    transfers.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(transfers)


_PLAN_CACHE = OrderedDict()
_PLAN_CACHE_MAX_SIZE = 1024


def _plan_geometry(src_mesh, src_placements, dst_mesh, dst_placements, global_shape):
    _validate_layout(src_mesh, src_placements, global_shape, "src")
    _validate_layout(dst_mesh, dst_placements, global_shape, "dst")
    key = (
        _mesh_signature(src_mesh),
        _placement_signature(src_placements, len(global_shape)),
        _mesh_signature(dst_mesh),
        _placement_signature(dst_placements, len(global_shape)),
        tuple(global_shape),
    )
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        _PLAN_CACHE.move_to_end(key)
        return cached

    src_ranks = _mesh_ranks(src_mesh)
    src_regions = _rank_regions(src_mesh, src_placements, global_shape)
    dst_regions = _rank_regions(dst_mesh, dst_placements, global_shape)
    destination_groups = _destination_groups(dst_mesh, dst_placements, dst_regions)
    transfers = _build_exact_plan(src_regions, src_ranks, destination_groups)
    result = (src_regions, dst_regions, destination_groups, transfers)
    _PLAN_CACHE[key] = result
    if len(_PLAN_CACHE) > _PLAN_CACHE_MAX_SIZE:
        _PLAN_CACHE.popitem(last=False)
    return result


# ---------------------------------------------------------------------------
# Replica communicator lifetime and caching
# ---------------------------------------------------------------------------


_SUBCOMM_CACHE = {}
_PROCESS_GROUP_COMM_IDS = {}
_PROCESS_GROUP_FINALIZERS = {}


def _evict_communicators(comm_ids):
    comm_ids = set(comm_ids)
    for key in list(_SUBCOMM_CACHE):
        if key[0] in comm_ids:
            del _SUBCOMM_CACHE[key]


def _finalize_process_group(process_group_id):
    _evict_communicators(_PROCESS_GROUP_COMM_IDS.pop(process_group_id, ()))
    _PROCESS_GROUP_FINALIZERS.pop(process_group_id, None)


def _parent_communicator_key(process_group):
    communicator_id = id(process_group.nccl_communicator)
    process_group_id = id(process_group)
    _PROCESS_GROUP_COMM_IDS.setdefault(process_group_id, set()).add(communicator_id)
    if process_group_id not in _PROCESS_GROUP_FINALIZERS:
        try:
            _PROCESS_GROUP_FINALIZERS[process_group_id] = weakref.finalize(
                process_group, _finalize_process_group, process_group_id
            )
        except TypeError:
            # Some extension wrappers cannot be weak-referenced.  Their cache
            # entries can still be removed with the public cleanup function.
            pass
    return communicator_id


def _active_replica_signature(destination_groups):
    return tuple(
        (representative, members)
        for region, members, representative in destination_groups
        if len(members) > 1 and _region_numel(region) > 0
    )


def _get_replica_subcommunicator(process_group, destination_groups, device):
    """Collectively split once into all nonempty destination replica groups.

    The nccl4py version used by the target environment can hang if some peers
    use NCCL_SPLIT_NOCOLOR.  Therefore ranks outside an active replica group
    join one unused leftover color.  This preserves one collective split and
    does not add any data movement.
    """
    communicator = process_group.nccl_communicator
    rank = int(process_group.rank)
    signature = _active_replica_signature(destination_groups)
    key = (_parent_communicator_key(process_group), signature)
    if key in _SUBCOMM_CACHE:
        return _SUBCOMM_CACHE[key]
    if not signature:
        _SUBCOMM_CACHE[key] = None
        return None

    color = len(signature)
    split_key = rank
    active = False
    for group_color, (representative, members) in enumerate(signature):
        if rank not in members:
            continue
        ordered_members = (representative,) + tuple(
            member for member in members if member != representative
        )
        color = group_color
        split_key = ordered_members.index(rank)
        active = True
        break

    # NCCL requires all parent-communicator work to be quiescent before a new
    # ncclCommSplit.  A signature is paid for once, so a conservative device
    # synchronization on cache miss is both safe across changing caller streams
    # and negligible for the large, repeated FFN transfers this API serves.
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        subcommunicator = communicator.split(color, split_key)
    result = subcommunicator if active else None
    _SUBCOMM_CACHE[key] = result
    return result


def clear_xferdtensor_golden_v3_caches(process_group=None):
    """Clear cached plans and split communicators.

    This is part of the distributed call protocol: when a process group may be
    used again, every rank in that group must call cleanup in the same order
    after all GPU work has been synchronized.  Asymmetric eviction can make one
    rank enter a future collective split while its peers reuse a cached split.
    Passing a process group removes only that group's communicators; omitting it
    clears every local V3 cache and is intended for coordinated teardown.
    """
    if process_group is None:
        _PLAN_CACHE.clear()
        _SUBCOMM_CACHE.clear()
        for finalizer in _PROCESS_GROUP_FINALIZERS.values():
            finalizer.detach()
        _PROCESS_GROUP_FINALIZERS.clear()
        _PROCESS_GROUP_COMM_IDS.clear()
        return

    process_group_id = id(process_group)
    communicator_ids = _PROCESS_GROUP_COMM_IDS.pop(process_group_id, set())
    communicator = getattr(process_group, "nccl_communicator", None)
    if communicator is not None:
        communicator_ids.add(id(communicator))
    _evict_communicators(communicator_ids)
    finalizer = _PROCESS_GROUP_FINALIZERS.pop(process_group_id, None)
    if finalizer is not None:
        finalizer.detach()


# ---------------------------------------------------------------------------
# Communication
# ---------------------------------------------------------------------------


def _validate_local_inputs(
    rank, src_tensor, dst_tensor, src_regions, dst_regions, device, dtype
):
    """Validate local buffers before entering communicator split or P2P."""
    src_local = _local_tensor(src_tensor)
    dst_local = _local_tensor(dst_tensor)
    for name, local, region in (
        ("Source", src_local, src_regions.get(rank)),
        ("Destination", dst_local, dst_regions.get(rank)),
    ):
        if region is None:
            if local is not None:
                raise ValueError(
                    f"{name} tensor was provided on non-{name.lower()} rank {rank}."
                )
            continue
        if local is None:
            raise ValueError(
                f"{name} rank {rank} did not receive a {name.lower()} tensor."
            )
        expected_shape = tuple(stop - start for start, stop in region)
        if tuple(local.shape) != expected_shape:
            raise ValueError(
                f"{name} rank {rank} local shape {tuple(local.shape)} does not "
                f"match planned shape {expected_shape}."
            )
        if local.dtype != dtype:
            raise ValueError(
                f"{name} rank {rank} dtype {local.dtype} does not match {dtype}."
            )
        if local.device != device:
            raise ValueError(
                f"{name} rank {rank} device {local.device} does not match {device}."
            )


def _stage_rank_operations(
    rank,
    src_tensor,
    dst_tensor,
    src_regions,
    dst_regions,
    transfers,
    device,
    dtype,
):
    src_local = _local_tensor(src_tensor)
    dst_local = _local_tensor(dst_tensor)
    src_region = src_regions.get(rank)
    dst_region = dst_regions.get(rank)

    sends = []
    receives = []
    for source, representative, overlap in transfers:
        if source == representative:
            if rank == source:
                source_view = src_local[_local_slices(overlap, src_region)]
                destination_view = dst_local[_local_slices(overlap, dst_region)]
                destination_view.copy_(source_view)
            continue

        if rank == source:
            source_view = src_local[_local_slices(overlap, src_region)]
            send_buffer = (
                source_view if source_view.is_contiguous() else source_view.contiguous()
            )
            sends.append((representative, send_buffer))
        elif rank == representative:
            destination_view = dst_local[_local_slices(overlap, dst_region)]
            if destination_view.is_contiguous():
                receives.append((source, destination_view, None))
            else:
                shape = tuple(stop - start for start, stop in overlap)
                staging = torch.empty(shape, device=device, dtype=dtype)
                receives.append((source, staging, destination_view))
    return sends, receives, dst_local


def _exchange_exact_overlaps(communicator, sends, receives, stream):
    if not sends and not receives:
        return
    import nccl.core as nccl_core

    stream_handle = int(stream.cuda_stream)
    nccl_core.group_start()
    try:
        for peer, buffer in sends:
            communicator.send(buffer, peer, stream=stream_handle)
        for peer, buffer, _destination_view in receives:
            communicator.recv(buffer, peer, stream=stream_handle)
    finally:
        nccl_core.group_end()

    # Copies are enqueued immediately after their receives on the same stream,
    # so staging references need not survive the call.
    for _peer, buffer, destination_view in receives:
        if destination_view is not None:
            destination_view.copy_(buffer)


def _broadcast_destination(subcommunicator, dst_local, stream):
    if subcommunicator is None or dst_local is None or dst_local.numel() == 0:
        return
    # DTensor local tensors are normally contiguous.  Preserve generic behavior
    # for a strided local tensor with one bounded local-shard temporary.
    buffer = dst_local if dst_local.is_contiguous() else dst_local.contiguous()
    subcommunicator.broadcast(
        sendbuf=buffer,
        recvbuf=buffer,
        root=0,
        stream=int(stream.cuda_stream),
    )
    if buffer is not dst_local:
        dst_local.copy_(buffer)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def xferdtensor_golden_v3(
    src_tensor,
    src_mesh,
    src_placement,
    dst_tensor,
    dst_mesh,
    dst_placement,
    process_group,
) -> None:
    """Reshard ``src_tensor`` into ``dst_tensor`` with exact data movement.

    This has the same seven-argument signature and in-place effect as
    ``xferdtensor_golden``.  Every process-group rank must call it in the same
    order.  The process group is expected to cover the union of source and
    destination meshes, as required by the reference implementation.
    """
    rank = int(process_group.rank)
    global_shape, device, dtype = _tensor_metadata(src_tensor, dst_tensor)
    src_regions, dst_regions, destination_groups, transfers = _plan_geometry(
        src_mesh,
        src_placement,
        dst_mesh,
        dst_placement,
        global_shape,
    )

    # Split is collective on the parent communicator, so complete/cache it
    # before enqueuing parent-communicator P2P work.
    stream = torch.cuda.current_stream(device=device)
    with torch.cuda.device(device), torch.cuda.stream(stream):
        _validate_local_inputs(
            rank, src_tensor, dst_tensor, src_regions, dst_regions, device, dtype
        )
        subcommunicator = _get_replica_subcommunicator(
            process_group, destination_groups, device
        )
        sends, receives, dst_local = _stage_rank_operations(
            rank,
            src_tensor,
            dst_tensor,
            src_regions,
            dst_regions,
            transfers,
            device,
            dtype,
        )
        _exchange_exact_overlaps(
            process_group.nccl_communicator, sends, receives, stream
        )
        # nccl4py enqueued every use on this same stream.  Drop packing and
        # staging references before replica fan-out so the caching allocator
        # can reuse their storage as soon as those preceding operations finish.
        sends.clear()
        receives.clear()
        _broadcast_destination(subcommunicator, dst_local, stream)
    return None


__all__ = [
    "clear_xferdtensor_golden_v3_caches",
    "xferdtensor_golden_v3",
]
