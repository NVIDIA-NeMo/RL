"""Performant DTensor resharding with split NCCL sub-communicators.

This module is intentionally **self-contained** so it can be dropped into
another codebase as a single file.  Its only dependencies are ``torch`` and
``nccl4py`` (``nccl.core``, imported lazily inside the transfer function — the
same communicator a ``StatelessProcessGroup`` already wraps).

Public API
----------
``xferdtensor_golden_perf(src_tensor, src_mesh, src_placement,
                          dst_tensor, dst_mesh, dst_placement, process_group)``

Same 7-argument signature and effect as the broadcast-based reference
``xferdtensor_golden``: on each destination rank, ``dst_tensor._local_tensor``
is filled with that rank's resharded shard.  But instead of broadcasting the
whole global tensor to every rank, it does two cheap things:

  1. **Re-partition** — compute the exact overlap between each source shard and
     each *distinct* destination region, and move those pieces (only the
     resharded bytes) to the region's representative replica via one batched
     ``ncclGroupStart``/``ncclGroupEnd`` point-to-point group.
  2. **Replicate** — fan each region out to its DP replicas with an
     ``ncclBroadcast`` on a per-replica-group sub-communicator
     (``ncclCommSplit``).  NCCL's collective is bandwidth-optimal and broadcasts
     only the small shard, so this beats the full-tensor broadcast even when the
     destination is heavily replicated (large gen DP).

The idea is adapted from the C++ ``nccl-reshard`` library (ring-forward +
node-aware replication); here the topology-aware pipelining is delegated to
NCCL's own collective to stay Python-only and simple.

Duck-typed interfaces (so it works across separate ``torch.distributed`` worlds):
  * ``src_mesh`` / ``dst_mesh``  — expose ``.mesh`` (or ``._mesh``): an int
    tensor of global ranks, shaped as the device mesh.
  * ``src_tensor`` / ``dst_tensor`` — expose ``.shape`` (global shape) and
    ``._local_tensor`` (this rank's contiguous local shard); may be ``None``
    when this rank is dst-only / src-only.
  * ``process_group`` — exposes ``.rank`` and ``.nccl_communicator``.  The
    communicator must provide ``.send`` / ``.recv`` / ``.split`` /
    ``.broadcast``.
"""

import weakref
from collections import OrderedDict
from contextlib import nullcontext
from typing import Optional

import torch
from torch.distributed._tensor import Replicate, Shard


# =========================================================================
# Mesh / shard geometry helpers (self-contained copies)
# =========================================================================


def _mesh_rank_tensor(mesh):
    mesh_tensor = getattr(mesh, "mesh", None)
    if mesh_tensor is None:
        mesh_tensor = getattr(mesh, "_mesh", None)
    if mesh_tensor is None:
        raise ValueError("mesh does not expose a `.mesh` / `._mesh` rank tensor.")
    return mesh_tensor


def _flatten_mesh_ranks(mesh):
    return [int(r) for r in _mesh_rank_tensor(mesh).flatten().tolist()]


def _get_tensor_meta(src_tensor, dst_tensor):
    if src_tensor is not None:
        return src_tensor.shape, src_tensor.device
    if dst_tensor is not None:
        return dst_tensor.shape, dst_tensor.device
    device = torch.device("cuda", torch.cuda.current_device())
    return None, device


def _local_tensor(tensor):
    """Return a tensor-like object's local tensor without requiring DTensor."""
    if tensor is None:
        return None
    local = getattr(tensor, "_local_tensor", None)
    if local is not None:
        return local
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return to_local()
    raise ValueError("tensor does not expose `._local_tensor` or `.to_local()`.")


def _get_mesh_coords(mesh, rank):
    mesh_tensor = _mesh_rank_tensor(mesh)
    coords = (mesh_tensor == rank).nonzero(as_tuple=False)
    if coords.numel() == 0:
        return None
    return coords[0].tolist()


def _compute_shard_slices(global_shape, mesh_shape, mesh_coords, placements):
    """Slices of the global tensor owned by ``mesh_coords`` under ``placements``.

    Mirrors DTensor's sequential sharding semantics.  This distinction matters
    when more than one mesh dimension shards the same tensor dimension and the
    global size is uneven: DTensor shards the first local chunk again, rather
    than treating the mesh dimensions as one flattened even split.
    """
    slices = [slice(None) for _ in range(len(global_shape))]
    shard_map = OrderedDict()
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            tensor_dim = _normalize_shard_dim(placement.dim, len(global_shape))
            shard_map.setdefault(tensor_dim, []).append(
                (mesh_dim, mesh_shape[mesh_dim], mesh_coords[mesh_dim])
            )

    for tensor_dim, shard_info in shard_map.items():
        start = 0
        local_size = int(global_shape[tensor_dim])
        for mesh_dim, size, coord in shard_info:
            if coord >= size:
                raise ValueError(f"Invalid mesh coord {coord} for mesh dim {mesh_dim}.")
            base, remainder = divmod(local_size, size)
            relative_start = coord * base + min(coord, remainder)
            local_size = base + int(coord < remainder)
            start += relative_start
        slices[tensor_dim] = slice(start, start + local_size)

    return slices


def _normalize_shard_dim(dim, tensor_ndim):
    normalized = dim + tensor_ndim if dim < 0 else dim
    if normalized < 0 or normalized >= tensor_ndim:
        raise ValueError(f"Shard dim {dim} is invalid for a {tensor_ndim}D tensor.")
    return normalized


def _validate_geometry(mesh, placements, global_shape, name):
    mesh_tensor = _mesh_rank_tensor(mesh)
    if len(placements) != mesh_tensor.ndim:
        raise ValueError(
            f"{name}_placement has {len(placements)} entries for a "
            f"{mesh_tensor.ndim}D mesh."
        )
    ndim = len(global_shape)
    for placement in placements:
        if not isinstance(placement, (Shard, Replicate)):
            raise NotImplementedError(
                "xferdtensor_golden_perf supports only Shard and Replicate "
                f"placements, got {type(placement).__name__}."
            )
        if isinstance(placement, Shard):
            try:
                _normalize_shard_dim(placement.dim, ndim)
            except ValueError as error:
                raise ValueError(
                    f"{name} Shard dim {placement.dim} is invalid for a {ndim}D tensor."
                ) from error


def _shard_region(mesh, placement, global_shape, rank):
    """Global region (tuple of (start, end) per dim) that ``rank`` owns, or ``None`` if ``rank`` is not in the mesh."""
    coords = _get_mesh_coords(mesh, rank)
    if coords is None:
        return None
    mesh_shape = list(_mesh_rank_tensor(mesh).shape)
    slices = _compute_shard_slices(global_shape, mesh_shape, coords, placement)
    region = []
    for dim, s in enumerate(slices):
        start = s.start if s.start is not None else 0
        stop = s.stop if s.stop is not None else int(global_shape[dim])
        region.append((int(start), int(stop)))
    return tuple(region)


def _intersect_regions(a, b):
    """Per-dim intersection of two regions, or ``None`` if disjoint."""
    out = []
    for (a0, a1), (b0, b1) in zip(a, b):
        lo, hi = max(a0, b0), min(a1, b1)
        if lo >= hi:
            return None
        out.append((lo, hi))
    return tuple(out)


def _local_slices(overlap, owner_region):
    """Slices addressing ``overlap`` inside a local tensor whose global extent is ``owner_region`` (subtract the owner's lower corner)."""
    return tuple(
        slice(lo - owner_region[d][0], hi - owner_region[d][0])
        for d, (lo, hi) in enumerate(overlap)
    )


# =========================================================================
# Transfer plan
# =========================================================================


def _dst_replica_groups(dst_region):
    """Group destination ranks by their (identical) region.

    Returns an ``OrderedDict`` ``region -> sorted[replica ranks]``.  Ranks that
    share a region are DP replicas holding identical data; the lowest-ranked is
    the representative (and becomes the broadcast root).
    """
    groups = OrderedDict()
    for q in sorted(dst_region):
        groups.setdefault(dst_region[q], []).append(q)
    for reg in groups:
        groups[reg].sort()
    return groups


def _build_transfer_plan(src_region, dst_region, src_ranks):
    """Cross-mesh transfer plan: deliver each *distinct* destination region to its representative replica exactly once.

    Returns ``phase0`` — a list of ``(holder, rep, overlap)``.  Source regions
    tile the global tensor (dedup DP-replicated sources to one holder); each
    region is sent only to its representative (lowest-ranked replica), NOT to
    every replica — replication is handled afterwards by a sub-communicator
    broadcast (NCCL's bandwidth-optimal collective), so cross-mesh traffic is
    independent of the destination DP factor.
    """
    region_holders = OrderedDict()
    for r in src_ranks:
        region_holders.setdefault(src_region[r], []).append(r)
    for reg in region_holders:
        region_holders[reg].sort()

    phase0 = []
    for dreg, replicas in _dst_replica_groups(dst_region).items():
        rep = replicas[0]
        for sreg, holders in region_holders.items():
            overlap = _intersect_regions(sreg, dreg)
            if overlap is None:
                continue
            # Match ``DTensor.full_tensor()`` / golden semantics exactly when a
            # caller accidentally supplies inconsistent Replicate values: use
            # the canonical lowest-ranked holder, unless the destination itself
            # already owns a local copy.
            holder = rep if rep in holders else holders[0]
            phase0.append((holder, rep, overlap))
    phase0.sort(key=lambda t: (t[0], t[1], t[2]))
    return phase0


def _build_source_plan(src_region, dst_region, src_ranks):
    """Single-collective-round plan for the common 'destination finer than source' case (e.g. train TP4 -> gen TP8): when every destination region is contained in exactly ONE source shard, each source holder can broadcast its whole shard directly to *all* destination ranks that consume it — doing the cross-mesh transfer AND the DP replication in one ``ncclBroadcast`` of just that shard (S/src_tp), which beats golden's full-tensor broadcast.

    Returns ``(holder_of, consumers_of, active_holders)`` or ``None`` if any
    destination region draws from more than one source shard (then the caller
    uses the general two-stage path).
      * ``holder_of``      — ``{dst_rank -> source holder rank}``
      * ``consumers_of``   — ``{holder rank -> sorted[dst_ranks it feeds]}``
      * ``active_holders`` — sorted holders that feed >= 1 destination rank
    """
    # A rank can join only one color in ncclCommSplit.  On overlapping meshes a
    # rank may simultaneously be a holder and a consumer of another holder, so
    # use the fully general fallback instead of constructing an invalid split.
    if set(src_ranks).intersection(dst_region):
        return None

    region_holders = OrderedDict()
    for r in src_ranks:
        region_holders.setdefault(src_region[r], []).append(r)
    for reg in region_holders:
        region_holders[reg].sort()

    holder_of = {}
    consumers_of = OrderedDict()
    holder_for_region = {
        region: holders[0] for region, holders in region_holders.items()
    }
    for q in sorted(dst_region):
        dreg = dst_region[q]
        overlapping = [
            sreg
            for sreg in region_holders
            if _intersect_regions(sreg, dreg) is not None
        ]
        if len(overlapping) != 1:
            return None  # multi-source destination region -> not eligible
        holder = holder_for_region[overlapping[0]]
        holder_of[q] = holder
        consumers_of.setdefault(holder, []).append(q)
    for h in consumers_of:
        consumers_of[h].sort()
    active_holders = sorted(consumers_of.keys())
    return holder_of, consumers_of, active_holders


# The transfer plan is a pure function of the *geometry* (mesh rank layouts,
# placements, global shape) — never the tensor data — so it is identical on
# every pass of a refit loop.  Cache it so repeated reshards of the same shape
# skip the per-rank ``.nonzero()`` mesh lookups and region intersection.
_PLAN_CACHE = OrderedDict()
_PLAN_CACHE_MAX_SIZE = 1024
# Sub-communicators (created by ncclCommSplit).  Creating them is collective and
# not free, so cache them.  ``_SUBCOMM_CACHE`` holds the DP-replica sub-comms
# (two-stage fallback); ``_SRC_SUBCOMM_CACHE`` holds the source-holder sub-comms
# (single-round fast path).  Keys include exact rank-membership signatures, so
# shapes with the same grouping reuse communicators while zero/uneven shapes
# that change grouping cannot collide.
_SUBCOMM_CACHE = {}
_SRC_SUBCOMM_CACHE = {}
# Sub-communicators must not keep transient PP process groups alive forever.
# A weak finalizer evicts every cache entry owned by a process group when the
# wrapper dies, avoiding communicator-pool exhaustion in long-running jobs.
_PROCESS_GROUP_COMM_IDS = {}
_PROCESS_GROUP_FINALIZERS = {}


def _placement_sig(placement, tensor_ndim=None):
    signature = []
    for item in placement:
        dim = getattr(item, "dim", None)
        if isinstance(item, Shard) and tensor_ndim is not None:
            dim = _normalize_shard_dim(dim, tensor_ndim)
        signature.append((type(item).__name__, dim))
    return tuple(signature)


def _mesh_sig(mesh):
    """Full, unambiguous mesh signature: the rank layout *and its shape*.

    Keying on the flattened rank set alone is NOT enough — two meshes can hold
    the same ranks but in transposed shapes (e.g. ``[DP2,TP4]`` vs
    ``[DP4,TP2]``), which yield different shard regions and therefore different
    transfer plans.  Include the shape so such meshes never collide in the
    cache (a collision there silently reuses the wrong plan → mismatched
    send/recv counts → NCCL deadlock).
    """
    t = _mesh_rank_tensor(mesh)
    return (tuple(t.shape), tuple(int(r) for r in t.flatten().tolist()))


def _plan_geometry(src_mesh, src_placement, dst_mesh, dst_placement, global_shape):
    """Return cached split-mode plans for this geometry.

    ``source_plan`` is the single-round source-broadcast plan (see
    ``_build_source_plan``) when eligible, else ``None`` (use the two-stage
    fallback).
    """
    key = (
        _mesh_sig(src_mesh),
        _placement_sig(src_placement, len(global_shape)),
        _mesh_sig(dst_mesh),
        _placement_sig(dst_placement, len(global_shape)),
        tuple(int(d) for d in global_shape),
    )
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        _PLAN_CACHE.move_to_end(key)
        return cached
    _validate_geometry(src_mesh, src_placement, global_shape, "src")
    _validate_geometry(dst_mesh, dst_placement, global_shape, "dst")
    src_ranks = _flatten_mesh_ranks(src_mesh)
    dst_ranks = _flatten_mesh_ranks(dst_mesh)
    src_region = {
        r: _shard_region(src_mesh, src_placement, global_shape, r) for r in src_ranks
    }
    dst_region = {
        r: _shard_region(dst_mesh, dst_placement, global_shape, r) for r in dst_ranks
    }
    phase0 = _build_transfer_plan(src_region, dst_region, src_ranks)
    source_plan = _build_source_plan(src_region, dst_region, src_ranks)
    result = (src_region, dst_region, phase0, source_plan)
    _PLAN_CACHE[key] = result
    if len(_PLAN_CACHE) > _PLAN_CACHE_MAX_SIZE:
        _PLAN_CACHE.popitem(last=False)
    return result


def _evict_comm_ids(comm_ids):
    comm_ids = set(comm_ids)
    for cache in (_SUBCOMM_CACHE, _SRC_SUBCOMM_CACHE):
        for key in list(cache):
            if key[0] in comm_ids:
                del cache[key]


def _finalize_process_group(process_group_id):
    comm_ids = _PROCESS_GROUP_COMM_IDS.pop(process_group_id, ())
    _evict_comm_ids(comm_ids)
    _PROCESS_GROUP_FINALIZERS.pop(process_group_id, None)


def _parent_comm_key(process_group):
    comm = process_group.nccl_communicator
    comm_id = id(comm)
    process_group_id = id(process_group)
    _PROCESS_GROUP_COMM_IDS.setdefault(process_group_id, set()).add(comm_id)
    if process_group_id not in _PROCESS_GROUP_FINALIZERS:
        try:
            _PROCESS_GROUP_FINALIZERS[process_group_id] = weakref.finalize(
                process_group, _finalize_process_group, process_group_id
            )
        except TypeError:
            # Extension wrappers without weakref support are uncommon.  They
            # can still be released explicitly with the public cache clearer.
            pass
    return comm_id


def clear_xferdtensor_golden_perf_caches(process_group=None):
    """Release cached plans/sub-communicators after outstanding work completes.

    With ``process_group`` only communicator caches owned by that group are
    evicted.  With no argument all geometry and communicator caches are reset.
    Callers must ensure no operation using those sub-communicators is in flight.
    """
    if process_group is None:
        _PLAN_CACHE.clear()
        _SUBCOMM_CACHE.clear()
        _SRC_SUBCOMM_CACHE.clear()
        for finalizer in _PROCESS_GROUP_FINALIZERS.values():
            finalizer.detach()
        _PROCESS_GROUP_FINALIZERS.clear()
        _PROCESS_GROUP_COMM_IDS.clear()
        return

    process_group_id = id(process_group)
    comm_ids = _PROCESS_GROUP_COMM_IDS.pop(process_group_id, set())
    comm = getattr(process_group, "nccl_communicator", None)
    if comm is not None:
        comm_ids.add(id(comm))
    _evict_comm_ids(comm_ids)
    finalizer = _PROCESS_GROUP_FINALIZERS.pop(process_group_id, None)
    if finalizer is not None:
        finalizer.detach()


def _replica_membership_sig(dst_region):
    return tuple(tuple(members) for members in _dst_replica_groups(dst_region).values())


def _source_membership_sig(source_plan):
    _holder_of, consumers_of, _active_holders = source_plan
    return tuple(
        (holder, tuple(consumers)) for holder, consumers in consumers_of.items()
    )


def _get_replica_subcomm(process_group, dst_mesh, dst_placement, global_shape):
    """Collectively split ``process_group``'s communicator into one sub-comm per DP-replica group (destination regions held by >1 rank), and return THIS rank's sub-comm — with the lowest-ranked replica as sub-rank 0 (the broadcast root) — or ``None`` if this rank holds no replicated region.

    ``ncclCommSplit`` is collective: EVERY rank in ``process_group`` must call
    it.  We do NOT use ``NCCL_SPLIT_NOCOLOR`` — in this nccl4py build the
    colored ranks hang in ``split`` when any peer passes NOCOLOR — so every rank
    gets a real color: replicated regions get their own color (the useful
    sub-comms), and all other ranks (sources + singleton destinations) share one
    "leftover" color whose sub-comm is created but never used.  When there are
    no replicated regions at all (pure re-partition), the split is skipped
    entirely (consistently on all ranks).  Cached per (parent comm, dst geom).
    """
    comm = process_group.nccl_communicator
    rank = process_group.rank
    # Key WITHOUT shape: the replica *grouping* (which ranks hold identical data)
    # is a function of (mesh, placement) only — not the tensor shape — so every
    # parameter with the same placement reuses one sub-comm.  This collapses
    # hundreds of per-shape splits into a handful (avoids exhausting NCCL's
    # communicator pool) and skips redundant collective splits.
    dst_region = {
        r: _shard_region(dst_mesh, dst_placement, global_shape, r)
        for r in _flatten_mesh_ranks(dst_mesh)
    }
    key = (
        _parent_comm_key(process_group),
        _mesh_sig(dst_mesh),
        _placement_sig(dst_placement, len(global_shape)),
        _replica_membership_sig(dst_region),
    )
    if key in _SUBCOMM_CACHE:
        return _SUBCOMM_CACHE[key]

    groups = _dst_replica_groups(dst_region)
    multi = sorted(reg for reg, members in groups.items() if len(members) > 1)
    if not multi:
        # No replication anywhere -> no rank needs a sub-comm; skip the split
        # (same decision on every rank, so no collective mismatch).
        _SUBCOMM_CACHE[key] = None
        return None

    region_color = {reg: i for i, reg in enumerate(multi)}
    leftover_color = len(multi)  # one shared group for all non-replica ranks
    my_region = dst_region.get(rank)
    in_replica_group = my_region in region_color
    color = region_color[my_region] if in_replica_group else leftover_color
    # key=rank so the lowest global rank in each group becomes sub-rank 0 (root).
    sub = comm.split(color, rank)
    result = sub if in_replica_group else None  # leftover sub-comm is unused
    _SUBCOMM_CACHE[key] = result
    return result


def _get_source_subcomm(
    process_group,
    src_mesh,
    src_placement,
    dst_mesh,
    dst_placement,
    source_plan,
    global_shape,
):
    """Collectively split into one sub-comm per active *source holder* — each holding {the holder} ∪ {every destination rank that consumes its shard}, with the holder (lowest rank) as sub-rank 0 (the broadcast root).  Returns THIS rank's sub-comm, or ``None`` if it is idle (a duplicate source replica).

    Same NOCOLOR-avoidance as ``_get_replica_subcomm`` (idle ranks share a
    leftover color).  Cached per (parent comm, src geom, dst geom); the grouping
    is shape-independent.
    """
    comm = process_group.nccl_communicator
    rank = process_group.rank
    key = (
        _parent_comm_key(process_group),
        _mesh_sig(src_mesh),
        _placement_sig(src_placement, len(global_shape)),
        _mesh_sig(dst_mesh),
        _placement_sig(dst_placement, len(global_shape)),
        _source_membership_sig(source_plan),
    )
    if key in _SRC_SUBCOMM_CACHE:
        return _SRC_SUBCOMM_CACHE[key]

    holder_of, consumers_of, active_holders = source_plan
    holder_color = {h: i for i, h in enumerate(active_holders)}
    leftover_color = len(active_holders)
    if rank in consumers_of:  # active holder -> root of its group
        color, active = holder_color[rank], True
    elif rank in holder_of:  # consumer -> joins its holder's group
        color, active = holder_color[holder_of[rank]], True
    else:  # idle (duplicate source replica)
        color, active = leftover_color, False
    sub = comm.split(color, rank)
    result = sub if active else None
    _SRC_SUBCOMM_CACHE[key] = result
    return result


def _source_prepare(
    rank,
    sub,
    src_tensor,
    dst_tensor,
    src_region,
    dst_region,
    source_plan,
    device,
    dtype,
):
    """For the source-broadcast path, return ``(buf, extract)`` for this rank: ``buf`` is the tensor to broadcast on ``sub`` (root=0), and ``extract`` is ``(dst_local, temp, src_slice)`` to copy in after the broadcast (consumer) or ``None`` (holder).  ``buf`` is ``None`` for idle ranks (no broadcast)."""
    if sub is None:
        return None, None
    holder_of, consumers_of, _ = source_plan
    if rank in consumers_of:
        # Active holder: broadcast our whole source shard to all consumers.
        return _local_tensor(src_tensor), None
    # Consumer: receive the holder's shard into a temp, then slice out our region.
    holder = holder_of[rank]
    h_region = src_region[holder]
    size = tuple(hi - lo for lo, hi in h_region)
    temp = torch.empty(size, device=device, dtype=dtype)
    dst_local = _local_tensor(dst_tensor)
    src_slice = _local_slices(dst_region[rank], h_region)
    return temp, (dst_local, temp, src_slice)


# =========================================================================
# Public API
# =========================================================================


def xferdtensor_golden_v2(
    src_tensor,
    src_mesh,
    src_placement,
    dst_tensor,
    dst_mesh,
    dst_placement,
    process_group,
    stream: Optional[torch.cuda.Stream] = None,
):
    """Reshard ``src_tensor`` -> ``dst_tensor``, beating the broadcast golden.

    Drop-in replacement for ``xferdtensor_golden`` with the same 7-argument
    signature and effect, but only the overlapping regions between source and
    destination shards move (no full-tensor broadcast).  Sub-communicators are
    created with ``ncclCommSplit`` and reused across calls.

    * **Fast path (destination finer than source — the common refit case):**
      every destination region comes from one source shard, so each source
      holder broadcasts its whole shard to all consuming ranks in ONE
      collective on a {holder ∪ consumers} sub-comm — cross-mesh transfer and
      DP replication together, broadcasting only ``tensor / src_tp``.  ~src_tp×
      less per-rank data than golden's full-tensor broadcast.
    * **General fallback:** (1) cross-mesh re-partition delivers each distinct
      region to its representative replica via batched point-to-point, then (2)
      a per-replica-group sub-comm ``ncclBroadcast`` fans it out to the DP
      replicas.
    Plans and sub-communicators are pure geometry, cached by shape/placement,
    so a refit loop pays planning + split cost only once.
    """
    rank = process_group.rank
    global_shape, device = _get_tensor_meta(src_tensor, dst_tensor)
    if global_shape is None:
        raise ValueError("Unable to infer tensor shape/device from src or dst tensor.")
    if stream is None:
        stream = torch.cuda.current_stream()
    sint = int(stream.cuda_stream)
    dtype = src_tensor.dtype if src_tensor is not None else dst_tensor.dtype

    src_region, dst_region, _phase0, source_plan = _plan_geometry(
        src_mesh, src_placement, dst_mesh, dst_placement, global_shape
    )

    if source_plan is not None:
        # Fast path: one source-holder broadcast per group (split is collective).
        sub = _get_source_subcomm(
            process_group,
            src_mesh,
            src_placement,
            dst_mesh,
            dst_placement,
            source_plan,
            global_shape,
        )
        with torch.cuda.stream(stream):
            buf, extract = _source_prepare(
                rank,
                sub,
                src_tensor,
                dst_tensor,
                src_region,
                dst_region,
                source_plan,
                device,
                dtype,
            )
            if buf is not None:
                sub.broadcast(sendbuf=buf, recvbuf=buf, root=0, stream=sint)
            if extract is not None:
                dst_local, temp, src_slice = extract
                dst_local.copy_(temp[src_slice])
        return

    # General fallback: cross-mesh P2P re-partition + replica sub-comm broadcast.
    sub = _get_replica_subcomm(process_group, dst_mesh, dst_placement, global_shape)
    sends, recvs, dst_local = _plan_and_stage(
        rank,
        src_tensor,
        src_mesh,
        src_placement,
        dst_tensor,
        dst_mesh,
        dst_placement,
        stream=stream,
    )
    _exchange_p2p(process_group.nccl_communicator, sends, recvs, stream)
    if sub is not None and dst_local is not None:
        sub.broadcast(sendbuf=dst_local, recvbuf=dst_local, root=0, stream=sint)


def _plan_and_stage(
    rank,
    src_tensor,
    src_mesh,
    src_placement,
    dst_tensor,
    dst_mesh,
    dst_placement,
    stream=None,
):
    """Compute this rank's cross-mesh (stage-0) plan, do local copies immediately, and return ``(sends, recvs, dst_local)``.

    ``sends`` = list of ``(peer, contiguous_send_buf)``.
    ``recvs`` = list of ``(peer, recv_buf, dst_view_or_None)`` — ``None`` view
    means received in place; otherwise a staged (strided) recv copied into the
    view after the group.  ``dst_local`` is this rank's local destination shard
    (the buffer the replica broadcast fans out), or ``None`` if not a dst rank.
    """
    global_shape, device = _get_tensor_meta(src_tensor, dst_tensor)
    if global_shape is None:
        raise ValueError("Unable to infer tensor shape/device from src or dst tensor.")
    dtype = src_tensor.dtype if src_tensor is not None else dst_tensor.dtype

    src_region, dst_region, phase0, _source_plan = _plan_geometry(
        src_mesh, src_placement, dst_mesh, dst_placement, global_shape
    )

    src_local = _local_tensor(src_tensor)
    dst_local = _local_tensor(dst_tensor)
    my_src_region = src_region.get(rank)
    my_dst_region = dst_region.get(rank)

    sends, recvs = [], []
    context = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with context:
        for holder, rep, overlap in phase0:
            if holder == rep:
                # Colocated (src holder is also the destination rank): local copy.
                if rank == holder:
                    ssl = _local_slices(overlap, my_src_region)
                    dsl = _local_slices(overlap, my_dst_region)
                    dst_local[dsl].copy_(src_local[ssl])
                continue
            if holder == rank:
                view = src_local[_local_slices(overlap, my_src_region)]
                buf = view if view.is_contiguous() else view.contiguous()
                sends.append((rep, buf))
            elif rep == rank:
                dst_view = dst_local[_local_slices(overlap, my_dst_region)]
                if dst_view.is_contiguous():
                    recvs.append((holder, dst_view, None))
                else:
                    size = tuple(hi - lo for lo, hi in overlap)
                    recvs.append(
                        (
                            holder,
                            torch.empty(size, device=device, dtype=dtype),
                            dst_view,
                        )
                    )
    return sends, recvs, dst_local


def _exchange_p2p(comm, sends, recvs, stream):
    """Issue all cross-mesh sends/recvs in one ncclGroupStart/End, then scatter staged (strided) receives into their destination views."""
    if not sends and not recvs:
        return
    import nccl.core as _nccl_core

    sint = int(stream.cuda_stream)
    _nccl_core.group_start()
    try:
        for peer, buf in sends:
            comm.send(buf, peer, stream=sint)
        for peer, buf, _ in recvs:
            comm.recv(buf, peer, stream=sint)
    finally:
        _nccl_core.group_end()
    with torch.cuda.stream(stream):
        for _, buf, dst_view in recvs:
            if dst_view is not None:
                dst_view.copy_(buf)
