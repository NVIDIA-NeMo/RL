"""Performant DTensor resharding: point-to-point re-partition + sub-comm broadcast.

This module is intentionally **self-contained** so it can be dropped into
another codebase as a single file.  Its only dependencies are ``torch`` and
``nccl4py`` (``nccl.core``, imported lazily inside the transfer function — the
same communicator a ``StatelessProcessGroup`` already wraps).

Public API
----------
``xferdtensor_golden_v2(src_tensor, src_mesh, src_placement,
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
  * ``process_group`` — exposes ``.rank`` and ``.nccl_communicator`` (an
    nccl4py ``Communicator`` with ``.send`` / ``.recv`` / ``.split`` /
    ``.broadcast``).
"""

from collections import OrderedDict
from typing import Optional

import torch
from torch.distributed._tensor import Shard

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


def _get_mesh_coords(mesh, rank):
    mesh_tensor = _mesh_rank_tensor(mesh)
    coords = (mesh_tensor == rank).nonzero(as_tuple=False)
    if coords.numel() == 0:
        return None
    return coords[0].tolist()


def _compute_shard_slices(global_shape, mesh_shape, mesh_coords, placements):
    """Slices of the global tensor owned by ``mesh_coords`` under ``placements``.

    Mirrors DTensor's even-sharding split: a tensor dim sharded over one or
    more mesh dims is cut into ``prod(mesh sizes)`` contiguous chunks (the
    first ``remainder`` chunks get one extra element).
    """
    slices = [slice(None) for _ in range(len(global_shape))]
    shard_map = {}
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_map.setdefault(placement.dim, []).append(
                (mesh_dim, mesh_shape[mesh_dim], mesh_coords[mesh_dim])
            )

    for tensor_dim, shard_info in shard_map.items():
        shard_info.sort(key=lambda item: item[0])
        num_chunks = 1
        for _, size, _ in shard_info:
            num_chunks *= size

        total_size = int(global_shape[tensor_dim])
        base = total_size // num_chunks
        remainder = total_size % num_chunks
        sizes = [base + 1 if i < remainder else base for i in range(num_chunks)]

        strides = []
        running = 1
        for _, size, _ in reversed(shard_info):
            strides.append(running)
            running *= size
        strides.reverse()

        linear_index = 0
        for (mesh_dim, size, coord), stride in zip(shard_info, strides):
            if coord >= size:
                raise ValueError(f"Invalid mesh coord {coord} for mesh dim {mesh_dim}.")
            linear_index += coord * stride

        start = sum(sizes[:linear_index])
        end = start + sizes[linear_index]
        slices[tensor_dim] = slice(start, end)

    return slices


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
    region_holders = OrderedDict()
    for r in src_ranks:
        region_holders.setdefault(src_region[r], []).append(r)
    for reg in region_holders:
        region_holders[reg].sort()

    holder_of = {}
    consumers_of = OrderedDict()
    for q in sorted(dst_region):
        dreg = dst_region[q]
        overlapping = [
            sreg
            for sreg in region_holders
            if _intersect_regions(sreg, dreg) is not None
        ]
        if len(overlapping) != 1:
            return None  # multi-source destination region -> not eligible
        holder = region_holders[overlapping[0]][0]
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
_PLAN_CACHE = {}
# Sub-communicators (created by ncclCommSplit).  Creating them is collective and
# not free, so cache them.  ``_SUBCOMM_CACHE`` holds the DP-replica sub-comms
# (two-stage fallback); ``_SRC_SUBCOMM_CACHE`` holds the source-holder sub-comms
# (single-round fast path).  Both keyed WITHOUT shape (grouping is shape-free).
_SUBCOMM_CACHE = {}
_SRC_SUBCOMM_CACHE = {}
# One reusable side stream per device for the batched 2-stream pipeline.
_SIDE_STREAM = {}


def _side_stream(device):
    s = _SIDE_STREAM.get(device)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _SIDE_STREAM[device] = s
    return s


def _placement_sig(placement):
    return tuple((type(p).__name__, getattr(p, "dim", None)) for p in placement)


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
    """Return ``(src_region, dst_region, phase0, source_plan)`` for this geometry, computing it once and caching by a data-independent signature.

    ``source_plan`` is the single-round source-broadcast plan (see
    ``_build_source_plan``) when eligible, else ``None`` (use the two-stage
    fallback).
    """
    key = (
        _mesh_sig(src_mesh),
        _placement_sig(src_placement),
        _mesh_sig(dst_mesh),
        _placement_sig(dst_placement),
        tuple(int(d) for d in global_shape),
    )
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached
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
    return result


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
    key = (
        id(comm),
        _mesh_sig(dst_mesh),
        _placement_sig(dst_placement),
    )
    if key in _SUBCOMM_CACHE:
        return _SUBCOMM_CACHE[key]

    dst_region = {
        r: _shard_region(dst_mesh, dst_placement, global_shape, r)
        for r in _flatten_mesh_ranks(dst_mesh)
    }
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
    process_group, src_mesh, src_placement, dst_mesh, dst_placement, source_plan
):
    """Collectively split into one sub-comm per active *source holder* — each holding {the holder} ∪ {every destination rank that consumes its shard}, with the holder (lowest rank) as sub-rank 0 (the broadcast root).  Returns THIS rank's sub-comm, or ``None`` if it is idle (a duplicate source replica).

    Same NOCOLOR-avoidance as ``_get_replica_subcomm`` (idle ranks share a
    leftover color).  Cached per (parent comm, src geom, dst geom); the grouping
    is shape-independent.
    """
    comm = process_group.nccl_communicator
    rank = process_group.rank
    key = (
        id(comm),
        _mesh_sig(src_mesh),
        _placement_sig(src_placement),
        _mesh_sig(dst_mesh),
        _placement_sig(dst_placement),
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
        return src_tensor._local_tensor, None
    # Consumer: receive the holder's shard into a temp, then slice out our region.
    holder = holder_of[rank]
    h_region = src_region[holder]
    size = tuple(hi - lo for lo, hi in h_region)
    temp = torch.empty(size, device=device, dtype=dtype)
    dst_local = (
        dst_tensor._local_tensor
        if hasattr(dst_tensor, "_local_tensor")
        else dst_tensor.to_local()
    )
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
    destination shards move (no full-tensor broadcast).  Both communication
    paths are NCCL collectives (bandwidth-optimal), adapting the C++
    ``nccl-reshard`` idea (source broadcast + replication) to Python by
    delegating the topology-aware pipelining to NCCL's own ``ncclBroadcast``.

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

    src_region, dst_region, phase0, source_plan = _plan_geometry(
        src_mesh, src_placement, dst_mesh, dst_placement, global_shape
    )

    if source_plan is not None:
        # Fast path: one source-holder broadcast per group (split is collective).
        sub = _get_source_subcomm(
            process_group, src_mesh, src_placement, dst_mesh, dst_placement, source_plan
        )
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
        rank, src_tensor, src_mesh, src_placement, dst_tensor, dst_mesh, dst_placement
    )
    _exchange_p2p(process_group.nccl_communicator, sends, recvs, sint)
    if sub is not None and dst_local is not None:
        sub.broadcast(sendbuf=dst_local, recvbuf=dst_local, root=0, stream=sint)


def xferdtensor_golden_v2_batched(transfers, process_group, stream=None):
    """Reshard many tensors with a 2-stream pipeline that overlaps the cross-mesh transfer with the sub-comm broadcasts.

    ``transfers`` is an iterable of 6-tuples
    ``(src_tensor, src_mesh, src_placement, dst_tensor, dst_mesh, dst_placement)``
    — the same per-tensor arguments as :func:`xferdtensor_golden_v2`.

    Resharding many small parameters one at a time leaves the GPU/NICs idle
    between latency-bound collectives.  This pipelines across parameters on two
    CUDA streams to hide that latency:

      * **Stream A** runs the cross-mesh point-to-point re-partition (on the
        *global* communicator).
      * **Stream B** runs every sub-communicator ``ncclBroadcast`` (source
        fast-path + replica replication).

    A CUDA event enforces each fallback param's stage-1→stage-2 dependency, so
    param N's replica broadcast (B) overlaps param N+1's cross-mesh P2P (A), and
    fast-path source broadcasts (B) overlap expert stage-1 (A).

    NCCL safety: each communicator's operations stay on exactly ONE stream
    (global→A, all sub-comms→B), so there is never a concurrent same-comm use —
    avoiding the classic multi-stream deadlock.  Every rank must pass
    ``transfers`` in the same order so posting order matches across ranks.
    """
    import nccl.core as _nccl_core

    rank = process_group.rank
    comm = process_group.nccl_communicator
    if stream is None:
        stream = torch.cuda.current_stream()
    stream_a = stream  # cross-mesh P2P (global comm)
    stream_b = _side_stream(stream.device)  # sub-comm broadcasts
    sa = int(stream_a.cuda_stream)
    sb = int(stream_b.cuda_stream)

    keepalive = []  # send/recv buffers must outlive the async NCCL ops
    extracts = []  # (dst_local, temp, slice) fast-path consumer copies (on B)
    used_b = False
    for st, sm, sp, dt, dm, dp in transfers:
        global_shape, device = _get_tensor_meta(st, dt)
        if global_shape is None:
            raise ValueError(
                "Unable to infer tensor shape/device from src or dst tensor."
            )
        dtype = st.dtype if st is not None else dt.dtype
        src_region, dst_region, phase0, source_plan = _plan_geometry(
            sm, sp, dm, dp, global_shape
        )

        if source_plan is not None:
            # Fast path: one source-holder broadcast, on stream B (no A dep).
            sub = _get_source_subcomm(process_group, sm, sp, dm, dp, source_plan)
            buf, extract = _source_prepare(
                rank, sub, st, dt, src_region, dst_region, source_plan, device, dtype
            )
            if buf is not None:
                sub.broadcast(sendbuf=buf, recvbuf=buf, root=0, stream=sb)
                keepalive.append(buf)
                used_b = True
            if extract is not None:
                extracts.append(extract)
            continue

        # Fallback: cross-mesh P2P on A, then replica broadcast on B (after A).
        sub = _get_replica_subcomm(process_group, dm, dp, global_shape)
        sends, recvs, dst_local = _plan_and_stage(rank, st, sm, sp, dt, dm, dp)
        if sends or recvs:
            keepalive.append((sends, recvs))
            _nccl_core.group_start()
            try:
                for peer, b in sends:
                    comm.send(b, peer, stream=sa)
                for peer, b, _ in recvs:
                    comm.recv(b, peer, stream=sa)
            finally:
                _nccl_core.group_end()
            # Scatter staged (strided) receives into their views, on A.
            with torch.cuda.stream(stream_a):
                for _, b, dst_view in recvs:
                    if dst_view is not None:
                        dst_view.copy_(b)
        if sub is not None and dst_local is not None:
            # B replicates this param only after A finished its cross-mesh recv.
            ev = torch.cuda.Event()
            ev.record(stream_a)
            stream_b.wait_event(ev)
            sub.broadcast(sendbuf=dst_local, recvbuf=dst_local, root=0, stream=sb)
            used_b = True

    # Fast-path consumers slice their region out of the received holder shard (B).
    if extracts:
        with torch.cuda.stream(stream_b):
            for dst_local, temp, src_slice in extracts:
                dst_local.copy_(temp[src_slice])
    # Make the caller's stream (A) wait for all of B's work.
    if used_b:
        ev = torch.cuda.Event()
        ev.record(stream_b)
        stream_a.wait_event(ev)
    # ``keepalive`` holds NCCL buffers until the caller's stream is joined above.
    del keepalive


def _plan_and_stage(
    rank, src_tensor, src_mesh, src_placement, dst_tensor, dst_mesh, dst_placement
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

    src_local = src_tensor._local_tensor if src_tensor is not None else None
    if dst_tensor is not None:
        dst_local = (
            dst_tensor._local_tensor
            if hasattr(dst_tensor, "_local_tensor")
            else dst_tensor.to_local()
        )
    else:
        dst_local = None
    my_src_region = src_region.get(rank)
    my_dst_region = dst_region.get(rank)

    sends, recvs = [], []
    for holder, rep, overlap in phase0:
        if holder == rep:
            # Colocated (src holder is also the destination rep): local copy.
            if rank == holder:
                ssl = _local_slices(overlap, my_src_region)
                dsl = _local_slices(overlap, my_dst_region)
                dst_local[dsl].copy_(src_local[ssl])
            continue
        if holder == rank:
            view = src_local[_local_slices(overlap, my_src_region)]
            sends.append((rep, view if view.is_contiguous() else view.contiguous()))
        elif rep == rank:
            dst_view = dst_local[_local_slices(overlap, my_dst_region)]
            if dst_view.is_contiguous():
                recvs.append((holder, dst_view, None))
            else:
                size = tuple(hi - lo for lo, hi in overlap)
                recvs.append(
                    (holder, torch.empty(size, device=device, dtype=dtype), dst_view)
                )
    return sends, recvs, dst_local


def _exchange_p2p(comm, sends, recvs, sint):
    """Issue all cross-mesh sends/recvs in one ncclGroupStart/End, then scatter staged (strided) receives into their destination views."""
    if not sends and not recvs:
        return
    import nccl.core as _nccl_core

    _nccl_core.group_start()
    try:
        for peer, buf in sends:
            comm.send(buf, peer, stream=sint)
        for peer, buf, _ in recvs:
            comm.recv(buf, peer, stream=sint)
    finally:
        _nccl_core.group_end()
    for _, buf, dst_view in recvs:
        if dst_view is not None:
            dst_view.copy_(buf)
