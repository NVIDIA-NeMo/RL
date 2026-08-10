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

"""Inter-node NIXL RDMA transport for teacher full-logits (host-staged).

Intra-node teacher->student logit transport uses CUDA IPC
(:func:`nemo_rl.models.policy.utils.get_handle_from_tensor` on the producer,
:func:`nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc` on the
consumer). ``cudaIpcOpenMemHandle`` can only map a buffer created by a process
on the SAME physical node, so a non-node-local teacher/student layout (e.g.
``teacher_dp != student_dp``) has no IPC path.

This module ships the *same* teacher-logit storage over NIXL (UCX / RDMA READ)
for the cross-node case, reusing the low-level ``NixlAgent`` primitives from
:mod:`nemo_rl.utils.checkpoint_engines.nixl`.

**Host-staged**: on these workers NIXL/UCX cannot register device (VRAM) memory
once NCCL is initialized (``register_memory`` on a CUDA tensor returns
``NIXL_ERR_BACKEND``), but pinned-host registration always works. So the
transport bounces through a pinned-host mirror, exactly like the checkpoint
engine's host path (``_allocate_transfer_buffer`` with ``pin_memory``):

- Producer: keep the GPU storage (still needed for the intra-node IPC path);
  allocate a pinned-host mirror, register *that*, and D2H-copy the GPU storage
  into it after the teacher forward (:func:`stage_teacher_storage_to_host`).
- Consumer: RDMA-READ the remote host mirror into a local pinned-host buffer,
  then H2D-copy into a GPU tensor (:func:`read_teacher_storage`).

The per-sample handle dict, TP/CP shard routing, and reassembly in
``x_token.loss_utils`` are transport-agnostic and unchanged; the consumer picks
IPC (same node) or NIXL (remote) per shard via ``producer_node_ip``.

The producer/consumer NIXL agents are process-global and lazily created (one per
worker process), analogous to the process-global CUDA context the IPC path
relies on.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import torch


# ---------------------------------------------------------------------------
# Process-global agents (one producer + one consumer NixlAgent per worker).
# ---------------------------------------------------------------------------

_PRODUCER_AGENT: Any = None
# Producer state for the single persistent storage buffer: data_ptr ->
# registered pinned-host mirror + its transfer descriptors.
_PRODUCER_STATE: dict[str, Any] = {}

_CONSUMER_AGENT: Any = None
# add_remote_agent is idempotent-per-peer; cache the resolved remote name by the
# producer agent name so repeated reads of the same teacher don't re-add it.
_REMOTE_AGENTS: dict[str, str] = {}

_LOCAL_NODE_IP: Optional[str] = None


def _local_node_ip() -> str:
    """This worker's node IP (cached); used to decide IPC vs NIXL per shard."""
    global _LOCAL_NODE_IP
    if _LOCAL_NODE_IP is None:
        import ray

        _LOCAL_NODE_IP = ray.util.get_node_ip_address().strip("[]")
    return _LOCAL_NODE_IP


def _new_nixl_agent() -> Any:
    # Reuse the checkpoint-engine NixlAgent wrapper (agent + zmq control plane)
    # so both RDMA users share one integration of the NIXL library.
    from nemo_rl.utils.checkpoint_engines.nixl import NixlAgent

    return NixlAgent()


def _producer_agent() -> Any:
    global _PRODUCER_AGENT
    if _PRODUCER_AGENT is None:
        _PRODUCER_AGENT = _new_nixl_agent()
    return _PRODUCER_AGENT


def _consumer_agent() -> Any:
    global _CONSUMER_AGENT
    if _CONSUMER_AGENT is None:
        _CONSUMER_AGENT = _new_nixl_agent()
    return _CONSUMER_AGENT


# ---------------------------------------------------------------------------
# Producer side (teacher worker) — register a pinned-host mirror, D2H per export.
# ---------------------------------------------------------------------------


def register_teacher_storage(storage: torch.Tensor) -> dict[str, Any]:
    """Register a pinned-host mirror of the teacher-logit storage; return a descriptor.

    Called once per export (its result is embedded in every per-sample handle).
    The host mirror is allocated and registered lazily, cached by
    ``storage.data_ptr()`` — the GPU storage is persistent and reused across
    steps (``ensure_teacher_ipc_buffer`` only reallocates on grow), so this
    normally registers once per worker and re-registers on realloc.

    NOTE: this only *registers* the host mirror; the GPU->host copy happens in
    :func:`stage_teacher_storage_to_host` after the teacher forward has written
    every microbatch into the GPU storage.

    ``content_uuid`` is regenerated every call so the consumer can dedupe RDMA
    reads of one export across the many shards/samples that share the buffer.
    """
    agent = _producer_agent()
    ptr = storage.data_ptr()
    if _PRODUCER_STATE.get("ptr") != ptr:
        prev_reg = _PRODUCER_STATE.get("reg")
        if prev_reg is not None:
            try:
                agent.agent.deregister_memory(prev_reg)
            except Exception:
                pass
        # Pinned-host mirror matching the GPU storage (RDMA-registerable where
        # VRAM is not). Pinned so the D2H copy in stage_* can be async.
        host_mirror = torch.empty(
            tuple(storage.shape), dtype=storage.dtype, pin_memory=True
        )
        reg = agent.agent.register_memory(host_mirror)
        descs = agent.agent.get_xfer_descs(host_mirror)
        _PRODUCER_STATE.update(
            ptr=ptr,
            host_mirror=host_mirror,
            reg=reg,
            descs=descs,
            agent_metadata=agent.get_agent_metadata(),
            node_ip=_local_node_ip(),
        )
    return {
        "agent_metadata": _PRODUCER_STATE["agent_metadata"],
        "remote_descs": _PRODUCER_STATE["descs"],
        "node_ip": _PRODUCER_STATE["node_ip"],
        "content_uuid": uuid.uuid4().hex,
    }


def stage_teacher_storage_to_host(storage: torch.Tensor) -> None:
    """D2H-copy the GPU storage into its registered pinned-host mirror.

    Call once after the teacher forward has written every microbatch into
    ``storage`` and after :func:`register_teacher_storage`, so the host mirror
    the consumer RDMA-READs holds this export's logits.
    """
    host_mirror = _PRODUCER_STATE.get("host_mirror")
    if host_mirror is None or _PRODUCER_STATE.get("ptr") != storage.data_ptr():
        raise RuntimeError(
            "stage_teacher_storage_to_host called before register_teacher_storage "
            "for the current storage buffer."
        )
    host_mirror.copy_(storage, non_blocking=True)
    # The consumer reads the host mirror over RDMA from another node; make sure
    # the D2H copy has landed before the descriptor is handed out.
    torch.cuda.synchronize(storage.device)


def release_teacher_storage() -> None:
    """Deregister the host mirror and reset producer state.

    Mirrors ``release_ipc_buffer``: called when the persistent storage is freed
    so the next export re-registers a fresh mirror.
    """
    global _PRODUCER_AGENT
    reg = _PRODUCER_STATE.get("reg")
    if reg is not None and _PRODUCER_AGENT is not None:
        try:
            _PRODUCER_AGENT.agent.deregister_memory(reg)
        except Exception:
            pass
    _PRODUCER_STATE.clear()


# ---------------------------------------------------------------------------
# Consumer side (student worker) — RDMA-READ remote host mirror, then H2D.
# ---------------------------------------------------------------------------


def _remote_agent(agent: Any, agent_metadata: dict[str, Any]) -> str:
    name = agent_metadata["agent_name"]
    resolved = _REMOTE_AGENTS.get(name)
    if resolved is None:
        resolved = agent.add_remote_agent(agent_metadata)
        _REMOTE_AGENTS[name] = resolved
    return resolved


def _wait_read_done(agent: Any, xfer_handle: Any, remote_agent: str) -> None:
    """Busy-poll a NIXL READ to completion (synchronous consumer context)."""
    nixl_agent = agent.agent
    progress = getattr(nixl_agent, "progress", None)
    while True:
        if callable(progress):
            progress()
        state = nixl_agent.check_xfer_state(xfer_handle)
        if state == "DONE":
            nixl_agent.release_xfer_handle(xfer_handle)
            return
        if state == "ERR":
            raise RuntimeError(f"NIXL teacher-logit READ from {remote_agent} failed.")


def read_teacher_storage(
    nixl_desc: dict[str, Any],
    storage_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: int,
) -> torch.Tensor:
    """RDMA-READ the producer's host mirror, then H2D into a GPU tensor.

    Reads the entire ``[N_mb, B, T_local, V_local]`` buffer (matching what the
    IPC path maps) into a local pinned-host buffer, then copies it to
    ``cuda:device`` so the caller can slice it identically. ``device`` is a CUDA
    device index (matches ``rebuild_cuda_tensor_from_ipc``'s ``device_id``).
    """
    agent = _consumer_agent()
    remote = _remote_agent(agent, nixl_desc["agent_metadata"])
    dst_host = torch.empty(tuple(storage_shape), dtype=dtype, pin_memory=True)
    reg = agent.agent.register_memory(dst_host)
    try:
        local_descs = agent.agent.get_xfer_descs(dst_host)
        notif = uuid.uuid4().bytes
        xfer_handle = agent.agent.initialize_xfer(
            "READ", local_descs, nixl_desc["remote_descs"], remote, notif
        )
        if agent.agent.transfer(xfer_handle) == "ERR":
            raise RuntimeError(
                f"NIXL teacher-logit READ from {remote} failed to start."
            )
        _wait_read_done(agent, xfer_handle, remote)
    finally:
        agent.agent.deregister_memory(reg)
    cuda_device = torch.device("cuda", device)
    dst = dst_host.to(cuda_device, non_blocking=True)
    torch.cuda.synchronize(cuda_device)
    return dst


def materialize_teacher_storage(
    handle: dict[str, Any],
    device: int,
    read_cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return the producer's ``[N_mb, B, T_local, V_local]`` storage for a shard.

    Transport dispatch: a shard whose producer is on this node (or that carries
    no NIXL descriptor, i.e. the transport is IPC-only) is mapped zero-copy via
    CUDA IPC; a remote shard is RDMA-READ once per export (``content_uuid``) and
    cached in ``read_cache`` so the many shards/samples sharing one buffer share
    a single READ.
    """
    from nemo_rl.models.policy.utils import rebuild_cuda_tensor_from_ipc

    nixl_desc = handle.get("nixl")
    if nixl_desc is None or nixl_desc["node_ip"] == _local_node_ip():
        return rebuild_cuda_tensor_from_ipc(handle["payload_ipc"], device)

    key = nixl_desc["content_uuid"]
    cached = read_cache.get(key)
    if cached is None:
        cached = read_teacher_storage(
            nixl_desc,
            tuple(handle["storage_shape"]),
            handle["dtype"],
            device,
        )
        read_cache[key] = cached
    return cached
