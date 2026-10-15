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

"""Force NCCL's lazy point-to-point connection setup to happen at startup.

NCCL connects peers lazily. Ring and tree collectives (``all_reduce``,
``all_gather``, ``reduce_scatter`` — everything a training step drives) use
connector index 0, which is wired up during ``ncclCommInitRank``. Point-to-point
operations (``send``/``recv``, and therefore ``gather`` and ``all_to_all``) use a
*separate* connector, index 1, which is left unconnected until the first p2p op
is enqueued. At that moment NCCL runs ``ncclP2PPreconnectFunc`` ->
``ncclTransportP2pSetup`` -> ``sendSetup`` -> ``ncclProxyConnect``, which attaches
a ``/dev/shm/nccl-*`` segment shared with the proxy thread.

On GB200/OCI-HSG that attach intermittently returns ENOENT and the run dies with
``NCCL Error 2: unhandled system error``. Two operations in a NeMo-RL Megatron
run are the first p2p on their respective communicators:

* the async-checkpoint plan ``gather``, on the **default** process group — see
  ``megatron/core/dist_checkpointing/strategies/torch.py``, which calls
  ``save_state_dict_async_plan`` with ``process_group=None, coordinator=0``, so
  every rank sends its plan to global rank 0; and
* the MoE token dispatch ``all_to_all``, on the **expert-parallel** group.

Both therefore run their preconnect deep into the job, under full generation and
lustre load. This module performs the same two operations at worker startup,
while the cluster is quiet, so the real ones find their transports already up.

A **tiny** payload is sufficient — connection setup does not depend on message
size. ``postTuneP2pRecordPreconnect`` in NCCL flags channels with
``for (int c = 0; c < comm->p2pnChannelsPerPeer; c++) comm->connectSend[peer] |=
(1ULL << channelId)``: the loop bound is a communicator-level constant and the
channel base comes from the init-time ``comm->p2pSchedule``, neither of which is
derived from the byte count. The flags are one-shot (``hasSeen``), so a later,
larger op over the same peers records no further preconnect and reuses what this
module established.

Set ``NRL_NCCL_PREWARM=0`` to disable.
"""

import os
import time
from typing import Optional

import torch
import torch.distributed as dist

__all__ = [
    "prewarm_enabled",
    "prewarm_checkpoint_gather",
    "prewarm_expert_alltoall",
]


def prewarm_enabled() -> bool:
    """Whether NCCL p2p pre-warming is enabled.

    Returns:
        True unless ``NRL_NCCL_PREWARM`` is set to ``0``.
    """
    return os.environ.get("NRL_NCCL_PREWARM", "1") != "0"


def _log(label: str, rank: int, elapsed_s: float, extra: str = "") -> None:
    # Rank 0 alone: 128 ranks logging startup chatter buries the driver log, and
    # rank 0 is the gather root, so it is the rank whose timing is interesting.
    if rank == 0:
        print(
            f"[NCCL_PREWARM] {label} completed in {elapsed_s:.2f}s{extra}",
            flush=True,
        )


def prewarm_checkpoint_gather() -> None:
    """Pre-connect the p2p transports used by the async-checkpoint plan gather.

    Mirrors the gather that Megatron's distributed checkpointing performs during
    save planning: same collective (``gather_object``), same process group (the
    default one, i.e. every training rank) and same root (global rank 0). Those
    three properties are what decide the set of (peer, channel) connections NCCL
    establishes, so pre-running it here connects exactly the set the real save
    will need.

    Must be called by every rank of the default process group, and is a
    collective: it blocks until all ranks arrive.

    Raises:
        RuntimeError: propagated from NCCL if the preconnect itself fails. This
            is deliberate — a failure here is the same fault that would
            otherwise kill the run at its first checkpoint, and surfacing it at
            startup costs seconds instead of hours.
    """
    if not prewarm_enabled():
        return
    assert dist.is_initialized(), (
        "prewarm_checkpoint_gather() requires an initialized process group"
    )

    # Root 0 and the default group are not configurable on purpose: they are
    # hardcoded on the Megatron side (strategies/torch.py sets `coordinator = 0`
    # and passes `process_group=None`), and a pre-warm over any other group or
    # root would connect the wrong peers while looking like it worked.
    root = 0
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    obj = {"nrl_nccl_prewarm": "checkpoint_gather", "rank": rank}
    gather_list: Optional[list[object]] = [None] * world_size if rank == root else None

    start = time.perf_counter()
    dist.gather_object(obj, gather_list, dst=root)
    elapsed = time.perf_counter() - start

    _log("checkpoint_gather", rank, elapsed, f" ({world_size} ranks -> rank {root})")


def prewarm_expert_alltoall(group: dist.ProcessGroup) -> None:
    """Pre-connect the p2p transports used by MoE token dispatch.

    The MoE token dispatcher calls ``all_to_all_single`` on the expert-parallel
    group (``megatron/core/transformer/moe/token_dispatcher.py``). Like
    ``gather``, that lowers to ``ncclSend``/``ncclRecv`` and so triggers the same
    lazy preconnect on its first invocation — which, in a run resumed from a
    checkpoint, lands in the middle of the first training step rather than at
    startup.

    Args:
        group: The expert-parallel process group to pre-warm. Must be the same
            group the token dispatcher uses.

    Raises:
        RuntimeError: propagated from NCCL if the preconnect fails; see
            :func:`prewarm_checkpoint_gather` for why this is not swallowed.
    """
    if not prewarm_enabled():
        return
    assert dist.is_initialized(), (
        "prewarm_expert_alltoall() requires an initialized process group"
    )

    group_size = dist.get_world_size(group)
    if group_size <= 1:
        return

    # One element per group member, which is the minimum shape all_to_all_single
    # accepts and enough to touch every peer.
    device = torch.device("cuda", torch.cuda.current_device())
    send = torch.zeros(group_size, dtype=torch.float32, device=device)
    recv = torch.empty_like(send)

    start = time.perf_counter()
    dist.all_to_all_single(recv, send, group=group)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    _log("expert_alltoall", dist.get_rank(), elapsed, f" ({group_size} ranks)")
