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

"""ModelExpress-brokered replacement for the nccl_reshard bootstrap.

This is the whole of the MX integration. ``xferdtensor`` consumes exactly one
thing from a process group -- ``.nccl_communicator`` -- and the packed
broadcast consumes one more, ``.broadcast``. So the MX path does not need its
own refit loop: it needs to produce an object with that surface whose
``ncclUniqueId`` came from ModelExpress rather than a ``TCPStore``, drop it
where ``StatelessProcessGroup`` normally goes, and let ``nccl_reshard_refit``
run unchanged.

Keeping it to that boundary is not just less code. It is what makes the two
transports comparable: they cannot drift apart, because below the bootstrap
they are the same code path.
"""

import os
from typing import Any

import torch


class MxProcessGroup:
    """``StatelessProcessGroup``'s surface, bootstrapped through ModelExpress.

    Deliberately duck-typed rather than a subclass: the only contract the refit
    path relies on is ``nccl_communicator`` plus ``broadcast``, and inheriting
    would drag in the ``TCPStore`` this exists to replace.
    """

    def __init__(
        self,
        *,
        unique_id: bytes,
        rank: int,
        world_size: int,
        root_unique_id: Any = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.nccl_communicator = None
        self._unique_id = unique_id
        # Rank zero keeps and consumes the exact object returned by
        # ncclGetUniqueId, matching StatelessProcessGroup. Peers reconstruct
        # it from the bytes MX brokered.
        self._root_unique_id = root_unique_id

    def init_nccl_communicator(self, device):
        from nccl.core.communicator import Communicator
        from nccl.core.utils import UniqueId

        # Mirror the native path: free cached blocks first so the communicator's
        # transport buffers have device-memory headroom.
        torch.cuda.empty_cache()
        with torch.cuda.device(device):
            unique_id = self._root_unique_id
            if unique_id is None:
                unique_id = UniqueId.from_bytes(self._unique_id)
            self.nccl_communicator = Communicator.init(
                nranks=self.world_size,
                rank=self.rank,
                unique_id=unique_id,
            )

            # Match the native StatelessProcessGroup bootstrap protocol. This
            # proves the newly-created communicator can execute one collective
            # before it is handed to the refit path.
            stream = torch.cuda.current_stream()
            data = (
                torch.ones(1, device=device)
                if self.rank == 0
                else torch.zeros(1, device=device)
            )
            self.broadcast(data, 0, stream=stream)
            stream.synchronize()
            if not torch.allclose(data, torch.ones(1, device=device)):
                raise RuntimeError("MX NCCL communicator bootstrap broadcast failed")

    def broadcast(self, tensor, src, stream=None):
        if stream is None:
            stream = torch.cuda.current_stream()
        self.nccl_communicator.broadcast(
            sendbuf=tensor, recvbuf=tensor, root=src, stream=int(stream.cuda_stream)
        )


class MxBootstrapState:
    """Rendezvous result plus the lane communicators built so far.

    Held across phases because lane creation is driven from the driver, one
    lane at a time with a barrier between them.
    """

    def __init__(self, membership, ids, device, worker_id):
        self.membership = membership
        self.ids = ids
        self.device = device
        self.worker_id = worker_id
        self.reshard_groups = {}
        self.broadcast_group = None
        self.rz = None
        self.channel = None
        self.timeout_s = 900.0


def mx_rendezvous(
    *,
    mx_server_url: str,
    model_name: str,
    role: str,
    index_in_role: int,
    slot_id: str,
    worker_id: str,
    trainer_slots: list,
    generator_slots: list,
    source_partition_count: int,
    source_partition=None,
    plan_digest: str = "",
    device=None,
    timeout_s: float = 900.0,
) -> MxBootstrapState:
    """Join the MX group and fetch every lane's identifier.

    No communicator is created here. Creation is a separate phase per lane so
    the driver can barrier between them.
    """
    if role not in ("TRAINER", "GENERATOR"):
        raise ValueError(f"unsupported MX collective role {role!r}")
    if os.environ.get("NCCL_COMM_ID"):
        raise RuntimeError(
            "NCCL_COMM_ID must be unset for the MX collective transport; "
            "it overrides ncclGetUniqueId and bypasses MX's per-lane bootstrap"
        )

    import grpc
    from modelexpress_rl.collective import CollectiveRendezvous, Role

    role_enum = Role.TRAINER if role == "TRAINER" else Role.GENERATOR
    channel = grpc.insecure_channel(mx_server_url)
    rz = CollectiveRendezvous(channel, rpc_timeout_s=60.0)

    membership = rz.join(
        model_name=model_name,
        trainer_slots=trainer_slots,
        generator_slots=generator_slots,
        source_partition_count=source_partition_count,
        slot_id=slot_id,
        worker_id=worker_id,
        role=role_enum,
        index_in_role=index_in_role,
        plan_digest=plan_digest,
        source_partition=source_partition,
    )

    # No identifier is minted here. ncclGetUniqueId opens the bootstrap
    # listening socket as a side effect, and that socket has to still be
    # accepting when the lane's peers dial in. Minting at rendezvous means it
    # must survive an actor-method return, a driver round-trip and a barrier;
    # peers then get "connection refused" from an address that looks correct.
    # Each lane mints its own identifier in the phase that uses it instead.
    if device is None:
        device = torch.cuda.current_device()
    state = MxBootstrapState(membership, {}, device, worker_id)
    state.rz = rz
    state.channel = channel
    state.timeout_s = timeout_s
    return state


def _await_lane_id(state: MxBootstrapState, lane_id: int) -> bytes:
    """Block until MX has this lane's identifier at the admitted epoch.

    Per-lane rather than whole-group: with mint-at-use the later lanes are not
    published yet, so waiting for group READY here would deadlock the very
    ordering it is meant to protect.
    """
    import time

    from modelexpress_rl import refit_collective_pb2 as pb
    from modelexpress_rl import refit_collective_pb2_grpc as pb_grpc

    stub = pb_grpc.RefitCollectiveServiceStub(state.channel)
    deadline = time.monotonic() + state.timeout_s
    while True:
        group = stub.GetCollectiveGroup(
            pb.GetCollectiveGroupRequest(group_id=state.membership.group_id),
            timeout=30.0,
        )
        if group.epoch != state.membership.epoch:
            raise RuntimeError(
                f"collective group epoch moved {state.membership.epoch} -> "
                f"{group.epoch} while bootstrapping lane {lane_id}"
            )
        for lane in group.lanes:
            if (
                lane.lane_id == lane_id
                and lane.bootstrap_epoch == state.membership.epoch
                and len(lane.nccl_unique_id) == 128
            ):
                return bytes(lane.nccl_unique_id)
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"lane {lane_id} identifier not published within {state.timeout_s:.0f}s"
            )
        time.sleep(0.2)


def mx_init_lane(state: MxBootstrapState, lane_id: int) -> None:
    """Create this worker's communicator for one lane, if it belongs to it.

    One lane at a time, cluster-wide, is not an optimisation to be undone:
    creating two different communicators concurrently across overlapping rank
    sets deadlocks NCCL. A worker not in ``lane_id`` returns immediately and
    waits at the driver's barrier rather than racing ahead into its next lane.
    """
    lane = next((l for l in state.membership.lanes if l.lane_id == lane_id), None)
    if lane is None:
        return
    root_unique_id = None
    minted_uid = None
    if lane.rank_in_lane == 0:
        # Mint and publish in the same breath as the init below, so the
        # listening socket ncclGetUniqueId opens is still accepting when the
        # peers dial it.
        from nccl.core.utils import get_unique_id

        root_unique_id = get_unique_id()
        minted_uid = bytes(root_unique_id.as_bytes)
        state.rz.publish_bootstrap(
            group_id=state.membership.group_id,
            epoch=state.membership.epoch,
            lane_id=lane.lane_id,
            worker_id=state.worker_id,
            nccl_unique_id=minted_uid,
        )
    uid = _await_lane_id(state, lane.lane_id)
    if minted_uid is not None and uid != minted_uid:
        raise RuntimeError(
            f"MX returned a different ncclUniqueId for lane {lane.lane_id} "
            "than rank zero published"
        )
    state.ids[lane.lane_id] = uid
    pg = MxProcessGroup(
        unique_id=uid,
        rank=lane.rank_in_lane,
        world_size=lane.world_size,
        root_unique_id=root_unique_id,
    )
    pg.init_nccl_communicator(device=state.device)
    if lane.kind == "BROADCAST":
        state.broadcast_group = pg
    else:
        state.reshard_groups[lane.lane_id] = pg


def mx_lane_order(source_partition_count: int) -> list:
    """Cluster-wide lane creation order.

    Broadcast first because every rank is in it, so the barrier that follows is
    a full-cluster sync point; then the reshard lanes in ascending order.
    """
    return [source_partition_count] + list(range(source_partition_count))


def build_mx_groups(**kwargs):
    """Single-process convenience wrapper: rendezvous then every lane in order.

    Safe only when one process drives all ranks (tests). Real deployments must
    use the phased API so the driver can barrier between lanes.
    """
    spc = kwargs["source_partition_count"]
    state = mx_rendezvous(**kwargs)
    for lane_id in mx_lane_order(spc):
        mx_init_lane(state, lane_id)
    return state.reshard_groups, state.broadcast_group, state.membership
