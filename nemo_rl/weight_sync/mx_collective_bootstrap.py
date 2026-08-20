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

import torch


class MxProcessGroup:
    """``StatelessProcessGroup``'s surface, bootstrapped through ModelExpress.

    Deliberately duck-typed rather than a subclass: the only contract the refit
    path relies on is ``nccl_communicator`` plus ``broadcast``, and inheriting
    would drag in the ``TCPStore`` this exists to replace.
    """

    def __init__(self, *, unique_id: bytes, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.nccl_communicator = None
        self._unique_id = unique_id

    def init_nccl_communicator(self, device):
        from nccl.core.communicator import Communicator
        from nccl.core.utils import UniqueId

        # Mirror the native path: free cached blocks first so the communicator's
        # transport buffers have device-memory headroom.
        torch.cuda.empty_cache()
        with torch.cuda.device(device):
            self.nccl_communicator = Communicator.init(
                nranks=self.world_size,
                rank=self.rank,
                unique_id=UniqueId.from_bytes(self._unique_id),
            )

    def broadcast(self, tensor, src, stream=None):
        if stream is None:
            stream = torch.cuda.current_stream()
        self.nccl_communicator.broadcast(
            sendbuf=tensor, recvbuf=tensor, root=src, stream=int(stream.cuda_stream)
        )


def build_mx_groups(
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
):
    """Join the MX group and return this worker's initialized lane groups.

    Returns ``(reshard_group, broadcast_group, lane_ids)``. The reshard group is
    this worker's own source partition on the trainer side; a generator joins
    every reshard lane, so it takes the one it was asked for.
    """
    import grpc
    from modelexpress_rl.collective import CollectiveRendezvous, Role
    from modelexpress_rl.collective.comm import new_unique_id

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

    # Rank 0 of a lane owes it an identifier. Publishing before waiting is what
    # lets the group reach READY at all: readiness requires every lane
    # bootstrapped for the current epoch.
    if membership.is_bootstrap_leader:
        for lane in membership.lanes:
            if lane.rank_in_lane == 0:
                rz.publish_bootstrap(
                    group_id=membership.group_id,
                    epoch=membership.epoch,
                    lane_id=lane.lane_id,
                    worker_id=worker_id,
                    nccl_unique_id=new_unique_id(),
                )

    group = rz.await_ready(
        group_id=membership.group_id, epoch=membership.epoch, timeout_s=timeout_s
    )
    ids = {lane.lane_id: bytes(lane.nccl_unique_id) for lane in group.lanes}

    if device is None:
        device = torch.cuda.current_device()

    reshard_groups = {}
    broadcast_group = None
    # Lane order matters: a generator joins every reshard lane, and the trainers
    # in each lane are blocked inside their own bootstrap until it does. Creating
    # them in ascending lane order is what makes the two sides unblock in the
    # same sequence rather than deadlocking against each other.
    for lane in sorted(membership.lanes, key=lambda l: l.lane_id):
        pg = MxProcessGroup(
            unique_id=ids[lane.lane_id],
            rank=lane.rank_in_lane,
            world_size=lane.world_size,
        )
        pg.init_nccl_communicator(device=device)
        if lane.kind == "BROADCAST":
            broadcast_group = pg
        else:
            reshard_groups[lane.lane_id] = pg

    return reshard_groups, broadcast_group, membership
