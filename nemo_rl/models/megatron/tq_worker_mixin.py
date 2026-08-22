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
"""TransferQueue awareness for Megatron-Core workers.

``TQWorkerMixin`` leaves ``_local_coords`` and ``_get_replica_group``
to the backend because only the backend knows its parallelism state.
Both answers are derived purely from ``megatron.core.parallel_state``,
so every Megatron worker -- policy and PPO value alike -- shares this
one implementation.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from megatron.core import parallel_state

from nemo_rl.data_plane.worker_mixin import TQWorkerMixin


class MegatronTQWorkerMixin(TQWorkerMixin):
    """Fills in the two parallelism hooks TQWorkerMixin declares abstract."""

    def _local_coords(self) -> dict[str, int]:
        if not torch.distributed.is_initialized():
            return {}
        return {
            "tensor_parallel": parallel_state.get_tensor_model_parallel_rank(),
            "context_parallel": parallel_state.get_context_parallel_rank(),
            "pipeline_parallel": parallel_state.get_pipeline_model_parallel_rank(),
        }

    def _get_replica_group(self) -> Optional[Any]:
        """Replica group = TP × CP × PP siblings within this DP rank.

        Always returns the real group so :meth:`_is_replica_leader` (used
        by both fetch and write-back) gives the correct single-writer
        answer even at CP=1 — gating on CP=1 here is what produced the
        ``-601 ILLEGAL_CLIENT`` duplicate-write bug. The fetch-path
        broadcast-vs-independent perf choice lives inside ``_fetch``
        keyed on ``replica_group.size()``.

        mcore exposes per-axis groups (``get_tensor_model_parallel_group``,
        ``get_context_parallel_group``, ``get_pipeline_model_parallel_group``)
        but no single combined group. We build the combined NCCL group
        once on first call by enumerating coordinates that share this
        rank's ``dp_rank``.
        """
        if not torch.distributed.is_initialized():
            return None
        cached = getattr(self, "_replica_group_cache", "uninit")
        if cached != "uninit":
            return cached

        world_size = torch.distributed.get_world_size()
        my_dp_rank = parallel_state.get_data_parallel_rank()
        # Collect global ranks that share this DP rank — they form the
        # replica group. Done collectively so every rank ends up with
        # the same ranks list and can pass it to new_group().
        my_replica_ranks_t = torch.full(
            (world_size,),
            -1,
            dtype=torch.long,
            device="cuda",
        )
        my_replica_ranks_t[torch.distributed.get_rank()] = my_dp_rank
        torch.distributed.all_reduce(
            my_replica_ranks_t, op=torch.distributed.ReduceOp.MAX
        )
        all_dp_ranks = my_replica_ranks_t.tolist()

        # Every (dp_rank → ranks) bucket must call new_group on its own
        # ranks list, but new_group itself must be called collectively
        # across the full world. Sort by dp_rank to keep call order
        # consistent across processes.
        groups: dict[int, Any] = {}
        for dp in sorted(set(all_dp_ranks)):
            ranks = [r for r, d in enumerate(all_dp_ranks) if d == dp]
            grp = torch.distributed.new_group(ranks=ranks, backend="nccl")
            groups[dp] = grp
        self._replica_group_cache = groups[my_dp_rank]
        return self._replica_group_cache
