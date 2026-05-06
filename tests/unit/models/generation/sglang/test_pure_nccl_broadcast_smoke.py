# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Minimal cross-process NCCL broadcast smoke test.

Two Ray actors, one GPU each, no SGLang, no Megatron, no AutoBridge.
The only nemo-rl code in the loop is ``init_process_group`` from
``nemo_rl.models.policy.utils``.

Pass / fail mapping:

* PASS → the host supports cross-process NCCL between two GPUs
  (P2P/IPC, SHM, or NET, whichever NCCL picks). Any failure of the
  full disag refit is in our higher-level code, not the transport.
* FAIL with ``Cuda failure 'invalid argument'`` → cross-GPU
  ``cudaIpcOpenMemHandle`` is blocked at the kernel layer (typically
  IOMMU). The full disag refit is therefore not a code bug; the host
  needs an admin-level fix or NCCL has to be forced onto a different
  transport (``NCCL_P2P_DISABLE=1`` etc.).
"""

from __future__ import annotations

import os
import socket

import pytest
import ray
import torch

pytestmark = pytest.mark.sglang


# ---------------------------------------------------------------------------
# NcclWorker — minimal Ray actor that holds one GPU and a custom NCCL group
# ---------------------------------------------------------------------------
@ray.remote(num_gpus=1)
class NcclWorker:
    """Single-GPU actor that participates in one cross-process NCCL group."""

    def __init__(self) -> None:
        self._pg = None

    def get_node_ip(self) -> str:
        import ray as _ray

        return _ray.util.get_node_ip_address()

    def find_free_port(self) -> int:
        with socket.socket() as s:
            s.bind(("", 0))
            return int(s.getsockname()[1])

    def device_info(self) -> dict:
        import torch as _torch

        return {
            "current_device": _torch.cuda.current_device(),
            "device_count": _torch.cuda.device_count(),
            "device_name": _torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }

    def setup_default_pg(self) -> None:
        """Init a 1-rank gloo default PG to mimic real-world conditions where
        the trainer or engine has already initialized the world before
        building the cross-process NCCL group on top."""
        import torch.distributed as dist

        if dist.is_initialized():
            torch.cuda.set_device(0)
            return
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            world_size=1,
            rank=0,
        )
        torch.cuda.set_device(0)

    def init_nccl_group(
        self,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        group_name: str,
    ) -> None:
        """Build the cross-process NCCL group via the in-tree helper."""
        from nemo_rl.models.policy.utils import init_process_group

        self._pg = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )

    def broadcast_send(self, n: int = 8, fill: float = 1.5) -> list:
        """Rank 0: broadcast a known tiny bf16 tensor."""
        import torch.distributed as dist

        tensor = torch.full((n,), fill, dtype=torch.bfloat16, device="cuda:0")
        dist.broadcast(tensor, src=0, group=self._pg)
        torch.cuda.synchronize()
        return tensor.cpu().tolist()

    def broadcast_recv(self, n: int = 8) -> list:
        """Rank 1: receive into a fresh buffer and return it."""
        import torch.distributed as dist

        tensor = torch.empty((n,), dtype=torch.bfloat16, device="cuda:0")
        dist.broadcast(tensor, src=0, group=self._pg)
        torch.cuda.synchronize()
        return tensor.cpu().tolist()

    def shutdown(self) -> None:
        import torch.distributed as dist

        if self._pg is not None:
            try:
                dist.destroy_process_group(self._pg)
            except Exception:
                pass
            self._pg = None
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------
def test_pure_nccl_broadcast(ray_cluster):
    """Two GPUs, two Ray actors, one bf16 tensor. Pass ⇔ host transport works.

    No SGLang, no Megatron, no AutoBridge — the only nemo-rl symbol used
    is ``init_process_group``. If this fails the same way the real disag
    refit does (``Cuda failure 'invalid argument'``), the failure is at
    NCCL transport, not in our higher-level code.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("test requires 2 GPUs visible to Ray")

    rank0 = NcclWorker.remote()
    rank1 = NcclWorker.remote()

    try:
        # 1. 1-rank default PG on each side (mimics Megatron / SGLang startup).
        ray.get(
            [rank0.setup_default_pg.remote(), rank1.setup_default_pg.remote()]
        )
        info0, info1 = ray.get(
            [rank0.device_info.remote(), rank1.device_info.remote()]
        )
        print(f"[pure-nccl] rank0={info0}")
        print(f"[pure-nccl] rank1={info1}")

        # 2. Pick a master address+port on rank 0.
        master_addr, master_port = ray.get(
            [rank0.get_node_ip.remote(), rank0.find_free_port.remote()]
        )
        print(f"[pure-nccl] master={master_addr}:{master_port}")

        # 3. Bring up the cross-process NCCL group on both ranks in parallel.
        ray.get(
            [
                rank0.init_nccl_group.remote(
                    master_addr, master_port, 0, 2, "smoke-pure"
                ),
                rank1.init_nccl_group.remote(
                    master_addr, master_port, 1, 2, "smoke-pure"
                ),
            ]
        )
        print("[pure-nccl] cross-process NCCL group up on both ranks")

        # 4. Broadcast (rank 0 → rank 1) and verify.
        sent_fut = rank0.broadcast_send.remote()
        recv_fut = rank1.broadcast_recv.remote()
        sent, received = ray.get([sent_fut, recv_fut])
        print(f"[pure-nccl] sent={sent}")
        print(f"[pure-nccl] received={received}")
        assert sent == received, (
            f"broadcast bytes diverged:\n  sent={sent}\n  received={received}"
        )
    finally:
        try:
            ray.get([rank0.shutdown.remote(), rank1.shutdown.remote()])
        except Exception:
            pass
        for actor in (rank0, rank1):
            try:
                ray.kill(actor)
            except Exception:
                pass
