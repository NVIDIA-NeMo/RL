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

"""Two Ray actors, two NCCL groups, one bf16 broadcast.

This is a narrow transport demo for the Megatron -> SGLang weight-update
failure mode:

1. Start two Ray actors, each reserving one GPU.
2. Initialize a normal/default NCCL process group across the two actors.
3. Initialize a second side-by-side NCCL group with the nemo-rl
   ``init_process_group`` helper, the same helper used for distributed
   SGLang weight updates after Megatron has already initialized torch.distributed.
4. Broadcast a tiny bf16 tensor through the second group.

If this fails only when NCCL P2P/NVSHM is enabled, the problem is already
reproducible without SGLang or Megatron. If it passes, the real failure needs
additional topology from SGLang/Megatron.
"""

from __future__ import annotations

import argparse
import os
import socket
from datetime import timedelta
from typing import Any

import pytest
import ray
import torch

pytestmark = pytest.mark.sglang

os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")


def _find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


@ray.remote(num_gpus=1)
class TwoGroupNcclActor:
    """Single-GPU actor holding one default PG and one custom weight-update PG."""

    def __init__(self) -> None:
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
        self._rank: int | None = None
        self._weight_update_pg = None

    def node_ip(self) -> str:
        import ray as _ray

        return _ray.util.get_node_ip_address()

    def device_info(self) -> dict[str, Any]:
        return {
            "rank": self._rank,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device()
            if torch.cuda.is_available()
            else None,
            "device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE", ""),
            "nccl_shm_disable": os.environ.get("NCCL_SHM_DISABLE", ""),
            "nccl_cumem_enable": os.environ.get("NCCL_CUMEM_ENABLE", ""),
            "nccl_debug": os.environ.get("NCCL_DEBUG", ""),
        }

    def init_default_nccl_pg(
        self,
        *,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
    ) -> dict[str, Any]:
        import torch.distributed as dist

        os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
        torch.cuda.set_device(0)
        self._rank = rank
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=120),
            )
        dist.barrier(device_ids=[0])
        return self.device_info()

    def default_group_all_reduce(self) -> float:
        import torch.distributed as dist

        tensor = torch.tensor(
            [float((self._rank or 0) + 1)], dtype=torch.float32, device="cuda:0"
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        return float(tensor.item())

    def init_weight_update_nccl_pg(
        self,
        *,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        group_name: str,
    ) -> None:
        from nemo_rl.models.policy.utils import init_process_group

        os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
        torch.cuda.set_device(0)
        self._weight_update_pg = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
            timeout=timedelta(seconds=120),
        )

    def broadcast_weight_update_tensor(
        self,
        *,
        n: int,
        src: int = 0,
        async_op: bool = True,
    ) -> dict[str, Any]:
        import torch.distributed as dist

        if self._weight_update_pg is None:
            raise RuntimeError("weight-update NCCL group is not initialized")

        rank = self._rank
        if rank is None:
            raise RuntimeError("default NCCL group is not initialized")

        if rank == src:
            tensor = (torch.arange(n, device="cuda:0", dtype=torch.float32) % 97).to(
                torch.bfloat16
            )
        else:
            tensor = torch.empty((n,), dtype=torch.bfloat16, device="cuda:0")

        work = dist.broadcast(
            tensor,
            src=src,
            group=self._weight_update_pg,
            async_op=async_op,
        )
        if async_op:
            work.wait()
        torch.cuda.synchronize()

        return {
            "rank": rank,
            "numel": tensor.numel(),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "first16": tensor[:16].float().cpu().tolist(),
            "last16": tensor[-16:].float().cpu().tolist(),
            "checksum": float(tensor.float().sum().item()),
        }

    def shutdown(self) -> None:
        import torch.distributed as dist

        if self._weight_update_pg is not None:
            try:
                dist.destroy_process_group(self._weight_update_pg)
            except Exception:
                pass
            self._weight_update_pg = None
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def run_two_actor_second_group_demo(*, tensor_numel: int = 2048) -> tuple[dict, dict]:
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    if torch.cuda.device_count() < 2:
        raise RuntimeError("demo requires at least 2 GPUs visible to Ray")

    rank0 = TwoGroupNcclActor.remote()
    rank1 = TwoGroupNcclActor.remote()
    try:
        master_addr = ray.get(rank0.node_ip.remote())
        default_port = _find_free_port()
        weight_update_port = _find_free_port()

        print(
            f"[two-group-demo] default_pg=tcp://{master_addr}:{default_port} "
            f"weight_update_pg=tcp://{master_addr}:{weight_update_port}",
            flush=True,
        )

        infos = ray.get(
            [
                rank0.init_default_nccl_pg.remote(
                    master_addr=master_addr,
                    master_port=default_port,
                    rank=0,
                    world_size=2,
                ),
                rank1.init_default_nccl_pg.remote(
                    master_addr=master_addr,
                    master_port=default_port,
                    rank=1,
                    world_size=2,
                ),
            ]
        )
        print(f"[two-group-demo] default_pg_infos={infos}", flush=True)

        reduced = ray.get(
            [
                rank0.default_group_all_reduce.remote(),
                rank1.default_group_all_reduce.remote(),
            ]
        )
        print(f"[two-group-demo] default_pg_all_reduce={reduced}", flush=True)
        assert reduced == [3.0, 3.0]

        group_name = "weight-update-two-ray-actors"
        ray.get(
            [
                rank0.init_weight_update_nccl_pg.remote(
                    master_addr=master_addr,
                    master_port=weight_update_port,
                    rank=0,
                    world_size=2,
                    group_name=group_name,
                ),
                rank1.init_weight_update_nccl_pg.remote(
                    master_addr=master_addr,
                    master_port=weight_update_port,
                    rank=1,
                    world_size=2,
                    group_name=group_name,
                ),
            ]
        )
        print("[two-group-demo] weight-update NCCL group ready", flush=True)

        sent, received = ray.get(
            [
                rank0.broadcast_weight_update_tensor.remote(n=tensor_numel),
                rank1.broadcast_weight_update_tensor.remote(n=tensor_numel),
            ]
        )
        print(f"[two-group-demo] sent={sent}", flush=True)
        print(f"[two-group-demo] received={received}", flush=True)
        comparable_keys = ("numel", "dtype", "first16", "last16", "checksum")
        assert {k: sent[k] for k in comparable_keys} == {
            k: received[k] for k in comparable_keys
        }
        return sent, received
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


def test_two_ray_actors_second_nccl_group_broadcast(ray_cluster):
    run_two_actor_second_group_demo()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-numel", type=int, default=2048)
    args = parser.parse_args()
    try:
        run_two_actor_second_group_demo(tensor_numel=args.tensor_numel)
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
