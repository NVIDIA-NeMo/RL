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

"""Ray-wrapped version of ``repro_pp_weight_update.py``.

The standalone repro is real SGLang plus a mock Megatron sender:

* receiver: a real bare ``sglang.srt.entrypoints.engine.Engine``
* sender: a hand-written trainer that reads real HF safetensors and broadcasts
  them with ``nemo_rl.models.policy.utils.init_process_group``

This test keeps those two roles, but wraps both sides in Ray actors. It answers
one narrow question: does adding Ray actor boundaries to the standalone repro
trigger the same NCCL failure as the real Megatron -> SGLangGeneration UT?
"""

from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path
from typing import Any

import pytest
import ray
import torch

pytestmark = pytest.mark.sglang

os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")

_DTYPE_FROM_STR = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def _find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _select_specs(
    specs: list[tuple[str, str, list[int]]],
    *,
    param_name: str | None,
    max_tensors: int,
) -> list[tuple[str, str, list[int]]]:
    if param_name:
        matches = [spec for spec in specs if spec[0] == param_name]
        if not matches:
            available = ", ".join(name for name, _, _ in specs[:20])
            raise RuntimeError(
                f"param {param_name!r} not found; first available params: {available}"
            )
        return matches
    if max_tensors > 0:
        return specs[:max_tensors]
    return specs


def _ray_get_best_effort(ref, *, label: str, timeout_s: float = 5.0):
    try:
        return ray.get(ref, timeout=timeout_s)
    except Exception as exc:
        print(f"[ray-wrapped-repro] cleanup {label} did not finish: {exc!r}", flush=True)
        return None


def _describe_ref(label: str, ref, ready_refs: set) -> str:
    if ref not in ready_refs:
        try:
            ray.cancel(ref, force=True)
        except Exception:
            pass
        return f"{label}=pending"
    try:
        return f"{label}={ray.get(ref)!r}"
    except Exception as exc:
        return f"{label}=raised {exc!r}"


@ray.remote(num_gpus=1)
class RaySGLangEngine:
    """Real bare SGLang Engine running inside one Ray actor."""

    def __init__(self) -> None:
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
        self._engine = None

    def start(self, *, model_path: str, tp: int, pp: int, dp: int) -> dict[str, Any]:
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
        from sglang.srt.entrypoints.engine import Engine

        self._engine = Engine(
            model_path=model_path,
            tp_size=tp,
            pp_size=pp,
            dp_size=dp,
            mem_fraction_static=0.5,
            log_level="info",
            random_seed=42,
            disable_cuda_graph=True,
        )
        return {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "nccl_cumem_enable": os.environ.get("NCCL_CUMEM_ENABLE", ""),
        }

    def init_weight_update_group(
        self,
        *,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
    ) -> tuple[bool, str]:
        if self._engine is None:
            raise RuntimeError("engine is not started")
        return self._engine.init_weights_update_group(
            master_address=master_addr,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend="nccl",
        )

    def update_one(
        self,
        *,
        name: str,
        dtype: str,
        shape: list[int],
        group_name: str,
    ):
        if self._engine is None:
            raise RuntimeError("engine is not started")
        return self._engine.update_weights_from_distributed(
            names=[name],
            dtypes=[dtype],
            shapes=[shape],
            group_name=group_name,
            flush_cache=False,
        )

    def destroy_weight_update_group(self, *, group_name: str):
        if self._engine is not None:
            return self._engine.destroy_weights_update_group(group_name=group_name)
        return False, "engine is not started"

    def shutdown(self) -> None:
        if self._engine is not None:
            try:
                self._engine.shutdown()
            finally:
                self._engine = None


@ray.remote(num_gpus=1)
class RayMockMegatronSender:
    """Mock Megatron rank 0 running inside one Ray actor."""

    def __init__(self) -> None:
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
        self._pg = None

    def node_ip(self) -> str:
        import ray as _ray

        return _ray.util.get_node_ip_address()

    def device_info(self) -> dict[str, Any]:
        return {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device()
            if torch.cuda.is_available()
            else None,
            "device_name": torch.cuda.get_device_name(0),
            "nccl_cumem_enable": os.environ.get("NCCL_CUMEM_ENABLE", ""),
        }

    def init_weight_update_group(
        self,
        *,
        master_addr: str,
        master_port: int,
        world_size: int,
        group_name: str,
    ) -> dict[str, Any]:
        from nemo_rl.models.policy.utils import init_process_group

        os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
        torch.cuda.set_device(0)
        self._pg = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=0,
            group_name=group_name,
        )
        return self.device_info()

    def broadcast_one(
        self,
        *,
        ckpt: str,
        weight_map: dict[str, str],
        name: str,
        dtype: str,
        group_name: str,
    ) -> dict[str, Any]:
        import torch.distributed as dist
        from safetensors import safe_open

        if self._pg is None:
            raise RuntimeError("weight-update process group is not initialized")

        shard = Path(ckpt) / weight_map[name]
        with safe_open(shard, framework="pt") as reader:
            tensor = reader.get_tensor(name)
        tensor = tensor.to(device="cuda:0", dtype=_DTYPE_FROM_STR[dtype]).contiguous()
        dist.broadcast(tensor, src=0, group=self._pg)
        torch.cuda.synchronize()
        return {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "checksum": float(tensor.float().sum().item()),
        }

    def destroy_weight_update_group(self) -> None:
        import torch.distributed as dist

        if self._pg is not None:
            try:
                dist.destroy_process_group(self._pg)
            finally:
                self._pg = None


def run_ray_wrapped_repro(
    *,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    param_name: str | None = "model.norm.weight",
    max_tensors: int = 1,
) -> list[dict[str, Any]]:
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    if torch.cuda.device_count() < 2:
        raise RuntimeError("ray-wrapped repro requires at least 2 visible GPUs")

    from repro_pp_weight_update import _ckpt_path, _collect_specs

    ckpt = _ckpt_path()
    specs, weight_map = _collect_specs(ckpt)
    selected_specs = _select_specs(
        specs,
        param_name=param_name,
        max_tensors=max_tensors,
    )
    print(
        f"[ray-wrapped-repro] ckpt={ckpt} selected={len(selected_specs)} "
        f"first={selected_specs[0][0]} last={selected_specs[-1][0]}",
        flush=True,
    )

    group_name = "ray_wrapped_repro_weight_update"
    world_size = 2
    rank_offset = 1

    engine = RaySGLangEngine.remote()
    sender = RayMockMegatronSender.remote()
    try:
        engine_info = ray.get(
            engine.start.remote(model_path=str(ckpt), tp=tp, pp=pp, dp=dp)
        )
        sender_info = ray.get(sender.device_info.remote())
        print(f"[ray-wrapped-repro] engine={engine_info}", flush=True)
        print(f"[ray-wrapped-repro] sender={sender_info}", flush=True)

        master_addr = ray.get(sender.node_ip.remote())
        master_port = _find_free_port()
        print(
            f"[ray-wrapped-repro] init group tcp://{master_addr}:{master_port}",
            flush=True,
        )
        sender_ref = sender.init_weight_update_group.remote(
            master_addr=master_addr,
            master_port=master_port,
            world_size=world_size,
            group_name=group_name,
        )
        engine_ref = engine.init_weight_update_group.remote(
            master_addr=master_addr,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
        )
        sender_group_info, engine_group_result = ray.get([sender_ref, engine_ref])
        print(
            f"[ray-wrapped-repro] sender_group={sender_group_info} "
            f"engine_group={engine_group_result}",
            flush=True,
        )
        success, message = engine_group_result
        if not success:
            raise RuntimeError(f"engine init_weights_update_group failed: {message}")

        results: list[dict[str, Any]] = []
        for i, (name, dtype, shape) in enumerate(selected_specs, start=1):
            recv_ref = engine.update_one.remote(
                name=name,
                dtype=dtype,
                shape=shape,
                group_name=group_name,
            )
            send_ref = sender.broadcast_one.remote(
                ckpt=str(ckpt),
                weight_map=weight_map,
                name=name,
                dtype=dtype,
                group_name=group_name,
            )
            ready, pending = ray.wait([send_ref, recv_ref], num_returns=2, timeout=60)
            if pending:
                ready_refs = set(ready)
                details = ", ".join(
                    [
                        _describe_ref("send", send_ref, ready_refs),
                        _describe_ref("recv", recv_ref, ready_refs),
                    ]
                )
                raise RuntimeError(
                    f"timed out waiting for weight update {name!r}; {details}"
                )
            send_result, recv_result = ray.get([send_ref, recv_ref])
            print(
                f"[ray-wrapped-repro] {i}/{len(selected_specs)} {name} "
                f"send={send_result} recv={recv_result}",
                flush=True,
            )
            success, message = recv_result
            if not success:
                raise RuntimeError(f"engine update_one failed for {name!r}: {message}")
            results.append(send_result)
        return results
    finally:
        _ray_get_best_effort(
            engine.destroy_weight_update_group.remote(group_name=group_name),
            label="engine.destroy_weight_update_group",
        )
        _ray_get_best_effort(
            sender.destroy_weight_update_group.remote(),
            label="sender.destroy_weight_update_group",
        )
        _ray_get_best_effort(engine.shutdown.remote(), label="engine.shutdown")
        for actor in (engine, sender):
            try:
                ray.kill(actor)
            except Exception:
                pass


def test_ray_wrapped_sglang_engine_mock_megatron_weight_update(ray_cluster):
    results = run_ray_wrapped_repro()
    assert results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--param-name", type=str, default="model.norm.weight")
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=1,
        help="Used only when --param-name is empty.",
    )
    args = parser.parse_args()

    param_name = args.param_name or None
    try:
        run_ray_wrapped_repro(
            tp=args.tp,
            pp=args.pp,
            dp=args.dp,
            param_name=param_name,
            max_tensors=args.max_tensors,
        )
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
