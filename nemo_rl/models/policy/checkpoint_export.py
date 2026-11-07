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

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Write serving checkpoints while every training rank finishes its gathers.

Adapted from inference-cookbook's Apache-2.0 NeMo-RL policy exporter.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

SHARD_BYTES = 4 * 2**30
_SKIP_SUFFIXES = (".safetensors", ".safetensors.index.json")
#: safetensors dtype names, as checkpoint headers spell them.
_TORCH_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}
#: The dtypes the export casts between. Other dtypes are written only if the policy holds them.
_FLOAT_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32, torch.float64})


class _ShardWriter:
    """Accumulate CPU tensors and write them as ``model-XXXXX-of-YYYYY.safetensors`` shards."""

    def __init__(self, path: Path, shard_bytes: int) -> None:
        self.path = path
        self.tmp = path.with_name(path.name + ".tmp")
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp.mkdir(parents=True)
        self.shard_bytes = shard_bytes
        self._pending: dict[str, torch.Tensor] = {}
        self._pending_bytes = 0
        self._shards: list[tuple[str, list[str]]] = []
        self.total_bytes = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if self._pending and self._pending_bytes + tensor.nbytes > self.shard_bytes:
            self._flush()
        self._pending[name] = tensor
        self._pending_bytes += tensor.nbytes
        self.total_bytes += tensor.nbytes

    def _flush(self) -> None:
        if not self._pending:
            return
        fname = f"model-{len(self._shards):05d}.safetensors"
        save_file(self._pending, str(self.tmp / fname), metadata={"format": "pt"})
        self._shards.append((fname, list(self._pending)))
        self._pending, self._pending_bytes = {}, 0

    def finish(self, config_dir: str | None) -> dict[str, Any]:
        self._flush()
        total = len(self._shards)
        weight_map: dict[str, str] = {}
        for i, (fname, names) in enumerate(self._shards):
            final = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
            os.replace(self.tmp / fname, self.tmp / final)
            weight_map.update({n: final for n in names})
        (self.tmp / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": self.total_bytes},
                    "weight_map": weight_map,
                },
                indent=1,
            )
        )
        if config_dir:
            for entry in Path(config_dir).iterdir():
                if entry.is_file() and not entry.name.endswith(_SKIP_SUFFIXES):
                    shutil.copy2(entry, self.tmp / entry.name)
        shutil.rmtree(self.path, ignore_errors=True)
        self.tmp.replace(self.path)
        return {
            "path": str(self.path),
            "tensors": len(weight_map),
            "bytes": self.total_bytes,
            "shards": total,
        }


def _any_rank_failed(failed: bool) -> bool:
    """Whether any rank failed, agreed by every rank (a MAX all-reduce of a 0/1 flag)."""
    if not torch.distributed.is_initialized():
        return failed
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    flag = torch.tensor([1 if failed else 0], dtype=torch.int32, device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


def _target_dtype(
    name: str, tensor: torch.Tensor, dtypes: dict[str, str] | None
) -> torch.dtype:
    """The dtype to write ``name`` in: the checkpoint's, when the policy's tensor converts to it.

    A cast to FP8 or an integer type ignores a quantized checkpoint's scales. It writes wrong
    values that still match the checkpoint's layout. So the only casts are between the
    floating-point dtypes in ``_FLOAT_DTYPES``.
    """
    if not dtypes or name not in dtypes:
        return tensor.dtype
    target = _TORCH_DTYPES.get(dtypes[name])
    if target is None:
        raise ValueError(
            f"{name} is {dtypes[name]} in the deployment's checkpoint, a safetensors dtype the "
            "export does not know"
        )
    if target in (torch.float32, torch.float64) and tensor.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError(
            f"{name}: cannot restore {target} precision from {tensor.dtype}"
        )
    if target == tensor.dtype or {target, tensor.dtype} <= _FLOAT_DTYPES:
        return target
    raise ValueError(
        f"{name} is {dtypes[name]} in the deployment's checkpoint and {tensor.dtype} in the "
        "policy; the export casts only between F16, BF16, F32 and F64 (a quantized checkpoint "
        "needs its scales too), so it cannot write that dtype"
    )


def export_hf_checkpoint(
    params: Any,
    path: str,
    allowed_names: list[str] | None,
    config_dir: str | None,
    shard_bytes: int,
    dtypes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Iterate ``params`` on every rank; rank 0 writes the checkpoint.

    ``dtypes`` maps tensor names to their safetensors dtype in the deployment's checkpoint. Rank 0
    writes each of those tensors in that dtype. Other tensors keep the dtype ``params`` yields.

    The generator runs a collective per tensor (it gathers TP/PP shards). A rank that stops
    iterating early leaves its peers blocked in the next collective until the distributed
    timeout. So a write failure on rank 0 (for example, a full disk) stops the writing but not the
    iteration. After the last tensor, the ranks agree on whether any rank failed, and all raise
    together.

    Raises:
        RuntimeError: The export failed on this rank or on another rank.
    """
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    allowed = set(allowed_names) if allowed_names else None
    error: BaseException | None = None
    writer: _ShardWriter | None = None
    if rank == 0:
        try:
            writer = _ShardWriter(Path(path), shard_bytes)
        except Exception as exc:  # noqa: BLE001 - reported once every rank has finished
            error = exc
    seen: set[str] = set()
    for name, tensor in params:
        if allowed is not None and name not in allowed:
            continue
        seen.add(name)
        if writer is not None and error is None:
            try:
                target = _target_dtype(name, tensor, dtypes)
                writer.add(
                    name, tensor.detach().to("cpu", target, copy=True).contiguous()
                )
            except Exception as exc:  # noqa: BLE001 - keep iterating so peers do not block
                error = exc
    result: dict[str, Any] = {"rank": rank, "tensors": len(seen)}
    if rank == 0 and error is None:
        if allowed is not None and seen != allowed:
            missing = sorted(allowed - seen)[:5]
            error = RuntimeError(
                f"export is missing {len(allowed - seen)} tensors the deployment holds, "
                f"e.g. {missing}"
            )
        else:
            try:
                result = {**writer.finish(config_dir), "rank": rank}  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001 - reported below, on every rank
                error = exc
    if _any_rank_failed(error is not None):
        if writer is not None:
            shutil.rmtree(writer.tmp, ignore_errors=True)
        if error is not None:
            raise RuntimeError(
                f"checkpoint export failed on rank {rank}: {error}"
            ) from error
        raise RuntimeError(
            f"checkpoint export failed on another rank; rank {rank} stops too"
        )
    return result
