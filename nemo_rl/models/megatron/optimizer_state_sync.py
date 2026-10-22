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

"""Refresh hybrid Adam parameters after distributed checkpoint state is loaded."""

import gc
import re
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import (
        HybridDeviceOptimizer,
    )


def release_unused_pinned_memory_after_checkpoint() -> int:
    """Return unused checkpoint-era pinned blocks to the host operating system.

    Hybrid's load hook rebuilds its pinned CPU working parameters. Tensor
    references can be released while CUDA's host allocator keeps those old
    blocks cached, increasing a colocated critic's initialization peak. Wait
    for outstanding copies, then release only unoccupied cached blocks. This
    leaves live masters, Adam state, model parameters and RNG untouched.
    """
    if not torch.cuda.is_initialized():
        return 0
    empty_host_cache = getattr(torch._C, "_host_emptyCache", None)
    if empty_host_cache is None:
        return 0
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory.host_memory_stats()["allocated_bytes.current"]
    empty_host_cache()
    after = torch.cuda.memory.host_memory_stats()["allocated_bytes.current"]
    return max(0, before - after)


def checkpoint_bf16_master_dtypes(
    checkpoint_path: str | Path,
) -> dict[int, torch.dtype] | None:
    """Read the stored BF16 master formats before building a DCP load template.

    MCore initializes HDO state with a dummy CPU Adam step, so the live state's
    dtype does not describe the source checkpoint. NeMo-RL uses dp_reshardable
    optimizer buckets; inspect their actual Torch DCP metadata per chained
    optimizer. Legacy non-DCP checkpoints load their tensors directly.
    """
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    directory = Path(checkpoint_path)
    if not (directory / ".metadata").is_file():
        tracker = directory / "latest_checkpointed_iteration.txt"
        if tracker.is_file():
            iteration = tracker.read_text().strip()
            directory = directory / (
                "release" if iteration == "release" else f"iter_{int(iteration):07d}"
            )
    if not (directory / ".metadata").is_file():
        if (directory / "common.pt").is_file() and not (
            directory / "metadata.json"
        ).is_file():
            return None
        raise RuntimeError(f"Optimizer checkpoint metadata is missing at {directory}")
    metadata = FileSystemReader(directory).read_metadata()
    formats: dict[int, torch.dtype] = {}
    for name, value in metadata.state_dict_metadata.items():
        match = re.match(
            r"^(?:chained_(\d+)\.)?optimizer\.distributed\..*\.dtype_\(torch\.bfloat16,.*\.param$",
            name,
        )
        if match is None:
            continue
        if not isinstance(value, TensorStorageMetadata):
            raise RuntimeError(f"Optimizer master metadata is not a tensor: {name}")
        dtype = value.properties.dtype
        if dtype not in (torch.int16, torch.float32):
            raise RuntimeError(
                f"Unsupported BF16 optimizer master dtype {dtype} for {name}"
            )
        index = int(match.group(1) or 0)
        if index in formats and formats[index] != dtype:
            raise RuntimeError(
                f"Mixed BF16 master formats in chained optimizer {index}"
            )
        formats[index] = dtype
    return formats


@contextmanager
def preserve_hybrid_optimizer_state_dtypes(
    optimizer: "HybridDeviceOptimizer", *, bf16_master_dtype: torch.dtype | None
) -> Iterator[None]:
    """Keep FusedAdam int16 remainders intact through Torch's state loader.

    Hybrid's own pre-hook temporarily substitutes FP32 parameters. Torch then
    casts every non-step state tensor to FP32, including int16 remainders. The
    distributed checkpoint loader uses that state to build its tensor template;
    preserve the original int16 tensors so checkpoint I/O retains their format.
    The source metadata is required because MCore's dummy Adam step creates
    FP32 state even when the saved checkpoint contains int16 remainders. Both
    hooks are removed after the complete distributed load, including errors.
    """
    remainders: dict[torch.Tensor, torch.Tensor] = {}

    def remember(hybrid: "HybridDeviceOptimizer", state_dict: dict[str, Any]) -> None:
        remainders.clear()
        saved_ids = chain.from_iterable(g["params"] for g in state_dict["param_groups"])
        parameters = chain.from_iterable(g["params"] for g in hybrid.param_groups)
        for saved_id, parameter in zip(saved_ids, parameters, strict=True):
            master = state_dict["state"].get(saved_id, {}).get("master_param")
            if not isinstance(master, torch.Tensor):
                continue
            original = hybrid.fp32_param_to_orig_param.get(parameter, parameter)
            if original.dtype != torch.bfloat16:
                continue
            if bf16_master_dtype == torch.int16:
                # The initial FP32 state comes from MCore's dummy Adam step.
                # Its values are placeholders, not checkpoint master data. Give
                # DCP an int16 destination before it loads the actual remainder.
                remainders[original] = (
                    master
                    if master.dtype == torch.int16
                    else torch.empty_like(master, dtype=torch.int16)
                )
            elif bf16_master_dtype is None and master.dtype == torch.int16:
                remainders[original] = master

    def restore(hybrid: "HybridDeviceOptimizer") -> None:
        for parameter, master in remainders.items():
            hybrid.state[parameter]["master_param"] = master
        remainders.clear()

    before = optimizer.register_load_state_dict_pre_hook(remember)
    after = optimizer.register_load_state_dict_post_hook(restore)
    try:
        yield
    finally:
        after.remove()
        before.remove()
        remainders.clear()


@torch.no_grad()
def refresh_hybrid_optimizer_after_checkpoint(
    optimizer: "HybridDeviceOptimizer",
) -> frozenset[torch.Tensor]:
    """Refresh state assigned after HybridDeviceOptimizer's normal load hook.

    GPU FusedAdam checkpoints may store a BF16 master's low 16 bits as int16.
    Transformer Engine reconstructs the full FP32 value from the rounded BF16
    parameter and the signed remainder (common/multi_tensor/adam.cu). Adding
    that remainder to the BF16 value's FP32 bit representation also undoes the
    high-word rounding, without losing any saved master precision. Return the
    parameters reconstructed from remainders so callers preserve their original
    checkpoint BF16 weights, including TE's halfway rounding convention.
    """
    for parameter, working in optimizer.param_to_fp32_param.items():
        loaded = optimizer.state.get(parameter, {}).get("master_param")
        if (
            not isinstance(loaded, torch.Tensor)
            or loaded.shape != working.shape
            or working.dtype != torch.float32
            or not (
                loaded.dtype == torch.float32
                or (loaded.dtype == torch.int16 and parameter.dtype == torch.bfloat16)
            )
        ):
            raise RuntimeError(
                "Hybrid optimizer resume requires a loaded FP32 master or BF16 "
                "parameter with int16 remainders for every working parameter"
            )
    for parameter, cpu_parameter in optimizer.gpu_params_map_cpu_copy.items():
        if parameter.dtype != torch.float32:
            continue
        loaded = optimizer.state.get(parameter, {}).get("master_param")
        if loaded is not None and (
            not isinstance(loaded, torch.Tensor)
            or loaded.dtype != torch.float32
            or loaded.shape != cpu_parameter.shape
        ):
            raise RuntimeError(
                "Native FP32 optimizer parameters require FP32 checkpoint masters"
            )
    # Validate every entry before modifying any working parameters. Reuse the
    # existing FP32 allocation instead of allocating a second full master copy.
    remainder_parameters: set[torch.Tensor] = set()
    for parameter, working in optimizer.param_to_fp32_param.items():
        state = optimizer.state[parameter]
        loaded = state["master_param"]
        if loaded.dtype == torch.int16:
            working.copy_(parameter)
            working.view(torch.int32).add_(
                loaded.to(device=working.device, dtype=torch.int32)
            )
            state["master_param"] = working
            remainder_parameters.add(parameter)
    # MCore puts dummy step tensors in LocalNonpersistentObject entries. Its
    # final parameter-state load can overwrite the restored step with that
    # placeholder. The common param-group step is the authoritative saved
    # Adam counter; restore it after every distributed tensor assignment.
    for group in optimizer.param_groups:
        saved_step = group.get("step")
        if saved_step is None:
            continue
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            step = state.get("step")
            if isinstance(step, torch.Tensor):
                step.fill_(saved_step)
            else:
                state["step"] = torch.tensor(float(saved_step), dtype=torch.float32)
    optimizer._sync_hdo_state_to_sub_optimizers()
    # Native HDO has now copied the loaded FP32 masters into the parameters
    # CPU Adam actually updates. Release the redundant checkpoint allocations
    # before constructing a colocated critic. Its first step would make the
    # same alias, but a resumed actor must not retain both copies until then.
    for parameter, working in optimizer.param_to_fp32_param.items():
        optimizer.state[parameter]["master_param"] = working
    for parameter, cpu_parameter in optimizer.gpu_params_map_cpu_copy.items():
        if parameter.dtype == torch.float32:
            # The loaded model parameter already is the full FP32 master.
            # GPU FusedAdam updates FP32 parameters directly and can leave its
            # redundant master_param entry stale. Refresh the CPU Adam parameter
            # from the exact FP32 model value and keep its saved state aligned.
            cpu_parameter.copy_(parameter)
            optimizer.state[parameter]["master_param"] = cpu_parameter
    return frozenset(remainder_parameters)
