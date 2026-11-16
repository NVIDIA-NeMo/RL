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
"""Refit BF16 policy weights into vLLM's Inkling2 model (vllm-project/vllm#67325).

vLLM's Inkling2 serves routed experts only as NVFP4 or MXFP8, and its
``load_weights`` expects the whole checkpoint in one call with exact dtypes.
NeMo-RL refits BF16 HF-format tensors in buckets, so this patch wraps
``Inkling2ForConditionalGeneration.load_weights`` to:

* quantize ``mlp.experts.{w13,w2}_weight`` to MXFP8 (e4m3 values with UE8M0
  scales per 32 input elements, vLLM's own ``mxfp8_quantize``) when the live
  parameter is MXFP8, emitting the matching ``*_weight_scale`` tensor;
* cast every other tensor to the live parameter's dtype;
* accept buckets that cover only part of the model.

Serve with ``load_format: dummy`` and an ``hf_quant_config.json`` that marks
the experts ``mxfp8``; the first refit then fills every weight. Installed in
vLLM worker processes when ``NRL_VLLM_INKLING2_MXFP8_REFIT=1`` (set it through
``policy.generation.vllm_cfg.env_vars``).
"""

import logging
import os
import re
from collections.abc import Iterable, Iterator
from typing import Any

import torch

logger = logging.getLogger(__name__)

ENV_FLAG = "NRL_VLLM_INKLING2_MXFP8_REFIT"
_EXPERT_WEIGHT = re.compile(r"\.mlp\.experts\.w(13|2)_weight$")
_installed = False


def _quantize_experts_mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[E, N, K]`` bf16 -> e4m3 ``[E, N, K]`` and UE8M0 ``[E, N, K // 32]``."""
    from vllm.models.inkling2.ops.mxfp8 import mxfp8_quantize

    experts, rows, cols = weight.shape
    values, scales = mxfp8_quantize(
        weight.to(torch.bfloat16).reshape(experts * rows, cols).contiguous()
    )
    return values.view(experts, rows, cols), scales.view(experts, rows, cols // 32)


def _install_supported_token_tiles() -> None:
    """Skip expert-GEMM token tiles with no kernel for the model's shapes.

    The TRT-LLM-gen MXFP8 kernel of the smallest token tile (8) has a K tile
    of 512, so models with a hidden or expert intermediate size below that
    (the tiny test checkpoint has 256) fail ``expert_plan``. Small batches
    then use the next supported tile instead.
    """
    from vllm.models.inkling2 import moe as moe_mod
    from vllm.models.inkling2.ops import trtllm_bmm

    original_load_kernels = trtllm_bmm.load_kernels

    def load_kernels(weights: Any, weight_scales: Any, topk: int) -> None:
        nvfp4 = weights[0].dtype == torch.uint8
        pack = 2 if nvfp4 else 1
        shapes = tuple((w.shape[1], w.shape[2] * pack) for w in weights)
        supported = []
        for max_tokens, tile in trtllm_bmm._TILES:
            try:
                trtllm_bmm.expert_plan(
                    1, topk, weights[0].shape[0], nvfp4, shapes, tile
                )
            except AssertionError:
                continue
            supported.append((max_tokens, tile))
        if not supported:
            raise RuntimeError(f"No Inkling2 expert GEMM tile supports shapes {shapes}")
        if len(supported) != len(trtllm_bmm._TILES):
            supported[-1] = (trtllm_bmm._TILES[-1][0], supported[-1][1])
            logger.warning(
                "Inkling2 expert shapes %s: using token tiles %s", shapes, supported
            )
            trtllm_bmm._TILES = tuple(supported)
        original_load_kernels(weights, weight_scales, topk)

    moe_mod.load_kernels = load_kernels


def _install_narrow_combine() -> None:
    """Let the MoE combine kernel run with hidden sizes below its 512 block."""
    from vllm.models.inkling2.ops import moe as ops_moe

    def _combine(down: Any, src2dst: Any, topk: int, out: Any, sp: Any) -> None:
        hidden = down.shape[1]
        tokens = src2dst.numel() // topk
        block = min(512, hidden & -hidden)
        assert block >= 32, f"hidden size {hidden} is not a multiple of 32"
        ops_moe._combine_kernel[(tokens, hidden // block)](
            down,
            src2dst,
            out,
            None if sp is None else sp.slot_bases,
            tokens,
            1 if sp is None else sp.shard(tokens)[2],
            hidden,
            topk,
            block,
            tokens <= ops_moe.LAUNCH_DEPENDENTS_MAX_TOKENS,
            sp is not None,
            ops_moe.HALO_ROWS,
            num_warps=2,
            launch_pdl=True,
        )

    ops_moe._combine = _combine


def _install_flashinfer_multicast_memory_compat() -> None:
    """Match Inkling2's MNNVL all-reduce buffers to FlashInfer >= 0.7.3.

    vLLM's Inkling2 overrides ``SymmDeviceMemory._get_allocation_prop(buf_size)``
    to use minimum allocation/multicast granularities. FlashInfer 0.7.3 calls it
    as ``(buf_size, handle_type)``, which breaks every TP > 1 engine. Same
    override, forwarding the handle type.
    """
    import torch.distributed as dist
    from vllm.models.inkling2.ops import allreduce as allreduce_mod
    from vllm.utils.math_utils import round_up

    def _multicast_memory(size: int, group: Any) -> Any:
        from cuda.bindings import driver as cuda
        from flashinfer.comm.mnnvl import (
            SymmDeviceMemory,
            TorchDistBackend,
            checkCudaErrors,
        )

        class MinimumGranularityMemory(SymmDeviceMemory):
            def _get_allocation_prop(self, buf_size: int, *args: Any) -> Any:
                allocation_prop, mc_prop = super()._get_allocation_prop(buf_size, *args)
                granularity = checkCudaErrors(
                    cuda.cuMemGetAllocationGranularity(
                        allocation_prop,
                        cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                    )
                )
                self.allocation_size = round_up(
                    buf_size + self.SIGNAL_PAD_SIZE, granularity
                )
                if mc_prop is None:
                    return allocation_prop, mc_prop
                mc_prop.size = self.allocation_size
                self._mc_granularity = checkCudaErrors(
                    cuda.cuMulticastGetGranularity(
                        mc_prop,
                        cuda.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_MINIMUM,
                    )
                )
                granularity = max(granularity, self._mc_granularity)
                self.allocation_size = mc_prop.size = round_up(
                    mc_prop.size, granularity
                )
                return allocation_prop, mc_prop

        return MinimumGranularityMemory(
            size,
            group.size(),
            dist.get_rank(group),
            torch.accelerator.current_device_index(),
            TorchDistBackend(group=group),
        )

    allreduce_mod._multicast_memory = _multicast_memory


def _install_cross_domain_allreduce_fallback() -> None:
    """Use NCCL all-reduce for TP groups that span NVLink domains.

    ``Inkling2AllReduceWorkspace.create`` checks only that every rank supports
    multicast, so a TP 8 engine whose two nodes sit in different NVL72 racks
    fails in FlashInfer's MNNVL handle exchange. FlashInfer
    raises on every rank alike (it compares the gathered fabric IDs), so
    returning ``None`` -- vLLM's own "no MNNVL" result -- keeps ranks consistent.
    """
    from vllm.models.inkling2.ops.allreduce import Inkling2AllReduceWorkspace as ws

    original_create = ws.create.__func__

    def create(cls: Any, group: Any, hidden_size: int) -> Any:
        try:
            return original_create(cls, group, hidden_size)
        except RuntimeError as error:
            if "NVLink domain" not in str(error):
                raise
            logger.warning(
                "Inkling2 TP group spans NVLink domains; using NCCL all-reduce: %s",
                str(error).splitlines()[0],
            )
            return None

    ws.create = classmethod(create)


def install() -> None:
    """Patch vLLM's Inkling2 ``load_weights`` in this process. Idempotent."""
    global _installed
    if _installed:
        return
    from vllm.models.inkling2.model import Inkling2ForConditionalGeneration as model_cls

    _install_supported_token_tiles()
    _install_narrow_combine()
    _install_flashinfer_multicast_memory_compat()
    _install_cross_domain_allreduce_fallback()

    original_load = model_cls.load_weights

    def load_weights(
        self: Any, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        params = dict(self.named_parameters())

        def live_param(name: str) -> torch.nn.Parameter | None:
            for old, new in self._renames:
                name = name.replace(old, new)
            for old, (new, _) in self._shards.items():
                if old in name:
                    name = name.replace(old, new)
                    break
            return params.get(name)

        def converted() -> Iterator[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                param = live_param(name)
                if param is None or param.dtype == tensor.dtype:
                    yield name, tensor
                elif _EXPERT_WEIGHT.search(name) and param.dtype == torch.float8_e4m3fn:
                    values, scales = _quantize_experts_mxfp8(tensor)
                    yield name, values
                    yield f"{name}_scale", scales
                else:
                    yield name, tensor.to(param.dtype)

        try:
            return original_load(self, converted())
        except ValueError as error:
            # Raised only after every tensor of the bucket was loaded.
            if not str(error).startswith("Missing checkpoint tensors"):
                raise
            return set()

    model_cls.load_weights = load_weights
    _installed = True
    logger.warning("Installed NeMo-RL MXFP8 refit for vLLM Inkling2")


def maybe_install() -> None:
    """Install the patch when ``NRL_VLLM_INKLING2_MXFP8_REFIT=1``."""
    if os.environ.get(ENV_FLAG) == "1":
        install()
