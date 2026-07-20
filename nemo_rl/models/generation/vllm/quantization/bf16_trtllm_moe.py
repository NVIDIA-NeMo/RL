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
"""Refit-idempotent FlashInfer TRTLLM MoE processing for BF16 vLLM engines.

vLLM's UnquantizedFusedMoEMethod._setup_kernel rebinds w13_weight/w2_weight to
brand-new 4D block-layout tensors via replace_parameter, so the second refit
load_weights streams checkpoint-layout shards into a block-layout parameter
and fails (shard_dim=0 on a 3D-expected tensor), and a second processing pass
would double-shuffle. These patches keep the checkpoint-layout parameters as
permanent load targets and recompute the TRTLLM block layout idempotently on
every processing pass, mirroring process_weights_after_loading_mxfp8_moe in
fp8.py. They are inert unless the layer's selected unquantized MoE backend is
FLASHINFER_TRTLLM; all other backends fall through to the original vLLM code.

The pure permutation/layout helpers live in bf16_trtllm_moe_utils.py so they
stay importable (and CPU-testable) without a vLLM install.
"""

import inspect
import os
from dataclasses import dataclass
from unittest.mock import patch

import ray
import torch
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import CoreEngineProcManager

from nemo_rl.models.generation.vllm.quantization.bf16_trtllm_moe_utils import (
    EPILOGUE_TILE_M,
    block_layout_alias,
    gather_rows_to_block_layout,
    swap_w13_to_w31_row_indices,
)


@dataclass(frozen=True)
class Bf16TrtllmMoeConfig:
    model_parallel_size: int = 1


global_bf16_trtllm_config: Bf16TrtllmMoeConfig = None

bf16_trtllm_patches_applied = False
bf16_trtllm_vllm_patches: list = []

original_run_engine_core = EngineCoreProc.run_engine_core
original_core_engine_proc_manager_init = CoreEngineProcManager.__init__

_original_process_weights_after_loading = None
_original_apply_monolithic = None
_original_forward_native = None

# One-shot flag for NRL_BF16_TRTLLM_VERIFY: compare the first refit
# reprocessing against a from-scratch vLLM reference on one layer only.
bf16_trtllm_shuffle_verified = False


def _my_core_engine_proc_manager_init(*args, **kwargs):
    kwargs["vllm_config"].nrl_bf16_trtllm_cfg = global_bf16_trtllm_config
    return original_core_engine_proc_manager_init(*args, **kwargs)


def _my_run_engine_core(*args, **kwargs):
    cfg = kwargs["vllm_config"].nrl_bf16_trtllm_cfg
    del kwargs["vllm_config"].nrl_bf16_trtllm_cfg
    monkey_patch_vllm_ray_executor(cfg)
    if hasattr(kwargs["vllm_config"], "nrl_fp8_cfg"):
        # precision=fp8 with unquantized (bf16) MoE layers: this spawned
        # process imported the module fresh, so original_run_engine_core is
        # the raw vLLM entry point, not fp8's wrapper; chain explicitly so
        # the fp8 patches are applied in this process too.
        from nemo_rl.models.generation.vllm.quantization.fp8 import (
            my_run_engine_core as fp8_my_run_engine_core,
        )

        return fp8_my_run_engine_core(*args, **kwargs)
    return original_run_engine_core(*args, **kwargs)


def monkey_patch_vllm_ray_executor(config: Bf16TrtllmMoeConfig) -> None:
    global bf16_trtllm_patches_applied
    if config.model_parallel_size > 1:
        # Patch collective_rpc so every vLLM TP/PP worker applies the MoE
        # patches before model init (same mechanism as fp8.py).
        from vllm.v1.executor.ray_executor import RayDistributedExecutor

        original_collective_rpc = RayDistributedExecutor.collective_rpc

        def patched_collective_rpc(self, *args, **kwargs):
            global bf16_trtllm_patches_applied
            if not bf16_trtllm_patches_applied:
                futures = [
                    worker.execute_method.remote(apply_bf16_trtllm_moe_patches, config)
                    for worker in self.workers
                ]
                [ray.get(future) for future in futures]
                bf16_trtllm_patches_applied = True

            return original_collective_rpc(self, *args, **kwargs)

        RayDistributedExecutor.collective_rpc = patched_collective_rpc
    else:
        # For single GPU there is no ray executor, so patch directly.
        apply_bf16_trtllm_moe_patches(None, config)


def apply_bf16_trtllm_moe_patches(self, config: Bf16TrtllmMoeConfig) -> None:
    global bf16_trtllm_patches_applied, global_bf16_trtllm_config
    global _original_process_weights_after_loading
    global _original_apply_monolithic
    global _original_forward_native
    assert not bf16_trtllm_patches_applied

    global_bf16_trtllm_config = config

    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    _original_process_weights_after_loading = (
        UnquantizedFusedMoEMethod.process_weights_after_loading
    )
    _original_apply_monolithic = UnquantizedFusedMoEMethod.apply_monolithic
    _original_forward_native = UnquantizedFusedMoEMethod.forward_native

    method_path = (
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
        ".UnquantizedFusedMoEMethod"
    )
    bf16_trtllm_vllm_patches.extend(
        [
            patch(
                f"{method_path}.process_weights_after_loading",
                process_weights_after_loading_bf16_trtllm_moe,
            ),
            patch(f"{method_path}.apply_monolithic", apply_monolithic_bf16_trtllm_moe),
            patch(f"{method_path}.forward_native", forward_native_bf16_trtllm_moe),
        ]
    )
    for p in bf16_trtllm_vllm_patches:
        p.start()

    bf16_trtllm_patches_applied = True


def init_bf16_trtllm_moe(vllm_cfg, model_parallel_size: int) -> None:
    """Install the refit-idempotent TRTLLM MoE patches in every engine process.

    Must run before vLLM model init so the first processing pass already keeps
    the checkpoint layout; refit-time hooks would be too late (the engine's
    initial process_weights_after_loading has already rebound the parameters).
    """
    global global_bf16_trtllm_config
    global_bf16_trtllm_config = Bf16TrtllmMoeConfig(
        model_parallel_size=model_parallel_size
    )

    if vllm_cfg["async_engine"]:
        # For async engine, vLLM spawns a process per engine core, so patch
        # the spawn entry point to apply our patches inside each process.
        EngineCoreProc.run_engine_core = _my_run_engine_core
        CoreEngineProcManager.__init__ = _my_core_engine_proc_manager_init
    else:
        monkey_patch_vllm_ray_executor(global_bf16_trtllm_config)


def _is_flashinfer_trtllm_backend(self) -> bool:
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoeBackend,
    )

    return self.unquantized_backend == UnquantizedMoeBackend.FLASHINFER_TRTLLM


def _install_refit_load_counters(layer) -> None:
    """Count weight loads into this layer through its fused-MoE weight loaders.

    apply/forward assert the layer was reprocessed after its last load: a
    load_weights that is not followed by a full process_weights_after_loading
    pass (e.g. the modelopt real-quant ZMQ refit or an MTP draft-weight
    reload) would leave checkpoint-layout bytes in the storage the TRTLLM
    kernel reads through the block-layout apply views.
    """
    layer._bf16_trtllm_load_count = 0
    layer._bf16_trtllm_processed_count = 0
    for param in (layer.w13_weight, layer.w2_weight):
        loader = getattr(param, "weight_loader", None)
        if loader is None:
            continue

        def counting_loader(param, loaded_weight, *args, _loader=loader, **kwargs):
            layer._bf16_trtllm_load_count += 1
            return _loader(param, loaded_weight, *args, **kwargs)

        param.weight_loader = counting_loader


def _assert_processed_after_load(layer) -> None:
    if getattr(layer, "_bf16_trtllm_load_count", 0) != getattr(
        layer, "_bf16_trtllm_processed_count", 0
    ):
        raise RuntimeError(
            "BF16 FlashInfer TRTLLM MoE layer received a weight load without "
            "a full process_weights_after_loading pass, so the kernel would "
            "read stale block-layout bytes. This load path (e.g. the modelopt "
            "real-quant ZMQ refit or an MTP draft-weight reload) is not "
            "supported by the refit-idempotent BF16 TRTLLM patches; set "
            "NRL_BF16_TRTLLM_REFIT=0 to disable them."
        )


def _bf16_trtllm_row_permutations(
    layer,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    is_gated: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Composed row-index permutations for the batched TRTLLM shuffle.

    swap_w13_to_w31, reorder_rows_for_gated_act_gemm, and shuffle_matrix_a are
    all input-independent row permutations, so composing their indices once
    reproduces vLLM's per-expert pipeline as a single gather per tensor.
    Cached on the layer as CPU tensors; device copies are made per call because
    tensors allocated while vLLM loads the model land in the sleep-mode weights
    pool, whose contents are discarded at sleep_level=2.
    """
    perm_w13 = getattr(layer, "_bf16_trtllm_perm_w13", None)
    perm_w2 = getattr(layer, "_bf16_trtllm_perm_w2", None)
    if perm_w13 is None or perm_w2 is None:
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        cache_permute_indices: dict = {}
        perm_w13 = _maybe_get_cached_w3_w1_permute_indices(
            cache_permute_indices,
            w13_weight[0].view(torch.uint8),
            EPILOGUE_TILE_M,
            is_gated_act_gemm=is_gated,
        ).cpu()
        if is_gated:
            # vLLM applies the flashinfer permutation to the already-swapped
            # [w3; w1] tensor; fold the swap into the composed gather.
            perm_w13 = swap_w13_to_w31_row_indices(w13_weight.shape[1])[perm_w13]
        perm_w2 = get_w2_permute_indices_with_cache(
            cache_permute_indices,
            w2_weight[0].view(torch.uint8),
            EPILOGUE_TILE_M,
        ).cpu()
        layer._bf16_trtllm_perm_w13 = perm_w13
        layer._bf16_trtllm_perm_w2 = perm_w2
    device = w13_weight.device
    return perm_w13.to(device), perm_w2.to(device)


def _shuffle_bf16_trtllm_moe_batched(
    layer,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    is_gated: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-permute both stacked expert weights as one gather per tensor.

    Returns block-layout uint8 views over the shared scratch buffers; callers
    copy_ them into the persistent destinations.
    """
    perm_w13, perm_w2 = _bf16_trtllm_row_permutations(
        layer, w13_weight, w2_weight, is_gated
    )
    return gather_rows_to_block_layout(w13_weight, w2_weight, perm_w13, perm_w2)


def _shuffle_bf16_trtllm_moe_reference(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    is_gated: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """From-scratch vLLM pipeline, kept for NRL_BF16_TRTLLM_VERIFY."""
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        convert_moe_weights_to_flashinfer_trtllm_block_layout,
        swap_w13_to_w31,
    )

    if is_gated:
        w13_weight = swap_w13_to_w31(w13_weight)
    cache_permute_indices: dict = {}
    # is_gated_act was added in a later vLLM 0.20 patch release; older wheels
    # hardcode the gated layout, which is only valid when is_gated is True.
    params = inspect.signature(
        convert_moe_weights_to_flashinfer_trtllm_block_layout
    ).parameters
    if "is_gated_act" in params:
        return convert_moe_weights_to_flashinfer_trtllm_block_layout(
            cache_permute_indices,
            w13_weight,
            w2_weight,
            is_gated_act=is_gated,
        )
    if not is_gated:
        raise RuntimeError(
            "Installed vLLM's convert_moe_weights_to_flashinfer_trtllm_block_layout "
            "does not support non-gated MoE; cannot build the verify reference. "
            "Unset NRL_BF16_TRTLLM_VERIFY for this model."
        )
    return convert_moe_weights_to_flashinfer_trtllm_block_layout(
        cache_permute_indices,
        w13_weight,
        w2_weight,
    )


def process_weights_after_loading_bf16_trtllm_moe(self, layer) -> None:
    """Refit-idempotent replacement for UnquantizedFusedMoEMethod processing.

    For the FLASHINFER_TRTLLM backend only: w13_weight/w2_weight keep their
    checkpoint layout and weight_loader forever (every refit load works
    unchanged), while the TRTLLM block layout is recomputed from them into
    w13_weight_for_apply/w2_weight_for_apply, 4D views aliasing the same
    storage. The moe kernel is built once; refit passes only rewrite bytes,
    never rebind parameters, so CUDA graphs stay valid.
    """
    if not _is_flashinfer_trtllm_backend(self):
        return _original_process_weights_after_loading(self, layer)

    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        make_unquantized_moe_kernel,
    )

    if (
        layer.w13_weight.dtype != torch.bfloat16
        or layer.w2_weight.dtype != torch.bfloat16
    ):
        raise ValueError(
            "Unquantized MoE Backend FlashInfer TRTLLM requires bfloat16 weights"
        )

    is_gated = layer.moe_config.is_act_and_mul
    w13_weight = layer.w13_weight.data
    w2_weight = layer.w2_weight.data
    first_load = not hasattr(layer, "w13_weight_for_apply")

    global bf16_trtllm_shuffle_verified
    verify = (
        os.getenv("NRL_BF16_TRTLLM_VERIFY") == "1"
        and not bf16_trtllm_shuffle_verified
        and not first_load
    )
    # Snapshot the reference before the in-place copy below overwrites the
    # checkpoint bytes with the block layout.
    reference = (
        _shuffle_bf16_trtllm_moe_reference(w13_weight, w2_weight, is_gated)
        if verify
        else None
    )

    w13_blocked, w2_blocked = _shuffle_bf16_trtllm_moe_batched(
        layer, w13_weight, w2_weight, is_gated
    )

    if first_load:
        layer.w13_weight_for_apply = block_layout_alias(layer.w13_weight)
        layer.w2_weight_for_apply = block_layout_alias(layer.w2_weight)
        _install_refit_load_counters(layer)
    # The gathers above fully copied the checkpoint bytes into scratch, so
    # overwriting the aliased parameter storage here is safe.
    layer.w13_weight_for_apply.view(torch.uint8).copy_(w13_blocked)
    layer.w2_weight_for_apply.view(torch.uint8).copy_(w2_blocked)
    layer._bf16_trtllm_processed_count = layer._bf16_trtllm_load_count

    if reference is not None:
        for got, want, tensor_name in zip(
            (layer.w13_weight_for_apply, layer.w2_weight_for_apply),
            reference,
            ("w13_weight", "w2_weight"),
        ):
            assert torch.equal(got.view(torch.uint8), want.view(torch.uint8)), (
                "BF16 TRTLLM refit reprocessing mismatch vs from-scratch "
                f"reference: {tensor_name}"
            )
        bf16_trtllm_shuffle_verified = True
        print(
            "[NRL_BF16_TRTLLM_VERIFY] refit-reprocessed MoE weights match the "
            "from-scratch vLLM reference bit-exactly"
        )

    # Build the kernel on initial load only (same as upstream _setup_kernel
    # but without replace_parameter). Refit calls must not rebuild it: the
    # runner captured it, and they may lack set_current_vllm_config context.
    self.moe_quant_config = self.get_fused_moe_quant_config(layer)
    if self.moe_kernel is None:
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_unquantized_moe_kernel(
            quant_config=self.moe_quant_config,
            moe_config=self.moe,
            backend=self.unquantized_backend,
            experts_cls=self.experts_cls,
            routing_tables=layer._maybe_init_expert_routing_tables(),
            shared_experts=layer.shared_experts,
        )


def apply_monolithic_bf16_trtllm_moe(self, layer, x, router_logits, input_ids=None):
    """apply_monolithic redirected to the block-layout apply tensors."""
    if not _is_flashinfer_trtllm_backend(self):
        return _original_apply_monolithic(self, layer, x, router_logits, input_ids)
    assert self.is_monolithic
    assert self.moe_kernel is not None
    _assert_processed_after_load(layer)
    return self.moe_kernel.apply_monolithic(
        x,
        getattr(layer, "w13_weight_for_apply", layer.w13_weight),
        getattr(layer, "w2_weight_for_apply", layer.w2_weight),
        router_logits,
        activation=layer.activation,
        global_num_experts=layer.global_num_experts,
        expert_map=layer.expert_map,
        apply_router_weight_on_input=layer.apply_router_weight_on_input,
        num_expert_group=layer.num_expert_group,
        topk_group=layer.topk_group,
        e_score_correction_bias=layer.e_score_correction_bias,
        routed_scaling_factor=layer.routed_scaling_factor,
    )


def forward_native_bf16_trtllm_moe(
    self, layer, x, topk_weights, topk_ids, shared_experts_input
):
    """forward_native redirected to the block-layout apply tensors."""
    if not _is_flashinfer_trtllm_backend(self):
        return _original_forward_native(
            self, layer, x, topk_weights, topk_ids, shared_experts_input
        )
    assert self.moe_kernel is not None
    _assert_processed_after_load(layer)
    return self.moe_kernel.apply(
        hidden_states=x,
        w1=getattr(layer, "w13_weight_for_apply", layer.w13_weight),
        w2=getattr(layer, "w2_weight_for_apply", layer.w2_weight),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=layer.activation,
        apply_router_weight_on_input=layer.apply_router_weight_on_input,
        global_num_experts=layer.global_num_experts,
        expert_map=layer.expert_map,
        shared_experts_input=shared_experts_input,
    )
