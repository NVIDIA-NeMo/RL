# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
from typing import Any

import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.quantization.calib.max import MaxCalibrator
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from modelopt.torch.quantization.plugins.vllm import disable_compilation

from nemo_rl.modelopt.utils import resolve_quant_cfg
from nemo_rl.models.generation.vllm.vllm_backend import NixlVllmWorker


@contextmanager
def _tolerate_dummy_weight_nan_amax():
    """Scope-locally make `MaxCalibrator.collect` zero-fill on fully-NaN inputs.

    When this prolog runs, vLLM's model still has *uninitialized / dummy*
    weights — the real weights only arrive later via refit from the
    Megatron policy worker. Cumulative BF16 matmuls on dummy weights can
    overflow at deeper layers (e.g. Nemotron-3-Nano's Mamba `out_proj`
    at layer 4) and produce NaN, which then cascades to every downstream
    quantizer's input during dummy calibration.

    The dummy-calibration amax is *meant* to be discarded — the prolog
    sentinels every enabled quantizer's `_amax` to `-1.0` immediately
    afterwards, and Megatron's real amax is loaded by the temporary loaders
    attached in `VllmQuantInternalWorkerExtension` during refit
    (`max(-1.0, real)=real`). So a fully-NaN input here should produce zero
    amax rather than crash the prolog.

    Scoping this monkey-patch to the prolog (instead of editing
    `MaxCalibrator.collect` in modelopt) keeps modelopt's source pristine
    and limits the workaround to the single dummy-weight code path that
    needs it. Genuine numerical NaN at runtime — when the calibrator is
    no longer active — would still be caught by the production callsite.

    Nonfinite dummy activations are sanitized before calibration reduce. The
    patch is active only inside the dummy-weight prolog, before runtime
    generation starts and before real amax values are loaded.
    """
    _original_collect = MaxCalibrator.collect

    @torch.no_grad()
    def _safe_collect(self, x):
        if x.device.type != "meta" and x.numel() > 0 and not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        _original_collect(self, x)

    MaxCalibrator.collect = _safe_collect
    try:
        yield
    finally:
        MaxCalibrator.collect = _original_collect


def _drop_nonclass_quant_registry_keys() -> None:
    """Drop non-class keys from ModelOpt's QuantModuleRegistry.

    vLLM >= 0.25 turned FusedMoE/SharedFusedMoE into factory functions, but
    ModelOpt's vLLM plugin still registers them as QuantModuleRegistry keys.
    Registry lookups run issubclass(<module type>, <key>) over all keys
    (modelopt/torch/opt/dynamic.py::_get_registered_nn_class) and raise
    TypeError on a function key. MoE fakequant is re-registered against the
    vLLM >= 0.25 module layout in _register_routed_experts_quant_module.
    """
    import inspect

    from modelopt.torch.quantization.nn import QuantModuleRegistry

    dropped = [key for key in QuantModuleRegistry._registry if not inspect.isclass(key)]
    if dropped:
        # Record what was purged. Today this is only vllm_FusedMoE, whose key
        # became a factory function in 0.25 and which is re-registered against
        # RoutedExperts below. If vLLM turns another layer class into a factory,
        # this loop would silently drop that quantization too -- e.g. a
        # RowParallelLinear factory would leave the fakequant model quantizing
        # MoE and nothing else, with no error anywhere.
        print(f"Dropping non-class QuantModuleRegistry keys: {dropped}")
    for key in dropped:
        QuantModuleRegistry.unregister(key)


def _register_routed_experts_quant_module() -> None:
    """Register ModelOpt MoE fakequant for vLLM >= 0.25's RoutedExperts layout.

    ModelOpt's vLLM plugin targets the pre-0.25 FusedMoE class, so with
    vLLM >= 0.25 no MoE quantizers are inserted at all and refit crashes on
    the incoming expert amax keys. The expert weights and kernels now live on
    the nested RoutedExperts module, so an adapted _QuantFusedMoEBase is
    registered there:

    - Quantizer buffers land at ``...experts.routed_experts.*_quantizer``,
      which is exactly the parameter path model ``load_weights`` derives from
      incoming per-expert ``experts.N.<proj>.input_quantizer._amax`` keys via
      the expert mapping, so refit amax loading works unchanged.
    - RoutedExperts owns ``w13_weight``/``w2_weight``, keeping the base
      class's ``B is self.w13_weight`` kernel-operand checks valid.
    - The fakequant kernel swap wraps ``forward_modular``/
      ``forward_monolithic`` (RoutedExperts.forward asserts it must not be
      called) instead of the base class's ``forward``.
    """
    from functools import partial

    from modelopt.torch.quantization.nn import QuantModuleRegistry
    from modelopt.torch.quantization.plugins import vllm as modelopt_vllm
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    if any(key is RoutedExperts for key in QuantModuleRegistry._registry):
        return

    class _QuantVllmRoutedExperts(modelopt_vllm._QuantFusedMoEBase):
        def _setup(self):
            # Mirrors _QuantFusedMoEBase._setup, minus its assert that
            # quant_method is exactly the pre-0.25 UnquantizedFusedMoEMethod
            # class object.
            desc_input = modelopt_vllm.QuantLinearConvBase.default_quant_desc_input
            desc_weight = modelopt_vllm.QuantLinearConvBase.default_quant_desc_weight
            desc_output = modelopt_vllm.QuantLinearConvBase.default_quant_desc_output
            self.w13_input_quantizer = TensorQuantizer(desc_input)
            self.w2_input_quantizer = TensorQuantizer(desc_input)
            self.w13_weight_quantizer = TensorQuantizer(desc_weight)
            self.w2_weight_quantizer = TensorQuantizer(desc_weight)
            self.w13_output_quantizer = TensorQuantizer(desc_output)
            self.w2_output_quantizer = TensorQuantizer(desc_output)
            self.w13_output_quantizer.disable()
            self.w2_output_quantizer.disable()
            quant_method_name = type(self.quant_method).__name__
            assert "Unquantized" in quant_method_name, (
                "MoE fakequant requires unquantized vLLM experts, got "
                f"{quant_method_name}"
            )
            self.parallel_state = modelopt_vllm.create_parallel_state()

        @contextmanager
        def _fakequant_kernels(self):
            # Patch every module that binds the MoE GEMM kernels, not just
            # fused_moe.fused_moe (all ModelOpt's plugin covers): vLLM 0.25's
            # modular TritonExperts imports invoke_fused_moe_triton_kernel
            # into experts.triton_moe at import time, so without patching
            # that namespace the executing callsite keeps the original
            # kernel and fakequant silently never engages (input quantizers
            # collect no amax during calibration and refit KeyErrors on the
            # incoming amax buffers).
            import importlib

            # Not wrapped in try/except: swallowing an ImportError here would
            # recreate exactly the silent failure described above. `originals`
            # cannot catch it either, since it is non-empty as soon as
            # vllm_fused_moe_package contributes a name, which it always does.
            # If this module ever moves, failing loudly is the point.
            packages = [
                modelopt_vllm.vllm_fused_moe_package,
                importlib.import_module(
                    "vllm.model_executor.layers.fused_moe.experts.triton_moe"
                ),
            ]
            names = modelopt_vllm._FUSED_MOE_KERNEL_CANDIDATES
            originals: list[tuple[Any, str, Any]] = []
            for package in packages:
                for name in names:
                    kernel = getattr(package, name, None)
                    if kernel is None:
                        continue
                    originals.append((package, name, kernel))
            assert originals, "no fused-MoE kernels found to patch for fakequant"
            try:
                for package, name, kernel in originals:
                    setattr(
                        package,
                        name,
                        partial(
                            self.invoke_fused_moe_quantized,
                            original_kernel=kernel,
                        ),
                    )
                yield
            finally:
                for package, name, kernel in originals:
                    setattr(package, name, kernel)

        def forward_modular(self, *args, **kwargs):
            with self._fakequant_kernels():
                return super().forward_modular(*args, **kwargs)

        def forward_monolithic(self, *args, **kwargs):
            with self._fakequant_kernels():
                return super().forward_monolithic(*args, **kwargs)

    QuantModuleRegistry.register({RoutedExperts: "vllm_RoutedExperts"})(
        _QuantVllmRoutedExperts
    )


def _fakequant_run_prolog_worker(self) -> None:
    def calibrate_loop(model: Any = None) -> None:
        self.model_runner._dummy_run(1, skip_eplb=True, remove_lora=False)

    _drop_nonclass_quant_registry_keys()
    _register_routed_experts_quant_module()

    quant_cfg = resolve_quant_cfg(os.environ["VLLM_QUANT_CFG"])
    print(f"quant_cfg: {quant_cfg}")

    model = self.model_runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()

    with disable_compilation(model), _tolerate_dummy_weight_nan_amax():
        print("quantizing model...")
        mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        mtq.print_quant_summary(model)

    # we are using dummy data for calibration, we expect the amax is loaded from the actor
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and module.is_enabled:
            module._is_active = True
            # For easier merging of amax across q,k,v or experts
            if module.amax is not None:
                module.amax.fill_(-1.0)
            # we disable weight quantizers for CUDA graph capture.
            if name.endswith("weight_quantizer"):
                module.disable()


def _make_nvfp4_moe_input_scales_writable(model: torch.nn.Module) -> None:
    """Backport writable shared NVFP4 MoE scales for vLLM 0.25.1."""
    scale_names = ("w13_input_scale", "w2_input_scale")
    for module in model.modules():
        for name in scale_names:
            scale = getattr(module, name, None)
            if not isinstance(scale, torch.nn.Parameter) or 0 not in scale.stride():
                continue

            scale.data = scale.detach().clone()


def _process_moe_reload(
    layer: torch.nn.Module,
    *,
    quant_method: Any,
    process_weights: Callable[[torch.nn.Module], None],
) -> None:
    # Import only in the real-quant worker; fake quant does not use native reload.
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    reloading = get_layerwise_info(layer).kernel_tensors is not None
    previous_config = quant_method.moe_quant_config
    previous_kernel = quant_method.moe_kernel
    process_weights(layer)
    config = quant_method.moe_quant_config
    for name in ("a1_gscale", "a2_gscale"):
        current = getattr(config, name)
        if reloading:
            previous = getattr(previous_config, name)
            if current is None or previous is None:
                if current is not previous:
                    raise ValueError("MoE scale metadata changed during refit")
            elif (
                current.shape != previous.shape
                or current.dtype != previous.dtype
                or current.device != previous.device
            ):
                raise ValueError("MoE scale metadata changed during refit")
    for name in ("a1_gscale", "a2_gscale"):
        scale = getattr(config, name)
        if scale is not None:
            layer.register_buffer(f"_nrl_{name}", scale)
    if reloading:
        # Native copy-back updates the registered scales, including our two
        # reciprocals. Keep the kernel that references that original storage;
        # a newly built kernel would retain temporary postprocessing tensors.
        quant_method.moe_quant_config = previous_config
        quant_method.moe_kernel = previous_kernel


def _preserve_moe_reload_scales(model: torch.nn.Module) -> None:
    """Keep CUDA-graph scale storage stable across native ModelOpt MoE reloads."""
    # These backend types are optional outside vLLM's real-quant worker.
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptFp8MoEMethod,
        ModelOptNvFp4FusedMoE,
    )

    for layer in model.modules():
        method = getattr(layer, "quant_method", None)
        if isinstance(method, ModelOptNvFp4FusedMoE):
            if method.nvfp4_backend not in (
                NvFp4MoeBackend.VLLM_CUTLASS,
                NvFp4MoeBackend.FLASHINFER_TRTLLM,
                NvFp4MoeBackend.FLASHINFER_CUTLASS,
                NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED,
                NvFp4MoeBackend.MARLIN,
            ):
                raise ValueError(
                    "Graph-safe ModelOpt real-quant refit does not support NVFP4 MoE "
                    f"backend {method.nvfp4_backend.name}. Choose a supported MoE backend."
                )
        elif isinstance(method, ModelOptFp8MoEMethod):
            # Static FP8 on these backends also creates unregistered reciprocals.
            # Other FP8 backends keep their native storage lifecycle unchanged.
            if method.fp8_backend not in (
                Fp8MoeBackend.FLASHINFER_CUTLASS,
                Fp8MoeBackend.HPC,
            ):
                continue
        else:
            continue
        # Install after cold loading: checkpoint metadata excludes these derived
        # buffers. Persistent buffers participate in native reload copy-back;
        # nonpersistent buffers absent from the checkpoint would be skipped.
        for name in ("a1_gscale", "a2_gscale"):
            scale = getattr(method.moe_quant_config, name)
            if scale is not None:
                layer.register_buffer(f"_nrl_{name}", scale)
        method.process_weights_after_loading = partial(
            _process_moe_reload,
            quant_method=method,
            process_weights=method.process_weights_after_loading,
        )


class RealQuantWorker(NixlVllmWorker):
    """vLLM worker with graph-safe native ModelOpt MoE reloads."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Worker processes have their own registry. Import only on the native
        # real-quant route; the fake-quant plugin must remain unchanged.
        from nemo_rl.modelopt.models.generation.vllm_quant_kv import (
            register_modelopt_kv_configs,
        )

        register_modelopt_kv_configs()
        super().__init__(*args, **kwargs)
        self._nrl_kv_cache_awake = True
        self._nrl_kv_graphs_need_recapture = False

    def _on_kv_scales_changed(self) -> None:
        self._nrl_kv_graphs_need_recapture = True
        if self._nrl_kv_cache_awake:
            self._recapture_kv_graphs()

    def _recapture_kv_graphs(self) -> None:
        from nemo_rl.modelopt.models.generation.vllm_quant_kv import recapture_kv_graphs

        try:
            recapture_kv_graphs(self.model_runner)
            self._nrl_kv_graphs_need_recapture = False
        except Exception as error:
            self._nrl_layerwise_reload_failure = error
            raise

    def sleep(self, level: int = 1) -> None:
        super().sleep(level=level)
        self._nrl_kv_cache_awake = False

    def wake_up(self, tags: list[str] | None = None) -> None:
        super().wake_up(tags=tags)
        if tags is None or "kv_cache" in tags:
            self._nrl_kv_cache_awake = True
            if self._nrl_kv_graphs_need_recapture:
                self._recapture_kv_graphs()

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        model = self.model_runner.model
        if hasattr(model, "unwrap"):
            model = model.unwrap()
        _make_nvfp4_moe_input_scales_writable(model)
        _preserve_moe_reload_scales(model)


class FakeQuantWorker(NixlVllmWorker):
    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        model = self.model_runner.model
        if hasattr(model, "unwrap"):
            model = model.unwrap()
        with disable_compilation(model):
            return super().determine_available_memory()

    def compile_or_warm_up_model(self) -> float:
        print(
            "os.environ.get('VLLM_QUANT_CFG'): ", os.environ.get("VLLM_QUANT_CFG", None)
        )
        if os.environ.get("VLLM_QUANT_CFG", None) is not None:
            _fakequant_run_prolog_worker(self)
        return super().compile_or_warm_up_model()
