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
from contextlib import contextmanager
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
    afterwards, and Megatron's real amax is loaded via
    `vllm_quant_backend.input_amax_loader` during refit (`max(-1.0,
    real)=real`). So a fully-NaN input here should produce zero amax
    rather than crash the prolog.

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

    for key in list(QuantModuleRegistry._registry):
        if not inspect.isclass(key):
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

            packages = [modelopt_vllm.vllm_fused_moe_package]
            try:
                packages.append(
                    importlib.import_module(
                        "vllm.model_executor.layers.fused_moe.experts.triton_moe"
                    )
                )
            except ImportError:
                pass
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
