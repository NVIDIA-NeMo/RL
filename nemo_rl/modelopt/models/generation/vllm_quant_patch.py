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
from vllm.v1.worker.gpu_worker import Worker as BaseWorker

from nemo_rl.modelopt.utils import resolve_quant_cfg


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


def _fakequant_run_prolog_worker(self) -> None:
    def calibrate_loop(model: Any = None) -> None:
        self.model_runner._dummy_run(1, skip_eplb=True, remove_lora=False)

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


class FakeQuantWorker(BaseWorker):
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
