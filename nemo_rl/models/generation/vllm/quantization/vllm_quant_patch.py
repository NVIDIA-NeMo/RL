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
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from vllm.v1.worker.gpu_worker import Worker as BaseWorker

from nemo_rl.models.modelopt.quant_config import CUSTOM_CONFIG


@contextmanager
def disable_compilation(model):
    do_not_compile = True
    if hasattr(model, "model"):
        do_not_compile = model.model.do_not_compile
        model.model.do_not_compile = True
    elif hasattr(model, "language_model"):
        do_not_compile = model.language_model.model.do_not_compile
        model.language_model.model.do_not_compile = True
    else:
        raise ValueError("Model does not have a model or language_model attribute")

    try:
        yield
    finally:
        if hasattr(model, "model"):
            model.model.do_not_compile = do_not_compile
        elif hasattr(model, "language_model"):
            model.language_model.model.do_not_compile = do_not_compile


def _fakequant_run_prolog_worker(self) -> None:
    def calibrate_loop(model: Any = None) -> None:
        self.model_runner._dummy_run(1, skip_eplb=True, remove_lora=False)

    quant_cfg_name = os.environ.get("VLLM_QUANT_CFG", None)
    quant_cfg = CUSTOM_CONFIG.get(quant_cfg_name, None) or getattr(mtq, quant_cfg_name)
    assert quant_cfg is not None, f"quantization config {quant_cfg_name} not found"
    print(f"quant_cfg: {quant_cfg}")

    model = self.model_runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()

    with disable_compilation(model):
        print("quantizing model...")
        mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        mtq.print_quant_summary(model)

    # we are using dummy data for calibration, we expect the amax is loaded from the actor
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and module.is_enabled:
            module._is_active = True
            # For easiser to merge amax of q,k,v, or experts
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

    def compile_or_warm_up_model(self) -> None:
        print(
            "os.environ.get('VLLM_QUANT_CFG'): ", os.environ.get("VLLM_QUANT_CFG", None)
        )
        if os.environ.get("VLLM_QUANT_CFG", None) is not None:
            _fakequant_run_prolog_worker(self)
        super().compile_or_warm_up_model()
