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

import types
from contextlib import ExitStack, contextmanager

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
import vllm  # noqa: F401
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension


class VllmQuantInternalWorkerExtension(VllmInternalWorkerExtension):
    @contextmanager
    def _patch_named_parameters_to_include_buffers(self, model):
        """Temporarily patches model.named_parameters() to also yield named_buffers().

        This allows vLLM's load_weights (which typically iterates named_parameters)
        to discover and load into buffers (like amax) as if they were parameters.
        Also attaches a default weight_loader to buffers so they can be loaded.
        """
        print("patching named parameters to include buffers")
        original_named_parameters = model.named_parameters
        buffers_with_loader = []

        def input_amax_loader(param, loaded_weight, *args, **kwargs):
            param.copy_(torch.max(param, loaded_weight))

        def weight_amax_loader(param, loaded_weight, *args, **kwargs):
            if param.numel() == 1:
                param.copy_(torch.max(param, loaded_weight))
            else:
                raise ValueError("Now only tensor-wise quantization is supported.")

        def new_named_parameters(self, *args, **kwargs):
            yield from original_named_parameters(*args, **kwargs)
            for name, buf in self.named_buffers(*args, **kwargs):
                if "_quantizer" not in name:
                    continue
                if not hasattr(buf, "weight_loader"):
                    if "input_quantizer" in name:
                        buf.weight_loader = input_amax_loader
                    elif "weight_quantizer" in name:
                        buf.weight_loader = weight_amax_loader
                    # print("buf", name, buf.item())
                    buffers_with_loader.append(buf)
                yield name, buf

        # module.module.decoder.layers.6.mlp.shared_experts.linear_fc1.weight_quantizer    TensorQuantizer((2, 1) bit fake block_sizes={-1: 16, 'type': 'dynamic
        # ', 'scale_bits': (4, 3)}, amax=0.6406 calibrator=MaxCalibrator quant)
        # (MegatronQuantPolicyWorker pid=927455) module.module.decoder.layers.6.mlp.shared_experts.linear_fc1.output_quantizer    TensorQuantizer(disabled)
        # (MegatronQuantPolicyWorker pid=927455) module.module.decoder.layers.6.mlp.shared_experts.linear_fc2.input_quantizer     TensorQuantizer((2, 1) bit fake block_sizes={-1: 16, 'type': 'dynamic
        # ', 'scale_bits': (4, 3)}, amax=17.2500 calibrator=MaxCalibrator quant)
        # (MegatronQuantPolicyWorker pid=927455) module.module.decoder.layers.6.mlp.shared_experts.linear_fc2.weight_quantizer    TensorQuantizer((2, 1) bit fake block_sizes={-1: 16, 'type': 'dynamic
        # ', 'scale_bits': (4, 3)}, amax=0.9336 calibrator=MaxCalibrator quant)
        # (MegatronQuantPolicyWorker pid=927455) module.module.decoder.layers.6.mlp.shared_experts.linear_fc2.output_quantizer

        model.named_parameters = types.MethodType(new_named_parameters, model)
        try:
            # print("calling patch named parameters to include buffers")
            yield
        finally:
            model.named_parameters = original_named_parameters
            for buf in buffers_with_loader:
                if hasattr(buf, "weight_loader"):
                    del buf.weight_loader

    @contextmanager
    def _fold_weight(self, model: nn.Module):
        """Enable quantizers and fold weight after loading weights."""
        print("folding weight context")
        if hasattr(model, "unwrap"):
            model = model.unwrap()

        try:
            for _, module in model.named_modules():
                if (
                    isinstance(module, TensorQuantizer)
                    and hasattr(module, "_is_active")
                    and module._is_active
                ):
                    # print(f"enabling quantizer: {module}")
                    module.enable()
            # print(f"vllm model before fold: {model}")
            yield
        finally:
            for name, module in model.named_modules():
                if (
                    isinstance(module, TensorQuantizer)
                    and hasattr(module, "_is_active")
                    and module._is_active
                    and module.amax is not None
                    and module.amax.item() <= 0.0
                ):
                    print(
                        f"quantizer {name} is active but has amax <= 0.0, amax:{module.amax}"
                    )
            mtq.fold_weight(model, keep_attrs=True)
            # print(f"vllm model after fold: {model}")

    def _load_weights(self, weights):
        """Load weights and fold weight after loading weights."""
        with ExitStack() as contexts:
            contexts.enter_context(self._fold_weight(self.model_runner.model))
            # we need to patch the children of the root model
            # For LLM: model.model
            # For VLM: model.language_model, model.vision_model, etc.
            for _, child in self.model_runner.model.named_children():
                contexts.enter_context(
                    self._patch_named_parameters_to_include_buffers(child)
                )
            return super()._load_weights(weights)

    def export_amax(self) -> dict[str, torch.Tensor]:
        """Export amax buffers from the model for testing/debugging."""
        try:
            model = self.model_runner.model
            return {
                n: b.detach().cpu()
                for n, b in model.named_buffers()
                if n.endswith("amax")
            }
        except AttributeError:
            return {}
