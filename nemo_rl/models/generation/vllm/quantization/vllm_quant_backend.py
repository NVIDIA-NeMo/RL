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
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
        "please run at least once with the environment variable NRL_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
    )


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
            # 1. Yield actual parameters
            yield from original_named_parameters(*args, **kwargs)
            # 2. Yield buffers as if they were parameters
            for name, buf in self.named_buffers(*args, **kwargs):
                if "_quantizer" not in name:
                    continue
                if not hasattr(buf, "weight_loader"):
                    if "input_quantizer" in name:
                        buf.weight_loader = input_amax_loader
                    elif "weight_quantizer" in name:
                        buf.weight_loader = weight_amax_loader
                    # print("buf", name, buf.shape)
                    buffers_with_loader.append(buf)
                yield name, buf

        # Bind the new method to the instance
        model.named_parameters = types.MethodType(new_named_parameters, model)
        try:
            # print("calling patch named parameters to include buffers")
            yield
        finally:
            # Restore original method
            model.named_parameters = original_named_parameters
            # Clean up weight_loader from buffers
            for buf in buffers_with_loader:
                if hasattr(buf, "weight_loader"):
                    del buf.weight_loader

    @contextmanager
    def _fold_weight(self, model: nn.Module):
        print("folding weight context")
        for _, module in model.named_children():
            if (
                isinstance(module, TensorQuantizer)
                and hasattr(module, "_is_active")
                and module._is_active
            ):
                module.enable()
        try:
            yield
        finally:
            mtq.fold_weight(model, keep_attrs=True)

    def _load_weights(self, weights):
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
