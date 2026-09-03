# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import types
from contextlib import ExitStack, contextmanager

import torch
import vllm  # noqa: F401
import zmq
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

from nemo_rl.modelopt.utils import MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS
from nemo_rl.models.generation.vllm.checkpoint_engine import VllmCheckpointEngineMixin
from nemo_rl.models.generation.vllm.vllm_backend import (
    VllmInternalWorkerExtension,
    WeightUpdateTransport,
)


class VllmQuantInternalWorkerExtension(VllmInternalWorkerExtension):
    _QUANT_AMAX_SUFFIXES = (
        "input_quantizer._amax",
        "k_bmm_quantizer._amax",
        "v_bmm_quantizer._amax",
    )

    def _supports_unquantized_flashinfer_trtllm_refit(self) -> bool:
        return False

    def maybe_init_zmq(self) -> None:
        """Use a longer timeout only for ModelOpt real-quant refits."""
        super().maybe_init_zmq()
        if self._is_real_quant_model():
            self.zmq_socket.setsockopt(zmq.SNDTIMEO, MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS)
            self.zmq_socket.setsockopt(zmq.RCVTIMEO, MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS)

    def _is_real_quant_model(self) -> bool:
        quantization = self.model_runner.vllm_config.model_config.quantization
        return bool(quantization and str(quantization).startswith("modelopt"))

    def _uses_native_layerwise_refit(self, transport: WeightUpdateTransport) -> bool:
        if self._is_real_quant_model():
            return transport in ("ipc", "collective")
        return super()._uses_native_layerwise_refit(transport)

    def _weight_update_errors_are_fatal(self) -> bool:
        return self._is_real_quant_model()

    def _synchronize_before_ipc_data_ack(self) -> None:
        if self._is_real_quant_model():
            torch.accelerator.synchronize()
            return
        super()._synchronize_before_ipc_data_ack()

    @contextmanager
    def _patch_named_parameters_to_include_buffers(self, model):
        """Temporarily expose fake-quant activation amax buffers to loaders."""
        original_named_parameters = model.named_parameters
        patched_quantizer_buffers = []

        def amax_loader(param, loaded_weight, *args, **kwargs):
            param.copy_(torch.max(param, loaded_weight))

        def new_named_parameters(self, *args, **kwargs):
            yield from original_named_parameters(*args, **kwargs)
            for name, buf in self.named_buffers(*args, **kwargs):
                if not name.endswith(
                    VllmQuantInternalWorkerExtension._QUANT_AMAX_SUFFIXES
                ):
                    continue
                if not hasattr(buf, "weight_loader"):
                    buf.weight_loader = amax_loader
                    patched_quantizer_buffers.append(buf)
                yield name, buf

        model.named_parameters = types.MethodType(new_named_parameters, model)
        try:
            yield
        finally:
            model.named_parameters = original_named_parameters
            for buf in patched_quantizer_buffers:
                del buf.weight_loader

    @contextmanager
    def _attach_input_quantizer_amax_loaders(self, model):
        """Attach loaders used by vLLM's module-local fake-quant refit path."""

        def input_amax_loader(param, loaded_weight, *args, **kwargs):
            param.copy_(torch.max(param, loaded_weight))

        attached = []
        for name, buf in model.named_buffers():
            if "input_quantizer" not in name:
                continue
            if not hasattr(buf, "weight_loader"):
                buf.weight_loader = input_amax_loader
                attached.append(buf)
        try:
            yield
        finally:
            for buf in attached:
                del buf.weight_loader

    def _load_weights(self, weights):
        if self._is_real_quant_model():

            def owned_weights():
                for name, tensor in weights:
                    yield name, tensor.detach().clone()

            with torch.device(self.device):
                self._load_full_hf_weights(list(owned_weights()))
            return

        remapped_weights = []
        for name, weight in weights:
            for suffix in self._QUANT_AMAX_SUFFIXES:
                if (
                    "_bmm_quantizer" in suffix
                    and name.endswith(suffix)
                    and not name.endswith(f".attn.{suffix}")
                ):
                    name = f"{name[: -len(suffix)]}attn.{suffix}"
                    break
            remapped_weights.append((name, weight))

        with ExitStack() as contexts:
            for _, child in self.model_runner.model.named_children():
                contexts.enter_context(
                    self._patch_named_parameters_to_include_buffers(child)
                )
            contexts.enter_context(
                self._attach_input_quantizer_amax_loaders(self.model_runner.model)
            )
            return super()._load_weights(remapped_weights)

    def get_weight_snapshot(self, name: str) -> torch.Tensor:
        """Return a CPU copy of a named parameter for before/after comparison."""
        model = self.model_runner.model
        for parameter_name, parameter in model.named_parameters():
            if parameter_name == name:
                return parameter.detach().cpu().clone()
        raise KeyError(f"Parameter '{name}' not found in model")

    def get_quantizer_stats(self) -> dict:
        """Return summary statistics for fake-quant TensorQuantizer modules."""
        total = 0
        enabled = 0
        with_amax = 0
        positive_amax = 0
        kv_amax = {}
        for name, module in self.model_runner.model.named_modules():
            if not isinstance(module, TensorQuantizer):
                continue
            total += 1
            if module.is_enabled:
                enabled += 1
                if hasattr(module, "amax") and module.amax is not None:
                    with_amax += 1
                    if (module.amax > 0).all():
                        positive_amax += 1
                    if name.endswith(("k_bmm_quantizer", "v_bmm_quantizer")):
                        kv_amax[name] = module.amax.detach().cpu().clone()
        return {
            "total": total,
            "enabled": enabled,
            "with_amax": with_amax,
            "positive_amax": positive_amax,
            "kv_amax": kv_amax,
        }


class VllmQuantInternalWorkerExtensionWithCheckpointEngine(
    VllmCheckpointEngineMixin, VllmQuantInternalWorkerExtension
):
    """ModelOpt worker extension with checkpoint-engine refit support."""
