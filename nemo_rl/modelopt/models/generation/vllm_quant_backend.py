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

import os
import types
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

import torch
import vllm  # noqa: F401
import zmq
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

from nemo_rl.modelopt.utils import (
    MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS,
)
from nemo_rl.models.generation.vllm.checkpoint_engine import VllmCheckpointEngineMixin
from nemo_rl.models.generation.vllm.vllm_backend import (
    IPCWeightManifestError,
    VllmInternalWorkerExtension,
    WeightUpdateFinalizer,
    WeightUpdateTransport,
)


class VllmQuantInternalWorkerExtension(VllmInternalWorkerExtension):
    _QUANT_AMAX_SUFFIXES = (
        "input_quantizer._amax",
        "k_bmm_quantizer._amax",
        "v_bmm_quantizer._amax",
    )

    def maybe_init_zmq(self) -> None:
        """Use a longer timeout only for ModelOpt real-quant refits."""
        super().maybe_init_zmq()
        if self._is_real_quant_model():
            self.zmq_socket.setsockopt(zmq.SNDTIMEO, MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS)
            self.zmq_socket.setsockopt(zmq.RCVTIMEO, MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS)

    def _is_real_quant_model(self) -> bool:
        return os.environ.get("VLLM_MODELOPT_REAL_QUANT", "0") == "1"

    @contextmanager
    def _weight_update_lifecycle(
        self, transport: WeightUpdateTransport
    ) -> Iterator[WeightUpdateFinalizer]:
        """Use vLLM's native layerwise reload lifecycle for real quantization."""
        if not self._is_real_quant_model():
            with super()._weight_update_lifecycle(transport) as finalize:
                yield finalize
            return

        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )

        model = self.model_runner.model

        def finalize() -> None:
            try:
                with set_current_vllm_config(self.model_runner.vllm_config):
                    with torch.device(self.device):
                        finalize_layerwise_reload(model, self.model_config)
                # Fence completion for both collective return and the IPC
                # COMPLETE acknowledgment. Data-batch ACKs use the hook below.
                torch.accelerator.synchronize()
            except Exception as error:
                if transport == "ipc":
                    raise RuntimeError(
                        f"ModelOpt real-quant refit post-processing failed: {error}"
                    ) from error
                raise

        try:
            # Layerwise loading may reconstruct backend CustomOps as soon as a
            # layer becomes complete. Keep vLLM's worker config available for
            # that online processing as well as deferred finalization.
            with set_current_vllm_config(self.model_runner.vllm_config):
                with torch.device(self.device):
                    initialize_layerwise_reload(model)
                yield finalize
        except IPCWeightManifestError as error:
            raise RuntimeError(
                f"ModelOpt real-quant refit rejected: {error}"
            ) from error
        except Exception as error:
            if transport == "collective":
                raise RuntimeError(
                    "ModelOpt real-quant collective refit failed"
                ) from error
            raise

    def _weight_update_errors_are_fatal(self) -> bool:
        return self._is_real_quant_model()

    def _synchronize_before_ipc_data_ack(self) -> None:
        """Fence all accelerator streams used by ModelOpt post-load methods."""
        if self._is_real_quant_model():
            torch.accelerator.synchronize()
            return
        super()._synchronize_before_ipc_data_ack()

    @contextmanager
    def _patch_named_parameters_to_include_buffers(self, model):
        """Temporarily expose activation-quantizer amax buffers as parameters.

        Weights arrive pre-folded from the Megatron side, so weight-quantizer
        buffers are skipped. Input and KV-cache amax values use the same vLLM
        weight-loading path.
        """
        original_named_parameters = model.named_parameters
        quant_amax_suffixes = self._QUANT_AMAX_SUFFIXES
        patched_quantizer_buffers = []

        def amax_loader(param, loaded_weight, *args, **kwargs):
            # Input amax may fan in; K/V amax is fixed after calibration.
            param.copy_(torch.max(param, loaded_weight))

        def new_named_parameters(self, *args, **kwargs):
            yield from original_named_parameters(*args, **kwargs)
            for name, buf in self.named_buffers(*args, **kwargs):
                if not name.endswith(quant_amax_suffixes):
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
        """Eagerly attach weight_loaders to input_quantizer amax buffers.

        vLLM >= 0.25 loads refit weights through per-module
        ``load_weights`` (e.g. ``LinearBase.load_weights``), which resolves
        targets via ``getattr`` and calls ``param.weight_loader(param,
        loaded_weight, shard_id)`` directly — it never iterates
        ``model.named_parameters()``, so the lazy attach in
        ``_patch_named_parameters_to_include_buffers`` no longer fires and
        quantizer amax buffers arrive without a loader (AttributeError:
        'Tensor' object has no attribute 'weight_loader').
        """

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
        """Load fake-quant state or canonical ModelOpt deployment tensors."""
        if self._is_real_quant_model():
            from vllm.model_executor.model_loader.reload import (
                detach_layerwise_weights_from_source,
            )

            weights = list(weights)
            try:
                with torch.device(self.device):
                    return super()._load_weights(weights)
            finally:
                with torch.device(self.device):
                    detach_layerwise_weights_from_source(
                        self.model_runner.model,
                        (weight for _, weight in weights),
                    )

        # MBridge exports K/V amax with the HF-semantic attention path, such as
        # ``self_attn.k_bmm_quantizer._amax``. ModelOpt installs these quantizers
        # on vLLM's inner Attention module, whose runtime path contains ``.attn``.
        # Insert only that ModelOpt-owned path segment here; the normal vLLM
        # loader still owns model-specific HF mapping and PP-local filtering.
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
        for n, p in model.named_parameters():
            if n == name:
                return p.detach().cpu().clone()
        raise KeyError(f"Parameter '{name}' not found in model")

    def get_quantizer_stats(self) -> dict:
        """Return summary statistics for all TensorQuantizer modules.

        Matches the interface of MegatronQuantPolicyWorker.get_quantizer_stats().
        """
        total = 0
        enabled = 0
        with_amax = 0
        positive_amax = 0
        kv_amax = {}
        model = self.model_runner.model
        for name, module in model.named_modules():
            if isinstance(module, TensorQuantizer):
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
