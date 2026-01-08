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

        # model.layers.0.self_attn.qkv_proj.weight_quantizer
        # layers.0.self_attn.qkv_proj.input_quantizer._amax

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

    # @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_via_ipc_zmq")
    # def update_weights_via_ipc_zmq(self) -> bool:
    #     """Receive and update model weights via ZMQ IPC socket.

    #     Returns:
    #         bool: True if weights were successfully updated.
    #     """
    #     buffer = None
    #     weights = None

    #     try:
    #         self.maybe_init_zmq()
    #         while True:
    #             # Blocking receive with timeout (this is the main operation)
    #             payload = self.zmq_socket.recv_pyobj()

    #             if payload == IPCProtocol.COMPLETE:
    #                 # means the update is done
    #                 from vllm.model_executor.model_loader.utils import (
    #                     process_weights_after_loading,
    #                 )

    #                 process_weights_after_loading(self.model_runner.model, self.model_config, self.device)
    #                 self.zmq_socket.send(IPCProtocol.ACK.value.encode())
    #                 break

    #             ipc_handle, list_keys, used_bytes = payload
    #             buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device.index)

    #             weights = []
    #             offset = 0
    #             for key in list_keys:
    #                 shape, dtype = self.state_dict_info[key]  # pyrefly
    #                 if isinstance(shape, list):
    #                     shape = torch.Size(shape)
    #                 size_in_bytes = dtype.itemsize * shape.numel()
    #                 weights.append(
    #                     (
    #                         key,
    #                         buffer[offset : offset + size_in_bytes].view(dtype=dtype).view(shape),
    #                     )
    #                 )
    #                 aligned_size = calculate_aligned_size(size_in_bytes)
    #                 offset += aligned_size
    #             assert (
    #                 offset == used_bytes
    #             ), "Offset is not equal to used bytes, usually indicate inaccurate info like keys or cached dtype in state_dict_info"
    #             # Load weights into the model
    #             from nemo_rl.models.generation import fp8

    #             print("update weghts: ", [key for key, _ in weights])

    #             if fp8.is_fp8_model(self.model_runner.vllm_config):
    #                 # the fp8 load_weights additionally casts bf16 weights into fp8
    #                 fp8.load_weights(weights, self.model_runner)
    #             else:
    #                 with (
    #                     self._fold_weight(self.model_runner.model),
    #                     self._patch_named_parameters_to_include_buffers(self.model_runner.model.model),
    #                 ):
    #                     self.model_runner.model.load_weights(weights=weights)

    #             torch.cuda.current_stream().synchronize()

    #             # CRITICAL: Delete views before ACK to prevent corruption.
    #             # 'weights' contains views into IPC shared memory. Even though load_weights()
    #             # copied the data, Python may not garbage collect these view objects immediately.
    #             # If sender reuses the buffer before GC runs, old views would read corrupted data.
    #             # Explicit del ensures immediate cleanup before sending ACK.
    #             del weights, buffer
    #             weights = None
    #             buffer = None
    #             self.zmq_socket.send(IPCProtocol.ACK.value.encode())

    #         # Process weights after loading for FP8 KV cache
    #         self._maybe_process_fp8_kv_cache()

    #         gc.collect()
    #         torch.cuda.empty_cache()
    #         return True
    #     except Exception as e:
    #         print(
    #             f"Error in VllmInternalWorkerExtension.update_weights_via_ipc_zmq: {e}.\n" f"{traceback.format_exc()}"
    #         )
    #         return False

    # @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_from_collective")
    # def update_weights_from_collective(self) -> bool:
    #     """Update the model weights from collective communication."""
    #     assert self.state_dict_info is not None, (
    #         "state_dict_info is not prepared. " "Please call prepare_refit_info when initializing the worker."
    #     )

    #     def _load_model_weights(weights, model_runner):
    #         """Load model weights.

    #         Args:
    #             weights: List[(name, tensor)]
    #             model_runner: vLLM ModelRunner

    #         Returns:
    #             None
    #         """
    #         from nemo_rl.models.generation import fp8

    #         if fp8.is_fp8_model(model_runner.vllm_config):
    #             # the fp8 load_weights additionally casts bf16 weights into fp8
    #             fp8.load_weights(weights, model_runner)
    #         else:
    #             with (
    #                 self._fold_weight(model_runner.model),
    #                 self._patch_named_parameters_to_include_buffers(model_runner.model),
    #             ):
    #                 model_runner.model.load_weights(weights=weights)

    #     load_model_weight_func = lambda x: _load_model_weights(x, self.model_runner)

    #     try:
    #         packed_broadcast_consumer(
    #             iterator=iter(self.state_dict_info.items()),
    #             group=self.model_update_group,
    #             src=0,
    #             post_unpack_func=load_model_weight_func,
    #         )

    #         # Process weights after loading for FP8 KV cache
    #         self._maybe_process_fp8_kv_cache()

    #     except Exception as e:
    #         print(f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}")
    #         return False

    #     return True
