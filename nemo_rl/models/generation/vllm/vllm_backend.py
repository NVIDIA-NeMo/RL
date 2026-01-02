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
import gc
import traceback
from typing import Any

import torch
import zmq

from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer
from nemo_rl.utils.weights import (
    is_base_model_weight_name,
    is_lora_weight_name,
)

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
        "please run at least once with the environment variable NRL_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
    )


class VllmInternalWorkerExtension:
    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        """Initialize the collective communication."""
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        # Place vLLM ranks after all training ranks so all training workers can join
        rank = train_world_size + rank_prefix + local_rank

        pg = StatelessProcessGroup.create(
            host=ip, port=port, rank=rank, world_size=world_size
        )
        self.model_update_group = PyNcclCommunicator(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            pg, device=self.device
        )

    def report_device_id(self) -> str:
        """Retrieve the UUID of the current CUDA device."""
        from nemo_rl.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def get_zmq_address(self):
        """Get the ZMQ address for the current device."""
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self):
        """Initialize the ZMQ socket if it doesn't exist."""
        if not hasattr(self, "zmq_socket"):
            self.zmq_context = zmq.Context()  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            self.zmq_socket = self.zmq_context.socket(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
                zmq.REP
            )
            self.zmq_socket.setsockopt(
                zmq.SNDTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(
                zmq.RCVTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.connect(self.get_zmq_address())

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming.

        Args:
            state_dict_info (dict): A dictionary containing the info for refit.
                e.g. {tensor_name: (shape, dtype)}
        """
        self.state_dict_info = state_dict_info  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
        # self.vllm_state_dict_keys = self.model_runner.model.state_dict().keys()

    def _maybe_process_fp8_kv_cache(self) -> None:
        """Process weights after loading for FP8 KV cache (static scales)."""
        use_fp8_kv_cache = False
        if hasattr(self.model_runner.vllm_config, "cache_config"):
            kv_cache_dtype = getattr(
                self.model_runner.vllm_config.cache_config, "cache_dtype", None
            )
            use_fp8_kv_cache = (
                kv_cache_dtype is not None and "fp8" in str(kv_cache_dtype).lower()
            )

        if not use_fp8_kv_cache:
            return

        # FP8 KV cache: process KV scales after weight loading
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        # Get target device for processing
        target_device = next(self.model_runner.model.parameters()).device

        # Call process_weights_after_loading to handle KV scales
        process_weights_after_loading(
            self.model_runner.model,
            self.model_runner.model_config,
            target_device,
        )

    def apply_lora_patches(self) -> None:
        """Apply LoRA patches inside the vLLM worker process."""
        try:
            from nemo_rl.models.generation.lora import apply_lora_patches

            apply_lora_patches()
        except Exception as e:
            print(f"Failed to apply LoRA patches in worker extension: {e}")
            import traceback as _tb

            print(_tb.format_exc())
            raise e

    def _apply_weight_name_mapping(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        """Apply weight name mapping if LoRA is enabled."""

        def map_param_name(param_name: str) -> str:
            lora_mgr = self.model_runner.model.lora_manager
            supported_modules = lora_mgr.supported_lora_modules
            packed_modules_mapping = lora_mgr.packed_modules_mapping

            parts = param_name.split(".")
            if len(parts) < 2:
                return param_name

            base_name = ".".join(parts[:-2])  # prefix
            module_name = parts[-2]  # e.g. q_proj/k_proj/v_proj/gate_proj/up_proj/...
            field_name = parts[-1]  # weight/bias

            resolved_module_name = module_name
            for packed_name, member_names in packed_modules_mapping.items():
                if module_name in member_names:
                    resolved_module_name = packed_name
                    break

            # use resolved_module_name for checking, but return the original module_name
            if resolved_module_name in supported_modules:
                if base_name != "":
                    return f"{base_name}.{module_name}.base_layer.{field_name}"
                else:
                    return f"{module_name}.base_layer.{field_name}"
            return param_name

        new_weights = []
        for name, w in weights:
            new_name = map_param_name(name)
            new_weights.append((new_name, w))
        return new_weights

    def _apply_loaded_weights(
        self,
        weights: list[tuple[str, torch.Tensor]],
        lora_config: dict[str, Any],
        refit_base_model_weights: bool,
        refit_lora_weights: bool,
    ) -> None:
        """Apply loaded weights to model or LoRA based on flags.

        This unifies the duplicate logic used by both IPC and collective paths.
        """
        from nemo_rl.models.generation import fp8

        runner = self.model_runner

        if fp8.is_fp8_model(runner.vllm_config):
            # the fp8 load_weights additionally casts bf16 weights into fp8
            fp8.load_weights(weights, runner)
            return

        if refit_base_model_weights:
            if lora_config and "enabled" in lora_config and lora_config["enabled"]:
                weights = self._apply_weight_name_mapping(weights)
            runner.model.load_weights(weights=weights)
            return

        if refit_lora_weights:
            assert lora_config, (
                "lora_config is not provided, can not refit lora weights"
            )
            from nemo_rl.models.generation.lora import (
                LoRARequestWithCfgAndWeights,
                get_vllm_lora_metadata,
            )

            lora_cfg_dict = dict(
                {
                    "r": lora_config["dim"],
                    "lora_alpha": lora_config["alpha"],
                    "target_modules": lora_config["target_modules"],
                }
            )
            lora_metadata = get_vllm_lora_metadata()
            # Note: We don't need to remove the lora if it is already set max_loras = 1
            self.remove_lora(lora_id=lora_metadata["lora_int_id"])
            lora_request = LoRARequestWithCfgAndWeights(
                **lora_metadata,
                lora_cfg=lora_cfg_dict,
                lora_weights=dict({name: tensor for name, tensor in weights}),
            )
            try:
                self.add_lora(lora_request=lora_request)
            except Exception as e:
                print(
                    f"Error in VllmInternalWorkerExtension._apply_loaded_weights: {e}"
                )
                print(traceback.format_exc())
                raise e
            # self.add_lora(lora_request=lora_request)
            return

        raise ValueError(
            "refit_base_model_weights and refit_lora_weights cannot be both False"
        )

    @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_via_ipc_zmq")
    def update_weights_via_ipc_zmq(
        self,
        lora_config: dict[str, Any] = {},
        refit_base_model_weights: bool = False,
        refit_lora_weights: bool = True,
    ) -> bool:
        """Receive and update model weights via ZMQ IPC socket.

        Returns:
            bool: True if weights were successfully updated.
        """
        buffer = None
        weights = None

        try:
            self.maybe_init_zmq()
            while True:
                # Blocking receive with timeout (this is the main operation)
                payload = self.zmq_socket.recv_pyobj()

                if payload == IPCProtocol.COMPLETE:
                    # means the update is done
                    from vllm.model_executor.model_loader.utils import (
                        process_weights_after_loading,
                    )

                    process_weights_after_loading(
                        self.model_runner.model, self.model_config, self.device
                    )
                    self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                    break

                ipc_handle, list_keys, used_bytes = payload
                buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device.index)

                weights = []
                offset = 0
                for key in list_keys:
                    shape, dtype = self.state_dict_info[key]  # pyrefly
                    if isinstance(shape, list):
                        shape = torch.Size(shape)
                    size_in_bytes = dtype.itemsize * shape.numel()
                    weights.append(
                        (
                            key,
                            buffer[offset : offset + size_in_bytes]
                            .view(dtype=dtype)
                            .view(shape),
                        )
                    )
                    aligned_size = calculate_aligned_size(size_in_bytes)
                    offset += aligned_size
                assert offset == used_bytes, (
                    "Offset is not equal to used bytes, usually indicate inaccurate info like keys or cached dtype in state_dict_info"
                )
                # Load weights into the model or LoRA
                self._apply_loaded_weights(
                    weights=weights,
                    lora_config=lora_config,
                    refit_base_model_weights=refit_base_model_weights,
                    refit_lora_weights=refit_lora_weights,
                )

                torch.cuda.current_stream().synchronize()

                # CRITICAL: Delete views before ACK to prevent corruption.
                # 'weights' contains views into IPC shared memory. Even though load_weights()
                # copied the data, Python may not garbage collect these view objects immediately.
                # If sender reuses the buffer before GC runs, old views would read corrupted data.
                # Explicit del ensures immediate cleanup before sending ACK.
                del weights, buffer
                weights = None
                buffer = None
                self.zmq_socket.send(IPCProtocol.ACK.value.encode())

            # Process weights after loading for FP8 KV cache
            self._maybe_process_fp8_kv_cache()

            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_via_ipc_zmq: {e}.\n"
                f"{traceback.format_exc()}"
            )
            return False

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_collective"
    )
    def update_weights_from_collective(
        self,
        lora_config: dict[str, Any] = {},
        refit_base_model_weights: bool = True,
        refit_lora_weights: bool = False,
    ) -> bool:
        """Update the model weights from collective communication."""
        assert self.state_dict_info is not None, (
            "state_dict_info is not prepared. "
            "Please call prepare_refit_info when initializing the worker."
        )

        def _filtered_state_dict_iterator():
            """Iterator that yields only base model weights when skip_base_model_weights is True."""
            for name, tensor_tuple in self.state_dict_info.items():
                # Skip base model weights if skip_base_model_weights is True
                if is_base_model_weight_name(name) and not refit_base_model_weights:
                    continue
                if is_lora_weight_name(name) and not refit_lora_weights:
                    continue
                yield name, tensor_tuple

        load_model_weight_func = lambda weights: self._apply_loaded_weights(
            weights=weights,
            lora_config=lora_config,
            refit_base_model_weights=refit_base_model_weights,
            refit_lora_weights=refit_lora_weights,
        )

        try:
            packed_broadcast_consumer(
                iterator=_filtered_state_dict_iterator(),
                group=self.model_update_group,
                src=0,
                post_unpack_func=load_model_weight_func,
            )

            # Process weights after loading for FP8 KV cache
            self._maybe_process_fp8_kv_cache()

        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}"
            )
            return False

        return True

    def cleanup(self) -> None:
        """Shutdown and cleanup resources."""
        # Close ZMQ socket and context if they exist
        if hasattr(self, "zmq_socket"):
            self.zmq_socket.close()
            self.zmq_context.term()

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
