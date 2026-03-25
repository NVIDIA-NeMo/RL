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
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        # Place vLLM ranks after all training ranks so all training workers can join
        rank = train_world_size + rank_prefix + local_rank

        self.model_update_group = StatelessProcessGroup(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            master_address=ip, port=port, rank=rank, world_size=world_size
        )
        self.model_update_group.init_nccl_communicator(device=self.device)

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

    @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_via_ipc_zmq")
    def update_weights_via_ipc_zmq(self) -> bool:
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
                # Load weights into the model
                from nemo_rl.models.generation.vllm.quantization import fp8

                if fp8.is_fp8_model(self.model_runner.vllm_config):
                    # the fp8 load_weights additionally casts bf16 weights into fp8
                    fp8.load_weights(weights, self.model_runner)
                else:
                    self.model_runner.model.load_weights(weights=weights)

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
    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        assert self.state_dict_info is not None, (
            "state_dict_info is not prepared. "
            "Please call prepare_refit_info when initializing the worker."
        )

        def _load_model_weights(weights, model_runner):
            """Load model weights.

            Args:
                weights: List[(name, tensor)]
                model_runner: vLLM ModelRunner

            Returns:
                None
            """
            from nemo_rl.models.generation.vllm.quantization import fp8

            if fp8.is_fp8_model(model_runner.vllm_config):
                # the fp8 load_weights additionally casts bf16 weights into fp8
                fp8.load_weights(weights, model_runner)
            else:
                model_runner.model.load_weights(weights=weights)

        load_model_weight_func = lambda x: _load_model_weights(x, self.model_runner)

        try:
            packed_broadcast_consumer(
                iterator=iter(self.state_dict_info.items()),
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

    def prepare_nccl_reshard_refit_info(self, refit_info: dict) -> None:
        """Store per-layer param metadata for nccl_reshard-based refit."""
        self.nccl_reshard_refit_info = (  # pyrefly: ignore[implicitly-defined-attribute]
            refit_info
        )

    def _build_hf_to_vllm_mapping(self, refit_info):
        """Build mapping from HF param names to vLLM (param, dim0_slice).

        vLLM merges certain HF params into combined tensors:
          - q_proj + k_proj + v_proj  → qkv_proj  (concat along dim 0)
          - gate_proj + up_proj       → gate_up_proj (concat along dim 0)
          - lm_head may be tied to embed_tokens

        For TP>1, vLLM shards merged params along dim 0. Each TP rank stores
        [q_shard, k_shard, v_shard] locally. We compute LOCAL slices by scaling
        global offsets proportionally: local_offset = global_offset * local_dim0 / global_dim0.

        Returns:
            dict: hf_name → (vllm_param_tensor, dim0_slice or None)
                  If dim0_slice is None, the HF param maps 1:1 to the vLLM param.
                  If dim0_slice is a slice, the HF param occupies that LOCAL slice.
        """
        vllm_params = dict(self.model_runner.model.named_parameters())
        mapping = {}

        # Collect all HF param names and their global shapes from refit_info
        hf_shapes = {}
        for layer_name in refit_info["layer_names"]:
            for p in refit_info["per_layer_params"][layer_name]:
                hf_shapes[p["name"]] = tuple(p["global_shape"])

        # Merge rules: (list of HF suffixes) → vLLM suffix, concat along dim 0
        MERGE_RULES = [
            (["q_proj.weight", "k_proj.weight", "v_proj.weight"], "qkv_proj.weight"),
            (["q_proj.bias", "k_proj.bias", "v_proj.bias"], "qkv_proj.bias"),
            (["gate_proj.weight", "up_proj.weight"], "gate_up_proj.weight"),
        ]

        for hf_name in hf_shapes:
            # 1) Direct match
            if hf_name in vllm_params:
                mapping[hf_name] = (vllm_params[hf_name], None)
                continue

            # 2) Check merge rules
            matched = False
            for hf_suffixes, vllm_suffix in MERGE_RULES:
                for i, suffix in enumerate(hf_suffixes):
                    if hf_name.endswith(suffix):
                        prefix = hf_name[: -len(suffix)]
                        vllm_name = prefix + vllm_suffix
                        if vllm_name in vllm_params:
                            vllm_param = vllm_params[vllm_name]
                            local_dim0 = vllm_param.shape[0]

                            # Collect global dim0 sizes for all components
                            global_sizes = []
                            for s in hf_suffixes:
                                full_name = prefix + s
                                global_sizes.append(
                                    hf_shapes[full_name][0]
                                    if full_name in hf_shapes
                                    else 0
                                )
                            global_dim0 = sum(global_sizes)

                            # Compute LOCAL sizes per component.
                            # Linear interpolation (global_size * local_dim0 / global_dim0) fails
                            # when vLLM replicates KV heads (num_kv_heads < tp_size), because
                            # q/k/v proportions change between global and local.
                            tp_size = torch.distributed.get_world_size()
                            naive_local_sizes = [gs // tp_size for gs in global_sizes]
                            if sum(naive_local_sizes) == local_dim0:
                                local_sizes = naive_local_sizes
                            else:
                                # KV head replication: q divides evenly, k/v are replicated
                                local_sizes = [global_sizes[0] // tp_size]
                                num_rest = len(global_sizes) - 1
                                rest = local_dim0 - local_sizes[0]
                                for _ in range(num_rest):
                                    local_sizes.append(rest // num_rest)
                            local_offset = sum(local_sizes[:i])
                            local_size = local_sizes[i]

                            mapping[hf_name] = (
                                vllm_param,
                                slice(local_offset, local_offset + local_size),
                            )
                            matched = True
                        break
                if matched:
                    break

            # 3) lm_head tied to embed_tokens
            if not matched and hf_name == "lm_head.weight":
                if "model.embed_tokens.weight" in vllm_params:
                    mapping[hf_name] = (
                        vllm_params["model.embed_tokens.weight"],
                        None,
                    )
                    matched = True

            if not matched:
                mapping[hf_name] = (None, None)

        return mapping

    def _compute_tp_local_slice(self, full_tensor, param_name, tp_rank, tp_size):
        """Compute TP-local slice from a full global tensor for merged params.

        Handles KV head replication: when num_kv_heads < tp_size, vLLM gives
        each TP rank max(1, num_kv_heads // tp_size) KV heads instead of a
        naive 1/tp_size shard of the global k/v tensor.
        """
        global_dim0 = full_tensor.shape[0]

        if any(s in param_name for s in ("k_proj", "v_proj")):
            hf_config = self.model_runner.model_config.hf_config
            num_kv_heads = hf_config.num_key_value_heads
            head_dim: int = hf_config.hidden_size // hf_config.num_attention_heads
            if hasattr(hf_config, "head_dim"):
                head_dim = hf_config.head_dim
            if num_kv_heads >= tp_size:
                num_kv_heads_per_tp: int = num_kv_heads // tp_size
                start_head = tp_rank * num_kv_heads_per_tp
            else:
                num_kv_heads_per_tp = max(1, num_kv_heads // tp_size)
                start_head = (tp_rank * num_kv_heads) // tp_size
            start = start_head * head_dim
            end = start + num_kv_heads_per_tp * head_dim
            return full_tensor[start:end]
        else:
            # Standard column-parallel shard (q_proj, gate_proj, up_proj)
            shard_size = global_dim0 // tp_size
            start = tp_rank * shard_size
            return full_tensor[start : start + shard_size]

    def nccl_reshard_refit(self) -> bool:
        """Receive weights from training workers via xferdtensor_golden.

        Writes directly into vLLM parameters using an HF→vLLM name mapping
        that handles merged params (qkv_proj, gate_up_proj). This approach
        is compatible with the future nccl_reshard path where each gen worker
        receives only its local shard.

        For merged params (qkv_proj, gate_up_proj), receives the full global
        tensor with all-Replicate placement, then locally computes the correct
        TP slice. This correctly handles KV head replication (num_kv_heads < tp).
        """
        from torch.distributed.tensor.placement_types import Replicate

        from nemo_rl.distributed.nccl_reshard_utils import (
            TensorWrapper,
            xferdtensor_golden,
        )

        hf_to_vllm = self._build_hf_to_vllm_mapping(self.nccl_reshard_refit_info)
        tp_rank = torch.distributed.get_rank()
        tp_size = torch.distributed.get_world_size()

        rank = self.model_update_group.rank
        for layer_name in self.nccl_reshard_refit_info["layer_names"]:
            params = self.nccl_reshard_refit_info["per_layer_params"][layer_name]
            if rank == 0:
                print(
                    f"[vLLM nccl_reshard_refit] layer={layer_name} num_params={len(params)}",
                    flush=True,
                )
            for param_info in params:
                name = param_info["name"]
                vllm_param, dim0_slice = hf_to_vllm.get(name, (None, None))

                if vllm_param is None and rank == 0:
                    print(
                        f"[vLLM nccl_reshard_refit] WARNING: no vLLM mapping for '{name}', "
                        f"broadcast will proceed but data discarded",
                        flush=True,
                    )
                    xferdtensor_golden(
                        src_tensor=None,
                        src_mesh=param_info["src_mesh_info"],
                        src_placement=param_info["src_placements"],
                        dst_tensor=None,
                        dst_mesh=param_info["dst_mesh_info"],
                        dst_placement=param_info["dst_placements"],
                        process_group=self.model_update_group,
                        global_shape=param_info["global_shape"],
                        dtype=param_info["dtype"],
                        param_name=name,
                    )
                elif dim0_slice is not None:
                    # Merged param (qkv_proj or gate_up_proj).
                    # Receive full tensor with all-Replicate, then locally compute
                    # the correct TP slice. This handles KV head replication.
                    global_shape = param_info["global_shape"]
                    full_buf = torch.empty(
                        global_shape, device=self.device, dtype=vllm_param.dtype
                    )
                    all_replicate = [Replicate() for _ in param_info["dst_placements"]]
                    xferdtensor_golden(
                        src_tensor=None,
                        src_mesh=param_info["src_mesh_info"],
                        src_placement=param_info["src_placements"],
                        dst_tensor=TensorWrapper(full_buf),
                        dst_mesh=param_info["dst_mesh_info"],
                        dst_placement=all_replicate,
                        process_group=self.model_update_group,
                        global_shape=param_info["global_shape"],
                        dtype=param_info["dtype"],
                        param_name=name,
                    )
                    local_data = self._compute_tp_local_slice(
                        full_buf, name, tp_rank, tp_size
                    )
                    vllm_param.data[dim0_slice].copy_(local_data)
                    del full_buf
                else:
                    # Direct 1:1 mapping — xferdtensor handles sharding
                    xferdtensor_golden(
                        src_tensor=None,
                        src_mesh=param_info["src_mesh_info"],
                        src_placement=param_info["src_placements"],
                        dst_tensor=TensorWrapper(vllm_param),
                        dst_mesh=param_info["dst_mesh_info"],
                        dst_placement=param_info["dst_placements"],
                        process_group=self.model_update_group,
                        global_shape=param_info["global_shape"],
                        dtype=param_info["dtype"],
                        param_name=name,
                    )

        self._maybe_process_fp8_kv_cache()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
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
