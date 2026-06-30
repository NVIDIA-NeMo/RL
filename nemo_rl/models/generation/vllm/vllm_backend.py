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
import math
import re
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


def fix_gemma3_vision_weight_name(key: str) -> str:
    """Re-insert the `vision_model` segment into Gemma3 vision-tower weights.

    When performing refit, the vision-tower weight paths are flattened. This unflattens them.
    """
    return re.sub(
        r"vision_tower\.(?!vision_model\.)", "vision_tower.vision_model.", key
    )


def _read_mtp_layer_weights_from_checkpoint(
    model_path: str, mtp_layer_indices: set[int]
) -> list[tuple[str, torch.Tensor]]:
    """Read only the MTP draft layer weights from a sharded HF safetensors checkpoint.

    Uses the checkpoint's ``model.safetensors.index.json`` to open only the
    shards that contain the requested transformer layer indices, so the
    multi-terabyte base-model weights are never read from disk.

    Args:
        model_path: Path to the HF checkpoint directory.
        mtp_layer_indices: Transformer layer indices belonging to the MTP module(s).

    Returns:
        A list of ``(weight_name, tensor)`` pairs for the requested layers, with
        tensors on CPU.
    """
    import json
    import os

    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    layer_re = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    shard_to_names: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        match = layer_re.search(name)
        if match is not None and int(match.group(1)) in mtp_layer_indices:
            shard_to_names.setdefault(shard, []).append(name)

    weights: list[tuple[str, torch.Tensor]] = []
    for shard, names in shard_to_names.items():
        with safe_open(
            os.path.join(model_path, shard), framework="pt", device="cpu"
        ) as reader:
            for name in names:
                weights.append((name, reader.get_tensor(name)))
    return weights


class VllmInternalWorkerExtension:
    _pending_kv_cache_scales: dict[str, float]
    _KV_SCALE_RE = re.compile(
        r"^(?P<prefix>.*\.(?:self_attn|self_attention))(?:(?:\.attn)?\.(?P<kind>q)_scale|\.(?P<kv_kind>[kv])_scale)$"
    )
    _Q_SCALE_UNSUPPORTED_BACKENDS = {"TRITON_ATTN", "ROCM_ATTN"}

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

    @classmethod
    def _parse_kv_scale_name(cls, name: str) -> tuple[str, str] | None:
        match = cls._KV_SCALE_RE.match(name)
        if match is None:
            return None
        kind = match.group("kind") or match.group("kv_kind")
        return f"{match.group('prefix')}.attn", kind

    @staticmethod
    def _without_model_prefix(name: str) -> str:
        return name.removeprefix("model.")

    def _iter_attention_modules(self):
        model = getattr(self.model_runner, "model", None)
        if model is not None and hasattr(model, "named_modules"):
            yield from model.named_modules()

        vllm_config = getattr(self.model_runner, "vllm_config", None)
        context = getattr(vllm_config, "compilation_config", None)
        static_context = getattr(context, "static_forward_context", None)
        if isinstance(static_context, dict) and static_context:
            yield from static_context.items()

    def _attention_module_map(self) -> dict[str, torch.nn.Module]:
        modules: dict[str, torch.nn.Module] = {}
        for name, module in self._iter_attention_modules():
            if all(
                hasattr(module, attr) for attr in ("_q_scale", "_k_scale", "_v_scale")
            ):
                modules[name] = module
                modules[self._without_model_prefix(name)] = module
        return modules

    @staticmethod
    def _get_attention_backend_name(module: torch.nn.Module) -> str:
        get_attn_backend = getattr(module, "get_attn_backend", None)
        if callable(get_attn_backend):
            backend = get_attn_backend()
            if hasattr(backend, "get_name"):
                return str(backend.get_name())
        backend = getattr(module, "attn_backend", None)
        if hasattr(backend, "get_name"):
            return str(backend.get_name())
        backend = getattr(module, "backend", None)
        if backend is not None:
            return str(backend)
        return module.impl.__class__.__name__ if hasattr(module, "impl") else "unknown"

    @classmethod
    def _q_scale_supported(cls, module: torch.nn.Module) -> bool:
        backend_name = cls._get_attention_backend_name(module)
        if backend_name in cls._Q_SCALE_UNSUPPORTED_BACKENDS:
            return False
        impl_module = getattr(getattr(module, "impl", None), "__module__", "")
        return not (
            impl_module.endswith(".triton_attn") or impl_module.endswith(".rocm_attn")
        )

    @staticmethod
    def _platform_adjust_scale(value: float) -> float:
        from vllm.platforms import current_platform

        return value * 2 if current_platform.is_fp8_fnuz() else value

    @staticmethod
    def _set_scale(module: torch.nn.Module, kind: str, value: float) -> None:
        attr = f"_{kind}_scale"
        tensor = getattr(module, attr)
        with torch.no_grad():
            tensor.copy_(torch.tensor(value, dtype=tensor.dtype, device=tensor.device))
        setattr(module, f"_{kind}_scale_float", float(value))

    @staticmethod
    def _invalidate_attention_scale_cache(module: torch.nn.Module) -> None:
        impl = getattr(module, "impl", None)
        if impl is not None:
            for attr in ("bmm1_scale", "bmm2_scale", "o_sf_scale"):
                if hasattr(impl, attr):
                    setattr(impl, attr, None)
        if hasattr(module, "_o_scale_float"):
            setattr(module, "_o_scale_float", None)

    def apply_kv_cache_scales(self, kv_scales: dict[str, float] | None = None) -> dict:
        scales = (
            kv_scales
            if kv_scales is not None
            else getattr(self, "_pending_kv_cache_scales", {})
        )
        modules = self._attention_module_map()
        resolved = []
        missing = []

        for transport_name, raw_value in scales.items():
            parsed = self._parse_kv_scale_name(transport_name)
            if parsed is None:
                missing.append(transport_name)
                continue
            module_name, kind = parsed
            module = modules.get(module_name) or modules.get(
                self._without_model_prefix(module_name)
            )
            if module is None:
                missing.append(transport_name)
                continue

            raw_float = float(raw_value)
            if kind == "q" and raw_float != 1.0 and not self._q_scale_supported(module):
                backend_name = self._get_attention_backend_name(module)
                raise RuntimeError(
                    f"Attention backend {backend_name} does not support non-1.0 q_scale for {transport_name}."
                )
            value = self._platform_adjust_scale(raw_float)
            resolved.append((module_name, module, kind, value))

        if missing:
            raise KeyError(
                "Failed to apply FP8 KV cache scales for missing or malformed keys: "
                f"{missing[:8]}"
            )

        applied = {"q": 0, "k": 0, "v": 0}
        applied_values = []
        backends = {}
        for module_name, module, kind, value in resolved:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Invalid FP8 KV cache scale for {module_name}.{kind}: {value}"
                )
            self._set_scale(module, kind, value)
            self._invalidate_attention_scale_cache(module)
            applied[kind] += 1
            applied_values.append(value)
            backends[module_name] = self._get_attention_backend_name(module)

        non_default = sum(
            not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
            for value in applied_values
        )
        if applied_values and non_default == 0:
            raise RuntimeError(
                "FP8 KV cache scales are all 1.0; calibrated scales were not applied."
            )
        summary = {
            "applied": applied,
            "total": sum(applied.values()),
            "min": min(applied_values) if applied_values else None,
            "max": max(applied_values) if applied_values else None,
            "non_default": non_default,
            "backends": backends,
        }
        if kv_scales is None:
            self._pending_kv_cache_scales = {}
        print(
            "FP8 KV cache scales applied: "
            f"total={summary['total']}, applied={summary['applied']}, "
            f"min={summary['min']}, max={summary['max']}, "
            f"non_default={summary['non_default']}",
            flush=True,
        )
        return summary

    def get_kv_cache_scale_snapshot(self) -> dict[str, dict[str, float | str]]:
        snapshot: dict[str, dict[str, float | str]] = {}
        for module_name, module in self._attention_module_map().items():
            if module_name.startswith("model."):
                continue
            transport_prefix = f"model.{module_name}"
            if transport_prefix.endswith(".self_attn.attn"):
                base = transport_prefix.removesuffix(".attn")
                names = {
                    "q": f"{transport_prefix}.q_scale",
                    "k": f"{base}.k_scale",
                    "v": f"{base}.v_scale",
                }
            else:
                names = {
                    "q": f"{transport_prefix}.q_scale",
                    "k": f"{transport_prefix}.k_scale",
                    "v": f"{transport_prefix}.v_scale",
                }
            backend = self._get_attention_backend_name(module)
            for kind, key in names.items():
                tensor = getattr(module, f"_{kind}_scale")
                snapshot[key] = {
                    "tensor": float(
                        tensor.detach().float().cpu().reshape(-1)[0].item()
                    ),
                    "host": float(getattr(module, f"_{kind}_scale_float")),
                    "backend": backend,
                }
        return snapshot

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
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        # Get target device for processing
        target_device = next(self.model_runner.model.parameters()).device

        # Call process_weights_after_loading to handle KV scales
        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(
                self.model_runner.model,
                self.model_runner.model_config,
                target_device,
            )

    @staticmethod
    def _split_policy_and_draft_weights(
        weights: list[tuple[str, torch.Tensor]],
    ) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
        """Split trainer-owned draft weights from policy weights.

        This path is only used for the Eagle3 online-training flow, where the
        trainer exports draft parameters under a `draft.` prefix before sending
        them to vLLM.
        This implementation is specific to the eagle model. For MTP, we can add
        similar logic to this function to split weights and send it to the drafter.
        The "draft." prefix is added here https://github.com/isomap/RL/blob/d3a5e1396d00f82fb888d9ec6800687a23bb4017/nemo_rl/models/policy/workers/megatron_policy_worker.py#L967-L997
        """
        policy_weights = []
        draft_weights = []
        for key, tensor in weights:
            if key.startswith("draft."):
                draft_weights.append((key.removeprefix("draft."), tensor))
            else:
                policy_weights.append((key, tensor))
        return policy_weights, draft_weights

    @staticmethod
    def _trim_vocab_padding(
        draft_model: torch.nn.Module,
        draft_weights: list[tuple[str, torch.Tensor]],
    ) -> list[tuple[str, torch.Tensor]]:
        """Trim padded vocab dimensions from draft weights.

        Megatron pads vocab to a multiple, but vLLM 0.20's autoloader
        strictly asserts loaded_weight.shape[0] == org_vocab_size on
        VocabParallelEmbedding layers. Each such layer may have a
        different org_vocab_size (e.g. embed_tokens uses vocab_size
        while lm_head uses draft_vocab_size), so we match each weight
        to its target module by name.
        """
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        vocab_sizes: dict[str, int] = {}
        for name, module in draft_model.named_modules():
            if isinstance(module, VocabParallelEmbedding):
                vocab_sizes[name] = module.org_vocab_size

        if not vocab_sizes:
            return draft_weights

        trimmed = []
        for key, tensor in draft_weights:
            for mod_name, org_vocab_size in vocab_sizes.items():
                leaf = mod_name.rsplit(".", 1)[-1]
                if leaf in key and tensor.shape[0] > org_vocab_size:
                    tensor = tensor[:org_vocab_size]
                    break
            trimmed.append((key, tensor))
        return trimmed

    def _load_draft_weights(
        self, draft_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        if not draft_weights:
            return

        draft_owner = getattr(self.model_runner, "drafter", None)
        draft_model = getattr(draft_owner, "model", None) if draft_owner else None

        if draft_model is None:
            print(
                "[draft] Received draft weights but vLLM drafter is unavailable; skipping draft update."
            )
            return
        draft_weights = self._trim_vocab_padding(draft_model, draft_weights)
        draft_model.load_weights(weights=draft_weights)

    def load_mtp_weights_from_disk(self, model_path: str) -> bool:
        """Load only the MTP (multi-token-prediction) draft weights from disk.

        Used when an MTP speculative-decoding policy runs with
        ``load_format="dummy"``: the main model receives real weights via refit,
        but the MTP draft layer is not covered by refit (the trainer runs with
        ``mtp_num_layers=0``), so its weights must come from the checkpoint. Only
        the MTP layer(s) are read, avoiding a full base-model load (~1.3 TB for
        DeepSeek-V3) on every inference replica.

        Args:
            model_path: Path to the HF checkpoint directory.

        Returns:
            bool: True if MTP weights were loaded.
        """
        draft_owner = getattr(self.model_runner, "drafter", None)
        draft_model = getattr(draft_owner, "model", None) if draft_owner else None
        if draft_model is None:
            print("[mtp] Drafter unavailable; cannot load MTP weights from disk.")
            return False

        predictor = draft_model.model
        mtp_layer_indices = set(
            range(
                predictor.mtp_start_layer_idx,
                predictor.mtp_start_layer_idx + predictor.num_mtp_layers,
            )
        )
        weights = _read_mtp_layer_weights_from_checkpoint(model_path, mtp_layer_indices)
        if not weights:
            raise ValueError(
                f"No MTP layer weights for layers {sorted(mtp_layer_indices)} "
                f"found in checkpoint at {model_path}. The checkpoint must "
                f"include MTP layer weights to run deepseek_mtp speculative decoding."
            )

        self._load_draft_weights(weights)

        # The MTP block contains MoE experts whose weights need post-load
        # processing (e.g. grouped-GEMM layout), matching the main-model path.
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        draft_model_config = (
            self.model_runner.vllm_config.speculative_config.draft_model_config
        )
        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(draft_model, draft_model_config, self.device)
        print(
            f"[mtp] Loaded MTP draft weights for layers "
            f"{sorted(mtp_layer_indices)} from {model_path}"
        )
        return True

    def _load_weights(self, weights):
        """Load weights with Gemma3 vision-tower weight name fix, FP8, and draft-weight support.

        Applies Gemma3 vision-tower weight name fix if needed, splits policy/draft
        weights, applies FP8 conversion if needed, and loads draft weights
        into the drafter model.
        """
        from nemo_rl.models.generation.vllm.quantization import fp8

        if (
            "Gemma3ForConditionalGeneration"
            in self.model_runner.vllm_config.model_config.architectures
        ):
            for idx, (key, weight) in enumerate(weights):
                weights[idx] = (fix_gemma3_vision_weight_name(key), weight)

        weights = [
            (name, weight)
            for name, weight in weights
            if self._parse_kv_scale_name(name) is None
        ]
        policy_weights, draft_weights = self._split_policy_and_draft_weights(weights)
        if fp8.is_fp8_model(self.model_runner.vllm_config):
            fp8.load_weights(policy_weights, self.model_runner)
        else:
            self.model_runner.model.load_weights(weights=policy_weights)

        self._load_draft_weights(draft_weights)

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
                    from vllm.config import set_current_vllm_config
                    from vllm.model_executor.model_loader.utils import (
                        process_weights_after_loading,
                    )

                    with set_current_vllm_config(self.model_runner.vllm_config):
                        process_weights_after_loading(
                            self.model_runner.model, self.model_config, self.device
                        )
                    self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                    break

                ipc_handle, list_keys, used_bytes = payload
                buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device.index)

                weight = None
                weights = []
                offset = 0
                for key in list_keys:
                    shape, dtype = self.state_dict_info[key]  # pyrefly
                    if isinstance(shape, list):
                        shape = torch.Size(shape)

                    # Get the weight from the buffer
                    size_in_bytes = dtype.itemsize * shape.numel()
                    weight = (
                        buffer[offset : offset + size_in_bytes]
                        .view(dtype=dtype)
                        .view(shape)
                    )
                    weights.append((key, weight))

                    # Move offset to the next weight
                    aligned_size = calculate_aligned_size(size_in_bytes)
                    offset += aligned_size

                assert offset == used_bytes, (
                    "Offset is not equal to used bytes, usually indicate inaccurate info like keys or cached dtype in state_dict_info"
                )

                # Load weights into the model
                self._load_weights(weights)

                torch.cuda.current_stream().synchronize()

                # CRITICAL: Delete views before ACK to prevent corruption.
                # 'weights' contains views into IPC shared memory. Even though load_weights()
                # copied the data, Python may not garbage collect these view objects immediately.
                # If sender reuses the buffer before GC runs, old views would read corrupted data.
                # Explicit del ensures immediate cleanup before sending ACK.
                del weight, weights, buffer
                weight = None
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

        load_model_weight_func = self._load_weights

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
