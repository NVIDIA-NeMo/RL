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
import os
import types
from contextlib import ExitStack, contextmanager

import torch
import vllm  # noqa: F401
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

from nemo_rl.modelopt.utils import (
    matches_quant_ignore_pattern,
)
from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension
from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
)

_FUSED_MODELOPT_MOE_SUFFIXES = {
    ".experts.up_proj": "up_proj.weight",
    ".experts.up_proj_scale": "up_proj.weight_scale",
    ".experts.up_proj_scale_2": "up_proj.weight_scale_2",
    ".experts.w2_weight": "down_proj.weight",
    ".experts.w2_weight_scale": "down_proj.weight_scale",
    ".experts.w2_weight_scale_2": "down_proj.weight_scale_2",
}


def _match_fused_modelopt_moe_weight(name: str) -> tuple[str, str] | None:
    return next(
        (
            (suffix, target)
            for suffix, target in _FUSED_MODELOPT_MOE_SUFFIXES.items()
            if name.endswith(suffix)
        ),
        None,
    )


def _is_fused_modelopt_moe_weight(name: str) -> bool:
    return _match_fused_modelopt_moe_weight(name) is not None


def _batch_fused_modelopt_moe_weights(
    weights: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Map non-gated fused ModelOpt payloads to vLLM checkpoint names.

    Super's large expert weights and block scales stay batched so vLLM can
    tensor-parallel-shard the full ``[E, ...]`` tensor at once.  Its scalar
    loader still requires an expert id, so only the tiny per-expert global
    scales are exposed as scalar views.
    """
    batched: list[tuple[str, torch.Tensor]] = []
    for name, tensor in weights:
        matched = _match_fused_modelopt_moe_weight(name)
        if matched is None:
            batched.append((name, tensor))
            continue

        suffix, target = matched
        prefix = name[: -len(suffix)]
        if tensor.ndim == 0:
            raise ValueError(
                f"Fused ModelOpt MoE tensor must have an expert dimension: {name}"
            )

        if not target.endswith("weight_scale_2"):
            batched.append((f"{prefix}.experts.0.{target}", tensor))
            continue

        if tensor.ndim == 1:
            expert_scales = tensor
        elif tensor.ndim == 2 and tensor.shape[1] == 1:
            expert_scales = tensor[:, 0]
        else:
            raise ValueError(
                f"Expected one global scale per expert for {name}, got "
                f"shape {tuple(tensor.shape)}"
            )

        batched.extend(
            (f"{prefix}.experts.{expert_id}.{target}", expert_scale)
            for expert_id, expert_scale in enumerate(expert_scales.unbind(0))
        )

    return batched


def _supports_batched_modelopt_moe_load(model: torch.nn.Module) -> bool:
    """Return whether every ModelOpt MoE layer owns the complete expert set."""
    found_modelopt_moe = False
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method.__class__.__name__ != "ModelOptNvFp4FusedMoE":
            continue
        found_modelopt_moe = True
        quant_config = getattr(quant_method, "quant_config", None)
        backend = getattr(quant_method, "nvfp4_backend", None)
        if (
            not getattr(quant_config, "_nrl_weight_only_w4a16", False)
            or getattr(
                backend,
                "value",
                backend,
            )
            != "MARLIN"
        ):
            return False
        if getattr(module, "_expert_map", None) is not None:
            return False
        local_num_experts = getattr(module, "local_num_experts", None)
        global_num_experts = getattr(module, "global_num_experts", None)
        if (
            not isinstance(local_num_experts, int)
            or local_num_experts != global_num_experts
        ):
            return False
    return found_modelopt_moe


if os.environ.get("VLLM_MODELOPT_REAL_QUANT", "0") == "1":
    from nemo_rl.modelopt.models.generation.vllm_modelopt_patch import (
        apply_modelopt_nvfp4_patches,
    )

    apply_modelopt_nvfp4_patches()


class VllmQuantInternalWorkerExtension(VllmInternalWorkerExtension):
    def _is_real_quant_model(self) -> bool:
        return os.environ.get("VLLM_MODELOPT_REAL_QUANT", "0") == "1"

    @contextmanager
    def _patch_named_parameters_to_include_buffers(self, model):
        """Temporarily patches model.named_parameters() to also yield input_quantizer buffers.

        Weights arrive pre-folded from the Megatron side, so only input_quantizer
        amax buffers need to be loaded. Weight quantizer buffers are skipped.
        """
        original_named_parameters = model.named_parameters
        # input_quantizer buffers we attached a weight_loader to and must
        # clean up on exit; pre-existing loaders (if any) are left untouched.
        patched_quantizer_buffers = []

        def input_amax_loader(param, loaded_weight, *args, **kwargs):
            param.copy_(torch.max(param, loaded_weight))

        def new_named_parameters(self, *args, **kwargs):
            yield from original_named_parameters(*args, **kwargs)
            for name, buf in self.named_buffers(*args, **kwargs):
                if "input_quantizer" not in name:
                    continue
                if not hasattr(buf, "weight_loader"):
                    buf.weight_loader = input_amax_loader
                    patched_quantizer_buffers.append(buf)
                yield name, buf

        model.named_parameters = types.MethodType(new_named_parameters, model)
        try:
            yield
        finally:
            model.named_parameters = original_named_parameters
            for buf in patched_quantizer_buffers:
                del buf.weight_loader

    def _load_weights(self, weights):
        """Load pre-folded weights and input_quantizer amax buffers.

        Weights arrive already folded from the Megatron side (weight_quantizer
        applied during export), so no fold_weight step is needed here.
        """
        if self._is_real_quant_model():
            quant_config = (
                self.model_runner.vllm_config.model_config.hf_config.quantization_config
            )
            ignore_patterns = quant_config.get("ignore", []) or []
            filtered = []
            for name, weight in weights:
                suffix = name.rsplit(".", 1)[-1]
                ignored = matches_quant_ignore_pattern(name, ignore_patterns)
                if ignored and suffix in {"weight_scale", "weight_scale_2"}:
                    continue

                filtered.append((name, weight))
            if any(_is_fused_modelopt_moe_weight(name) for name, _ in filtered):
                supports_batched_moe = getattr(
                    self,
                    "_nrl_supports_batched_modelopt_moe_load",
                    None,
                )
                if supports_batched_moe is None:
                    supports_batched_moe = _supports_batched_modelopt_moe_load(
                        self.model_runner.model
                    )
                    self._nrl_supports_batched_modelopt_moe_load = supports_batched_moe
                if not supports_batched_moe:
                    raise RuntimeError(
                        "Fused ModelOpt MoE refits require W4A16 Marlin with all "
                        "experts local; vLLM expert parallelism is unsupported"
                    )
                weights = _batch_fused_modelopt_moe_weights(filtered)
            else:
                weights = filtered
            if not weights:
                return None
            return super()._load_weights(weights)

        with ExitStack() as contexts:
            for _, child in self.model_runner.model.named_children():
                contexts.enter_context(
                    self._patch_named_parameters_to_include_buffers(child)
                )
            return super()._load_weights(weights)

    def update_weights_via_ipc_zmq(self) -> bool:
        """Receive and update weights through CUDA IPC."""
        if not self._is_real_quant_model():
            return super().update_weights_via_ipc_zmq()

        from nemo_rl.modelopt.models.generation.vllm_modelopt_patch import (
            modelopt_process_weights_after_loading,
            prepare_modelopt_for_weight_reload,
        )

        prepare_modelopt_for_weight_reload(self.model_runner.model, device=self.device)
        self.maybe_init_zmq()
        buffer = None
        weight = None
        weights = None
        while True:
            payload = self.zmq_socket.recv_pyobj()

            if payload == IPCProtocol.COMPLETE:
                modelopt_process_weights_after_loading(self.model_runner.model)
                torch.cuda.synchronize()
                self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                break

            ipc_handle, list_keys, used_bytes = payload
            buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device.index)

            weights = []
            offset = 0
            for key in list_keys:
                shape, dtype = self.state_dict_info[key]
                if isinstance(shape, list):
                    shape = torch.Size(shape)

                size_in_bytes = dtype.itemsize * shape.numel()
                weight = (
                    buffer[offset : offset + size_in_bytes]
                    .view(dtype=dtype)
                    .view(shape)
                )
                weights.append((key, weight))

                offset += calculate_aligned_size(size_in_bytes)

            assert offset == used_bytes, (
                "Offset is not equal to used bytes, usually indicate inaccurate "
                "info like keys or cached dtype in state_dict_info"
            )

            self._load_weights(weights)
            torch.cuda.synchronize()

            # Match the base vLLM receiver: the loop variable is the final IPC
            # tensor view and must be dropped together with the list and base
            # buffer before ACK permits the sender to reuse or release storage.
            del weight, weights, buffer
            weight = None
            weights = None
            buffer = None
            self.zmq_socket.send(IPCProtocol.ACK.value.encode())

        self._maybe_process_fp8_kv_cache()
        gc.collect()
        torch.cuda.empty_cache()
        return True

    def update_weights_from_collective(self) -> bool:
        """Receive and update weights through collective communication."""
        if not self._is_real_quant_model():
            return super().update_weights_from_collective()

        from nemo_rl.modelopt.models.generation.vllm_modelopt_patch import (
            modelopt_process_weights_after_loading,
            prepare_modelopt_for_weight_reload,
        )

        prepare_modelopt_for_weight_reload(self.model_runner.model, device=self.device)
        result = super().update_weights_from_collective()
        if result:
            modelopt_process_weights_after_loading(self.model_runner.model)
        return result

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
        model = self.model_runner.model
        for _, module in model.named_modules():
            if isinstance(module, TensorQuantizer):
                total += 1
                if module.is_enabled:
                    enabled += 1
                    if hasattr(module, "amax") and module.amax is not None:
                        with_amax += 1
                        if (module.amax > 0).all():
                            positive_amax += 1
        return {
            "total": total,
            "enabled": enabled,
            "with_amax": with_amax,
            "positive_amax": positive_amax,
        }
