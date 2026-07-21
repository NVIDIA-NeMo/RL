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
from typing import Any, Optional

import torch
import vllm  # noqa: F401
import zmq
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

from nemo_rl.modelopt.utils import (
    MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS,
    matches_quant_ignore_pattern,
)
from nemo_rl.models.generation.vllm.checkpoint_engine import VllmCheckpointEngineMixin
from nemo_rl.models.generation.vllm.vllm_backend import (
    IPCWeightManifestError,
    VllmInternalWorkerExtension,
    WeightUpdateFinalizer,
    WeightUpdateTransport,
)

_FUSED_MODELOPT_MOE_SUFFIXES = {
    ".experts.w13_weight": "w13_weight",
    ".experts.w13_weight_scale": "w13_weight_scale",
    ".experts.w13_weight_scale_2": "w13_weight_scale_2",
    ".experts.w2_weight": "down_proj.weight",
    ".experts.w2_weight_scale": "down_proj.weight_scale",
    ".experts.w2_weight_scale_2": "down_proj.weight_scale_2",
    ".experts.w13_input_scale": "w13_input_scale",
    ".experts.w2_input_scale": "w2_input_scale",
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


def _w13_num_shards_from_state_dict_info(
    state_dict_info: dict[str, Any],
    *,
    require_input_scales: bool = False,
) -> dict[str, int]:
    """Validate complete fused-MoE families and resolve their W13 layout."""
    num_shards_by_prefix: dict[str, int] = {}
    input_shards_by_prefix: dict[str, int] = {}
    targets_by_prefix: dict[str, set[str]] = {}
    for name, (shape, _dtype) in state_dict_info.items():
        matched = _match_fused_modelopt_moe_weight(name)
        if matched is None:
            continue
        suffix, target = matched
        prefix = name[: -len(suffix)]
        if target.startswith("down_proj."):
            target = "w2_" + target.removeprefix("down_proj.")
        targets_by_prefix.setdefault(prefix, set()).add(target)
        if target == "w13_input_scale":
            if len(shape) == 1:
                input_shards = 1
            elif len(shape) == 2 and shape[1] in {1, 2}:
                input_shards = shape[1]
            else:
                raise ValueError(
                    f"Expected one or two W13 input scales per expert for {name}, "
                    f"got {tuple(shape)}"
                )
            input_shards_by_prefix[prefix] = input_shards
        if target != "w13_weight_scale_2":
            continue
        if len(shape) == 1:
            num_shards = 1
        elif len(shape) == 2 and shape[1] in {1, 2}:
            num_shards = shape[1]
        else:
            raise ValueError(
                f"Expected one or two W13 global scales per expert for {name}, "
                f"got {tuple(shape)}"
            )
        num_shards_by_prefix[prefix] = num_shards

    required_targets = {
        "w13_weight",
        "w13_weight_scale",
        "w13_weight_scale_2",
        "w2_weight",
        "w2_weight_scale",
        "w2_weight_scale_2",
    }
    if require_input_scales:
        required_targets.update({"w13_input_scale", "w2_input_scale"})
    for prefix, targets in targets_by_prefix.items():
        missing = required_targets - targets
        if missing:
            raise RuntimeError(
                f"Incomplete ModelOpt MoE export family for {prefix}: "
                f"missing {sorted(missing)}"
            )
    if set(num_shards_by_prefix) != set(targets_by_prefix):
        missing = set(targets_by_prefix) - set(num_shards_by_prefix)
        raise RuntimeError(
            "ModelOpt MoE export families are missing W13 global scales: "
            f"{sorted(missing)}"
        )
    if require_input_scales:
        mismatched = {
            prefix
            for prefix, num_shards in num_shards_by_prefix.items()
            if input_shards_by_prefix.get(prefix) != num_shards
        }
        if mismatched:
            raise RuntimeError(
                "ModelOpt MoE W13 input/global scale layouts disagree for: "
                f"{sorted(mismatched)}"
            )
    return num_shards_by_prefix


def _batch_fused_modelopt_moe_weights(
    weights: list[tuple[str, torch.Tensor]],
    *,
    w13_num_shards_by_prefix: dict[str, int],
) -> list[tuple[str, torch.Tensor]]:
    """Map fused ModelOpt payloads to vLLM per-projection checkpoint names.

    Large expert weights and block scales stay batched so vLLM can
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

        if target in {"w13_weight", "w13_weight_scale"}:
            target_suffix = "weight" if target == "w13_weight" else "weight_scale"
            if w13_num_shards_by_prefix.get(prefix) == 1:
                batched.append(
                    (
                        f"{prefix}.experts.0.up_proj.{target_suffix}",
                        tensor,
                    )
                )
                continue
            if tensor.ndim < 2 or tensor.shape[1] % 2 != 0:
                raise ValueError(
                    f"Expected fused gate/up tensor with an even projection "
                    f"dimension for {name}, got {tuple(tensor.shape)}"
                )
            gate, up = tensor.chunk(2, dim=1)
            batched.extend(
                (
                    f"{prefix}.experts.0.{projection}.{target_suffix}",
                    shard,
                )
                for projection, shard in (
                    ("gate_proj", gate),
                    ("up_proj", up),
                )
            )
            continue

        if target == "w13_input_scale":
            if tensor.ndim == 1:
                tensor = tensor[:, None]
            if tensor.ndim != 2 or tensor.shape[1] not in {1, 2}:
                raise ValueError(
                    f"Expected one or two W13 input scales per expert for {name}, "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.shape[1] == 1:
                batched.extend(
                    (
                        f"{prefix}.experts.{expert_id}.up_proj.input_scale",
                        expert_scale[0],
                    )
                    for expert_id, expert_scale in enumerate(tensor.unbind(0))
                )
                continue
            for expert_id, expert_scale in enumerate(tensor.unbind(0)):
                batched.append(
                    (
                        f"{prefix}.experts.{expert_id}.gate_proj.input_scale",
                        expert_scale[0],
                    )
                )
                batched.append(
                    (
                        f"{prefix}.experts.{expert_id}.up_proj.input_scale",
                        expert_scale[1],
                    )
                )
            continue

        if target == "w2_input_scale":
            if tensor.ndim == 2 and tensor.shape[1] == 1:
                tensor = tensor[:, 0]
            if tensor.ndim != 1:
                raise ValueError(
                    f"Expected one down-projection input scale per expert for "
                    f"{name}, got {tuple(tensor.shape)}"
                )
            batched.extend(
                (f"{prefix}.experts.{expert_id}.down_proj.input_scale", scale)
                for expert_id, scale in enumerate(tensor.unbind(0))
            )
            continue

        if target == "w13_weight_scale_2":
            if tensor.ndim == 1:
                tensor = tensor[:, None]
            if tensor.ndim != 2 or tensor.shape[1] not in {1, 2}:
                raise ValueError(
                    f"Expected one or two W13 global scales per expert for {name}, "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.shape[1] == 1:
                batched.extend(
                    (
                        f"{prefix}.experts.{expert_id}.up_proj.weight_scale_2",
                        expert_scale[0],
                    )
                    for expert_id, expert_scale in enumerate(tensor.unbind(0))
                )
                continue
            for expert_id, expert_scale in enumerate(tensor.unbind(0)):
                batched.append(
                    (
                        f"{prefix}.experts.{expert_id}.gate_proj.weight_scale_2",
                        expert_scale[0],
                    )
                )
                batched.append(
                    (
                        f"{prefix}.experts.{expert_id}.up_proj.weight_scale_2",
                        expert_scale[1],
                    )
                )
            continue

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


def _detach_pending_layerwise_weights(
    reload_roots: tuple[torch.nn.Module, ...],
    source_storage_ptrs: set[int],
) -> None:
    """Own deferred weights before a transport buffer may be reused.

    Completed layers have already released their buffered arguments, so this
    clones only tensors from a layer split across transport batches. Only the
    cached layerwise-reload subgraphs are inspected.
    """
    if not source_storage_ptrs:
        return
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    for reload_root in reload_roots:
        for module in reload_root.modules():
            info = get_layerwise_info(module)
            for _, arguments in info.loaded_weights:
                loaded_weight = arguments.arguments.get("loaded_weight")
                if not isinstance(loaded_weight, torch.Tensor):
                    continue
                if loaded_weight.untyped_storage().data_ptr() in source_storage_ptrs:
                    arguments.arguments["loaded_weight"] = loaded_weight.clone()


def _iter_modelopt_quant_modules(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    """Return modules whose runtime layout is owned by vLLM ModelOpt methods."""
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE,
        ModelOptNvFp4LinearMethod,
    )

    method_types = (ModelOptNvFp4FusedMoE, ModelOptNvFp4LinearMethod)
    return [
        (module_name, module)
        for module_name, module in model.named_modules()
        if isinstance(getattr(module, "quant_method", None), method_types)
    ]


def _modelopt_layerwise_reload_roots(
    model: torch.nn.Module,
    *,
    include_fp8_kv_cache: bool,
) -> list[torch.nn.Module]:
    """Select disjoint roots that require vLLM's native reload lifecycle.

    Ordinary parameters are already updated in place by vLLM's checkpoint
    loaders.  Restricting layerwise reconstruction to ModelOpt runtime layouts
    and attention scale owners avoids materializing unrelated non-persistent
    buffers.  In vLLM 0.20, whole-model reconstruction can otherwise break a
    derived buffer that aliases a child parameter (for example Nemotron-H's
    ``conv_weights`` view of ``conv1d.weight``).
    """
    from vllm.model_executor.layers.attention import Attention, MLAAttention
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

    modelopt_modules = {module for _, module in _iter_modelopt_quant_modules(model)}
    attention_types = (Attention, MLAAttention)
    quant_roots: list[torch.nn.Module] = []
    attention_roots: list[torch.nn.Module] = []
    visited: set[torch.nn.Module] = set()

    def collect(module: torch.nn.Module) -> None:
        if module in visited:
            return
        visited.add(module)
        if (
            include_fp8_kv_cache
            and isinstance(module, attention_types)
            and isinstance(getattr(module, "quant_method", None), BaseKVCacheMethod)
            and "fp8" in str(getattr(module, "kv_cache_dtype", "auto")).lower()
        ):
            attention_roots.append(module)
            return
        if module in modelopt_modules:
            quant_roots.append(module)
            return
        for child in module.children():
            collect(child)

    collect(model)
    # Match vLLM's ordering contract: process quantized modules before the
    # attention owners that finalize KV-cache scales.
    return quant_roots + attention_roots


def _require_complete_modelopt_layerwise_reload(model: torch.nn.Module) -> None:
    """Reject ModelOpt layers that vLLM would otherwise finalize partially."""
    candidates = _iter_modelopt_quant_modules(model)

    if not candidates:
        return

    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    incomplete = []
    for module_name, module in candidates:
        info = get_layerwise_info(module)
        if info.load_numel_total is None:
            # A completed layer is processed and reset immediately by vLLM.
            continue
        if info.load_numel == info.load_numel_total:
            continue
        buffered = sorted({name for name, _ in info.loaded_weights})
        incomplete.append(
            f"{module_name or '<root>'}: {info.load_numel}/"
            f"{info.load_numel_total} elements, buffered={buffered}"
        )

    if incomplete:
        details = "; ".join(incomplete[:8])
        suffix = "; ..." if len(incomplete) > 8 else ""
        raise RuntimeError(
            "ModelOpt layerwise reload is incomplete for "
            f"{len(incomplete)} layer(s): {details}{suffix}"
        )


if os.environ.get("VLLM_MODELOPT_REAL_QUANT", "0") == "1":
    from nemo_rl.modelopt.models.generation.vllm_modelopt import (
        register_nemo_modelopt_nvfp4,
    )

    register_nemo_modelopt_nvfp4()


class VllmQuantInternalWorkerExtension(VllmInternalWorkerExtension):
    _nrl_w13_num_shards_by_prefix: dict[str, int]
    _nrl_modelopt_reload_roots: tuple[torch.nn.Module, ...] | None = None

    def maybe_init_zmq(self) -> None:
        """Use a longer timeout only for ModelOpt real-quant refits."""
        super().maybe_init_zmq()
        if self._is_real_quant_model():
            self.zmq_socket.setsockopt(zmq.SNDTIMEO, MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS)
            self.zmq_socket.setsockopt(zmq.RCVTIMEO, MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS)

    def _is_real_quant_model(self) -> bool:
        return os.environ.get("VLLM_MODELOPT_REAL_QUANT", "0") == "1"

    def _get_modelopt_reload_roots(self) -> tuple[torch.nn.Module, ...]:
        """Return the invariant ModelOpt layerwise-reload subgraphs."""
        if self._nrl_modelopt_reload_roots is None:
            self._nrl_modelopt_reload_roots = tuple(
                _modelopt_layerwise_reload_roots(
                    self.model_runner.model,
                    include_fp8_kv_cache=self._uses_fp8_kv_cache(),
                )
            )
        return self._nrl_modelopt_reload_roots

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
        reload_roots = self._get_modelopt_reload_roots()

        def finalize() -> None:
            try:
                with torch.device(self.device):
                    _require_complete_modelopt_layerwise_reload(model)
                    for reload_root in reload_roots:
                        finalize_layerwise_reload(reload_root, self.model_config)
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
                    for reload_root in reload_roots:
                        initialize_layerwise_reload(reload_root)
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

    def prepare_refit_info(
        self, state_dict_info: dict[str, Any]
    ) -> Optional[list[str]]:
        if not self._is_real_quant_model():
            return super().prepare_refit_info(state_dict_info)

        # Real quantization owns a separate refit handshake and must not import
        # the legacy FP8 quantization path.
        self.state_dict_info = state_dict_info
        self._get_modelopt_reload_roots()
        quant_config = (
            self.model_runner.vllm_config.model_config.hf_config.quantization_config
        )
        self._nrl_w13_num_shards_by_prefix = _w13_num_shards_from_state_dict_info(
            state_dict_info,
            require_input_scales=(
                str(quant_config.get("quant_algo", "")).upper() == "NVFP4"
            ),
        )
        if (
            self._nrl_w13_num_shards_by_prefix
            and self.model_runner.vllm_config.parallel_config.enable_expert_parallel
        ):
            raise RuntimeError(
                "Fused ModelOpt MoE refits require all experts local; "
                "vLLM expert parallelism is unsupported"
            )
        return None

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
            weights = list(weights)
            source_storage_ptrs = {
                tensor.untyped_storage().data_ptr() for _, tensor in weights
            }
            quant_config = (
                self.model_runner.vllm_config.model_config.hf_config.quantization_config
            )
            ignore_patterns = quant_config.get("ignore", []) or []
            filtered = []
            for name, weight in weights:
                suffix = name.rsplit(".", 1)[-1]
                ignored = matches_quant_ignore_pattern(name, ignore_patterns)
                if ignored and suffix in {
                    "weight_scale",
                    "weight_scale_2",
                    "input_scale",
                }:
                    continue

                filtered.append((name, weight))
            if any(
                _match_fused_modelopt_moe_weight(name) is not None
                for name, _ in filtered
            ):
                weights = _batch_fused_modelopt_moe_weights(
                    filtered,
                    w13_num_shards_by_prefix=self._nrl_w13_num_shards_by_prefix,
                )
            else:
                weights = filtered
            if not weights:
                return None
            try:
                with torch.device(self.device):
                    return super()._load_weights(weights)
            finally:
                with torch.device(self.device):
                    _detach_pending_layerwise_weights(
                        self._get_modelopt_reload_roots(),
                        source_storage_ptrs,
                    )

        with ExitStack() as contexts:
            for _, child in self.model_runner.model.named_children():
                contexts.enter_context(
                    self._patch_named_parameters_to_include_buffers(child)
                )
            return super()._load_weights(weights)

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


class VllmQuantInternalWorkerExtensionWithCheckpointEngine(
    VllmCheckpointEngineMixin, VllmQuantInternalWorkerExtension
):
    """ModelOpt worker extension with checkpoint-engine refit support."""
