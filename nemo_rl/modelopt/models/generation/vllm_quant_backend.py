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
import re
import types
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

import torch
import vllm  # noqa: F401
import zmq
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

from nemo_rl.modelopt.calibration_artifact import load_nvfp4_calibration
from nemo_rl.modelopt.models.generation.nvfp4_refit import (
    NVFP4Calibration,
    NVFP4RefitMode,
    serialize_bf16_nvfp4_group,
)
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
from nemo_rl.weight_sync.nccl_reshard_utils import (
    HFToLocalParamMap,
    LocalParamSpec,
    RefitCtx,
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
_ROUTED_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<expert>\d+)\."
    r"(?P<projection>gate|up|down)_proj\.weight$"
)
_GROUPED_ROUTED_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+\.experts)\."
    r"(?P<projection>gate|up|down)_proj\.weight$"
)
_UNSUPPORTED_BF16_NVFP4_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "qkv_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


def _vllm_calibration_provenance(model_config: Any) -> tuple[str, str]:
    """Return the model identity used to validate a calibration artifact."""
    model_id = model_config.model
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("vLLM model config requires a non-empty model id")

    configured_revision = model_config.revision
    if not isinstance(configured_revision, str) or not configured_revision:
        raise ValueError(
            "BF16 W4A4 calibration requires an explicit model revision in "
            "the vLLM model config"
        )

    resolved_revision = getattr(model_config.hf_config, "_commit_hash", None)
    if isinstance(resolved_revision, str) and resolved_revision:
        return model_id, resolved_revision
    return model_id, configured_revision


def _nvfp4_mode(quant_config: dict[str, Any]) -> NVFP4RefitMode:
    quant_algo = str(quant_config.get("quant_algo", "")).upper()
    if quant_algo == "NVFP4":
        return "w4a4"
    if quant_algo == "W4A16_NVFP4":
        return "w4a16"
    raise ValueError(
        "BF16 NCCL refit supports only ModelOpt NVFP4 or W4A16_NVFP4, "
        f"got quant_algo={quant_algo!r}"
    )


def _classify_bf16_routed_experts(
    state_dict_info: dict[str, Any],
    *,
    ignore_patterns: list[str],
) -> frozenset[str]:
    """Validate and return the routed-expert BF16 weights to quantize."""
    routed: set[str] = set()
    unsupported: set[str] = set()

    for name, (shape, dtype) in state_dict_info.items():
        if (
            dtype != torch.bfloat16
            or len(shape) != 2
            or not name.endswith(".weight")
            or matches_quant_ignore_pattern(name, ignore_patterns)
        ):
            continue
        match = _ROUTED_EXPERT_WEIGHT_RE.fullmatch(name)
        if match is not None:
            routed.add(name)
            continue
        if name.endswith(_UNSUPPORTED_BF16_NVFP4_SUFFIXES):
            unsupported.add(name)

    if unsupported:
        raise ValueError(
            "BF16-to-NVFP4 NCCL refit currently supports routed experts only; "
            "exclude unsupported QKVO/dense projection scope with "
            f"real_quant_ignore: {sorted(unsupported)}"
        )

    projections_by_expert: dict[tuple[str, int], set[str]] = {}
    for name in routed:
        match = _ROUTED_EXPERT_WEIGHT_RE.fullmatch(name)
        assert match is not None
        key = (match.group("prefix"), int(match.group("expert")))
        projections_by_expert.setdefault(key, set()).add(match.group("projection"))
    incomplete = {
        key: sorted({"gate", "up", "down"} - projections)
        for key, projections in projections_by_expert.items()
        if projections != {"gate", "up", "down"}
    }
    if incomplete:
        raise ValueError(
            "BF16-to-NVFP4 NCCL refit requires complete routed-expert "
            f"gate/up/down families; missing projections: {incomplete}"
        )
    return frozenset(routed)


def _local_refit_shape(param_info: dict[str, Any]) -> tuple[int, ...]:
    """Return the destination-local BF16 shape described by refit metadata."""
    mesh = param_info["dst_mesh_info"]
    mesh_tensor = getattr(mesh, "mesh", None)
    placements = param_info["dst_placements"]
    if mesh_tensor is None or len(placements) != mesh_tensor.ndim:
        raise ValueError(
            f"Invalid destination mesh metadata for {param_info['name']!r}"
        )

    local_shape = list(param_info["global_shape"])
    for mesh_dim, placement in enumerate(placements):
        shard_dim = getattr(placement, "dim", None)
        if shard_dim is None:
            continue
        shard_count = int(mesh_tensor.shape[mesh_dim])
        if local_shape[shard_dim] % shard_count != 0:
            raise ValueError(
                f"BF16-to-NVFP4 destination shape for {param_info['name']!r} "
                f"is not divisible by mesh shard count {shard_count}: "
                f"{tuple(local_shape)}"
            )
        local_shape[shard_dim] //= shard_count
    return tuple(local_shape)


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

    ``w2`` weights and block scales stay batched so vLLM can
    tensor-parallel-shard the full ``[E, ...]`` tensor at once.  Its scalar
    loader still requires an expert id, so only the tiny per-expert global
    scales are exposed as scalar views.

    Gated ``w13`` payloads are the exception on vLLM >= 0.25: they are emitted
    as per-expert 2-D shards instead, because ``RoutedExperts.load_weights``'
    fused-3D branch mis-transposes packed NVFP4. See the comment at the
    emission site below.
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
            # Emit per-expert 2-D shards rather than batched 3-D tensors:
            # gated models (e.g. Qwen3-MoE) route batched tensors through
            # vLLM 0.25's RoutedExperts.load_weights fused branch, whose
            # orientation heuristic compares the last dim against the
            # unpacked hidden size and mis-transposes packed NVFP4 weights
            # (K/2 uint8) and block scales (K/16). Per-expert 2-D loads take
            # the same weight_loader path as the initial disk load.
            gate, up = tensor.chunk(2, dim=1)
            batched.extend(
                (
                    f"{prefix}.experts.{expert_id}.{projection}.{target_suffix}",
                    expert_weight,
                )
                for projection, shard in (
                    ("gate_proj", gate),
                    ("up_proj", up),
                )
                for expert_id, expert_weight in enumerate(shard.unbind(0))
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
    _nrl_bf16_nvfp4_names: frozenset[str]
    _nrl_bf16_nvfp4_mode: NVFP4RefitMode
    _nrl_bf16_nvfp4_calibration: NVFP4Calibration | None
    _nrl_bf16_nvfp4_group_members: dict[str, tuple[str, ...]]
    _nrl_bf16_nvfp4_staging: dict[str, dict[str, torch.Tensor]]

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
                    if transport == "nccl_reshard":
                        self._require_complete_bf16_nvfp4_groups()
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
            if transport in {"collective", "nccl_reshard"}:
                raise RuntimeError(
                    f"ModelOpt real-quant {transport} refit failed"
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

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        super().prepare_refit_info(state_dict_info)
        if not self._is_real_quant_model():
            return

        quant_config = (
            self.model_runner.vllm_config.model_config.hf_config.quantization_config
        )
        ignore_patterns = quant_config.get("ignore", []) or []
        self._nrl_bf16_nvfp4_names = _classify_bf16_routed_experts(
            state_dict_info,
            ignore_patterns=ignore_patterns,
        )
        self._nrl_bf16_nvfp4_calibration = None
        self._nrl_bf16_nvfp4_group_members = {}
        self._nrl_bf16_nvfp4_staging = {}

        if self._nrl_bf16_nvfp4_names:
            self._nrl_bf16_nvfp4_mode = _nvfp4_mode(quant_config)
            self._nrl_w13_num_shards_by_prefix = {}
            if self._nrl_bf16_nvfp4_mode == "w4a4":
                calibration_path = os.environ.get("VLLM_MODELOPT_CALIBRATION_PATH")
                if not calibration_path:
                    raise ValueError(
                        "BF16 W4A4 NCCL refit requires VLLM_MODELOPT_CALIBRATION_PATH"
                    )
                quant_cfg = os.environ.get("VLLM_MODELOPT_CALIBRATION_QUANT_CFG")
                if not quant_cfg:
                    raise ValueError(
                        "BF16 W4A4 NCCL refit requires "
                        "VLLM_MODELOPT_CALIBRATION_QUANT_CFG"
                    )
                model_config = self.model_runner.vllm_config.model_config
                model_id, model_revision = _vllm_calibration_provenance(model_config)
                self._nrl_bf16_nvfp4_calibration = load_nvfp4_calibration(
                    calibration_path,
                    model_id=model_id,
                    model_revision=model_revision,
                    quant_cfg=quant_cfg,
                    expected_projection_names=self._nrl_bf16_nvfp4_names,
                )
        else:
            self._nrl_w13_num_shards_by_prefix = _w13_num_shards_from_state_dict_info(
                state_dict_info,
                require_input_scales=(
                    str(quant_config.get("quant_algo", "")).upper() == "NVFP4"
                ),
            )

        self._get_modelopt_reload_roots()
        if (
            self._nrl_w13_num_shards_by_prefix or self._nrl_bf16_nvfp4_names
        ) and self.model_runner.vllm_config.parallel_config.enable_expert_parallel:
            raise RuntimeError(
                "Fused ModelOpt MoE refits require all experts local; "
                "vLLM expert parallelism is unsupported"
            )

    def build_hf_to_local_param_map(self, refit_info: dict) -> HFToLocalParamMap:
        """Replace routed-expert destinations with BF16 receive scratch specs."""
        base_map = super().build_hf_to_local_param_map(refit_info)
        if not self._is_real_quant_model() or not self._nrl_bf16_nvfp4_names:
            return base_map

        grouped_by_prefix: dict[str, dict[str, str]] = {}
        param_info_by_name: dict[str, dict[str, Any]] = {}
        covered_names: set[str] = set()
        for layer_name in refit_info["layer_names"]:
            for param_info in refit_info["per_layer_params"][layer_name]:
                name = str(param_info["name"])
                match = _GROUPED_ROUTED_EXPERT_WEIGHT_RE.fullmatch(name)
                if match is None:
                    continue
                prefix = match.group("prefix")
                original_names = {
                    f"{prefix}.{expert_id}.{match.group('projection')}_proj.weight"
                    for expert_id in range(int(param_info["global_shape"][0]))
                }
                if not original_names.intersection(self._nrl_bf16_nvfp4_names):
                    continue
                if not original_names.issubset(self._nrl_bf16_nvfp4_names):
                    missing = sorted(
                        original_names.difference(self._nrl_bf16_nvfp4_names)
                    )
                    raise ValueError(
                        f"NVFP4 routed-expert metadata for {name!r} is incomplete: "
                        f"missing {missing}"
                    )
                projection = match.group("projection")
                projections = grouped_by_prefix.setdefault(prefix, {})
                if projection in projections:
                    raise ValueError(
                        f"Duplicate NVFP4 routed-expert metadata for {name!r}"
                    )
                projections[projection] = name
                param_info_by_name[name] = param_info
                covered_names.update(original_names)

        uncovered_names = sorted(self._nrl_bf16_nvfp4_names.difference(covered_names))
        if uncovered_names:
            raise ValueError(
                "NVFP4 routed-expert weights are missing grouped NCCL metadata: "
                f"{uncovered_names}"
            )

        group_members: dict[str, tuple[str, ...]] = {}
        for prefix, projections in grouped_by_prefix.items():
            if set(projections) != {"gate", "up", "down"}:
                raise ValueError(
                    f"NVFP4 routed-expert metadata for {prefix!r} requires "
                    f"gate/up/down, got {sorted(projections)}"
                )
            group_members[f"{prefix}.w13"] = (
                projections["gate"],
                projections["up"],
            )
            group_members[f"{prefix}.w2"] = (projections["down"],)
        self._nrl_bf16_nvfp4_group_members = group_members
        self._nrl_bf16_nvfp4_staging = {}

        specs = dict(base_map.specs)
        for name, param_info in param_info_by_name.items():
            base_spec = specs.get(name)
            if base_spec is None:
                raise ValueError(
                    f"NVFP4 routed-expert weight {name!r} has no vLLM target"
                )
            match = _GROUPED_ROUTED_EXPERT_WEIGHT_RE.fullmatch(name)
            assert match is not None
            completion_key = (
                f"{match.group('prefix')}.w2"
                if match.group("projection") == "down"
                else f"{match.group('prefix')}.w13"
            )
            local_shape = _local_refit_shape(param_info)
            if len(local_shape) != 3 or local_shape[-1] % 16 != 0:
                raise ValueError(
                    f"NVFP4 routed-expert scratch for {name!r} must be [E, M, K] "
                    f"with K divisible by 16, got {local_shape}"
                )

            def pre(
                _base: torch.Tensor,
                *,
                shape: tuple[int, ...] = local_shape,
            ) -> RefitCtx:
                return RefitCtx(
                    buf=torch.empty(
                        shape,
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                )

            def post(
                ctx: RefitCtx,
                *,
                group_key: str = completion_key,
                grouped_name: str = name,
            ) -> None:
                self._stage_bf16_nvfp4_group(
                    completion_key=group_key,
                    grouped_name=grouped_name,
                    weight=ctx.buf,
                )

            specs[name] = LocalParamSpec(base=base_spec.base, pre=pre, post=post)
        return HFToLocalParamMap(specs=specs)

    def _stage_bf16_nvfp4_group(
        self,
        *,
        completion_key: str,
        grouped_name: str,
        weight: torch.Tensor,
    ) -> None:
        """Serialize and load one complete grouped routed-expert family."""
        expected_names = self._nrl_bf16_nvfp4_group_members.get(completion_key)
        if expected_names is None or grouped_name not in expected_names:
            raise RuntimeError(
                f"Unknown NVFP4 completion group {completion_key!r} "
                f"for {grouped_name!r}"
            )
        staged = self._nrl_bf16_nvfp4_staging.setdefault(completion_key, {})
        if grouped_name in staged:
            raise RuntimeError(f"Duplicate NVFP4 grouped tensor {grouped_name!r}")
        staged[grouped_name] = weight
        if set(staged) != set(expected_names):
            return

        expert_counts = {tensor.shape[0] for tensor in staged.values()}
        if len(expert_counts) != 1:
            raise ValueError(
                f"NVFP4 completion group {completion_key!r} has inconsistent "
                f"expert counts {sorted(expert_counts)}"
            )
        expert_count = expert_counts.pop()
        serialized: list[tuple[str, torch.Tensor]] = []
        for expert_id in range(expert_count):
            tensors: dict[str, torch.Tensor] = {}
            for name in expected_names:
                match = _GROUPED_ROUTED_EXPERT_WEIGHT_RE.fullmatch(name)
                assert match is not None
                projection = match.group("projection")
                expert_name = (
                    f"{match.group('prefix')}.{expert_id}.{projection}_proj.weight"
                )
                tensors[expert_name] = staged[name][expert_id]
            serialized.extend(
                serialize_bf16_nvfp4_group(
                    tensors,
                    mode=self._nrl_bf16_nvfp4_mode,
                    calibration=self._nrl_bf16_nvfp4_calibration,
                )
            )

        self._nrl_bf16_nvfp4_staging.pop(completion_key)
        self._load_weights(serialized)

    def _require_complete_bf16_nvfp4_groups(self) -> None:
        staging = getattr(self, "_nrl_bf16_nvfp4_staging", {})
        if not staging:
            return
        group_members = getattr(self, "_nrl_bf16_nvfp4_group_members", {})
        incomplete = {
            group: sorted(set(group_members.get(group, ())).difference(staged))
            for group, staged in staging.items()
        }
        raise RuntimeError(f"Incomplete NVFP4 NCCL receive groups: {incomplete}")

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
            contexts.enter_context(
                self._attach_input_quantizer_amax_loaders(self.model_runner.model)
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
