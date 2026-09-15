# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""vLLM-side NVFP4 per-token W4A4 rollout quantization.

Routed-expert weights are quantized here, at refit weight-load time, from the
plain BF16 stream the Megatron training worker exports — a sibling of the
fp8/mxfp8 "real quant" rollout path (``quantization/fp8.py``). Megatron may
use TE NVFP4 for policy computation, but its master/export weights and the
refit transport remain BF16.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Optional, cast

import torch
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.utils import replace_parameter

from nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken_config import (
    DEFAULT_NVFP4_IGNORE,
    NVFP4_PERTOKEN_ZMQ_TIMEOUT_MS,
    NvFp4PerTokenRolloutConfig,
    boundary_layer_indices,
    module_layer_index,
)
from nemo_rl.models.generation.vllm.quantization.utils import (
    resolve_module_from_param_name,
)
from nemo_rl.models.generation.vllm.vllm_backend import (
    VllmInternalWorkerExtension,
    _ReloadWeightPreparer,
)

logger = init_logger(__name__)

__all__ = ["DEFAULT_NVFP4_IGNORE", "NvFp4PerTokenRolloutConfig"]

NVFP4_PER_TOKEN_METHOD = "nvfp4_pertoken"

_FP4_MAX = 6.0
_FP8_E4M3_MAX = 448.0
_AMAX_DENOMINATOR = _FP4_MAX * _FP8_E4M3_MAX

_registered = False
_pertoken_marker_printed = False

ProjectionRole = Literal["w1", "w2", "w3"]


@dataclass
class PendingHalf:
    """One half of a logical W1/W3 pair awaiting its partner.

    The tensor is cloned off the refit IPC buffer, which the sender recycles
    as soon as the batch is acknowledged (see ``policy/utils.py``'s
    ping-pong double buffering).
    """

    source_prefix: str
    role: Literal["w1", "w3"]
    tensor: torch.Tensor


@dataclass(frozen=True)
class ExpertWeightSpec:
    """One checkpoint projection classified by a vLLM expert mapping."""

    checkpoint_suffix: str
    logical_key: str
    role: ProjectionRole


@dataclass(frozen=True)
class RoutedExpertTarget:
    """Immutable refit inventory for one selected RoutedExperts container."""

    module: RoutedExperts
    module_name: str
    specs: tuple[ExpertWeightSpec, ...]
    specs_by_suffix: Mapping[str, ExpertWeightSpec]
    expected_roles: frozenset[tuple[str, ProjectionRole]]


def _build_expert_target(module_name: str, module: RoutedExperts) -> RoutedExpertTarget:
    """Build and validate the checkpoint W1/W2/W3 map exposed by vLLM."""
    projection_names = {
        "w1": module.ckpt_gate_proj_name,
        "w2": module.ckpt_down_proj_name,
        "w3": module.ckpt_up_proj_name,
    }
    specs_by_suffix: dict[str, ExpertWeightSpec] = {}
    for _param_name, weight_name, _expert_id, shard_id in module.get_expert_mapping():
        if shard_id not in projection_names:
            raise RuntimeError(
                f"[nvfp4_pertoken] {module_name} returned unsupported expert "
                f"shard {shard_id!r} for {weight_name!r}."
            )
        role = cast(ProjectionRole, shard_id)
        projection_name = projection_names[role]
        parts = weight_name.removesuffix(".").split(".")
        projection_positions = [
            index for index, part in enumerate(parts) if part == projection_name
        ]
        if len(projection_positions) != 1:
            raise RuntimeError(
                f"[nvfp4_pertoken] {module_name} cannot identify projection "
                f"{projection_name!r} in vLLM mapping name {weight_name!r}."
            )
        parts[projection_positions[0]] = "<projection>"
        logical_key = ".".join(parts)
        checkpoint_suffix = f"{weight_name.removesuffix('.')}.weight"
        spec = ExpertWeightSpec(
            checkpoint_suffix=checkpoint_suffix,
            logical_key=logical_key,
            role=role,
        )
        previous = specs_by_suffix.setdefault(checkpoint_suffix, spec)
        if previous != spec:
            raise RuntimeError(
                f"[nvfp4_pertoken] ambiguous vLLM expert mapping for "
                f"{module_name}: {checkpoint_suffix!r}."
            )

    roles_by_key: dict[str, set[ProjectionRole]] = {}
    for spec in specs_by_suffix.values():
        roles_by_key.setdefault(spec.logical_key, set()).add(spec.role)
    incomplete = {
        key: sorted({"w1", "w2", "w3"} - roles)
        for key, roles in roles_by_key.items()
        if roles != {"w1", "w2", "w3"}
    }
    if not specs_by_suffix or incomplete:
        raise RuntimeError(
            f"[nvfp4_pertoken] incomplete vLLM expert mapping for {module_name}: "
            f"missing roles {incomplete or 'all W1/W2/W3 entries'}."
        )

    specs = tuple(
        sorted(
            specs_by_suffix.values(),
            key=lambda spec: (spec.logical_key, spec.role, spec.checkpoint_suffix),
        )
    )
    return RoutedExpertTarget(
        module=module,
        module_name=module_name,
        specs=specs,
        specs_by_suffix=MappingProxyType(dict(specs_by_suffix)),
        expected_roles=frozenset((spec.logical_key, spec.role) for spec in specs),
    )


class NvFp4PerTokenQuantizer:
    """Quantizes routed-expert weights to NVFP4 during vLLM-side weight refit.

    Stateful only over a single W1/W3 pair per logical expert — never a whole
    layer — so memory stays bounded regardless of expert count. W1 and W3
    share one global scale (vLLM's
    ``ModelOptNvFp4FusedMoE.process_weights_after_loading`` collapses
    ``w13_weight_scale_2`` to column 0, so independently-scaled halves would
    decode W3 with W1's scale), so quantization is deferred until both halves
    arrive. Projection names and expert identities come exclusively from each
    destination ``RoutedExperts.get_expert_mapping()``; no model-family naming
    convention is assumed.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model
        self._targets: dict[int, RoutedExpertTarget] = {}
        self._all_target_suffixes: set[str] = set()
        self._pending: dict[tuple[int, str], PendingHalf] = {}
        self._seen_roles: dict[int, set[tuple[str, ProjectionRole]]] = {}
        self._quantized_events = 0
        self._build_inventory()

    def _build_inventory(self) -> None:
        quantized_layers: dict[int, str] = {}
        excluded_layers: dict[int, str] = {}
        boundary_patterns: list[str] | None = None
        for module_name, module in self._model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(module, RoutedExperts):
                num_fused_shared_experts = int(
                    getattr(module.expert_map_manager, "num_fused_shared_experts", 0)
                )
                if num_fused_shared_experts:
                    raise RuntimeError(
                        f"[nvfp4_pertoken] {module_name} fuses "
                        f"{num_fused_shared_experts} shared experts into RoutedExperts; "
                        "shared-expert NVFP4 is not supported."
                    )
                layer_index = module_layer_index(module_name)
                if isinstance(quant_method, ModelOptNvFp4PerTokenFusedMoE):
                    target = _build_expert_target(module_name, module)
                    self._targets[id(module)] = target
                    self._all_target_suffixes.update(
                        spec.checkpoint_suffix for spec in target.specs
                    )
                    quantized_layers[layer_index] = module_name
                    if boundary_patterns is None:
                        boundary_patterns = list(
                            getattr(quant_method.quant_config, "exclude_modules", ())
                            or ()
                        )
                elif isinstance(quant_method, UnquantizedFusedMoEMethod):
                    excluded_layers[layer_index] = module_name
                else:
                    raise RuntimeError(
                        f"[nvfp4_pertoken] {module_name} has unexpected routed "
                        f"expert quant method {type(quant_method).__name__}."
                    )
            elif isinstance(module, (LinearBase, ParallelLMHead)) and not isinstance(
                quant_method, UnquantizedLinearMethod
            ):
                raise RuntimeError(
                    f"[nvfp4_pertoken] ordinary linear {module_name} was assigned "
                    f"{type(quant_method).__name__}, expected UnquantizedLinearMethod."
                )

        self._verify_boundary(boundary_patterns, quantized_layers, excluded_layers)

        print(
            f"[nvfp4_pertoken] inventory: selected {len(self._targets)} "
            "RoutedExperts containers; ordinary linears remain BF16",
            flush=True,
        )

    @staticmethod
    def _verify_boundary(
        boundary_patterns: list[str] | None,
        quantized_layers: Mapping[int, str],
        excluded_layers: Mapping[int, str],
    ) -> None:
        """Fail closed when the BF16 boundary did not land where it was derived.

        The ignore patterns are module-name globs, so a model family whose
        routed experts are not named ``mlp.experts`` matches nothing and every
        boundary layer would silently be quantized while the trainer keeps it
        BF16. Compare the decisions this worker actually made against the layer
        indices the patterns encode, restricted to the layers this pipeline
        stage owns.
        """
        if boundary_patterns is None:
            # No local NVFP4 target: a pipeline stage may legitimately own only
            # boundary layers. The global zero-target check covers the rest.
            return
        expected = boundary_layer_indices(boundary_patterns)
        local = set(quantized_layers) | set(excluded_layers)
        should_exclude = expected & local
        wrongly_quantized = sorted(should_exclude & set(quantized_layers))
        wrongly_excluded = sorted(set(excluded_layers) - expected)
        if wrongly_quantized or wrongly_excluded:
            raise RuntimeError(
                "[nvfp4_pertoken] BF16 boundary did not match the routed-expert "
                "module names, so rollout precision disagrees with training. "
                f"patterns={boundary_patterns}; layers quantized but expected "
                f"BF16={[quantized_layers[i] for i in wrongly_quantized]}; "
                f"layers excluded but expected NVFP4="
                f"{[excluded_layers[i] for i in wrongly_excluded]}"
            )

    def _resolve_module(self, name: str) -> torch.nn.Module | None:
        resolution = resolve_module_from_param_name(self._model, name)
        return resolution.module if resolution is not None else None

    def reset(self) -> None:
        """Clear pending pairs, coverage, and this refit's liveness counter."""
        self._pending = {}
        self._seen_roles = {}
        self._quantized_events = 0

    @staticmethod
    def _classify(target: RoutedExpertTarget, name: str) -> ExpertWeightSpec | None:
        # Mapping keys are checkpoint-relative (for example
        # ``experts.17.w1.weight``), while the stream includes the model/layer
        # prefix. Probe the bounded set of dot-boundary suffixes instead of
        # scanning every expert mapping entry for every refit tensor.
        parts = name.split(".")
        for index in range(len(parts)):
            spec = target.specs_by_suffix.get(".".join(parts[index:]))
            if spec is not None:
                return spec
        return None

    def process(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        """Quantize matching expert weights in one refit batch.

        Non-target names and BF16 routed-expert layers pass through, cloned off
        the IPC buffer. A weight resolved to a selected target but absent from
        that target's vLLM mapping is fatal.

        The clone matters: vLLM's layerwise reload buffers every
        ``weight_loader`` call's arguments (including the tensor) and replays
        them at layer completion, which can land after the sender has
        recycled this batch's IPC buffer for a later one. Freshly-allocated
        quantized tensors are already safe; passthrough tensors are views
        into that buffer unless cloned here.
        """
        out: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            owner = self._resolve_module(name)
            if owner is None:
                if any(name.endswith(suffix) for suffix in self._all_target_suffixes):
                    raise RuntimeError(
                        f"[nvfp4_pertoken] could not resolve target-owned "
                        f"checkpoint weight {name!r}."
                    )
                out.append((name, tensor.clone()))
                continue
            target = self._targets.get(id(owner))
            if target is None:
                out.append((name, tensor.clone()))
                continue

            spec = self._classify(target, name)
            if spec is None:
                if name.endswith(".weight"):
                    raise RuntimeError(
                        f"[nvfp4_pertoken] target-owned weight {name!r} is not "
                        f"classified by {target.module_name}'s vLLM expert mapping."
                    )
                out.append((name, tensor.clone()))
                continue
            seen_roles = self._seen_roles.setdefault(id(target.module), set())
            role_key = (spec.logical_key, spec.role)
            if role_key in seen_roles:
                raise RuntimeError(
                    f"[nvfp4_pertoken] duplicate {spec.role} checkpoint weight "
                    f"for {target.module_name}:{spec.logical_key}."
                )
            seen_roles.add(role_key)
            source_prefix = name.removesuffix(".weight")
            if spec.role == "w2":
                out.extend(self._quantize(source_prefix, weight=tensor))
                continue

            key = (id(target.module), spec.logical_key)
            partner = self._pending.pop(key, None)
            if partner is None:
                self._pending[key] = PendingHalf(
                    source_prefix=source_prefix,
                    role=spec.role,
                    tensor=tensor.clone(),
                )
                continue
            if partner.role == spec.role:
                raise RuntimeError(
                    f"[nvfp4_pertoken] duplicate {spec.role} checkpoint weight "
                    f"for {target.module_name}:{spec.logical_key}."
                )
            first, second = (
                ((source_prefix, tensor), (partner.source_prefix, partner.tensor))
                if spec.role == "w1"
                else ((partner.source_prefix, partner.tensor), (source_prefix, tensor))
            )
            out.extend(self._quantize_pair(w1=first, w3=second))
        return out

    def _quantize(
        self, name_prefix: str, *, weight: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Quantize a single expert projection; emit its four checkpoint names."""
        weight_scale_2 = self._global_scale(weight)
        packed, scale = self._scaled_fp4_quant(weight, weight_scale_2)
        self._quantized_events += 1
        return self._emit(name_prefix, packed, scale, weight_scale_2)

    def _quantize_pair(
        self,
        *,
        w1: tuple[str, torch.Tensor],
        w3: tuple[str, torch.Tensor],
    ) -> list[tuple[str, torch.Tensor]]:
        """Quantize a logical W1/W3 pair under one shared global scale."""
        weight_scale_2 = self._global_scale(w1[1], w3[1])
        out: list[tuple[str, torch.Tensor]] = []
        for source_prefix, weight in (w1, w3):
            packed, scale = self._scaled_fp4_quant(weight, weight_scale_2)
            out.extend(self._emit(source_prefix, packed, scale, weight_scale_2))
        self._quantized_events += 1
        return out

    @staticmethod
    def _global_scale(*weights: torch.Tensor) -> torch.Tensor:
        amax = (
            torch.stack([w.abs().amax() for w in weights])
            .amax()
            .float()
            .clamp_min(1e-8)
        )
        return (amax / _AMAX_DENOMINATOR).reshape(())

    @staticmethod
    def _scaled_fp4_quant(
        weight: torch.Tensor, weight_scale_2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weight.ndim != 2 or weight.shape[1] % 16 != 0:
            raise RuntimeError(
                "[nvfp4_pertoken] expert checkpoint weights must be "
                "two-dimensional with an input dimension divisible by the "
                f"NVFP4 block size 16, got shape {tuple(weight.shape)}."
            )
        global_scale = weight_scale_2.reciprocal().reshape(1)
        packed, scale = ops.scaled_fp4_quant(
            weight, global_scale, is_sf_swizzled_layout=False, backend="none"
        )
        expected_scale_shape = (weight.shape[0], weight.shape[1] // 16)
        assert scale.shape == expected_scale_shape, (
            f"[nvfp4_pertoken] expected linear (non-swizzled) NVFP4 block-scale "
            f"shape {expected_scale_shape}, got {tuple(scale.shape)} — "
            "scaled_fp4_quant's default swizzled layout may have changed."
        )
        return packed, scale

    @staticmethod
    def _emit(
        name_prefix: str,
        packed: torch.Tensor,
        scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        return [
            (f"{name_prefix}.weight", packed),
            (f"{name_prefix}.weight_scale", scale),
            (f"{name_prefix}.weight_scale_2", weight_scale_2.to(torch.float32)),
            (
                f"{name_prefix}.input_scale",
                torch.ones((), device=packed.device, dtype=torch.float32),
            ),
        ]

    def finish(self) -> None:
        """Raise if any gate/up half never received its partner this refit.

        A non-empty ``pending`` at refit end means stale expert weights would
        silently survive under the previous refit's values — fail loud rather
        than let that pass unnoticed (mirrors
        ``_IPCWeightManifest.require_complete``).

        Every selected local target must receive its complete W1/W2/W3
        inventory. A PP worker owning only BF16 boundary layers may have no
        selected local target; the driver preflight separately rejects a
        globally dense-only model. Use print because Ray workers default to
        WARNING-level logging.
        """
        if self._pending:
            unpaired = [
                {
                    "module": self._targets[target_id].module_name,
                    "logical_expert": logical_key,
                    "received": pending.role,
                }
                for (target_id, logical_key), pending in sorted(
                    self._pending.items(), key=lambda item: item[0][1]
                )
            ]
            raise RuntimeError(
                "[nvfp4_pertoken] refit ended with unpaired expert projections: "
                f"{unpaired}"
            )
        missing: dict[str, list[tuple[str, str]]] = {}
        for target_id, target in self._targets.items():
            target_missing = sorted(
                target.expected_roles - self._seen_roles.get(target_id, set())
            )
            if target_missing:
                missing[target.module_name] = target_missing
        if missing:
            preview = {module_name: roles[:8] for module_name, roles in missing.items()}
            raise RuntimeError(
                "[nvfp4_pertoken] refit did not cover every selected W1/W2/W3 "
                f"target (first missing entries): {preview}"
            )
        if self._targets and self._quantized_events == 0:
            raise RuntimeError(
                "[nvfp4_pertoken] refit quantized 0 params — export naming and "
                "the model's quant_method assignment are out of sync."
            )
        print(
            f"[nvfp4_pertoken] refit: quantized {self._quantized_events} expert "
            f"weight groups across {len(self._targets)} RoutedExperts containers",
            flush=True,
        )


class ModelOptNvFp4PerTokenFusedMoE(ModelOptNvFp4FusedMoE):
    """W4A4 MoE: pre-quantized weights, per-token dynamic activation scales.

    The class NAME must contain "ModelOpt": vLLM's RoutedExperts.weight_loader
    duck-types NVFP4 scale loading on ``"ModelOpt" in
    self.quant_method.__class__.__name__`` (routed_experts.py); a rename
    silently drops expert scale params out of that branch and initial load
    fails with "quant method must be one of ['tensor','channel','group',
    'block']".
    """

    moe_quant_config: Any
    moe_kernel: Any

    def __init__(self, quant_config, moe_config) -> None:
        super().__init__(
            quant_config,  # pyrefly: ignore[bad-argument-count]
            moe_config,
        )
        if self.use_a16:
            raise ValueError(
                f"{NVFP4_PER_TOKEN_METHOD} requires a W4A4 NVFP4 checkpoint, "
                "got W4A16_NVFP4."
            )
        # make_nvfp4_moe_kernel silently drops per_token_activation for every
        # backend except FLASHINFER_TRTLLM — fail loudly instead of running
        # with stale static scales.
        if self.nvfp4_backend != NvFp4MoeBackend.FLASHINFER_TRTLLM:
            raise ValueError(
                f"{NVFP4_PER_TOKEN_METHOD} requires the FlashInfer TRT-LLM MoE "
                f"backend, got {self.nvfp4_backend}."
            )

    def process_weights_after_loading(self, layer) -> None:
        """Finalize reload-format weights for dynamic per-token activation FP4.

        This intentionally replaces the stock ModelOpt implementation in full.
        Stock processing reconstructs a kernel using checkpoint/static input
        scales; this method installs neutral activation scales, preserves dense
        reload storage, and rebuilds the FlashInfer TRT-LLM kernel with
        ``per_token_activation=True`` after every cold or warm refit.
        """
        # Neutral (1.0) global activation scales: the kernel derives per-token
        # scales at runtime, so the output scalars reduce to the weight scales.
        num_experts = layer.w13_input_scale.data.shape[0]
        device = layer.w13_weight.device
        ones = torch.ones(num_experts, device=device, dtype=torch.float32)
        replace_parameter(layer, "w13_input_scale", ones)
        replace_parameter(layer, "w2_input_scale", ones.clone())
        # Use print because the engine process does not configure INFO logging
        # for the nemo_rl logger tree.
        global _pertoken_marker_printed
        if not _pertoken_marker_printed:
            _pertoken_marker_printed = True
            print(
                f"[{NVFP4_PER_TOKEN_METHOD}] per-token NVFP4 activation scaling active",
                flush=True,
            )

        w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0].contiguous()

        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=self.nvfp4_backend,
            layer=layer,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_scale_2=w13_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=layer.w2_weight_scale_2,
            a2_scale=layer.w2_input_scale,
            is_act_and_mul=self.moe.is_act_and_mul,
        )

        # Stride-0 expanded scale views break layerwise-reload finalize
        # (param.data.copy_() into broadcast storage); contiguous is a no-op
        # for already-dense tensors.
        def _dense(t):
            return t.contiguous() if isinstance(t, torch.Tensor) else t

        replace_parameter(layer, "w13_weight", _dense(w13))
        replace_parameter(layer, "w13_weight_scale", _dense(w13_scale))
        replace_parameter(layer, "w13_weight_scale_2", _dense(w13_scale_2))
        replace_parameter(layer, "w13_input_scale", _dense(a13_scale))
        replace_parameter(layer, "w2_weight", _dense(w2))
        replace_parameter(layer, "w2_weight_scale", _dense(w2_scale))
        replace_parameter(layer, "w2_weight_scale_2", _dense(w2_scale_2))
        replace_parameter(layer, "w2_input_scale", _dense(a2_scale))

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.experts_cls is not None
        self.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.nvfp4_backend,
            routing_tables=layer._expert_routing_tables(),
            layer=layer,
            per_token_activation=True,
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)


class NvFp4PerTokenConfig(ModelOptNvFp4Config):
    """Routed-expert-only ModelOpt NVFP4 with per-token activations.

    W4A4 and block size 16 are intentionally fixed for this rollout contract;
    they are properties of the selected vLLM kernel and refit tensor format,
    not user-tunable recipe fields.
    """

    FusedMoEMethodCls = ModelOptNvFp4PerTokenFusedMoE

    def get_name(self):
        return NVFP4_PER_TOKEN_METHOD

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        """Quantize selected RoutedExperts and keep all ordinary linears BF16."""
        if isinstance(layer, RoutedExperts):
            if self.is_layer_excluded(prefix):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            num_fused_shared_experts = int(
                getattr(layer.expert_map_manager, "num_fused_shared_experts", 0)
            )
            if num_fused_shared_experts:
                raise ValueError(
                    f"{NVFP4_PER_TOKEN_METHOD} does not support RoutedExperts "
                    f"with {num_fused_shared_experts} fused shared experts "
                    f"({prefix})."
                )
            return self.FusedMoEMethodCls(
                quant_config=self, moe_config=layer.moe_config
            )
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return UnquantizedLinearMethod()
        # Preserve ModelOpt's attention/KV-cache handling; all remaining module
        # types are unquantized by the base implementation.
        return super().get_quant_method(layer, prefix)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        # Never auto-select from checkpoint metadata; only an explicit
        # quantization="nvfp4_pertoken" picks this config.
        if user_quant == NVFP4_PER_TOKEN_METHOD:
            return NVFP4_PER_TOKEN_METHOD
        return None


def register_nvfp4_pertoken() -> None:
    """Register the per-token NVFP4 config through vLLM's public API."""
    global _registered
    if _registered:
        return
    register_quantization_config(NVFP4_PER_TOKEN_METHOD)(NvFp4PerTokenConfig)
    _registered = True
    logger.info("Registered vLLM quantization method %r", NVFP4_PER_TOKEN_METHOD)


class NvFp4PerTokenWorkerExtension(VllmInternalWorkerExtension):
    """Refit transport for per-token NVFP4 rollouts.

    Quantizes routed-expert BF16 weights to NVFP4 at refit-load time
    (``NvFp4PerTokenQuantizer``), mirroring the fp8/mxfp8 real-quant rollout
    path (``quantization/fp8.py``). IPC weight updates enter vLLM through its
    native ``reload_weights`` API, which restores quantized params to load
    format and re-processes them afterwards while preserving stable kernel
    storage for CUDA graphs.
    """

    _quantizer: Optional[NvFp4PerTokenQuantizer] = None

    def _get_quantizer(self) -> NvFp4PerTokenQuantizer:
        if self._quantizer is None:
            self._quantizer = NvFp4PerTokenQuantizer(self.model_runner.model)
        return self._quantizer

    def _get_reload_weight_preparer(self) -> _ReloadWeightPreparer:
        return self._get_quantizer()

    def report_nvfp4_pertoken_target_count(self) -> int:
        """Report this TP/PP worker's selected routed-expert containers."""
        return len(self._get_quantizer()._targets)

    def maybe_init_zmq(self) -> None:
        """Use a longer ZMQ timeout.

        The first refit re-processes every layer (per-token kernel rebuild plus
        FlashInfer autotune) before acknowledging the update.
        """
        import zmq

        super().maybe_init_zmq()
        self.zmq_socket.setsockopt(zmq.SNDTIMEO, NVFP4_PERTOKEN_ZMQ_TIMEOUT_MS)
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, NVFP4_PERTOKEN_ZMQ_TIMEOUT_MS)

    def _weight_update_errors_are_fatal(self) -> bool:
        return True

    def _synchronize_before_ipc_data_ack(self) -> None:
        torch.accelerator.synchronize()


def _reject_conflicting_engine_kwargs(llm_kwargs: dict[str, Any]) -> None:
    """Reject explicit engine settings incompatible with per-token NVFP4."""
    conflicts = [
        key for key in ("worker_extension_cls", "quantization") if key in llm_kwargs
    ]
    if "load_format" in llm_kwargs and llm_kwargs["load_format"] != "dummy":
        conflicts.append("load_format")
    hf_overrides = llm_kwargs.get("hf_overrides")
    if isinstance(hf_overrides, dict) and "quantization_config" in hf_overrides:
        conflicts.append("hf_overrides.quantization_config")
    kernel_config = llm_kwargs.get("kernel_config")
    if isinstance(kernel_config, dict) and kernel_config.get(
        "enable_flashinfer_autotune"
    ) not in (None, False):
        conflicts.append("kernel_config.enable_flashinfer_autotune")
    if conflicts:
        raise ValueError(
            "nvfp4_pertoken cannot overwrite explicit vLLM settings: "
            + ", ".join(sorted(set(conflicts)))
        )


def configure_nvfp4_pertoken_engine_kwargs(
    llm_kwargs: dict[str, Any],
    ignore: list[str],
    *,
    explicit_engine_kwargs: dict[str, Any] | None = None,
) -> None:
    """Mutate vLLM engine kwargs for the per-token W4A4 rollout.

    ``explicit_engine_kwargs`` carries the untouched user configuration when
    the framework has already added defaults to ``llm_kwargs``. Direct callers
    may omit it to treat every supplied engine kwarg as explicit.

    - registers and selects the ``nvfp4_pertoken`` quantization method
    - overrides the HF quantization config (weights NVFP4, activations dynamic)
    - dummy initial load: params are NVFP4-shaped and the BF16 checkpoint on
      disk cannot fill them; the first refit (which always precedes the first
      generation) provides every weight
    - disables FlashInfer autotuning because its full-model warmup otherwise
      executes the dummy NVFP4 weights before that first refit
    - installs the refit worker extension
    """
    conflict_source = (
        llm_kwargs if explicit_engine_kwargs is None else explicit_engine_kwargs
    )
    _reject_conflicting_engine_kwargs(conflict_source)
    register_nvfp4_pertoken()
    llm_kwargs["quantization"] = NVFP4_PER_TOKEN_METHOD
    llm_kwargs["load_format"] = "dummy"
    kernel_config = llm_kwargs.setdefault("kernel_config", {})
    if not isinstance(kernel_config, dict):
        raise ValueError("nvfp4_pertoken requires kernel_config to be a mapping")
    kernel_config["enable_flashinfer_autotune"] = False
    hf_overrides = llm_kwargs.setdefault("hf_overrides", {})
    hf_overrides["quantization_config"] = build_nvfp4_pertoken_hf_quant_config(ignore)
    llm_kwargs["worker_extension_cls"] = (
        "nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken."
        "NvFp4PerTokenWorkerExtension"
    )


def build_nvfp4_pertoken_hf_quant_config(ignore: list[str]) -> dict[str, Any]:
    """HF ``quantization_config`` override for the per-token W4A4 rollout.

    A literal dict (no ModelOpt conversion helper): NVFP4 weights with
    block-16 e4m3 scales; activations dynamic (per-token global scales are
    derived inside the kernel, no ``input_scale`` tensors exist).
    """
    # Mirrors the quantization_config of ModelOpt NVFP4 HF checkpoints
    # (e.g. nvidia/Qwen3-30B-A3B-NVFP4 config.json) key-for-key — vLLM's
    # ModelOpt config parser is shape-sensitive (`ignore`, not
    # `exclude_modules`; `targets` inside the group). Only delta:
    # input_activations.dynamic=True since no input_scale tensors exist.
    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "producer": {"name": "modelopt"},
        "ignore": list(ignore),
        "config_groups": {
            "group_0": {
                "weights": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "input_activations": {
                    "dynamic": True,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "targets": ["Linear"],
            }
        },
    }
