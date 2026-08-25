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
fp8/mxfp8 "real quant" rollout path (``quantization/fp8.py``). The training
worker stays entirely unaware of NVFP4; the refit transport always carries
BF16.
"""

import re
from dataclasses import dataclass
from typing import Any, Optional

import torch
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.utils import replace_parameter

from nemo_rl.models.generation.vllm.quantization.fp8 import get_module_from_param_name
from nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken_config import (
    DEFAULT_NVFP4_IGNORE,
    NVFP4_PERTOKEN_ZMQ_TIMEOUT_MS,
    NvFp4PerTokenRolloutConfig,
)
from nemo_rl.models.generation.vllm.vllm_backend import (
    VllmInternalWorkerExtension,
    _ReloadWeightPreparer,
)

logger = init_logger(__name__)

__all__ = ["DEFAULT_NVFP4_IGNORE", "NvFp4PerTokenRolloutConfig"]

NVFP4_PER_TOKEN_METHOD = "nvfp4_pertoken"

_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.*\.experts)\.(?P<eid>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)

_FP4_MAX = 6.0
_FP8_E4M3_MAX = 448.0
_AMAX_DENOMINATOR = _FP4_MAX * _FP8_E4M3_MAX

_registered = False
_pertoken_marker_printed = False


@dataclass
class PendingHalf:
    """One half of a gate/up pair awaiting its partner.

    The tensor is cloned off the refit IPC buffer, which the sender recycles
    as soon as the batch is acknowledged (see ``policy/utils.py``'s
    ping-pong double buffering).
    """

    layer_prefix: str
    expert_id: int
    proj: str  # "gate_proj" | "up_proj"
    tensor: torch.Tensor


class NvFp4PerTokenQuantizer:
    """Quantizes routed-expert weights to NVFP4 during vLLM-side weight refit.

    Stateful only over a single gate/up pair per (layer, expert) — never a
    whole layer — so memory stays bounded regardless of expert count. Gate
    and up projections share one global scale (vLLM's
    ``ModelOptNvFp4FusedMoE.process_weights_after_loading`` collapses
    ``w13_weight_scale_2`` to column 0, so independently-scaled halves would
    decode the up projection with the gate's scale), so quantization of a
    pair is deferred until both halves have arrived. ``additional_ignore``
    expert layers are detected by probing the owning module's quant method
    rather than re-deriving the ignore-pattern match, since a config-level
    check cannot see how ``configure_nvfp4_pertoken_engine_kwargs`` resolved
    that layer's ``quant_method`` at engine build time.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model
        self._pending: dict[tuple[str, int], PendingHalf] = {}
        self._quantized_layer: dict[str, bool] = {}
        self._quantized_events = 0

    def reset(self) -> None:
        """Clear pending gate/up state and this refit's liveness counter."""
        self._pending = {}
        self._quantized_events = 0

    def _is_quantized_layer(self, layer_prefix: str) -> bool:
        cached = self._quantized_layer.get(layer_prefix)
        if cached is not None:
            return cached
        module = get_module_from_param_name(
            self._model, f"{layer_prefix}.0.gate_proj.weight"
        )
        is_quantized = isinstance(
            getattr(module, "quant_method", None), ModelOptNvFp4PerTokenFusedMoE
        )
        self._quantized_layer[layer_prefix] = is_quantized
        return is_quantized

    def process(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        """Quantize matching expert weights in one refit batch.

        Non-expert names and expert layers outside the quantized scope
        (``additional_ignore``) pass through, cloned off the IPC buffer.

        The clone matters: vLLM's layerwise reload buffers every
        ``weight_loader`` call's arguments (including the tensor) and replays
        them at layer completion, which can land after the sender has
        recycled this batch's IPC buffer for a later one. Freshly-allocated
        quantized tensors are already safe; passthrough tensors are views
        into that buffer unless cloned here.
        """
        out: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            match = _EXPERT_WEIGHT_RE.match(name)
            if match is None or not self._is_quantized_layer(match.group("prefix")):
                out.append((name, tensor.clone()))
                continue

            prefix = match.group("prefix")
            expert_id = int(match.group("eid"))
            proj = match.group("proj")

            if proj == "down_proj":
                out.extend(
                    self._quantize(f"{prefix}.{expert_id}.down_proj", weight=tensor)
                )
                continue

            key = (prefix, expert_id)
            partner = self._pending.pop(key, None)
            if partner is None:
                self._pending[key] = PendingHalf(
                    layer_prefix=prefix,
                    expert_id=expert_id,
                    proj=proj,
                    tensor=tensor.clone(),
                )
                continue

            gate, up = (
                (tensor, partner.tensor)
                if proj == "gate_proj"
                else (partner.tensor, tensor)
            )
            out.extend(self._quantize_pair(prefix, expert_id, gate=gate, up=up))
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
        self, prefix: str, expert_id: int, *, gate: torch.Tensor, up: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Quantize a gate/up pair under one shared global scale."""
        weight_scale_2 = self._global_scale(gate, up)
        out: list[tuple[str, torch.Tensor]] = []
        for proj_name, w in (("gate_proj", gate), ("up_proj", up)):
            packed, scale = self._scaled_fp4_quant(w, weight_scale_2)
            out.extend(
                self._emit(
                    f"{prefix}.{expert_id}.{proj_name}", packed, scale, weight_scale_2
                )
            )
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

        Also prints a per-refit liveness line and raises if nothing was
        quantized — a config/name mismatch would otherwise silently degrade to
        an all-BF16 refit. Use print because Ray workers default to
        WARNING-level logging.
        """
        if self._pending:
            raise RuntimeError(
                "[nvfp4_pertoken] refit ended with unpaired expert projections: "
                f"{sorted(self._pending)}"
            )
        print(
            f"[nvfp4_pertoken] refit: quantized {self._quantized_events} expert "
            "weight groups",
            flush=True,
        )
        if self._quantized_events == 0:
            raise RuntimeError(
                "[nvfp4_pertoken] refit quantized 0 params — export naming and "
                "the model's quant_method assignment are out of sync."
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
    """Stock ModelOpt NVFP4 config with per-token FusedMoE activations."""

    FusedMoEMethodCls = ModelOptNvFp4PerTokenFusedMoE

    def get_name(self):
        return NVFP4_PER_TOKEN_METHOD

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
    - installs the refit worker extension
    """
    conflict_source = (
        llm_kwargs if explicit_engine_kwargs is None else explicit_engine_kwargs
    )
    _reject_conflicting_engine_kwargs(conflict_source)
    register_nvfp4_pertoken()
    llm_kwargs["quantization"] = NVFP4_PER_TOKEN_METHOD
    llm_kwargs["load_format"] = "dummy"
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
