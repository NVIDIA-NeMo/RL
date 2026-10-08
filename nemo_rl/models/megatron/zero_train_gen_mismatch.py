# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed under an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Zero train/generation KL (batch-invariant Megatron) resolve, validate, and enable.

Only BF16 (no FP8) is supported.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, fields
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import torch
from packaging.version import Version

from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg

if TYPE_CHECKING:
    from nemo_rl.models.policy import PolicyConfig

# TransformerConfig fields the installed Megatron-Core must define for this mode.
MEGATRON_CORE_REQUIRED_CONFIG_FIELDS = (
    "batch_invariant_mode",
    "batch_invariant_backend",
    "batch_invariant_collective",
    "flash_attention_version",
)

TRANSFORMER_ENGINE_MIN_VERSION = Version("2.18")
FLASH_ATTN_MIN_VERSION = Version("2.8.1")
CUTEDSL_MIN_VERSION = Version("4.6.0.dev0")

_ZERO_KL_MEGATRON_DEFAULTS: dict[str, Any] = {
    "batch_invariant_mode": True,
    "moe_permute_fusion": False,
    "attention_backend": "flash",
    "flash_attention_version": 4,
    "batch_invariant_backend": "te_native",
    "batch_invariant_collective": "ordered",
}

_ZERO_KL_GENERATION_DEFAULTS: dict[str, Any] = {
    "logprobs_mode": "raw_logprobs",
    "enable_chunked_prefill": False,
}


@dataclass
class ZeroTrainGenValidation:
    """Collected validation outcome."""

    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_invalid(self, header: str) -> None:
        if not self.violations:
            return
        bullet = "\n".join(f"  - {msg}" for msg in self.violations)
        raise ValueError(f"{header}\n{bullet}")

    def emit_warnings(self) -> None:
        for msg in self.warnings:
            warnings.warn(msg, UserWarning, stacklevel=3)


def resolve_zero_train_gen_mismatch(config: PolicyConfig) -> None:
    """Apply zero-KL defaults; warn when overriding user recipe values."""
    if not config["megatron_cfg"].get("zero_train_gen_mismatch"):
        return

    mc = config["megatron_cfg"]
    generation = config.get("generation")
    defaults: list[tuple[dict[str, Any], str, dict[str, Any]]] = [
        (mc, "policy.megatron_cfg", _ZERO_KL_MEGATRON_DEFAULTS),
    ]
    if generation is not None:
        defaults.append(
            (
                generation["mcore_generation_config"],
                "policy.generation.mcore_generation_config",
                _ZERO_KL_GENERATION_DEFAULTS,
            )
        )
    for cfg, config_path, values in defaults:
        for key, value in values.items():
            if key in cfg and cfg[key] != value:
                warnings.warn(
                    f"zero_train_gen_mismatch=true overrides {config_path}.{key}"
                    f"={cfg[key]!r} with {value!r}: the configured value would "
                    "reintroduce train/generation mismatch.",
                    UserWarning,
                    stacklevel=2,
                )
            cfg[key] = value


def validate_zero_train_gen_mismatch(
    config: PolicyConfig,
    *,
    check_packages: bool = True,
    check_platform: bool = True,
) -> ZeroTrainGenValidation:
    """Run all zero-KL gates; return violations and non-fatal warnings."""
    out = ZeroTrainGenValidation()
    if not config["megatron_cfg"].get("zero_train_gen_mismatch"):
        return out

    _validate_backend(config, out)
    _validate_platform(config, out, check_device=check_platform)
    if check_packages:
        _validate_packages(out)
    _validate_precision(config, out)
    return out


def validate_batch_invariant_mode(config: PolicyConfig) -> ZeroTrainGenValidation:
    """Checks for ``batch_invariant_mode=True`` without enabling MCore kernels."""
    out = ZeroTrainGenValidation()
    megatron_cfg = config["megatron_cfg"]
    if not megatron_cfg.get("batch_invariant_mode"):
        return out

    required_fields = (
        "batch_invariant_backend",
        "batch_invariant_collective",
        "flash_attention_version",
    )
    missing_fields = [field for field in required_fields if field not in megatron_cfg]
    if missing_fields:
        out.violations.append(
            "batch_invariant_mode=True requires policy.megatron_cfg fields: "
            f"{', '.join(missing_fields)}."
        )

    if megatron_cfg.get("tensor_model_parallel_size") != 1:
        out.violations.append(
            "batch_invariant_mode=True currently requires "
            "policy.megatron_cfg.tensor_model_parallel_size=1."
        )
    if megatron_cfg.get("context_parallel_size") != 1:
        out.violations.append(
            "batch_invariant_mode=True currently requires training context "
            "parallel size 1."
        )
    if config["sequence_packing"]["enabled"]:
        out.violations.append(
            "batch_invariant_mode=True requires sequence_packing.enabled=False: "
            "packing changes microbatch composition and the packed log-prob path "
            "does not share generation's fused log-softmax."
        )
    if megatron_cfg.get("use_fused_linear_logprobs"):
        out.violations.append(
            "batch_invariant_mode=True is incompatible with "
            "use_fused_linear_logprobs=True because generation parity requires "
            "the shared float-log_softmax-gather path."
        )
    if megatron_cfg.get("attention_backend") != "flash":
        out.violations.append(
            "batch_invariant_mode=True requires "
            "policy.megatron_cfg.attention_backend='flash'."
        )
    fa_ver = megatron_cfg.get("flash_attention_version")
    if fa_ver not in (3, 4):
        out.violations.append(
            "batch_invariant_mode=True requires "
            "policy.megatron_cfg.flash_attention_version to be 3 or 4."
        )

    generation_cfg = config.get("generation")
    if generation_cfg is None or generation_cfg.get("backend") != "megatron":
        out.violations.append(
            "batch_invariant_mode=True requires policy.generation.backend='megatron'."
        )
    else:
        inference_cfg = merged_inference_megatron_cfg(config)
        if (
            inference_cfg.get("transformer_impl") == "inference_optimized"
            and config.get("precision") != "bfloat16"
        ):
            out.violations.append(
                "batch_invariant_mode=True with "
                "transformer_impl='inference_optimized' requires "
                "policy.precision='bfloat16'."
            )
        matching_fields = (
            "tensor_model_parallel_size",
            "context_parallel_size",
            "batch_invariant_mode",
            "batch_invariant_backend",
            "batch_invariant_collective",
            "attention_backend",
            "flash_attention_version",
        )
        mismatched_fields = [
            field
            for field in matching_fields
            if inference_cfg.get(field) != megatron_cfg.get(field)
        ]
        if mismatched_fields:
            out.violations.append(
                "Training and generation must use the same Megatron settings for "
                f"batch invariance: {', '.join(mismatched_fields)}."
            )

    return out


def allow_installed_flash_attn_4() -> None:
    """Enable MCore's FA4 path for the pinned ``flash-attn-4`` b19.

    MCore gates ``HAVE_FA4`` on ``>=4.0.0b20``, which needs a newer
    ``apache-tvm-ffi`` than we pin. The call signature is identical.
    """
    # Deferred: Megatron-Core exists only in worker venvs.
    import megatron.core.transformer.attention as mcore_attention

    if getattr(mcore_attention, "HAVE_FA4", False):
        return
    if _first_package_version(("flash-attn-4", "flash_attn_4")) is None:
        return
    # Deferred: optional dependency, checked above.
    from flash_attn.cute import flash_attn_varlen_func

    mcore_attention.flash_attn4_varlen_func = flash_attn_varlen_func
    mcore_attention.HAVE_FA4 = True


class _Tp1LocalLogprob:
    """At TP size 1, use the log-softmax Megatron Inference uses (bitwise)."""

    def __init__(self, original: Any, *, chunked: bool) -> None:
        self._original = original
        self._chunked = chunked

    def apply(self, *args: Any) -> Any:
        # Deferred: model_utils is heavy and only needed once the shim is installed.
        from nemo_rl.distributed import model_utils

        if self._chunked:
            logits, target, start, end, chunk_size, group, inference_only = args
        else:
            logits, target, start, end, group, inference_only = args
            chunk_size = None
        if group is None or torch.distributed.get_world_size(group) != 1:
            return self._original.apply(*args)
        return model_utils._tp_target_logprobs(
            logits,
            target,
            vocab_start_index=start,
            vocab_end_index=end,
            tp_group=None,
            chunk_size=chunk_size,
            inference_only=inference_only,
        )


def _use_fused_log_softmax_at_tp1() -> None:
    # Deferred: model_utils is heavy and only needed once the shim is installed.
    from nemo_rl.distributed import model_utils

    for name, chunked in (
        ("DistributedLogprob", False),
        ("ChunkedDistributedLogprob", True),
    ):
        current = getattr(model_utils, name)
        if not isinstance(current, _Tp1LocalLogprob):
            setattr(model_utils, name, _Tp1LocalLogprob(current, chunked=chunked))


def enable_batch_invariant_kernels(config: PolicyConfig) -> None:
    """Pin TE FA support and call MCore ``enable_batch_invariant_mode``."""
    megatron_cfg = config["megatron_cfg"]
    allow_installed_flash_attn_4()
    collective = megatron_cfg["batch_invariant_collective"]

    # Deferred: Megatron-Core exists only in worker venvs.
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        assert_te_supports_batch_invariant_attention,
    )
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        enable_batch_invariant_mode as enable_mcore_batch_invariant_mode,
    )

    assert_te_supports_batch_invariant_attention()
    enable_mcore_batch_invariant_mode(
        backend=megatron_cfg["batch_invariant_backend"], collective=collective
    )
    _use_fused_log_softmax_at_tp1()
    print(
        "[zero_train_gen_mismatch] batch-invariant kernels enabled: "
        f"backend={megatron_cfg['batch_invariant_backend']} "
        f"collective={collective} "
        f"flash_attention_version={megatron_cfg['flash_attention_version']} "
        f"CUTE_DSL_LIBS={os.environ.get('CUTE_DSL_LIBS')}",
        flush=True,
    )


def configure_zero_train_gen_mismatch(
    config: PolicyConfig,
    *,
    apply_kernels: bool,
) -> None:
    """Resolve, validate, and optionally enable batch-invariant kernels."""
    if not config["megatron_cfg"].get("zero_train_gen_mismatch"):
        return

    resolve_zero_train_gen_mismatch(config)
    result = validate_zero_train_gen_mismatch(
        config,
        check_packages=apply_kernels,
        check_platform=apply_kernels,
    )
    result.emit_warnings()
    result.raise_if_invalid(
        "policy.megatron_cfg.zero_train_gen_mismatch=true failed validation:"
    )

    bi_result = validate_batch_invariant_mode(config)
    bi_result.emit_warnings()
    bi_result.raise_if_invalid("batch_invariant_mode=True failed validation:")

    if not apply_kernels:
        return

    enable_batch_invariant_kernels(config)


def _validate_backend(config: PolicyConfig, out: ZeroTrainGenValidation) -> None:
    generation = config.get("generation")
    if generation is None or generation.get("backend") != "megatron":
        out.violations.append(
            "policy.generation.backend must be 'megatron' "
            f"(got {generation.get('backend') if generation else None!r})."
        )
        return

    if generation["colocated"]["enabled"]:
        out.violations.append(
            "zero_train_gen_mismatch does not support colocated generation; set "
            "policy.generation.colocated.enabled=false."
        )

    inference_cfg = merged_inference_megatron_cfg(config)
    impl = inference_cfg.get("transformer_impl")
    if impl != "inference_optimized":
        out.violations.append(
            f"generation transformer_impl must be 'inference_optimized' (got {impl!r})."
        )


def _validate_platform(
    config: PolicyConfig, out: ZeroTrainGenValidation, *, check_device: bool
) -> None:
    """Blackwell + FA4 only (mcore extra: flash-attn-4, CuteDSL). Hopper/FA3 not supported."""
    megatron_cfg = config["megatron_cfg"]
    fa_ver = megatron_cfg.get("flash_attention_version")
    if fa_ver != 4:
        out.violations.append(
            "zero_train_gen_mismatch requires policy.megatron_cfg."
            f"flash_attention_version=4 (got {fa_ver!r}). Hopper (FA3) is not "
            "supported; use Blackwell with the flash-attn-4 stack."
        )
        return

    if not check_device:
        return

    if not torch.cuda.is_available():
        return
    major, _minor = torch.cuda.get_device_capability()

    if major == 9:
        out.violations.append(
            "zero_train_gen_mismatch does not support Hopper (SM90); use "
            "Blackwell (SM100+) with flash_attention_version=4."
        )
    elif major < 10:
        out.violations.append(
            f"zero_train_gen_mismatch requires Blackwell (SM100+); got sm_{major}x."
        )


def _validate_megatron_core_features(out: ZeroTrainGenValidation) -> None:
    try:
        # Deferred: Megatron-Core exists only in worker venvs.
        from megatron.core.transformer.transformer_config import TransformerConfig
    except ImportError:
        out.violations.append(
            "Megatron-Core is not importable (required for zero_train_gen_mismatch)."
        )
        return
    defined = {f.name for f in fields(TransformerConfig)}
    missing = [f for f in MEGATRON_CORE_REQUIRED_CONFIG_FIELDS if f not in defined]
    if missing:
        out.violations.append(
            "Installed Megatron-Core lacks TransformerConfig fields required for "
            f"zero_train_gen_mismatch: {', '.join(missing)}."
        )


def _package_version(dist_name: str) -> Version | None:
    try:
        return Version(version(dist_name))
    except PackageNotFoundError:
        return None


def _first_package_version(dist_names: tuple[str, ...]) -> Version | None:
    for dist_name in dist_names:
        found = _package_version(dist_name)
        if found is not None:
            return found
    return None


def _validate_packages(out: ZeroTrainGenValidation) -> None:
    _validate_megatron_core_features(out)

    te_ver = _package_version("transformer_engine")
    if te_ver is None:
        out.violations.append(
            "transformer_engine is not installed (required for zero_train_gen_mismatch)."
        )
    elif te_ver < TRANSFORMER_ENGINE_MIN_VERSION:
        out.violations.append(
            f"transformer_engine>={TRANSFORMER_ENGINE_MIN_VERSION} required "
            f"(got {te_ver})."
        )

    fa = _package_version("flash_attn")
    if fa is None or fa < FLASH_ATTN_MIN_VERSION:
        out.violations.append(
            f"flash_attn>={FLASH_ATTN_MIN_VERSION} required (got {fa})."
        )

    fa4 = _first_package_version(("flash-attn-4", "flash_attn_4"))
    if fa4 is None:
        out.violations.append("flash-attn-4 is not installed (mcore extra).")
    cutedsl = _package_version("nvidia-cutlass-dsl")
    if cutedsl is None or cutedsl < CUTEDSL_MIN_VERSION:
        out.violations.append(
            f"nvidia-cutlass-dsl>={CUTEDSL_MIN_VERSION} required (got {cutedsl})."
        )


def validate_zero_train_gen_model_provider(
    config: PolicyConfig, model_cfg: Any
) -> None:
    """Reject unsupported architectures using the resolved Bridge provider."""
    if not config["megatron_cfg"].get("zero_train_gen_mismatch"):
        return

    unsupported_fields = [
        field_name
        for field_name in (
            "multi_latent_attention",
            "hybrid_layer_pattern",
            "experimental_attention_variant",
        )
        if getattr(model_cfg, field_name, None)
    ]
    if unsupported_fields:
        raise ValueError(
            "zero_train_gen_mismatch does not support the resolved model "
            f"architecture fields: {', '.join(unsupported_fields)}."
        )


def _validate_precision(config: PolicyConfig, out: ZeroTrainGenValidation) -> None:
    """Only BF16 without FP8 is supported, on both the train and generation side."""
    if config.get("precision") != "bfloat16":
        out.violations.append(
            "zero_train_gen_mismatch requires policy.precision='bfloat16' "
            f"(got {config.get('precision')!r})."
        )

    sides = [("policy.megatron_cfg", config["megatron_cfg"])]
    generation = config.get("generation")
    if generation is not None and generation.get("backend") == "megatron":
        sides.append(("generation", merged_inference_megatron_cfg(config)))
    for label, cfg in sides:
        if cfg.get("fp8_cfg") and cfg["fp8_cfg"].get("enabled"):
            out.violations.append(
                f"zero_train_gen_mismatch does not support FP8 ({label}.fp8_cfg.enabled=true)."
            )
