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

Precision gate (4) is BF16 vs MXFP8 from ``policy.precision`` and ``fp8_cfg`` only.
``te_precision_config_file`` is out of scope here (``setup._apply_precision_config``).
"""

from __future__ import annotations

import subprocess
import warnings
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

if TYPE_CHECKING:
    from nemo_rl.models.policy import PolicyConfig

# TE MXFP8 grouped MoE inference (Megatron-LM PR #6933) and batch-invariant follow-ups.
MEGATRON_CORE_MIN_COMMIT_SHA = "4e603390dfd64a599d1c23989209892c2326c122"

TRANSFORMER_ENGINE_MIN_VERSION = Version("2.18")
FLASH_ATTN_MIN_VERSION = Version("2.8.1")
FLASH_ATTN_4_MIN_VERSION = Version("4.0.0b20")
CUTEDSL_MIN_VERSION = Version("4.6.0.dev0")

PrecisionKind = Literal["bf16", "mxfp8"]

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
    if not config.get("megatron_cfg", {}).get("zero_train_gen_mismatch"):
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
    if not config.get("megatron_cfg", {}).get("zero_train_gen_mismatch"):
        return out

    _validate_backend(config, out)
    _validate_platform(config, out, check_device=check_platform)
    if check_packages:
        _validate_packages(out)
    _validate_model_architecture(config, out)
    _validate_precision(config, out)
    return out


def validate_batch_invariant_mode(config: PolicyConfig) -> ZeroTrainGenValidation:
    """Checks for ``batch_invariant_mode=True`` without enabling MCore kernels."""
    from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg

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


def enable_batch_invariant_kernels(config: PolicyConfig) -> None:
    """Pin TE FA support and call MCore ``enable_batch_invariant_mode``."""
    megatron_cfg = config["megatron_cfg"]
    collective = megatron_cfg["batch_invariant_collective"]

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


def configure_zero_train_gen_mismatch(
    config: PolicyConfig,
    *,
    apply_kernels: bool,
    register_moe_bi_fp8_skip: Callable[[], None],
) -> None:
    """Resolve, register Megatron config patch, validate, optionally enable kernels."""
    if not config.get("megatron_cfg", {}).get("zero_train_gen_mismatch"):
        return

    resolve_zero_train_gen_mismatch(config)
    register_moe_bi_fp8_skip()

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

    from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg

    inference_cfg = merged_inference_megatron_cfg(config)
    impl = inference_cfg.get("transformer_impl")
    if impl not in ("inference_optimized", "transformer_engine"):
        out.violations.append(
            "generation transformer_impl must be 'inference_optimized' or "
            f"'transformer_engine' (got {impl!r})."
        )
    elif impl == "transformer_engine":
        out.warnings.append(
            "zero_train_gen_mismatch: generation uses transformer_impl="
            "'transformer_engine'; use 'inference_optimized' on the generation "
            "worker for better performance."
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

    try:
        import torch

        if not torch.cuda.is_available():
            return
        major, _minor = torch.cuda.get_device_capability()
    except ImportError:
        return

    if major == 9:
        out.violations.append(
            "zero_train_gen_mismatch does not support Hopper (SM90); use "
            "Blackwell (SM100+) with flash_attention_version=4."
        )
    elif major < 10:
        out.violations.append(
            f"zero_train_gen_mismatch requires Blackwell (SM100+); got sm_{major}x."
        )


def _megatron_core_source_root() -> str | None:
    try:
        import megatron.core as mcore
    except ImportError:
        return None
    from pathlib import Path

    return str(Path(mcore.__file__).resolve().parent.parent.parent)


def _git_is_ancestor(ancestor_sha: str, head_sha: str, repo_root: str) -> bool:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor_sha, head_sha],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def _installed_megatron_core_commit() -> str | None:
    repo_root = _megatron_core_source_root()
    if repo_root is None:
        return None
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _validate_megatron_core_commit(min_sha: str, out: ZeroTrainGenValidation) -> None:
    repo_root = _megatron_core_source_root()
    if repo_root is None:
        out.violations.append(
            "Megatron-Core is not importable; cannot verify minimum commit "
            f"(required ancestor {min_sha})."
        )
        return
    head = _installed_megatron_core_commit()
    if head is None:
        out.violations.append(
            "Could not read Megatron-Core git HEAD; install from a git checkout "
            f"at or after {min_sha}."
        )
        return
    if min_sha == head:
        return
    if not _git_is_ancestor(min_sha, head, repo_root):
        out.violations.append(
            f"Megatron-Core commit {head} is not at or after required "
            f"zero_train_gen_mismatch minimum {min_sha}."
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
    _validate_megatron_core_commit(MEGATRON_CORE_MIN_COMMIT_SHA, out)

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
    if fa4 is None or fa4 < FLASH_ATTN_4_MIN_VERSION:
        out.violations.append(
            f"flash-attn-4>={FLASH_ATTN_4_MIN_VERSION} required "
            f"(got {fa4})."
        )
    cutedsl = _package_version("nvidia-cutlass-dsl")
    if cutedsl is None or cutedsl < CUTEDSL_MIN_VERSION:
        out.violations.append(
            f"nvidia-cutlass-dsl>={CUTEDSL_MIN_VERSION} required (got {cutedsl})."
        )


def _validate_model_architecture(
    config: PolicyConfig, out: ZeroTrainGenValidation
) -> None:
    megatron_cfg = config["megatron_cfg"]
    _deny_if_true(megatron_cfg, out, "multi_latent_attention", "MLA")
    _deny_if_true(megatron_cfg, out, "use_mla", "MLA")
    for key in (
        "hybrid_attention_ratio",
        "mamba_num_heads",
        "linear_attention",
    ):
        if megatron_cfg.get(key):
            out.violations.append(
                f"zero_train_gen_mismatch does not support linear-attention / "
                f"SSM hybrid config {key!r} yet."
            )


def _deny_if_true(
    megatron_cfg: Mapping[str, Any],
    out: ZeroTrainGenValidation,
    key: str,
    label: str,
) -> None:
    if megatron_cfg.get(key):
        out.violations.append(
            f"zero_train_gen_mismatch does not support {label} ({key}=true)."
        )


def _effective_train_precision(config: PolicyConfig) -> PrecisionKind | None:
    """Classify train or merged-gen config as bf16 vs mxfp8 for zero-KL parity."""
    megatron_cfg = config["megatron_cfg"]
    fp8_cfg = megatron_cfg.get("fp8_cfg") or {}
    if fp8_cfg.get("enabled"):
        if fp8_cfg.get("fp8_recipe") != "mxfp8":
            return None
        return "mxfp8"
    if config.get("precision") == "bfloat16":
        return "bf16"
    return None


def _validate_precision(config: PolicyConfig, out: ZeroTrainGenValidation) -> None:
    """Gate (4): allowed modes are BF16 or MXFP8; train and generation must agree."""
    train_kind = _effective_train_precision(config)
    if train_kind is None:
        fp8_cfg = config["megatron_cfg"].get("fp8_cfg") or {}
        if fp8_cfg.get("enabled"):
            out.violations.append(
                "zero_train_gen_mismatch supports mxfp8 (fp8_recipe='mxfp8') or "
                f"policy.precision='bfloat16' without FP8 (got fp8_recipe="
                f"{fp8_cfg.get('fp8_recipe')!r}, precision={config.get('precision')!r})."
            )
        else:
            out.violations.append(
                "zero_train_gen_mismatch requires policy.precision='bfloat16' "
                f"(got {config.get('precision')!r})."
            )
        return

    from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg

    inference_cfg = merged_inference_megatron_cfg(config)
    gen_kind = _effective_train_precision(
        {"megatron_cfg": inference_cfg, "precision": config.get("precision")}
    )
    if gen_kind != train_kind:
        out.violations.append(
            "zero_train_gen_mismatch requires matching train/generation precision "
            f"(train={train_kind!r}, generation={gen_kind!r})."
        )
