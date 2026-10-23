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

"""Zero train/generation KL (batch-invariant Megatron): validate the recipe, then enable.

The recipe sets every required value; this module only checks them. Supported
precisions are BF16 and MXFP8 (the same on train and generation).
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field, fields
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import torch
from packaging.version import Version

from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    need_top_k_or_top_p_filtering,
)
from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg
from nemo_rl.models.megatron.batch_invariant import batch_invariant_token_multiple

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

_ZERO_KL_MEGATRON_REQUIRED: dict[str, Any] = {
    "batch_invariant_mode": True,
    "moe_permute_fusion": False,
    "attention_backend": "flash",
    "flash_attention_version": 4,
    "batch_invariant_backend": "te_native",
    "batch_invariant_collective": "ordered",
}

_ZERO_KL_GENERATION_REQUIRED: dict[str, Any] = {}

# Workspace-free cuBLAS (fixed reduction order) for the te_native backend.
_TE_NATIVE_ENV_VARS = {
    "CUBLASLT_WORKSPACE_SIZE": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":0:0",
}

# Hybrid Mamba models: the SSM Triton ops read both at import. Without
# TRITON_CACHE_AUTOTUNING, MAMBA_DETERMINISTIC replaces autotuning with the cheapest
# config.
_HYBRID_MAMBA_ENV_VARS = {
    "MAMBA_DETERMINISTIC": "1",
    "TRITON_CACHE_AUTOTUNING": "1",
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

    _validate_required_values(config, out)
    _validate_backend(config, out)
    _validate_platform(out, check_device=check_platform)
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

    if (megatron_cfg.get("tensor_model_parallel_size") or 1) > 2:
        out.violations.append(
            "batch_invariant_mode=True currently requires "
            "policy.megatron_cfg.tensor_model_parallel_size <= 2: at TP >= 3 "
            "Megatron-Core's TP reductions sum in a message-layout dependent order."
        )
    if megatron_cfg.get("context_parallel_size") != 1:
        out.violations.append(
            "batch_invariant_mode=True currently requires training context "
            "parallel size 1."
        )
    token_multiple = batch_invariant_token_multiple(
        megatron_cfg["tensor_model_parallel_size"]
    )
    if config["make_sequence_length_divisible_by"] % token_multiple:
        out.violations.append(
            "batch_invariant_mode=True requires policy.make_sequence_length_divisible_by "
            f"to be a multiple of {token_multiple} to align with MCore generation "
            f"buckets (got {config['make_sequence_length_divisible_by']})."
        )
    dynamic_batching = config["dynamic_batching"]
    if (
        dynamic_batching["enabled"]
        and dynamic_batching["sequence_length_round"] % token_multiple
    ):
        out.violations.append(
            "batch_invariant_mode=True requires "
            f"policy.dynamic_batching.sequence_length_round to be a multiple of "
            f"{token_multiple} (got {dynamic_batching['sequence_length_round']})."
        )
    if megatron_cfg.get("batch_invariant_backend") == "te_native":
        env_vars = megatron_cfg.get("env_vars") or {}
        for key, value in _TE_NATIVE_ENV_VARS.items():
            if env_vars.get(key) != value:
                out.violations.append(
                    f"batch_invariant_backend='te_native' requires "
                    f"policy.megatron_cfg.env_vars.{key}={value!r} "
                    f"(got {env_vars.get(key)!r})."
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
        if inference_cfg.get("logprobs_mode") != "raw_logprobs":
            out.violations.append(
                "batch_invariant_mode=True requires "
                "policy.generation.mcore_generation_config.logprobs_mode='raw_logprobs' "
                f"(got {inference_cfg.get('logprobs_mode')!r})."
            )
        sampling = TrainingSamplingParams(
            top_k=generation_cfg["top_k"],
            top_p=generation_cfg["top_p"],
            temperature=generation_cfg["temperature"],
        )
        if sampling.temperature != 1.0 or need_top_k_or_top_p_filtering(sampling):
            out.violations.append(
                "batch_invariant_mode=True scores raw logprobs, so it requires "
                "policy.generation.temperature=1.0, top_p=1.0, top_k=null "
                f"(got temperature={sampling.temperature}, top_p={sampling.top_p}, "
                f"top_k={sampling.top_k})."
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


class _GenAlignedLogprob(torch.autograd.Function):
    """Megatron Inference's log-probs forward; the replaced function's backward.

    The forward gathers each sequence chunk's vocab shards (a no-op at TP 1)
    and runs the same full-row log-softmax as Megatron Inference, so it matches
    generation bitwise. The backward re-runs the replaced NeMo-RL function on
    the saved logits and returns its gradient, so for a given output gradient
    it is bitwise the gradient of a run without this mode.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]  Always ignore torch.autograd.Function.forward's type since it's always more specific than the base class
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        chunk_size: int | None,
        group: torch.distributed.ProcessGroup,
        inference_only: bool,
        original: Any,
        original_args: tuple[Any, ...],
    ) -> torch.Tensor:
        # Deferred: Megatron-Core exists only in worker venvs; model_utils is heavy.
        from megatron.core.tensor_parallel.mappings import (
            gather_from_tensor_model_parallel_region,
        )

        from nemo_rl.distributed import model_utils

        seq_len = int(target.shape[1])
        step = chunk_size or seq_len
        out_chunks = []
        for chunk_start in range(0, seq_len, step):
            chunk_end = min(seq_len, chunk_start + step)
            full_logits = gather_from_tensor_model_parallel_region(
                vocab_parallel_logits[:, chunk_start:chunk_end, :], group=group
            )
            out_chunks.append(
                model_utils._tp_target_logprobs(
                    full_logits,
                    target[:, chunk_start:chunk_end],
                    vocab_start_index=0,
                    vocab_end_index=full_logits.shape[-1],
                    tp_group=None,
                    inference_only=True,
                )
            )
            del full_logits

        if not inference_only:
            ctx.save_for_backward(vocab_parallel_logits, target)
            ctx.original = original
            ctx.original_args = original_args
        return torch.cat(out_chunks, dim=1)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[Any, ...]:
        vocab_parallel_logits, target = ctx.saved_tensors
        with torch.enable_grad():
            logits = vocab_parallel_logits.detach().requires_grad_()
            logprobs = ctx.original.apply(logits, target, *ctx.original_args, False)
            (grad_input,) = torch.autograd.grad(logprobs, logits, grad_outputs[0])
        return grad_input, None, None, None, None, None, None


class _FullVocabLogprob:
    """Forward with Megatron Inference's full-row log-softmax; backward unchanged.

    The distributed log-softmax sums per-rank partial normalizers, which rounds
    differently from Megatron Inference's single full-row reduction, so the
    forward all-gathers the vocab-parallel logits first. Gradients come from
    the replaced function (see ``_GenAlignedLogprob``).
    """

    def __init__(self, original: Any, *, chunked: bool) -> None:
        self._original = original
        self._chunked = chunked

    def apply(self, *args: Any) -> Any:
        logits, target, *original_args, inference_only = args
        group = original_args[-1]
        chunk_size = original_args[2] if self._chunked else None
        if group is None:
            return self._original.apply(*args)
        return _GenAlignedLogprob.apply(
            logits,
            target,
            chunk_size,
            group,
            inference_only,
            self._original,
            tuple(original_args),
        ).contiguous()


def _use_full_vocab_log_softmax() -> None:
    # Deferred: model_utils is heavy and only needed once the shim is installed.
    from nemo_rl.distributed import model_utils

    for name, chunked in (
        ("DistributedLogprob", False),
        ("ChunkedDistributedLogprob", True),
    ):
        current = getattr(model_utils, name)
        if not isinstance(current, _FullVocabLogprob):
            setattr(model_utils, name, _FullVocabLogprob(current, chunked=chunked))


def _enable_mcore_deterministic_tuning() -> None:
    """Pin Megatron-Core's Triton kernel configs when MAMBA_DETERMINISTIC=1 is set.

    Megatron-Core with ``megatron.core.tuning`` (NVIDIA/Megatron-LM#6076) no longer
    reads MAMBA_DETERMINISTIC. Its SSM kernels, and the hybrid batch-invariant check
    in TransformerConfig, follow ``set_deterministic_mode``, and ``install`` pins
    every in-scope autotuner without timing, so training and generation pick the
    same config per shape. Megatron's own training sets these from
    --deterministic-mode, which NeMo-RL does not go through. Older Megatron-Core
    reads the variable itself.
    """
    if not os.environ.get("MAMBA_DETERMINISTIC", "").startswith("1"):
        return
    try:
        from megatron.core import tuning
    except ImportError:
        return
    tuning.set_deterministic_mode(True)
    tuning.install(deterministic=True)


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
    _enable_mcore_deterministic_tuning()
    _use_full_vocab_log_softmax()
    print(
        "[zero_train_gen_mismatch] batch-invariant kernels enabled: "
        f"backend={megatron_cfg['batch_invariant_backend']} "
        f"collective={collective} "
        f"flash_attention_version={megatron_cfg['flash_attention_version']} "
        f"CUTE_DSL_LIBS={os.environ.get('CUTE_DSL_LIBS')}",
        flush=True,
    )


def validate_zero_train_gen_kl(
    config: PolicyConfig, *, check_environment: bool
) -> None:
    """Raise if the zero-KL / batch-invariant config is invalid.

    ``check_environment=False`` runs only the config checks, so the driver can
    call it before Ray allocates GPUs; workers also check the installed
    packages and the GPU.
    """
    if "megatron_cfg" not in config:  # Automodel policies
        return
    result = validate_zero_train_gen_mismatch(
        config,
        check_packages=check_environment,
        check_platform=check_environment,
    )
    result.emit_warnings()
    result.raise_if_invalid(
        "policy.megatron_cfg.zero_train_gen_mismatch=true failed validation:"
    )

    bi_result = validate_batch_invariant_mode(config)
    bi_result.emit_warnings()
    bi_result.raise_if_invalid("batch_invariant_mode=True failed validation:")


def _validate_required_values(
    config: PolicyConfig, out: ZeroTrainGenValidation
) -> None:
    """The recipe must set every value the preset needs; nothing is rewritten."""
    megatron_cfg = config["megatron_cfg"]
    for key, value in _ZERO_KL_MEGATRON_REQUIRED.items():
        if megatron_cfg.get(key) != value:
            out.violations.append(
                f"policy.megatron_cfg.{key} must be {value!r} "
                f"(got {megatron_cfg.get(key)!r})."
            )

    generation = config.get("generation")
    if generation is None or generation.get("backend") != "megatron":
        return  # reported by _validate_backend
    mcore_generation_config = generation["mcore_generation_config"]
    for key, value in _ZERO_KL_GENERATION_REQUIRED.items():
        if mcore_generation_config.get(key) != value:
            out.violations.append(
                f"policy.generation.mcore_generation_config.{key} must be "
                f"{value!r} (got {mcore_generation_config.get(key)!r})."
            )


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

    fp8_cfg = inference_cfg.get("fp8_cfg") or {}
    backend = inference_cfg.get("inference_grouped_gemm_backend")
    backend = getattr(backend, "value", backend)
    if (
        impl == "inference_optimized"
        and backend == "flashinfer"
        and fp8_cfg.get("enabled")
        and fp8_cfg.get("fp8_recipe") == "mxfp8"
    ):
        out.violations.append(
            "zero_train_gen_mismatch does not support "
            "inference_grouped_gemm_backend='flashinfer' with MXFP8: the FlashInfer "
            "kernel is batch-invariant but not bitwise identical to TE training. "
            "Use 'torch' or 'vllm'."
        )


def _validate_platform(out: ZeroTrainGenValidation, *, check_device: bool) -> None:
    """Blackwell + FA4 only (mcore extra: flash-attn-4, CuteDSL). Hopper/FA3 not supported."""
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
    """Reject unsupported architectures using the resolved Bridge provider.

    Hybrid Mamba models are supported with MAMBA_DETERMINISTIC=1 and
    TRITON_CACHE_AUTOTUNING=1, from policy.megatron_cfg.env_vars or the launch
    environment. With sequence packing, every packed sequence must also start on a
    Mamba chunk boundary.
    """
    if not config["megatron_cfg"].get("zero_train_gen_mismatch"):
        return

    if getattr(model_cfg, "is_hybrid_model", False) or "M" in (
        getattr(model_cfg, "hybrid_layer_pattern", None) or ""
    ):
        env_vars = config["megatron_cfg"].get("env_vars") or {}
        wrong = {}
        for key, value in _HYBRID_MAMBA_ENV_VARS.items():
            got = env_vars.get(key, os.environ.get(key))
            if got != value:
                wrong[key] = got
        if wrong:
            required = ", ".join(
                f"{k}={v!r}" for k, v in _HYBRID_MAMBA_ENV_VARS.items()
            )
            got = ", ".join(f"{k}={v!r}" for k, v in wrong.items())
            raise ValueError(
                "zero_train_gen_mismatch on a hybrid Mamba model requires "
                f"{required} together in policy.megatron_cfg.env_vars or the launch "
                f"environment (got {got})."
            )
        _validate_mamba_packed_sequence_alignment(config, model_cfg)

    unsupported_fields = [
        field_name
        for field_name in (
            "multi_latent_attention",
            "experimental_attention_variant",
        )
        if getattr(model_cfg, field_name, None)
    ]
    if unsupported_fields:
        raise ValueError(
            "zero_train_gen_mismatch does not support the resolved model "
            f"architecture fields: {', '.join(unsupported_fields)}."
        )


def _validate_mamba_packed_sequence_alignment(
    config: PolicyConfig, model_cfg: Any
) -> None:
    """Packed rows place each sequence at a multiple of make_sequence_length_divisible_by.

    Training's SSM scan cuts fixed chunks from the start of the packed row, while
    generation cuts them from each request's start; the two match only when every
    sequence starts on a chunk boundary.
    """
    if not config["sequence_packing"]["enabled"]:
        return
    chunk_size = model_cfg.mamba_chunk_size
    multiple = math.lcm(
        batch_invariant_token_multiple(
            config["megatron_cfg"]["tensor_model_parallel_size"]
        ),
        chunk_size,
    )
    divisor = config["make_sequence_length_divisible_by"]
    if divisor % multiple:
        raise ValueError(
            "zero_train_gen_mismatch with sequence packing on a hybrid Mamba model "
            f"requires policy.make_sequence_length_divisible_by to be a multiple of "
            f"{multiple} (mamba_chunk_size={chunk_size}), so each packed sequence "
            f"starts on a Mamba chunk boundary (got {divisor})."
        )


def _validate_precision(config: PolicyConfig, out: ZeroTrainGenValidation) -> None:
    """BF16, or FP8 with the MXFP8 recipe; train and generation must agree."""
    if config.get("precision") != "bfloat16":
        out.violations.append(
            "zero_train_gen_mismatch requires policy.precision='bfloat16' "
            f"(got {config.get('precision')!r})."
        )

    sides = [("policy.megatron_cfg", config["megatron_cfg"])]
    generation = config.get("generation")
    if generation is not None and generation.get("backend") == "megatron":
        sides.append(("generation", merged_inference_megatron_cfg(config)))
    kinds: dict[str, str] = {}
    for label, cfg in sides:
        fp8_cfg = cfg.get("fp8_cfg") or {}
        if not fp8_cfg.get("enabled"):
            kinds[label] = "bf16"
            continue
        kinds[label] = f"fp8:{fp8_cfg.get('fp8_recipe')}"
        if fp8_cfg.get("fp8_recipe") != "mxfp8":
            out.violations.append(
                "zero_train_gen_mismatch supports FP8 only with fp8_recipe='mxfp8' "
                f"(got {label}.fp8_cfg.fp8_recipe={fp8_cfg.get('fp8_recipe')!r})."
            )
    if len(set(kinds.values())) > 1:
        out.violations.append(
            "zero_train_gen_mismatch requires the same precision on the train and "
            f"generation side (got {kinds})."
        )
