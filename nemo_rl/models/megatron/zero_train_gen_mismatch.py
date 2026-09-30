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
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

from packaging.version import Version

if TYPE_CHECKING:
    from nemo_rl.models.policy import PolicyConfig

# Batch-invariant inference kernels and the batch_invariant_backend/collective
# knobs this mode configures.
#
# Has to be a commit on main. The previous value came from Megatron-LM PR #6933
# (TE MXFP8 grouped MoE inference), which is still an open PR, so it exists only
# on that branch -- no main-based checkout can satisfy it, whatever its date.
MEGATRON_CORE_MIN_COMMIT_SHA = "b005bf14c46b62169533e755ca2d4e62fe6b7e0a"

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

# Applied on top of the above only when generation selects the FlashInfer
# megakernel. batch_invariant_backend/collective are unchanged and still come
# from _ZERO_KL_MEGATRON_DEFAULTS: mega replaces the expert compute alone, so
# te_native still patches qkv/proj/attention and still swaps the inference
# router onto training's eager top-k, which is what makes per-token log-probs
# match. The 'ordered' collective is inert here -- it configures the NVLS
# dispatcher's cross-rank combine, and mega owns EP transport -- but is kept
# set because validate_batch_invariant_mode requires the field on both sides.
_ZERO_KL_MEGA_MEGATRON_DEFAULTS: dict[str, Any] = {
    # The megakernel hangs off InferenceGroupedMLP, which only the
    # inference_optimized layer spec builds, so a training side left on the
    # default transformer_engine has nothing to call. Set here rather than in
    # the recipe so it cannot come apart from moe_mega_training_forward below.
    "transformer_impl": "inference_optimized",
    # The backend is what selects the megakernel within that expert module, and
    # MCore validates it against moe_mega_training_forward, so the
    # generation-side value under mcore_generation_config is not enough.
    "inference_grouped_gemm_backend": "flashinfer_mega",
}

# Applied only to a training megatron_cfg. A dedicated generation model runs on
# merged_inference_megatron_cfg, which is also a megatron_cfg and is also
# resolved by this module, but has no backward for any of these to serve.
_ZERO_KL_MEGA_TRAIN_ONLY_DEFAULTS: dict[str, Any] = {
    # Without this the training forward runs TE while generation runs the
    # megakernel. They agree only to ~6e-3 relative, which is the mismatch this
    # whole mode exists to remove, so it is forced rather than defaulted.
    "moe_mega_training_forward": True,
    # The mega forward saves no intermediates, so the backward is built from a
    # recompute pass through the ordinary TE path.
    "activation_checkpointing": True,
    "recompute_granularity": "selective",
}
# The mega precisions that can hold zero-KL. Both build the kernel's weights
# themselves, which is what lets a refit reach the kernel instead of leaving
# generation on weights snapshotted at construction, and both run the same
# kernel on the training side, which is what keeps the two forwards bitwise.
# nvfp4 and fp8_fp4 have neither property: FlashInfer preprocesses and snapshots
# their weights, so a refit never lands.
#
# They differ in the backward. bf16 also matches TransformerEngine's gradient to
# rounding. mxfp8 does not -- its gradient comes from the bf16 recompute pass, a
# straight-through estimator -- so it additionally requires
# moe_mega_training_straight_through, which is the recipe saying it accepts that.
_ZERO_KL_MEGA_PRECISIONS: tuple[str, ...] = ("bf16", "mxfp8")
_ZERO_KL_MEGA_DEFAULT_PRECISION = "bf16"


def _requested_mega_precision(config: PolicyConfig) -> str:
    """The megakernel precision the recipe asked for, on either side.

    Read from one place and pushed to both, because the two sides being bitwise
    depends on them running the same kernel at the same precision -- and a
    recipe that set only the generation value would look correct while training
    silently ran bf16.
    """
    from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg

    if config.get("generation") is None:
        # A merged generation config is resolved as a megatron_cfg of its own,
        # and a colocated policy has no generation section; both carry it inline.
        precision = config["megatron_cfg"].get("inference_mega_precision")
    else:
        precision = merged_inference_megatron_cfg(config).get(
            "inference_mega_precision"
        )
    return precision or _ZERO_KL_MEGA_DEFAULT_PRECISION


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
    if mega_backend_selected(config):
        precision = _requested_mega_precision(config)
        defaults.append((mc, "policy.megatron_cfg", _ZERO_KL_MEGA_MEGATRON_DEFAULTS))
        # Pushed to the training side as well as the generation side. MCore
        # validates moe_mega_training_forward against the training config's own
        # precision, so a training side left at the bf16 default would run a
        # different kernel from generation and quietly lose parity.
        defaults.append(
            (mc, "policy.megatron_cfg", {"inference_mega_precision": precision})
        )
        # Guarded like the generation block above: a colocated config, or the
        # merged config a dedicated generation worker is resolved against, may
        # carry no generation section at all.
        if generation is not None:
            defaults.append(
                (
                    generation["mcore_generation_config"],
                    "policy.generation.mcore_generation_config",
                    {"inference_mega_precision": precision},
                )
            )
        if not mc.get("is_inference_model"):
            defaults.append(
                (mc, "policy.megatron_cfg", _ZERO_KL_MEGA_TRAIN_ONLY_DEFAULTS)
            )
            if precision != _ZERO_KL_MEGA_DEFAULT_PRECISION:
                # MCore rejects a quantized mega training forward without this,
                # and rejects the flag itself at bf16, so it cannot be a blanket
                # default on either side.
                defaults.append(
                    (
                        mc,
                        "policy.megatron_cfg",
                        {"moe_mega_training_straight_through": True},
                    )
                )
            # Appended rather than assigned: recompute_modules is a list the
            # recipe may already use for memory, and dropping its entries to add
            # ours would silently raise activation memory.
            modules = list(mc.get("recompute_modules") or [])
            if "moe" not in modules:
                mc["recompute_modules"] = modules + ["moe"]
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
        _validate_packages(out, mega=mega_backend_selected(config))
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


def mega_backend_selected(config: PolicyConfig) -> bool:
    """Whether generation runs the FlashInfer expert-parallel megakernel."""
    from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg

    generation = config.get("generation")
    if generation is None or generation.get("backend") != "megatron":
        return False
    backend = merged_inference_megatron_cfg(config).get(
        "inference_grouped_gemm_backend"
    )
    return getattr(backend, "value", backend) == "flashinfer_mega"


def _validate_mega_backend(
    config: PolicyConfig,
    inference_cfg: Mapping[str, Any],
    out: ZeroTrainGenValidation,
) -> None:
    """Gates specific to inference_grouped_gemm_backend='flashinfer_mega'."""
    megatron_cfg = config["megatron_cfg"]
    # A dedicated generation model is built from merged_inference_megatron_cfg
    # and its workers validate that merge as their megatron_cfg. Reading the
    # train-side knobs off it compares generation values against training
    # requirements, which rejects a correct setup: the merge legitimately
    # carries the generation cuda_graph_impl and no backward at all.
    trains = not megatron_cfg.get("is_inference_model")

    # resolve_zero_train_gen_mismatch forces these, so reaching here with them
    # unset means the caller validated a config it never resolved.
    if trains and not megatron_cfg.get("moe_mega_training_forward"):
        out.violations.append(
            "generation uses inference_grouped_gemm_backend='flashinfer_mega' but "
            "policy.megatron_cfg.moe_mega_training_forward is not set. The training "
            "forward would run TransformerEngine while generation runs the "
            "megakernel; the two agree only to ~6e-3 relative, which is the "
            "train/generation mismatch this mode removes."
        )
    precision = inference_cfg.get("inference_mega_precision")
    if precision not in _ZERO_KL_MEGA_PRECISIONS:
        out.violations.append(
            "zero_train_gen_mismatch requires inference_mega_precision in "
            f"{list(_ZERO_KL_MEGA_PRECISIONS)} (got {precision!r}). FlashInfer "
            "preprocesses and snapshots the other precisions' weights, so a refit "
            "never reaches the kernel and generation keeps sampling from the "
            "previous step's policy with nothing to show for it."
        )
    elif trains:
        # Both sides must run the same kernel at the same precision or the
        # forwards are not bitwise, which is the entire point of this mode.
        train_precision = (
            megatron_cfg.get("inference_mega_precision")
            or _ZERO_KL_MEGA_DEFAULT_PRECISION
        )
        if train_precision != precision:
            out.violations.append(
                "policy.megatron_cfg.inference_mega_precision="
                f"{train_precision!r} does not match generation's {precision!r}. "
                "The training forward and generation would run the megakernel at "
                "different precisions, so their outputs would not be bitwise equal."
            )
        # A quantized forward with a bf16 recomputed backward is a
        # straight-through estimator. The forward stays bitwise, so parity holds;
        # what is given up is gradient fidelity, which belongs in the recipe.
        if precision != _ZERO_KL_MEGA_DEFAULT_PRECISION and not megatron_cfg.get(
            "moe_mega_training_straight_through"
        ):
            out.violations.append(
                f"inference_mega_precision={precision!r} requires policy."
                "megatron_cfg.moe_mega_training_straight_through=true. The mega "
                "forward runs quantized while the backward comes from the bf16 "
                "recompute pass, so the gradient is that of the bf16 function -- a "
                "straight-through estimator, with the convergence consequences "
                "that implies."
            )
    if trains and (
        megatron_cfg.get("recompute_granularity") != "selective"
        or "moe" not in (megatron_cfg.get("recompute_modules") or [])
    ):
        out.violations.append(
            "moe_mega_training_forward requires policy.megatron_cfg."
            "recompute_granularity='selective' with 'moe' in recompute_modules: the "
            "mega forward saves no intermediates, so the backward is built from the "
            "recompute pass."
        )
    # Local CUDA graphs capture the MoE layer as one graph, which disables the
    # layer-level recompute the backward is built from. Generation is free to
    # graph: it has no backward, and its megakernel weights are repacked in
    # place on refit so replay sees them.
    if trains and megatron_cfg.get("cuda_graph_impl") == "local":
        out.violations.append(
            "moe_mega_training_forward is incompatible with "
            "policy.megatron_cfg.cuda_graph_impl='local'; use 'none' on the "
            "training side."
        )
    # The kernel raises instead of falling back when a rank exceeds the cap, so
    # an unset value is a latent prefill-time crash rather than a slow path.
    if not inference_cfg.get("inference_mega_max_tokens_per_rank"):
        out.violations.append(
            "inference_grouped_gemm_backend='flashinfer_mega' requires "
            "inference_mega_max_tokens_per_rank. It is a hard workspace cap: the "
            "kernel rejects a forward with more local tokens per EP rank, so it "
            "must cover the widest prefill, not just the decode width."
        )


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

    fp8_cfg = inference_cfg.get("fp8_cfg") or {}
    grouped_gemm_backend = inference_cfg.get("inference_grouped_gemm_backend")
    grouped_gemm_backend = getattr(grouped_gemm_backend, "value", grouped_gemm_backend)
    if (
        impl == "inference_optimized"
        and grouped_gemm_backend == "flashinfer"
        and fp8_cfg.get("enabled")
        and fp8_cfg.get("fp8_recipe") == "mxfp8"
    ):
        out.violations.append(
            "zero_train_gen_mismatch does not support "
            "inference_grouped_gemm_backend='flashinfer' with MXFP8: the FlashInfer "
            "kernel is batch-invariant but not bitwise identical to TE training. "
            "Use the Torch or vLLM MXFP8 exact-parity path."
        )
    if grouped_gemm_backend == "flashinfer_mega":
        _validate_mega_backend(config, inference_cfg, out)


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


def _validate_packages(out: ZeroTrainGenValidation, *, mega: bool = False) -> None:
    _validate_megatron_core_commit(MEGATRON_CORE_MIN_COMMIT_SHA, out)
    if mega:
        # No second commit gate for mega. The integration is not upstream yet,
        # so any SHA to pin would be a local one, which passes trivially on the
        # tree it was read from and fails everywhere else. The import check
        # below is what actually establishes the kernel is available.
        #
        # flashinfer.moe_ep ships in no published flashinfer release; the mcore
        # extra takes it from a git rev. Checked by import rather than by
        # version because the git build reports whatever version it was cut
        # from, which says nothing about whether the module is present.
        try:
            import flashinfer.moe_ep  # noqa: F401
        except ImportError as exc:
            out.violations.append(
                "inference_grouped_gemm_backend='flashinfer_mega' requires "
                f"flashinfer.moe_ep, which failed to import ({exc}). No released "
                "flashinfer wheel contains it; install the git rev pinned by the "
                "mcore extra in pyproject.toml."
            )

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
            f"flash-attn-4>={FLASH_ATTN_4_MIN_VERSION} required (got {fa4})."
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
