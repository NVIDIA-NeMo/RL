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
"""Resolution and validation of dLLM policy configuration.

Masked diffusion policies are incompatible with most of the machinery that
assumes causal attention and an exact token-level likelihood. Every such
combination is rejected here, at setup, rather than being allowed to run and
quietly produce a wrong likelihood -- which is the failure mode that matters,
because a subtly wrong ELBO still trains and still logs a plausible reward.
"""

from typing import Any, Optional

from nemo_rl.models.policy.dllm.config import DllmConfig


def dllm_config_from_policy(policy_cfg: Any) -> Optional[DllmConfig]:
    """Resolves the dLLM config from a policy config, or None if not in use.

    ``PolicyConfig`` is a ``TypedDict``, so the nested block arrives from
    omegaconf as a plain dict rather than a validated model. This is the single
    place that coercion happens, so defaults come from :class:`DllmConfig`'s
    fields instead of being re-invented at each call site.

    Args:
        policy_cfg: The policy config section, as loaded from YAML.

    Returns:
        The validated :class:`DllmConfig` when the block is present and enabled,
        otherwise ``None``.
    """
    raw = policy_cfg.get("dllm")
    if raw is None:
        return None
    cfg = raw if isinstance(raw, DllmConfig) else DllmConfig.model_validate(raw)
    return cfg if cfg.enabled else None


def validate_dllm_policy(policy_cfg: Any, loss_cfg: Any, grpo_cfg: Any = None) -> None:
    """Rejects dLLM configurations that would train against a wrong likelihood.

    Args:
        policy_cfg: The policy config section.
        loss_cfg: The ``loss_fn`` config section (a ``ClippedPGLossConfig`` or
            the equivalent mapping).
        grpo_cfg: The ``grpo`` config section, when available. Only used to
            reject async rollouts.

    Raises:
        ValueError: If the dLLM policy is combined with a feature that assumes
            causal attention, or with a loss configuration that is not the
            sequence-level objective the ELBO is meaningful under.
    """
    if dllm_config_from_policy(policy_cfg) is None:
        return

    def _enabled(section: str) -> bool:
        block = policy_cfg.get(section)
        return bool(block) and bool(block.get("enabled"))

    # Each of these shards, reorders, or reuses tokens in a way that assumes a
    # causal, per-token factorization. None of them are meaningful for a
    # bidirectional model scored at randomly masked positions.
    causal_only = {
        "sequence_packing": _enabled("sequence_packing"),
        "dynamic_batching": _enabled("dynamic_batching"),
        "megatron_cfg": _enabled("megatron_cfg"),
        # Router replay assumes one stable token->expert map per rollout. A dLLM
        # re-routes every position on every denoising step, so there is no
        # single map to replay.
        "router_replay": _enabled("router_replay"),
    }
    for name, is_on in causal_only.items():
        if is_on:
            raise ValueError(
                f"policy.{name} is not supported with policy.dllm.enabled=true: "
                "masked diffusion models are bidirectional and are scored at "
                f"masked positions, which {name} assumes away. Disable it."
            )

    dtensor_cfg = policy_cfg.get("dtensor_cfg")
    if dtensor_cfg and dtensor_cfg.get("context_parallel_size", 1) > 1:
        raise ValueError(
            "policy.dtensor_cfg.context_parallel_size > 1 is not supported with "
            "policy.dllm.enabled=true: context parallelism shards the sequence, "
            "but the ELBO masks positions across the whole sequence. Set it to 1."
        )

    # The dllm backend has no async engine, so
    # nemo_rl.models.generation.interfaces.should_use_async_rollouts returns
    # False for it. Asking for async_grpo would therefore run plain synchronous
    # rollouts and silently ignore every async setting.
    if grpo_cfg is not None:
        async_cfg = getattr(grpo_cfg, "async_grpo", None)
        if async_cfg is None and hasattr(grpo_cfg, "get"):
            async_cfg = grpo_cfg.get("async_grpo")
        enabled = getattr(async_cfg, "enabled", None)
        if enabled is None and hasattr(async_cfg, "get"):
            enabled = async_cfg.get("enabled")
        if enabled:
            raise ValueError(
                "grpo.async_grpo.enabled is not supported with "
                "policy.dllm.enabled=true: the dllm generation backend has no "
                "async engine, so rollouts would silently run synchronously. "
                "Set grpo.async_grpo.enabled=false."
            )
        seq_error_threshold = getattr(grpo_cfg, "seq_logprob_error_threshold", None)
        if seq_error_threshold is None and hasattr(grpo_cfg, "get"):
            seq_error_threshold = grpo_cfg.get("seq_logprob_error_threshold")
        if seq_error_threshold is not None:
            raise ValueError(
                "grpo.seq_logprob_error_threshold is not supported with "
                "policy.dllm.enabled=true: denoising does not expose generation-time "
                "log probabilities to compare with policy ELBOs. Set it to null."
            )

    dllm_cfg = dllm_config_from_policy(policy_cfg)
    _validate_dllm_generation(policy_cfg.get("generation"), dllm_cfg)
    _validate_dllm_loss(loss_cfg, dllm_cfg)


def _validate_dllm_generation(generation_cfg: Any, dllm_cfg: Any) -> None:
    """Rejects rollout settings the denoising sampler cannot honor."""
    if generation_cfg is None:
        return

    backend = generation_cfg.get("backend")
    if backend != "dllm":
        raise ValueError(
            f"policy.generation.backend is '{backend}', but policy.dllm.enabled "
            "is true. Masked diffusion models decode a fixed-width canvas and "
            "have no KV cache. SGLang serves LLaDA2.0 and SDAR, but not "
            "LLaDA-8B, and vLLM and TRT-LLM serve no diffusion models. "
            "Set policy.generation.backend='dllm'."
        )

    if not generation_cfg.get("colocated", {}).get("enabled", True):
        raise ValueError(
            "policy.generation.colocated.enabled must be true with "
            "policy.generation.backend='dllm': rollouts run in the training "
            "workers, so there is no separate inference cluster to place."
        )

    if generation_cfg.get("stop_strings"):
        raise ValueError(
            "policy.generation.stop_strings is not supported with "
            "policy.generation.backend='dllm': iterative denoising stops on token "
            "ids, and cannot incrementally match decoded strings. Set it to null "
            "and use stop_token_ids."
        )

    max_new_tokens = generation_cfg.get("max_new_tokens")
    if max_new_tokens is not None and max_new_tokens % dllm_cfg.block_length != 0:
        raise ValueError(
            f"policy.generation.max_new_tokens ({max_new_tokens}) must be a "
            f"multiple of policy.dllm.block_length ({dllm_cfg.block_length}): "
            "the generation region is denoised in whole blocks."
        )


def _validate_dllm_loss(loss_cfg: Any, dllm_cfg: Any) -> None:
    """Rejects loss settings the ELBO cannot be substituted into."""

    def _get(key: str, default: Any) -> Any:
        if hasattr(loss_cfg, key):
            return getattr(loss_cfg, key)
        if hasattr(loss_cfg, "get"):
            return loss_cfg.get(key, default)
        return default

    # The ELBO is a *sequence* likelihood: only its masked sum is meaningful, so
    # only a sequence-level ratio is well defined. A token-level ratio would
    # treat each position's ELBO contribution as its own likelihood, which it is
    # not -- positions not masked at a given quadrature point contribute zero.
    if not _get("sequence_level_importance_ratios", False):
        raise ValueError(
            "policy.dllm.enabled=true requires "
            "loss_fn.sequence_level_importance_ratios=true: the ELBO is a "
            "sequence-level likelihood, so per-token importance ratios are not "
            "well defined for it."
        )
    if _get("token_level_loss", True):
        raise ValueError(
            "policy.dllm.enabled=true requires loss_fn.token_level_loss=false, "
            "to match the sequence-level ratio (the GSPO-style objective GDPO "
            "builds on)."
        )
    # The correction divides by the generation-time token logprobs, which
    # iterative denoising does not produce. They are zero-filled, so leaving
    # this on would silently apply exp(prev - 0) as an importance weight.
    if _get("use_importance_sampling_correction", False):
        raise ValueError(
            "loss_fn.use_importance_sampling_correction=true is not supported "
            "with policy.dllm.enabled=true: denoising rollouts produce no "
            "per-token sampling log probabilities to correct against."
        )
    if _get("reference_policy_kl_penalty", 0.0) != 0:
        raise ValueError(
            "loss_fn.reference_policy_kl_penalty must be 0 with "
            "policy.dllm.enabled=true: the existing KL estimator expects token "
            "log probabilities, not per-position ELBO contributions."
        )
    if _get("use_kl_in_reward", False):
        raise ValueError(
            "loss_fn.use_kl_in_reward is not supported with "
            "policy.dllm.enabled=true: reward KL requires token log probabilities, "
            "not per-position ELBO contributions."
        )
    # The policy decides whether its log probabilities are position-aligned; the
    # loss decides whether to drop a column off every other per-token tensor to
    # match. Disagreement is an off-by-one at every position that still runs, so
    # require them to be stated consistently rather than inferring one silently.
    position_aligned = _get("position_aligned_logprobs", False)
    if position_aligned == dllm_cfg.shift_targets:
        raise ValueError(
            f"loss_fn.position_aligned_logprobs={position_aligned} is "
            f"inconsistent with policy.dllm.shift_targets="
            f"{dllm_cfg.shift_targets}. A policy that scores token i at "
            "position i (shift_targets=false) emits log probabilities as long "
            "as the sequence, which the loss must not shift "
            "(position_aligned_logprobs=true), and vice versa."
        )
