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

import warnings
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Optional, assert_never

from pydantic import BaseModel, BeforeValidator, Field


class ClippedPGLossConfig(BaseModel, extra="allow"):
    # Tag that selects this config among the policy-gradient loss configs (see
    # PolicyGradientLossConfig). Optional: an untagged ``loss_fn`` block is this
    # config. Excluded from dumps so serialized configs, and the checkpoint
    # fingerprints derived from them, are unchanged.
    kind: Literal["legacy_pg"] = Field(default="legacy_pg", exclude=True)

    # --- Loss type ---
    disable_ppo_ratio: bool = False
    token_level_loss: bool = True
    # If True, apply the off-policy importance-sampling correction at the
    # sequence level (one weight per generated sample), as in GSPO.
    # If False (default), correction is applied at the token level as in the
    # original GRPO paper.
    sequence_level_importance_ratios: bool = False

    # --- Clipping ---
    ratio_clip_min: float = 0.2
    ratio_clip_max: float = 0.2
    # Dual-clipping value (should be >1 if enabled; usually set to 3 empirically). None to disable.
    ratio_clip_c: Optional[float] = None

    # --- KL regularization ---
    reference_policy_kl_penalty: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    # Can be set to k1, k2, k3
    # For more details, see http://joschu.net/blog/kl-approx.html
    reference_policy_kl_type: Literal["k1", "k2", "k3"] = "k3"
    kl_input_clamp_value: Optional[float] = 20.0
    kl_output_clamp_value: Optional[float] = 10.0
    # If True, add KL penalty to reward instead of loss (used by Reinforce++)
    use_kl_in_reward: bool = False

    # --- Importance sampling correction ---
    # Async GRPO requires importance sampling correction enabled
    # Set to true when async_grpo.enabled is true
    use_importance_sampling_correction: bool = False
    # --- Truncated importance sampling ---
    # Type of truncated importance sampling:
    #   "tis"          – clamp IS weights to [min, max], where min defaults to 0
    #   "icepop"       – zero out tokens with IS weight outside [min, max]
    #   "adaptive-ppo-clip" – advantage-aware token filtering with an
    #                         entropy-adaptive upper bound
    #   "seq-mask-tis" – zero out sequences by geometric-mean IS ratio, non-truncated token IS correction
    truncated_importance_sampling_type: Optional[
        Literal["tis", "icepop", "adaptive-ppo-clip", "seq-mask-tis"]
    ] = None
    truncated_importance_sampling_ratio: Optional[float] = None
    # Lower bound for TIS clipping, or ICE-POP / adaptive-ppo-clip /
    # seq-mask-tis filtering
    truncated_importance_sampling_ratio_min: Optional[float] = None
    # adaptive-ppo-clip only; ignored by the other truncated importance
    # sampling types. The positive-advantage upper bound is
    # truncated_importance_sampling_ratio + k. After each training step, k rises
    # by k_step if approx_entropy is below the target entropy and falls by k_step
    # if it is above, clamped to [0, k_max]. Tokens with IS weight above
    # outer_bound are dropped regardless of advantage sign.
    truncated_importance_sampling_adaptive_target_entropy: float = Field(
        default=0.3, ge=0.0
    )
    truncated_importance_sampling_adaptive_k_step: float = Field(default=0.25, gt=0.0)
    truncated_importance_sampling_adaptive_k_max: float = Field(default=1.0, ge=0.0)
    truncated_importance_sampling_adaptive_outer_bound: float = Field(
        default=50.0, gt=0.0
    )

    # --- On-policy ---
    # (default off) loss formulation improvements (docs/guides/grpo.md#loss)
    use_on_policy_kl_approximation: bool = False
    # If True, force the PPO ratio to 1.0 while preserving its gradient.
    # Generation-to-trainer importance sampling remains independently enabled.
    # NOTE: This should only be used when doing exactly one update per rollout
    # (i.e., num_prompts_per_step * num_generations_per_prompt == train_global_batch_size)
    force_on_policy_ratio: bool = False
    # Evaluate the algorithm's seq_logprob_error_threshold in the training loss, then
    # normalize accumulated gradients over survivors before the optimizer step.
    # Opt-in; supported only for token-level, force-on-policy Megatron GRPO
    # (non-streaming or SingleController) and SingleController PPO with ppo_epochs=1.
    seq_logprob_error_in_loss: bool = False
    # If True, use CISPO (Clipped IS-weight Policy Optimization) from MiniMax-M1.
    use_cispo: bool = False
    # VAPO: weight μ for positive-example NLL loss on correct samples.
    # L = L_PPO + μ·L_NLL(correct)   (arXiv:2504.05118, Eq. 10)
    # Set to 0 to disable.
    positive_example_nll_weight: float = 0.0
    # Diagnostic metrics are useful for monitoring, but are not needed for
    # the objective. ``full`` preserves the historical metric surface;
    # ``minimal`` keeps only objective/importance-sampling bookkeeping.
    metrics_level: Literal["full", "minimal"] = "full"
    # Compile the tensor-only actor objective when supported by the active
    # backend. The eager implementation remains the default path.
    enable_torch_compile: bool = False
    # Score centering (arXiv:2609.20807): per token, add A * sum_v sg(c_v) log p_v
    # over the sampler's top-k ids so the update subtracts the sampler-expected
    # weighted score. Needs policy.generation.top_logprobs > 0; supports IS off,
    # token-level untruncated IS, TIS, ICEPOP and adaptive-ppo-clip.
    score_centering: bool = False


class CISPOLossConfig(BaseModel, extra="forbid"):
    """CISPO (https://arxiv.org/abs/2506.13585), selected with ``kind: cispo``.

    Carries only the settings CISPO supports, so a setting of another objective
    (``disable_ppo_ratio``, ``sequence_level_importance_ratios``,
    ``ratio_clip_c``) is rejected when the config is loaded. Field names,
    defaults and meanings match ClippedPGLossConfig, which documents them.
    """

    kind: Literal["cispo"] = "cispo"
    # Fixed by the objective. Kept as fields so trainer code that reads these
    # switches, and configs that still spell them out, keep working.
    use_cispo: Literal[True] = True
    token_level_loss: Literal[True] = True

    # --- Clipping ---
    ratio_clip_min: float = 0.2
    ratio_clip_max: float = 0.2

    # --- KL regularization ---
    reference_policy_kl_penalty: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    reference_policy_kl_type: Literal["k1", "k2", "k3"] = "k3"
    kl_input_clamp_value: float | None = 20.0
    kl_output_clamp_value: float | None = 10.0
    use_kl_in_reward: bool = False
    use_on_policy_kl_approximation: bool = False

    # --- Importance sampling correction ---
    use_importance_sampling_correction: bool = False
    truncated_importance_sampling_ratio: float | None = None
    truncated_importance_sampling_ratio_min: float | None = None
    truncated_importance_sampling_type: (
        Literal["tis", "icepop", "adaptive-ppo-clip", "seq-mask-tis"] | None
    ) = None
    truncated_importance_sampling_adaptive_target_entropy: float = Field(
        default=0.3, ge=0.0
    )
    truncated_importance_sampling_adaptive_k_step: float = Field(default=0.25, gt=0.0)
    truncated_importance_sampling_adaptive_k_max: float = Field(default=1.0, ge=0.0)
    truncated_importance_sampling_adaptive_outer_bound: float = Field(
        default=50.0, gt=0.0
    )

    # --- On-policy ---
    force_on_policy_ratio: bool = False
    seq_logprob_error_in_loss: bool = False

    positive_example_nll_weight: float = 0.0
    metrics_level: Literal["full", "minimal"] = "full"
    enable_torch_compile: bool = False
    score_centering: bool = False

    def as_clipped_pg_config(self) -> ClippedPGLossConfig:
        """Express these settings as the config of the shared loss implementation."""
        return ClippedPGLossConfig.model_validate(self.model_dump(exclude={"kind"}))


# Switches of ClippedPGLossConfig that select an objective, and the kind that
# replaces each. They keep working, with a warning, until they are removed.
_DEPRECATED_SWITCHES = {
    "use_cispo": "kind: cispo",
}


def _select_legacy_config(value: Any) -> Any:
    """Read a ``loss_fn`` block that names no kind as ClippedPGLossConfig."""
    if not isinstance(value, Mapping):
        return value
    if "kind" in value:
        return value
    for switch, replacement in _DEPRECATED_SWITCHES.items():
        if value.get(switch):
            warnings.warn(
                f"loss_fn.{switch} is deprecated and will be removed in a future "
                f"release. Select the objective with loss_fn.{replacement}.",
                FutureWarning,
            )
    return {"kind": "legacy_pg", **value}


# The ``loss_fn`` block of the policy-gradient trainers, discriminated on
# ``kind`` so each objective-specific config carries only the settings its
# objective supports. An untagged block stays the all-in-one
# ClippedPGLossConfig, so existing configs load unchanged.
PolicyGradientLossConfig = Annotated[
    ClippedPGLossConfig | CISPOLossConfig,
    Field(discriminator="kind"),
    BeforeValidator(_select_legacy_config),
]


def as_clipped_pg_config(cfg: PolicyGradientLossConfig) -> ClippedPGLossConfig:
    """Return the settings of any loss config that ClippedPGLossFn implements."""
    match cfg:
        case CISPOLossConfig():
            return cfg.as_clipped_pg_config()
        case ClippedPGLossConfig():
            return cfg
        case _:
            assert_never(cfg)
