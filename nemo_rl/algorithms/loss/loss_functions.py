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
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    Optional,
    TypedDict,
    TypeVar,
)

import torch
from pydantic import BaseModel, Field, model_validator

from nemo_rl.algorithms.loss.clipped_pg import (
    adaptive_ppo_clip_kept_mask,
    clipped_pg_actor_objective,
    clipped_pg_diagnostic_metrics,
    materialize_scalar_metrics,
)
from nemo_rl.algorithms.loss.draft import (
    DEFAULT_DRAFT_TOKEN_CHUNK_SIZE,
    DraftLossStats,
    streaming_vocab_parallel_soft_ce,
)
from nemo_rl.algorithms.loss.interfaces import (
    SEQ_LOGPROB_KEEP_MASK_KEY,
    LossFunction,
    LossInputType,
    LossType,
    MetricNormalizer,
    make_objective_reduction,
)
from nemo_rl.algorithms.loss.score_centering import compute_score_centering_term
from nemo_rl.algorithms.utils import (
    calculate_kl,
    compute_seq_logprob_errors,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.telemetry.vocabulary import TeedMetric, register_teed_metrics
from nemo_rl.utils.outdated_config_checks import reject_outdated_token_level_loss
from nemo_rl.utils.tensor_ops import masked_mean

if TYPE_CHECKING:
    # Import-time only: nemo_rl.algorithms.opd imports the data plane, which
    # would pull a heavy dependency chain into every loss-function consumer.
    from nemo_rl.algorithms.opd import OnPolicyDistillationFullConfig

Tensor = TypeVar("Tensor", bound=torch.Tensor)

KL_PENALTY_KEY = "kl_penalty"
APPROX_ENTROPY_KEY = "approx_entropy"

#: Teed rows for the two metrics the losses below compute. The kl_penalty key
#: is named for the penalty, but the value has the coefficient divided back
#: out, so the series it feeds really is the divergence.
LOSS_TEED_METRICS = (
    TeedMetric(
        KL_PENALTY_KEY,
        "rl.kl.divergence",
        description="KL divergence from the reference policy.",
    ),
    TeedMetric(
        APPROX_ENTROPY_KEY,
        "rl.entropy",
        description="Approximate policy entropy.",
    ),
)

register_teed_metrics(LOSS_TEED_METRICS)


class DraftCrossEntropyLossConfig(TypedDict):
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup]
    # Tokens per FP32 vocab tile in the streaming soft-CE reduction. Defaults to
    # DEFAULT_DRAFT_TOKEN_CHUNK_SIZE; see policy.draft.token_chunk_size.
    token_chunk_size: NotRequired[int]


class DraftCrossEntropyLossDataDict(TypedDict):
    teacher_logits: Tensor
    student_logits: Tensor
    token_mask: Tensor
    sample_mask: Tensor
    student_vocab_indices: NotRequired[Tensor]


class DraftCrossEntropyLossFn(LossFunction):
    """Compute the auxiliary soft-target cross-entropy used for draft-model training."""

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.DRAFT

    def __init__(
        self,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        token_chunk_size: int = DEFAULT_DRAFT_TOKEN_CHUNK_SIZE,
    ):
        self.vocab_parallel_group = vocab_parallel_group
        self.token_chunk_size = token_chunk_size

    def __call__(
        self,
        teacher_logits: Tensor,
        student_logits: Tensor,
        token_mask: Tensor,
        data: BatchedDataDict[DraftCrossEntropyLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce the masked per-token draft loss to a scalar."""
        stats = self.loss_stats(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            token_mask=token_mask,
            data=data,
        )
        return stats.normalized(
            normalization_counts=global_valid_toks.reshape(1),
        )

    def loss_stats(
        self,
        teacher_logits: Tensor,
        student_logits: Tensor,
        token_mask: Tensor,
        data: BatchedDataDict[DraftCrossEntropyLossDataDict],
    ) -> DraftLossStats:
        """Return raw one-bin EAGLE statistics for deferred normalization."""
        mask = token_mask * data["sample_mask"].unsqueeze(-1)
        return streaming_vocab_parallel_soft_ce(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            token_chunk_size=self.token_chunk_size,
            tp_group=self.vocab_parallel_group,
        )


LossNormalization = Literal["global_token_mean", "sequence_mean", "group_token_mean"]


class ClippedPGLossConfig(BaseModel, extra="allow"):
    # --- Loss type ---
    disable_ppo_ratio: bool = False
    # None selects global_token_mean.
    # group_token_mean requires SingleController GRPO with Megatron.
    normalization: LossNormalization | None = None
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

    @model_validator(mode="after")
    def reject_removed_loss_selector(self) -> "ClippedPGLossConfig":
        reject_outdated_token_level_loss(self.model_extra or {})
        return self

    @property
    def resolved_normalization(self) -> LossNormalization:
        """Return the selected mode, treating None as global_token_mean."""
        return self.normalization or "global_token_mean"


class ClippedPGLossDataDict(TypedDict):
    """Required keys for the Clipped Policy Gradient loss function."""

    input_ids: torch.Tensor
    advantages: torch.Tensor
    prev_logprobs: torch.Tensor
    generation_logprobs: torch.Tensor
    reference_policy_logprobs: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
    sampler_topk_token_ids: NotRequired[torch.Tensor]
    sampler_topk_logprobs: NotRequired[torch.Tensor]
    group_token_mean_inverse_tokens: NotRequired[torch.Tensor]
    __extra__: Any


class ClippedPGLossFn(LossFunction):
    """Generalized Clipped Policy Gradient loss function w/ KL regularization.

    This implements:

    - PPO (Clipped) - https://arxiv.org/abs/1707.06347
    - GRPO - https://arxiv.org/abs/2402.03300
    - REINFORCE/RLOO (set disable_ppo_ratio = True and ignores ratio_clip_min/ratio_clip_max) - https://arxiv.org/abs/2402.14740
    - GSPO (set sequence_level_importance_ratios = True and normalization = "sequence_mean") - https://arxiv.org/abs/2507.18071
    - CISPO (set use_cispo = True) - https://arxiv.org/abs/2506.13585
    - Truly on-policy (set force_on_policy_ratio = True to force ratio = 1.0, requires one update per rollout)

    Formula:
    L(θ) = E_t [ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ] - β * KL(π_θ || π_ref)

    where:
    - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio
    - A_t is the advantage estimate
    - ε is the clip parameter (ratio_clip_min/ratio_clip_max)
        - As proposed in the DAPO paper (https://arxiv.org/pdf/2503.14476),
          we allow setting a distinct minimum and maximum value for the clip parameter (set to the same value for PPO/GRPO/etc.)
            - ratio_clip_min: minimum value for the clip parameter
            - ratio_clip_max: maximum value for the clip parameter
    - β is the KL penalty coefficient (reference_policy_kl_penalty)
    - KL(π_θ || π_ref) is the KL divergence between the current policy and reference policy (Schulman Approx.)

    For REINFORCE/RLOO (when disable_ppo_ratio=True), the formula simplifies to:
    L(θ) = E_t [ log π_θ(a_t|s_t) * A_t ] - β * KL(π_θ || π_ref)

    Formula (CISPO):
    L(θ) = E_t [ sg(clip(r_t(θ), 1-ε_low, 1+ε_high)) * A_t * log π_θ(a_t|s_t) ]


    Also supports "Dual-Clipping" from https://arxiv.org/pdf/1912.09729, which
    imposes an additional upper bound on the probability ratio when advantages are negative.
    This prevents excessive policy updates. $rA << 0$ -> $cA$(clipped)
    The loss function is modified to the following when A_t < 0:
    L(θ) = E_t [ max(min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t), c * A_t) ] - β * KL(π_θ || π_ref)

    where:
    - c is the dual-clip parameter (ratio_clip_c), which must be greater than 1 and is
      usually set as 3 empirically.

    Due to potential numerical instability, we cast the logits to float32 before computing the loss.
    """

    input_type = LossInputType.LOGPROB

    def __init__(
        self,
        cfg: ClippedPGLossConfig,
        use_fused_linear_logprobs: bool = False,
        opd_full: Optional["OnPolicyDistillationFullConfig"] = None,
        *,
        seq_logprob_error_threshold: float | None = None,
    ):
        """Initialize the loss and its worker normalization requirements.

        Args:
            cfg: Policy-gradient loss configuration.
            use_fused_linear_logprobs: Whether the model returns precomputed
                next-token logprobs instead of logits.
            opd_full: Optional full-vocabulary distillation configuration.
            seq_logprob_error_threshold: Required when
                ``cfg.seq_logprob_error_in_loss`` is enabled; otherwise unused.
                The loss emits survivor counts but normalizes by
                the original global counts. The worker must aggregate survivor
                counts across all microbatches and data-parallel ranks, then
                rescale gradients and normalized metrics before clipping or
                stepping the optimizer. See ``requires_survivor_normalization``.
        """
        self.normalization = cfg.resolved_normalization
        self.seq_logprob_error_in_loss = cfg.seq_logprob_error_in_loss
        self.seq_logprob_error_threshold = seq_logprob_error_threshold
        if self.seq_logprob_error_in_loss:
            if seq_logprob_error_threshold is None:
                raise ValueError(
                    "loss_fn.seq_logprob_error_in_loss requires seq_logprob_error_threshold"
                )
            if (
                not cfg.force_on_policy_ratio
                or self.normalization != "global_token_mean"
            ):
                raise ValueError(
                    "In-loss sequence filtering requires force_on_policy_ratio "
                    "and normalization=global_token_mean"
                )
            if cfg.positive_example_nll_weight != 0:
                raise ValueError("In-loss sequence filtering does not support NLL")
        # When True, the model forward is patched to return precomputed next-token
        # logprobs (via chunked linear CE fusion) instead of full logits. This is
        # consumed by prepare_loss_input, which short-circuits the logits->logprobs
        # conversion. See nemo_rl/distributed/model_utils.py for the fused forward.
        self.use_fused_linear_logprobs = use_fused_linear_logprobs
        self.opd_full = opd_full if opd_full is not None and opd_full.enabled else None
        self.input_type = (
            LossInputType.OPD_FULL
            if self.opd_full is not None
            else LossInputType.LOGPROB
        )
        if self.opd_full is not None:
            _validate_opd_full_loss_config(cfg, use_fused_linear_logprobs)
        self.disable_ppo_ratio = cfg.disable_ppo_ratio
        self.ratio_clip_min = cfg.ratio_clip_min
        self.ratio_clip_max = cfg.ratio_clip_max
        self.ratio_clip_c = cfg.ratio_clip_c  # set to None to disable dual-clipping
        if self.ratio_clip_c is not None:
            assert self.ratio_clip_c > 1, (
                f"ratio_clip_c must exceed 1 representing a lower bound of the ratios, got {self.ratio_clip_c}."
            )
        self.reference_policy_kl_penalty = (
            cfg.reference_policy_kl_penalty if not cfg.use_kl_in_reward else 0
        )
        self.reference_policy_kl_type = cfg.reference_policy_kl_type
        self.kl_input_clamp_value = cfg.kl_input_clamp_value
        self.kl_output_clamp_value = cfg.kl_output_clamp_value
        self.use_importance_sampling_correction = cfg.use_importance_sampling_correction
        # Type of truncated importance sampling:
        # "tis" | "icepop" | "adaptive-ppo-clip" | "seq-mask-tis"
        self.truncated_importance_sampling_type = cfg.truncated_importance_sampling_type
        self.truncated_importance_sampling_ratio = (
            cfg.truncated_importance_sampling_ratio
        )
        # Lower bound for TIS clipping, or ICE-POP / adaptive-ppo-clip /
        # seq-mask-tis filtering
        self.truncated_importance_sampling_ratio_min = (
            cfg.truncated_importance_sampling_ratio_min
        )
        # adaptive-ppo-clip controller settings
        self.truncated_importance_sampling_adaptive_target_entropy = (
            cfg.truncated_importance_sampling_adaptive_target_entropy
        )
        self.truncated_importance_sampling_adaptive_k_step = (
            cfg.truncated_importance_sampling_adaptive_k_step
        )
        self.truncated_importance_sampling_adaptive_k_max = (
            cfg.truncated_importance_sampling_adaptive_k_max
        )
        self.truncated_importance_sampling_adaptive_outer_bound = (
            cfg.truncated_importance_sampling_adaptive_outer_bound
        )
        self.use_on_policy_kl_approximation = cfg.use_on_policy_kl_approximation
        self.force_on_policy_ratio = cfg.force_on_policy_ratio  # Force ratio to 1.0

        # Whether to compute importance weights per-sequence instead of per-token.
        self.sequence_level_importance_ratios = cfg.sequence_level_importance_ratios
        self.positive_example_nll_weight = cfg.positive_example_nll_weight
        self.metrics_level = cfg.metrics_level
        self.enable_torch_compile = cfg.enable_torch_compile
        # Compile the actor objective lazily on the worker after Ray has
        # serialized the loss function.
        self._compiled_actor_objective = None
        self.objective_reduction = make_objective_reduction(self.normalization)
        self.loss_type = self.objective_reduction.loss_type
        # Authoritative check for every construction path (legacy trainers, tests);
        # validate_single_controller_config repeats it to fail before cluster start.
        self.objective_reduction.validate_loss_options(
            sequence_level_importance_ratios=cfg.sequence_level_importance_ratios,
            opd_full=self.opd_full is not None,
            positive_example_nll_weight=cfg.positive_example_nll_weight,
            seq_logprob_error_in_loss=cfg.seq_logprob_error_in_loss,
        )
        if self.sequence_level_importance_ratios:
            assert self.loss_type == LossType.SEQUENCE_LEVEL, (
                "sequence-level importance sampling (e.g. GSPO) is mutually exclusive with token-level loss"
            )

        self.use_cispo = cfg.use_cispo
        if self.use_cispo:
            assert not self.disable_ppo_ratio, (
                "use_cispo is incompatible with disable_ppo_ratio; "
                "CISPO needs the pi_theta/pi_theta_old ratio but disable_ppo_ratio removes it"
            )
            if self.force_on_policy_ratio:
                assert (
                    self.use_importance_sampling_correction
                    and self.truncated_importance_sampling_type == "tis"
                ), "CISPO with force_on_policy_ratio requires IS correction with TIS"
            assert not self.sequence_level_importance_ratios, (
                "use_cispo is incompatible with sequence_level_importance_ratios; "
                "CISPO uses token-level importance weights"
            )
            assert self.ratio_clip_c is None, (
                "use_cispo is incompatible with dual clipping (ratio_clip_c); "
                "the dual-clip block runs after the CISPO loss assembly and would "
                "silently overwrite it. Set ratio_clip_c=null when use_cispo=True."
            )
            assert self.loss_type == LossType.TOKEN_LEVEL, (
                "use_cispo requires global_token_mean or group_token_mean normalization."
            )
        if self.truncated_importance_sampling_type is not None:
            assert self.use_importance_sampling_correction, (
                "truncated importance sampling is only supported when use_importance_sampling_correction is True"
            )
            assert self.truncated_importance_sampling_type in (
                "tis",
                "icepop",
                "adaptive-ppo-clip",
                "seq-mask-tis",
            ), (
                "truncated_importance_sampling_type must be 'tis', 'icepop', "
                "'adaptive-ppo-clip', or 'seq-mask-tis', "
                f"got {self.truncated_importance_sampling_type}"
            )
            assert (
                self.truncated_importance_sampling_ratio is not None
                and self.truncated_importance_sampling_ratio > 0
            ), "truncated_importance_sampling_ratio should be positive"
            if self.truncated_importance_sampling_ratio_min is not None:
                assert (
                    self.truncated_importance_sampling_ratio_min
                    <= self.truncated_importance_sampling_ratio
                ), (
                    "truncated_importance_sampling_ratio_min must be <= "
                    "truncated_importance_sampling_ratio"
                )
            if self.truncated_importance_sampling_type in (
                "icepop",
                "adaptive-ppo-clip",
                "seq-mask-tis",
            ):
                assert self.truncated_importance_sampling_ratio_min is not None, (
                    "truncated_importance_sampling_ratio_min should be set when "
                    "truncated_importance_sampling_type is 'icepop', "
                    "'adaptive-ppo-clip', or 'seq-mask-tis'"
                )
            if self.truncated_importance_sampling_type == "seq-mask-tis":
                assert not self.sequence_level_importance_ratios, (
                    "seq-mask-tis uses token-level IS correction with sequence-level masking, "
                    "and is incompatible with sequence_level_importance_ratios=True"
                )
            if self.truncated_importance_sampling_type == "adaptive-ppo-clip":
                assert self.force_on_policy_ratio, (
                    "adaptive-ppo-clip is intended for async RL; set "
                    "force_on_policy_ratio=True to use it"
                )
                assert self.metrics_level == "full", (
                    "adaptive-ppo-clip adapts its upper bound from approx_entropy, "
                    "which only metrics_level='full' computes"
                )
                assert not self.sequence_level_importance_ratios, (
                    "adaptive-ppo-clip uses token-level advantage-aware filtering "
                    "and is incompatible with sequence_level_importance_ratios=True"
                )

        # adaptive-ppo-clip controller state. The driver advances it once per
        # training step; workers receive it with the serialized loss function.
        self.cur_adaptive_ppo_clip_k: float | None = (
            0.0 if self.has_adaptive_loss_state() else None
        )

        self.score_centering = cfg.score_centering
        if self.score_centering:
            assert self.opd_full is None and not use_fused_linear_logprobs, (
                "score_centering needs the policy-gradient loss on full logits"
            )
            assert not self.sequence_level_importance_ratios and (
                not self.use_importance_sampling_correction
                or self.truncated_importance_sampling_type
                in (None, "tis", "icepop", "adaptive-ppo-clip")
            ), (
                "score_centering composes only with importance sampling off or "
                "token-level untruncated IS, 'tis', 'icepop', or 'adaptive-ppo-clip'"
            )
        self._score_centering_is_mode = (
            (self.truncated_importance_sampling_type or "is")
            if self.use_importance_sampling_correction
            else "off"
        )

        # Advertise, per returned metric, the global denominator it was
        # normalized by (see MetricNormalizer). Built here — next to the flags
        # that pick the denominators — so split-API trainers can undo the
        # placeholder global_valid_*=1 normalization without maintaining a
        # consumer-side table. Keep in sync with __call__'s return dict.
        self.metric_normalizations: dict[str, MetricNormalizer] = {
            # Normalized like the gradient (by the objective reduction).
            "loss": MetricNormalizer.LOSS_DIVISOR,
            KL_PENALTY_KEY: MetricNormalizer.LOSS_DIVISOR,
            # Token-normalized diagnostics, independent of loss_type.
            "probs_ratio": MetricNormalizer.TOKENS,
            "probs_ratio_clamped": MetricNormalizer.TOKENS,
            "token_mult_prob_error": MetricNormalizer.TOKENS,
            "token_abs_logprob_error": MetricNormalizer.TOKENS,
            "gen_kl_error": MetricNormalizer.TOKENS,
            "policy_kl_error": MetricNormalizer.TOKENS,
            "js_divergence_error": MetricNormalizer.TOKENS,
            APPROX_ENTROPY_KEY: MetricNormalizer.TOKENS,
            # Keyed on sequence_level_importance_ratios, NOT loss_type.
            "sampling_importance_ratio": (
                MetricNormalizer.SEQUENCES
                if self.sequence_level_importance_ratios
                else MetricNormalizer.TOKENS
            ),
            # Raw count — the downstream per-microbatch sum IS the value.
            "policy_support_excluded_tokens": MetricNormalizer.NONE,
            "num_valid_samples": MetricNormalizer.NONE,
            "gen_prev_logprob_mismatch_tokens": MetricNormalizer.NONE,
            "seq_logprob_error_valid_tokens": MetricNormalizer.NONE,
            "seq_logprob_error_valid_seqs": MetricNormalizer.NONE,
            "num_masked_seqs_by_logprob_error": MetricNormalizer.NONE,
            # Normalized by the microbatch's own correct-token count, not a
            # global factor — already a per-microbatch mean.
            "positive_nll_loss": MetricNormalizer.NONE,
            # Extrema — combined downstream with min/max, never scaled.
            "probs_ratio_min": MetricNormalizer.NONE,
            "probs_ratio_max": MetricNormalizer.NONE,
            "probs_ratio_clamped_min": MetricNormalizer.NONE,
            "probs_ratio_clamped_max": MetricNormalizer.NONE,
        }
        if self.metrics_level == "minimal":
            minimal_metrics = {
                "loss",
                "kl_penalty",
                "num_valid_samples",
                "policy_support_excluded_tokens",
                "positive_nll_loss",
                "token_mult_prob_error",
                "gen_prev_logprob_mismatch_tokens",
                "token_abs_logprob_error",
            }
            if self.use_importance_sampling_correction:
                minimal_metrics.add("sampling_importance_ratio")
            if self.seq_logprob_error_in_loss:
                minimal_metrics.update(
                    {
                        "seq_logprob_error_valid_tokens",
                        "seq_logprob_error_valid_seqs",
                        "num_masked_seqs_by_logprob_error",
                    }
                )
            self.metric_normalizations = {
                name: normalizer
                for name, normalizer in self.metric_normalizations.items()
                if name in minimal_metrics
            }
        if self.truncated_importance_sampling_type is not None:
            # Keyed on the TIS type, NOT loss_type: seq-mask-tis masks whole
            # sequences (÷ global_valid_seqs); tis/icepop/adaptive-ppo-clip
            # are token-level.
            self.metric_normalizations["is_oob_ratio"] = (
                MetricNormalizer.SEQUENCES
                if self.truncated_importance_sampling_type == "seq-mask-tis"
                else MetricNormalizer.TOKENS
            )
        if self.score_centering:
            self.metric_normalizations["score_centering_head_mass"] = (
                MetricNormalizer.TOKENS
            )
        if self.opd_full is not None:
            # opd_full returns its own metric set from _opd_full_call; the
            # policy-gradient diagnostics above are never produced there.
            self.metric_normalizations = {
                "loss": MetricNormalizer.LOSS_DIVISOR,
                KL_PENALTY_KEY: MetricNormalizer.LOSS_DIVISOR,
                "num_valid_samples": MetricNormalizer.NONE,
                "opd_full_reverse_kl": MetricNormalizer.TOKENS,
                "opd_full_reverse_kl_min": MetricNormalizer.NONE,
                "opd_full_reverse_kl_max": MetricNormalizer.NONE,
            }
            if self.opd_full.validate_decomposition:
                self.metric_normalizations.update(
                    {
                        "opd_full_entropy": MetricNormalizer.TOKENS,
                        "opd_full_cross_entropy": MetricNormalizer.TOKENS,
                        "opd_full_decomposition_error": MetricNormalizer.NONE,
                    }
                )

    def has_adaptive_loss_state(self) -> bool:
        """Whether this loss carries controller state across training steps."""
        return self.truncated_importance_sampling_type == "adaptive-ppo-clip"

    def set_adaptive_ppo_clip_k(self, k: float) -> None:
        """Set k, clamped to [0, k_max]; no-op without adaptive-ppo-clip."""
        if not self.has_adaptive_loss_state():
            return

        self.cur_adaptive_ppo_clip_k = min(
            self.truncated_importance_sampling_adaptive_k_max,
            max(0.0, float(k)),
        )

    def _get_adaptive_ppo_clip_bounds(self) -> tuple[float, float]:
        """adaptive-ppo-clip's ``(lo, hi)`` this step; ``hi`` includes the offset k."""
        tis_min = self.truncated_importance_sampling_ratio_min
        tis_max = self.truncated_importance_sampling_ratio
        cur_k = self.cur_adaptive_ppo_clip_k
        assert tis_min is not None and tis_max is not None and cur_k is not None
        return tis_min, tis_max + cur_k

    def _get_score_centering_is_bounds(self) -> tuple[float, float] | None:
        """Bounds that set the score-centering head and tail masses this step."""
        if self._score_centering_is_mode in ("tis", "icepop"):
            assert self.truncated_importance_sampling_ratio is not None
            return (
                self.truncated_importance_sampling_ratio_min or 0.0,
                self.truncated_importance_sampling_ratio,
            )
        elif self._score_centering_is_mode == "adaptive-ppo-clip":
            # Read per call: the controller moves the upper bound between steps.
            return self._get_adaptive_ppo_clip_bounds()
        return None

    def update_adaptive_loss_state(self, entropy: float) -> None:
        """Step k once after a training step to steer entropy toward the target.

        Entropy above the target narrows the positive-advantage upper bound;
        entropy below it widens the bound. k is clamped to [0, k_max].
        """
        if not self.has_adaptive_loss_state():
            return

        cur_k = self.cur_adaptive_ppo_clip_k
        assert cur_k is not None
        target_entropy = self.truncated_importance_sampling_adaptive_target_entropy
        if entropy > target_entropy:
            cur_k -= self.truncated_importance_sampling_adaptive_k_step
        elif entropy < target_entropy:
            cur_k += self.truncated_importance_sampling_adaptive_k_step
        self.set_adaptive_ppo_clip_k(cur_k)

    @property
    def requires_survivor_normalization(self) -> bool:
        """Whether the worker must renormalize using in-loss survivor counts.

        Uses the same switch as filtering so the loss and worker agree.
        """
        return self.seq_logprob_error_in_loss

    def __call__(
        self,
        next_token_logprobs: Optional[Tensor] = None,
        data: Optional[BatchedDataDict[ClippedPGLossDataDict]] = None,
        global_valid_seqs: Optional[torch.Tensor] = None,
        global_valid_toks: Optional[torch.Tensor] = None,
        opd_full_divergence: Optional[Tensor] = None,
        opd_full_entropy: Optional[Tensor] = None,
        opd_full_cross_entropy: Optional[Tensor] = None,
        policy_support_mask: Optional[Tensor] = None,
        policy_logprobs_at_sampler_topk: Optional[Tensor] = None,
        global_objective_denominator: Optional[Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Clipped Policy Gradient RL loss, or the full-vocabulary MOPD reverse KL.

        Which objective runs is fixed at construction by ``opd_full``.

        Args:
            next_token_logprobs: Sampled-token log-probabilities ``[B, S - 1]``.
                Required on the policy-gradient branch; on the ``opd_full``
                branch only when ``reference_policy_kl_penalty`` is non-zero.
            policy_logprobs_at_sampler_topk: Log-probabilities at the sampler's top-k
                ids ``[B, S - 1, k]``, required when ``score_centering`` is on.
            data: Microbatch with masks, advantages, and prior log-probabilities.
            global_valid_seqs: Global valid-sequence count for normalization.
            global_valid_toks: Global valid-token count for normalization.
            policy_support_mask: Current top-k/top-p support, supplied only with filtering.
                Negative infinity in other policy logprobs is also excluded.
            opd_full_divergence: Per-token reverse KL ``[B, S - 1]``, required on
                the ``opd_full`` branch.
            opd_full_entropy: Optional ``sum_v p_s log p_s`` diagnostic.
            opd_full_cross_entropy: Optional ``-sum_v p_s log p_t`` diagnostic.
            global_objective_denominator: Streaming split-API divisor passed by
                ``StreamingObjectiveLoss`` for reductions with
                ``requires_explicit_denominator``; None on synchronous paths,
                where ``loss_divisor`` falls back to ``global_valid_toks`` or
                ``global_valid_seqs``.

        Returns:
            Tuple of the scalar loss and its metric dict.
        """
        assert data is not None, "ClippedPGLossFn requires data"
        assert global_valid_seqs is not None, (
            "ClippedPGLossFn requires global_valid_seqs"
        )
        assert global_valid_toks is not None, (
            "ClippedPGLossFn requires global_valid_toks"
        )
        if self.opd_full is not None:
            return self._opd_full_call(
                data=data,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
                next_token_logprobs=next_token_logprobs,
                opd_full_divergence=opd_full_divergence,
                opd_full_entropy=opd_full_entropy,
                opd_full_cross_entropy=opd_full_cross_entropy,
                global_objective_denominator=global_objective_denominator,
            )
        assert next_token_logprobs is not None, (
            "ClippedPGLossFn requires next_token_logprobs"
        )
        curr_logprobs = next_token_logprobs
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        advantages = data["advantages"][:, 1:]
        # Skip loading prev_logprobs when force_on_policy_ratio=True (will use curr_logprobs instead)
        prev_logprobs = (
            None if self.force_on_policy_ratio else data["prev_logprobs"][:, 1:]
        )
        generation_logprobs = data["generation_logprobs"][:, 1:]
        if self.reference_policy_kl_penalty != 0:
            reference_policy_logprobs = data["reference_policy_logprobs"][:, 1:]
            curr_logprobs_unfiltered = data.get(
                "curr_logprobs_unfiltered", curr_logprobs
            )

        # Preserve the global-batch denominators when a microbatch has
        # rows without response loss tokens.
        sample_mask = sample_mask * token_mask.bool().any(dim=-1)
        mask = token_mask * sample_mask.unsqueeze(-1)
        full_metrics = self.metrics_level == "full"
        need_importance_weights = (
            full_metrics or self.use_importance_sampling_correction
        )

        # For truly on-policy training, use curr_logprobs as prev_logprobs
        # This avoids computing prev_logprobs upstream
        if self.force_on_policy_ratio:
            prev_logprobs = curr_logprobs.detach()

        # Only top-k/top-p can mark -inf as an expected support mismatch.
        # Keep NaN/+inf observable, and preserve strict checks without filtering.
        if policy_support_mask is not None:
            policy_support_mask = (
                policy_support_mask
                & ~torch.isneginf(prev_logprobs)
                & ~torch.isneginf(generation_logprobs)
            )
        seq_error_token_mask = token_mask
        if policy_support_mask is not None:
            seq_error_token_mask = token_mask * policy_support_mask.to(token_mask.dtype)
        seq_error_metrics = {}
        if self.seq_logprob_error_in_loss:
            errors, _ = compute_seq_logprob_errors(
                policy_logprobs=curr_logprobs.detach(),
                generation_logprobs=generation_logprobs,
                token_mask=seq_error_token_mask,
                sample_mask=sample_mask,
            )
            filtered_sample_mask = sample_mask * (
                errors <= self.seq_logprob_error_threshold
            ).to(sample_mask.dtype)
            seq_error_metrics["num_masked_seqs_by_logprob_error"] = (
                (sample_mask - filtered_sample_mask).sum().item()
            )
            seq_error_metrics[SEQ_LOGPROB_KEEP_MASK_KEY] = (
                (errors <= self.seq_logprob_error_threshold).float().cpu().tolist()
            )
            sample_mask = filtered_sample_mask
            mask = token_mask * sample_mask.unsqueeze(-1)
            seq_error_metrics["seq_logprob_error_valid_tokens"] = mask.sum().item()
            seq_error_metrics["seq_logprob_error_valid_seqs"] = sample_mask.sum().item()
            prev_logprobs = curr_logprobs.detach()

        # Reference KL uses the unfiltered policy and retains its own token set.
        # Actor support mismatches must not discard otherwise valid KL terms.
        kl_token_mask = token_mask
        kl_sample_mask = sample_mask
        kl_mask = mask
        kl_generation_logprobs = torch.where(kl_mask.bool(), generation_logprobs, 0.0)
        if self.reference_policy_kl_penalty != 0:
            # Sanitize before KL arithmetic even without sequence rejection;
            # multiplying a masked nonfinite result by zero is too late.
            curr_logprobs_unfiltered = torch.where(
                kl_mask.bool(), curr_logprobs_unfiltered, 0.0
            )
            reference_policy_logprobs = torch.where(
                kl_mask.bool(), reference_policy_logprobs, 0.0
            )

        # Keep support information until all three policy distributions are
        # available. Replacing -inf with zero earlier turns an impossible token
        # into log-probability one in token/sequence importance ratios.
        if policy_support_mask is not None:
            token_mask = token_mask * policy_support_mask.to(token_mask.dtype)
            sample_mask = sample_mask * token_mask.bool().any(dim=-1).to(
                sample_mask.dtype
            )
        mask = token_mask * sample_mask.unsqueeze(-1)
        curr_logprobs = torch.where(mask.bool(), curr_logprobs, 0.0)
        prev_logprobs = torch.where(mask.bool(), prev_logprobs, 0.0)
        generation_logprobs = torch.where(mask.bool(), generation_logprobs, 0.0)
        if self.score_centering:
            advantages = torch.where(mask.bool(), advantages.detach(), 0.0)
        if self.seq_logprob_error_in_loss:
            seq_error_metrics["seq_logprob_error_valid_tokens"] = mask.sum().item()
            seq_error_metrics["seq_logprob_error_valid_seqs"] = sample_mask.sum().item()

        per_response_weights = self.objective_reduction.per_response_weights(data)
        objective_denominator = self.objective_reduction.loss_divisor(
            global_valid_toks=global_valid_toks,
            global_valid_seqs=global_valid_seqs,
            global_objective_denominator=global_objective_denominator,
        )

        # Calculate KL regularization.
        if self.reference_policy_kl_penalty != 0:
            # When top-k/top-p filtering is enabled, we need special handling for KL:
            # - reference_policy_logprobs is computed **without** filtering (see use_reference_model)
            # - curr_logprobs/prev_logprobs are computed **with** filtering (for actor loss compatibility)
            # - For KL, we need curr_logprobs **without** filtering to be consistent with ref logprobs
            # - For importance weights, we also use unfiltered curr_logprobs_unfiltered since we're
            #   reweighting samples from π_gen_filtered to π_curr_unfiltered

            # On-policy KL approximation
            # KL samples come from the optimized policy, so the KL loss must include
            # the score-function gradient through the sampling probability; see
            # https://arxiv.org/abs/2506.09477v1. In the non-IS case,
            # exp(x - x.detach()) has forward value 1 while preserving that gradient.
            if self.use_on_policy_kl_approximation:
                # See: docs/guides/grpo.md#on-policy-kl-approximation
                kl_importance_weights = torch.exp(
                    curr_logprobs_unfiltered - kl_generation_logprobs
                )
            else:
                kl_importance_weights = torch.exp(
                    curr_logprobs_unfiltered - curr_logprobs_unfiltered.detach()
                )
            kl_importance_weights = torch.nan_to_num(
                kl_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
            )

            # Compute KL loss
            kl = self.reference_policy_kl_penalty * calculate_kl(
                logprobs=curr_logprobs_unfiltered,
                logprobs_reference=reference_policy_logprobs,
                kl_type=self.reference_policy_kl_type,
                input_clamp_value=self.kl_input_clamp_value,
                output_clamp_value=self.kl_output_clamp_value,
                importance_sampling_weights=kl_importance_weights,
            )

            # Reduce KL loss
            kl = self.objective_reduction.reduce(
                kl,
                token_mask=kl_token_mask,
                sample_mask=kl_sample_mask,
                mask=kl_mask,
                denominator=objective_denominator,
                per_response_weights=per_response_weights,
            )
        else:
            kl = curr_logprobs.new_zeros(())

        # -------------------------------------------------------------
        # Off-policy (actor) importance-sampling correction
        # -------------------------------------------------------------
        _is_filter_metrics: dict = {}  # populated for tis / icepop / adaptive-ppo-clip / seq-mask-tis
        # See: docs/guides/grpo.md#importance-sampling-correction
        if not need_importance_weights:
            # No objective or emitted metric needs the actor/generation
            # importance weights in this mode.
            actor_importance_weights_expanded = None
        elif self.sequence_level_importance_ratios:
            # importance weight w_i = exp(Σ_t (log π_actor − log π_behaviour))
            seq_lp_diff = ((prev_logprobs - generation_logprobs) * mask).sum(dim=-1)
            actor_importance_weights = torch.exp(seq_lp_diff).detach()
            actor_importance_weights = torch.nan_to_num(
                actor_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
            )
            # Broadcast to token dimension so we can reuse existing reduction
            actor_importance_weights_expanded = actor_importance_weights.unsqueeze(-1)
        else:
            # Token-level correction
            actor_importance_weights_expanded = torch.exp(
                prev_logprobs - generation_logprobs
            )
            actor_importance_weights_expanded = torch.nan_to_num(
                actor_importance_weights_expanded, nan=0.0, posinf=0.0, neginf=0.0
            )
        # ---- Truncated Importance Sampling ----
        # "tis"          – clamp IS weights to [min, max], where min defaults to 0
        # "icepop"       – zero out tokens whose IS weight ∉ [min, max]   (ref bounds: 0.5–5)
        # "adaptive-ppo-clip"
        #                – zero out A<0 tokens whose IS weight < min, A>0 tokens
        #                  whose IS weight > max + k (k adapts to entropy), and
        #                  every token whose IS weight exceeds the outer bound
        #   Paper: https://microsoft.ai/pdf/mai-thinking-1.pdf
        # "seq-mask-tis" – zero out entire sequences whose geometric-mean
        #                  IS ratio ∉ [min, max]; retained sequences keep
        #                  raw (non-truncated) token-level IS weights      (ref bounds: 0.999–1.002)
        #   Blog: https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda
        # is_oob_ratio: fraction of tokens (tis/icepop/adaptive-ppo-clip) or sequences (seq-mask-tis)
        # whose importance weight falls outside the truncation bounds. Each microbatch
        # contributes its out-of-bounds count divided by the *global* valid token/seq
        # count, so the np.sum aggregation in grpo.py recovers the correct global fraction.
        if self.truncated_importance_sampling_ratio is not None:
            if self.truncated_importance_sampling_type == "tis":
                tis_min = self.truncated_importance_sampling_ratio_min
                if tis_min is None:
                    tis_min = 0.0
                token_oob_mask = (
                    actor_importance_weights_expanded
                    > self.truncated_importance_sampling_ratio
                ) | (actor_importance_weights_expanded < tis_min)
                _is_filter_metrics = {
                    "is_oob_ratio": masked_mean(
                        token_oob_mask.float(),
                        mask,
                        global_normalization_factor=global_valid_toks,
                    ),
                }
                actor_importance_weights_expanded = torch.clamp(
                    actor_importance_weights_expanded,
                    min=tis_min,
                    max=self.truncated_importance_sampling_ratio,
                )
            elif self.truncated_importance_sampling_type == "icepop":
                token_kept_mask = (
                    actor_importance_weights_expanded
                    >= self.truncated_importance_sampling_ratio_min
                ) & (
                    actor_importance_weights_expanded
                    <= self.truncated_importance_sampling_ratio
                )
                _is_filter_metrics = {
                    "is_oob_ratio": masked_mean(
                        (~token_kept_mask).float(),
                        mask,
                        global_normalization_factor=global_valid_toks,
                    ),
                }
                actor_importance_weights_expanded = torch.where(
                    token_kept_mask,
                    actor_importance_weights_expanded,
                    torch.zeros_like(actor_importance_weights_expanded),
                )
            elif self.truncated_importance_sampling_type == "adaptive-ppo-clip":
                tis_min, tis_max = self._get_adaptive_ppo_clip_bounds()
                token_kept_mask = adaptive_ppo_clip_kept_mask(
                    actor_importance_weights_expanded,
                    advantages=advantages,
                    lower_bound=tis_min,
                    upper_bound=tis_max,
                    outer_bound=self.truncated_importance_sampling_adaptive_outer_bound,
                )
                _is_filter_metrics = {
                    "is_oob_ratio": masked_mean(
                        (~token_kept_mask).float(),
                        mask,
                        global_normalization_factor=global_valid_toks,
                    ),
                }
                actor_importance_weights_expanded = torch.where(
                    token_kept_mask,
                    actor_importance_weights_expanded,
                    torch.zeros_like(actor_importance_weights_expanded),
                )
            elif self.truncated_importance_sampling_type == "seq-mask-tis":
                # geo_mean_i = exp( mean_t( log(π_prev / π_gen) ) )
                log_is_ratio = torch.nan_to_num(
                    prev_logprobs - generation_logprobs,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                seq_log_is_ratio_mean = masked_mean(
                    log_is_ratio, token_mask, dim=-1
                )  # [B]
                seq_geomean_is_ratio = torch.exp(seq_log_is_ratio_mean).detach()  # [B]
                seq_kept_mask = (
                    (
                        seq_geomean_is_ratio
                        >= self.truncated_importance_sampling_ratio_min
                    )
                    & (seq_geomean_is_ratio <= self.truncated_importance_sampling_ratio)
                ).float()  # [B]
                _is_filter_metrics = {
                    "is_oob_ratio": masked_mean(
                        1.0 - seq_kept_mask,
                        sample_mask,
                        global_normalization_factor=global_valid_seqs,
                    ),
                }
                actor_importance_weights_expanded = (
                    actor_importance_weights_expanded * seq_kept_mask.unsqueeze(-1)
                )
            else:
                raise ValueError(
                    f"Invalid truncated importance sampling type: {self.truncated_importance_sampling_type}"
                )

        actor_importance_weights = actor_importance_weights_expanded
        del actor_importance_weights_expanded
        if self.use_importance_sampling_correction:
            importance_weights_to_use = actor_importance_weights
        else:
            importance_weights_to_use = None

        if self.enable_torch_compile and self._compiled_actor_objective is None:
            self._compiled_actor_objective = torch.compile(
                clipped_pg_actor_objective,
                dynamic=True,
                fullgraph=True,
                mode="max-autotune-no-cudagraphs",
            )
        actor_objective = self._compiled_actor_objective or clipped_pg_actor_objective
        actor_loss, ratios, ratios_clamped = actor_objective(
            curr_logprobs,
            prev_logprobs=prev_logprobs,
            advantages=advantages,
            token_mask=token_mask,
            mask=mask,
            sample_mask=sample_mask,
            importance_weights=importance_weights_to_use,
            objective_denominator=objective_denominator,
            ratio_clip_min=self.ratio_clip_min,
            ratio_clip_max=self.ratio_clip_max,
            ratio_clip_c=(self.ratio_clip_c if self.ratio_clip_c is not None else -1.0),
            disable_ppo_ratio=self.disable_ppo_ratio,
            force_on_policy_ratio=self.force_on_policy_ratio,
            sequence_level_importance_ratios=self.sequence_level_importance_ratios,
            use_cispo=self.use_cispo,
            reduction=self.objective_reduction,
            per_response_weights=per_response_weights,
        )

        score_centering_metrics: dict[str, Tensor] = {}
        if self.score_centering:
            assert policy_logprobs_at_sampler_topk is not None, (
                "score_centering requires policy_logprobs_at_sampler_topk from prepare_loss_input"
            )
            centering, head_mass = compute_score_centering_term(
                policy_logprobs_at_sampler_topk,
                sampler_topk_logprobs=data["sampler_topk_logprobs"][:, 1:],
                importance_sampling_mode=self._score_centering_is_mode,
                is_bounds=self._get_score_centering_is_bounds(),
                token_mask=mask,
                # Score centering under top-k/top-p always replays the sampler's
                # support; only filtering supplies policy_support_mask.
                sampler_replay=policy_support_mask is not None,
                adaptive_ppo_clip_advantages=advantages,
                adaptive_ppo_clip_outer_bound=self.truncated_importance_sampling_adaptive_outer_bound,
            )
            # Reduce the correction exactly like the actor objective it corrects.
            actor_loss = actor_loss + self.objective_reduction.reduce(
                advantages * centering,
                token_mask=token_mask,
                sample_mask=sample_mask,
                mask=mask,
                denominator=objective_denominator,
                per_response_weights=per_response_weights,
            )
            score_centering_metrics["score_centering_head_mass"] = masked_mean(
                head_mass, mask, global_normalization_factor=global_valid_toks
            )

        # Metric: sampling importance ratio (mean over samples)
        # See: docs/guides/grpo.md#sampling-importance-ratio
        if need_importance_weights:
            if self.sequence_level_importance_ratios:
                sample_importance_ratio = masked_mean(
                    actor_importance_weights.squeeze(-1),
                    sample_mask,
                    global_normalization_factor=global_valid_seqs,
                )
            else:
                sample_importance_ratio = masked_mean(
                    actor_importance_weights,
                    mask,
                    global_normalization_factor=global_valid_toks,
                )

        # -----------------------------------------------------------------
        # VAPO: positive-example NLL loss on correct samples (reward > 0)
        # L = L_PPO + μ · L_NLL(correct)
        # -----------------------------------------------------------------
        nll_loss = curr_logprobs.new_zeros(())
        if self.positive_example_nll_weight > 0 and "rewards" in data:
            correct_sample_mask = (data["rewards"] > 0).float()  # [batch]
            correct_mask = mask * correct_sample_mask.unsqueeze(-1)
            correct_valid_toks = correct_mask.sum()
            nll_loss = torch.where(
                correct_valid_toks > 0,
                masked_mean(
                    -curr_logprobs,
                    correct_mask,
                    global_normalization_factor=correct_valid_toks,
                ),
                curr_logprobs.new_zeros(()),
            )

        loss = actor_loss + kl + self.positive_example_nll_weight * nll_loss
        # Keep all reductions on device and perform one scalar materialization
        # for the Python-facing metric dictionary. This avoids a synchronization
        # for every individual metric while preserving the historical values.

        # If you provided a global_valid_{seqs/toks}, all metrics here are globally normalized
        # by either sequence or token count, depending on particular metric.
        # To get the true metric, you'll need to sum over the microbatch.
        metric_tensors = {"loss": loss.detach()}
        with torch.no_grad():
            lp_error = torch.abs(generation_logprobs - prev_logprobs)
            metric_tensors["gen_prev_logprob_mismatch_tokens"] = (
                (lp_error != 0) & mask.bool()
            ).sum()
            metric_tensors["token_abs_logprob_error"] = masked_mean(
                lp_error, mask, global_normalization_factor=global_valid_toks
            )
            metric_tensors["token_mult_prob_error"] = masked_mean(
                torch.exp(lp_error * mask),
                mask,
                global_normalization_factor=global_valid_toks,
            )
        if full_metrics:
            metric_tensors.update(
                clipped_pg_diagnostic_metrics(
                    curr_logprobs=curr_logprobs,
                    prev_logprobs=prev_logprobs,
                    generation_logprobs=generation_logprobs,
                    ratios=ratios,
                    ratios_clamped=ratios_clamped,
                    mask=mask,
                    global_valid_toks=global_valid_toks,
                    reference_policy_kl_type=self.reference_policy_kl_type,
                )
            )
        if need_importance_weights:
            metric_tensors["sampling_importance_ratio"] = sample_importance_ratio

        metric_tensors.update(
            {
                KL_PENALTY_KEY: (
                    kl / self.reference_policy_kl_penalty
                    if self.reference_policy_kl_penalty != 0
                    else curr_logprobs.new_zeros(())
                ),
                "num_valid_samples": sample_mask.sum(),
                "positive_nll_loss": nll_loss,
                **_is_filter_metrics,
                **score_centering_metrics,
            }
        )
        metric_tensors["policy_support_excluded_tokens"] = (kl_mask - mask).sum()
        return loss, {
            **materialize_scalar_metrics(metric_tensors, loss.dtype),
            **seq_error_metrics,
        }

    def _opd_full_call(
        self,
        *,
        data: BatchedDataDict[ClippedPGLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        next_token_logprobs: Optional[Tensor],
        opd_full_divergence: Optional[Tensor],
        opd_full_entropy: Optional[Tensor],
        opd_full_cross_entropy: Optional[Tensor],
        global_objective_denominator: Optional[Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Full-vocabulary MOPD objective: the exact reverse KL is the whole loss.

        ``prepare_loss_input`` reconstructs the teacher distribution and runs the
        distributed divergence kernel, so this only masks and normalizes. No
        policy-gradient ratio, advantage, or importance-sampling term applies:
        the reverse KL is an exact expectation over the student distribution and
        does not depend on which token was sampled.

        Args:
            data: Microbatch with ``token_mask`` and ``sample_mask``.
            global_valid_seqs: Global valid-sequence count for normalization.
            global_valid_toks: Global valid-token count for normalization.
            next_token_logprobs: Current-policy sampled-token log-probabilities,
                only needed when ``reference_policy_kl_penalty`` is non-zero.
            opd_full_divergence: Per-token reverse KL ``[B, S - 1]``.
            opd_full_entropy: Optional ``sum_v p_s log p_s`` diagnostic.
            opd_full_cross_entropy: Optional ``-sum_v p_s log p_t`` diagnostic.
            global_objective_denominator: Streaming split-API divisor forwarded
                from ``__call__``; see its docstring.

        Returns:
            Tuple of the scalar loss and its metric dict.

        Raises:
            ValueError: If the divergence tensor is missing.
        """
        if opd_full_divergence is None:
            raise ValueError(
                "opd_full requires opd_full_divergence from prepare_loss_input."
            )
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        mask = token_mask * sample_mask.unsqueeze(-1)

        objective_denominator = self.objective_reduction.loss_divisor(
            global_valid_toks=global_valid_toks,
            global_valid_seqs=global_valid_seqs,
            global_objective_denominator=global_objective_denominator,
        )
        per_response_weights = self.objective_reduction.per_response_weights(data)

        def reduce_like_loss(per_token_values: Tensor) -> torch.Tensor:
            """Reduce a per-token tensor exactly like the policy-gradient objective."""
            return self.objective_reduction.reduce(
                per_token_values,
                token_mask=token_mask,
                sample_mask=sample_mask,
                mask=mask,
                denominator=objective_denominator,
                per_response_weights=per_response_weights,
            )

        actor_loss = reduce_like_loss(opd_full_divergence)

        kl = torch.tensor(0.0, device=actor_loss.device)
        if self.reference_policy_kl_penalty != 0:
            if next_token_logprobs is None:
                raise ValueError(
                    "opd_full with reference_policy_kl_penalty != 0 requires "
                    "next_token_logprobs from prepare_loss_input: the penalty must "
                    "be differentiable through the current policy."
                )
            # Unlike the exact divergence above, this penalty is sampled, so its
            # gradient needs the score-function term through the sampling
            # probability, as in the policy-gradient branch.
            # exp(x - x.detach()) is 1 in the forward pass and supplies it.
            kl_importance_weights = torch.nan_to_num(
                torch.exp(next_token_logprobs - next_token_logprobs.detach()),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            kl = self.reference_policy_kl_penalty * calculate_kl(
                logprobs=next_token_logprobs,
                logprobs_reference=data["reference_policy_logprobs"][:, 1:],
                kl_type=self.reference_policy_kl_type,
                input_clamp_value=self.kl_input_clamp_value,
                output_clamp_value=self.kl_output_clamp_value,
                importance_sampling_weights=kl_importance_weights,
            )
            kl = reduce_like_loss(kl)

        loss = actor_loss + kl

        with torch.no_grad():
            masked_divergence = opd_full_divergence.detach()[mask.bool()]
            # Token-normalized diagnostics use global_valid_toks regardless of
            # loss_type, matching their declared MetricNormalizer.TOKENS.
            reverse_kl_metric = masked_mean(
                opd_full_divergence.detach(),
                mask,
                global_normalization_factor=global_valid_toks,
            )
            metrics: dict[str, Any] = {
                "loss": loss.item(),
                # Report the raw KL, undoing the coefficient. Guard on the
                # coefficient: `kl` is 0-dim, so its `numel()` is always 1 and
                # cannot stand in for "the penalty is active".
                KL_PENALTY_KEY: (
                    kl.item() / self.reference_policy_kl_penalty
                    if self.reference_policy_kl_penalty
                    else 0
                ),
                "num_valid_samples": sample_mask.sum().item(),
                "opd_full_reverse_kl": reverse_kl_metric.item(),
                "opd_full_reverse_kl_min": (
                    masked_divergence.min().item()
                    if masked_divergence.numel() > 0
                    else float("inf")
                ),
                "opd_full_reverse_kl_max": (
                    masked_divergence.max().item()
                    if masked_divergence.numel() > 0
                    else float("-inf")
                ),
            }
            if (
                opd_full_entropy is not None
                and opd_full_cross_entropy is not None
                and self.opd_full is not None
                and self.opd_full.validate_decomposition
            ):
                # reverse_kl == sum_v p_s log p_s + (-sum_v p_s log p_t)
                decomposition = (
                    opd_full_entropy.detach() + opd_full_cross_entropy.detach()
                )
                metrics["opd_full_entropy"] = masked_mean(
                    -opd_full_entropy.detach(),
                    mask,
                    global_normalization_factor=global_valid_toks,
                ).item()
                metrics["opd_full_cross_entropy"] = masked_mean(
                    opd_full_cross_entropy.detach(),
                    mask,
                    global_normalization_factor=global_valid_toks,
                ).item()
                metrics["opd_full_decomposition_error"] = (
                    (decomposition - opd_full_divergence.detach())
                    .abs()
                    .mul(mask)
                    .max()
                    .item()
                )

        return loss, metrics


def _validate_opd_full_loss_config(
    cfg: ClippedPGLossConfig, use_fused_linear_logprobs: bool
) -> None:
    """Reject loss knobs that have no code path under the full-vocabulary objective.

    ``opd_full`` replaces the whole policy-gradient objective with an exact
    reverse KL, so the ratio, CISPO, and importance-sampling machinery is dead
    code on that branch. Erroring out beats silently ignoring user config.

    Args:
        cfg: The clipped-PG loss config supplied by the user.
        use_fused_linear_logprobs: Whether the fused linear-CE forward is on.

    Raises:
        ValueError: If any incompatible option is set.
    """
    if use_fused_linear_logprobs:
        raise ValueError(
            "opd_full is incompatible with use_fused_linear_logprobs: the fused "
            "forward returns precomputed logprobs, but the full-vocabulary "
            "reverse KL needs the raw vocabulary-parallel logits."
        )

    # Policy-gradient ratio machinery: no ratio exists under opd_full.
    if not cfg.disable_ppo_ratio:
        raise ValueError("opd_full requires loss_fn.disable_ppo_ratio=true.")
    if cfg.ratio_clip_c is not None:
        raise ValueError("opd_full is incompatible with dual PPO clipping.")
    if cfg.use_cispo:
        raise ValueError("opd_full is incompatible with CISPO.")
    if cfg.force_on_policy_ratio:
        raise ValueError("opd_full is incompatible with force_on_policy_ratio.")
    if cfg.use_on_policy_kl_approximation:
        raise ValueError(
            "opd_full is incompatible with use_on_policy_kl_approximation: it "
            "reweights the reference-KL term by a generation-to-current ratio "
            "that only the policy-gradient branch computes. The opd_full branch "
            "never reads it, so leaving it on would be silently ignored."
        )
    if cfg.sequence_level_importance_ratios:
        raise ValueError(
            "opd_full is incompatible with sequence_level_importance_ratios."
        )

    # Importance-sampling machinery: the reverse KL is an exact expectation over
    # the student distribution, so no sampling-distribution correction applies.
    if cfg.use_importance_sampling_correction:
        raise ValueError(
            "opd_full is incompatible with use_importance_sampling_correction: the "
            "full-vocabulary reverse KL is an exact expectation and carries no "
            "sampled-token score-function term to correct."
        )
    if cfg.truncated_importance_sampling_type is not None:
        raise ValueError(
            "opd_full is incompatible with truncated_importance_sampling_type."
        )

    # Advantage/reward-shaped terms: opd_full never reads advantages or rewards.
    if cfg.positive_example_nll_weight != 0:
        raise ValueError(
            "opd_full is incompatible with a non-zero positive_example_nll_weight."
        )

    if cfg.use_kl_in_reward:
        raise ValueError(
            "opd_full is incompatible with use_kl_in_reward: it routes the "
            "reference KL through the reward, which reaches the objective only "
            "via advantages, and opd_full ignores advantages entirely. It also "
            "zeroes the loss-side reference_policy_kl_penalty, so the penalty "
            "would be dropped on both paths. Set use_kl_in_reward=false; the "
            "loss-side penalty is applied directly."
        )

    if cfg.reference_policy_kl_penalty != 0:
        warnings.warn(
            "opd_full is already a KL objective against the teacher; "
            "reference_policy_kl_penalty adds a second KL against the frozen "
            "reference policy on top of it.",
            stacklevel=3,
        )


class NLLLossFn(LossFunction):
    """Negative Log Likelihood Loss function."""

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(self, use_fused_linear_logprobs: bool = False):
        self.use_fused_linear_logprobs = use_fused_linear_logprobs
        # See MetricNormalizer — split-API trainers use this to undo the
        # placeholder global_valid_*=1 normalization per metric.
        self.metric_normalizations: dict[str, MetricNormalizer] = {
            "loss": MetricNormalizer.TOKENS,
            "num_unmasked_tokens": MetricNormalizer.NONE,
            "num_valid_samples": MetricNormalizer.NONE,
        }

    def __call__(
        self,
        next_token_logprobs: Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: Tensor | None,
        global_valid_toks: Tensor,
        dpo_loss: bool = False,
        dpo_average_log_probs: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # logits shape: [batch_size, seq_len, vocab_size]
        # Get the next token logits for each position
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        mask = token_mask * sample_mask.unsqueeze(-1)

        if dpo_loss:
            ## shape: [batch_size]
            num_unmasked_tokens = torch.sum(mask, -1)
            ## multiply by sample_mask to zero out invalid samples
            loss = -torch.sum(next_token_logprobs * mask, dim=-1)
            if dpo_average_log_probs:
                loss = loss / num_unmasked_tokens.clamp(min=1)
        else:
            ## single scalar loss
            ## scale by the total number of tokens in the batch
            loss = -masked_mean(
                next_token_logprobs,
                mask,
                global_normalization_factor=global_valid_toks,
            )

        return loss, {
            "loss": loss.item() if loss.ndim == 0 else loss,
            "num_unmasked_tokens": mask.sum().item(),
            "num_valid_samples": sample_mask.sum().item(),
        }


class PreferenceLossDataDict(TypedDict):
    """Keys for a preference loss.

    ``pair_index`` and ``is_chosen`` make pair reconstruction robust to row
    reordering by sequence packing. They remain optional for compatibility
    with existing interleaved DPO batches.
    """

    input_ids: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
    pair_index: NotRequired[torch.Tensor]
    is_chosen: NotRequired[torch.Tensor]


class PreferenceLossFn(LossFunction):
    """Preference Loss function.

    Optimizes the model to prefer chosen responses over rejected ones

    The preference loss is computed as:
    L_pref(θ) = -E[log(σ(β * (r_chosen - r_rejected)))]

    where:
    - σ is the sigmoid function
    - β is a scaling factor (ex: `reference_policy_kl_penalty` in DPO)
    - r_chosen and r_rejected are the rewards for chosen and rejected responses

    Returns:
        tuple[torch.Tensor, dict]: A tuple containing:
            - The preference loss value
            - A dictionary with metrics including:
                - loss: Preference loss
                - accuracy: Fraction of examples where chosen response has higher reward
    """

    loss_type = LossType.SEQUENCE_LEVEL
    input_type = LossInputType.LOGIT

    def split_output_tensor(
        self,
        tensor: Tensor,
        data: Optional[PreferenceLossDataDict] = None,
    ) -> tuple[Tensor, Tensor]:
        """Return pair-aligned chosen and rejected rows.

        Sequence packing may reorder rows, so positional slicing is only safe
        for legacy batches that do not carry pair metadata.
        """
        if data is None or "pair_index" not in data or "is_chosen" not in data:
            return tensor[::2], tensor[1::2]

        is_chosen = data["is_chosen"].to(device=tensor.device, dtype=torch.bool)
        pair_index = data["pair_index"].to(device=tensor.device)
        if (
            is_chosen.numel() != tensor.shape[0]
            or pair_index.numel() != tensor.shape[0]
        ):
            raise ValueError(
                "pair_index and is_chosen must contain one entry per preference row."
            )
        chosen_rows = torch.nonzero(is_chosen, as_tuple=True)[0]
        rejected_rows = torch.nonzero(~is_chosen, as_tuple=True)[0]
        chosen_pairs = pair_index.index_select(0, chosen_rows)
        rejected_pairs = pair_index.index_select(0, rejected_rows)
        chosen_order = chosen_pairs.argsort(stable=True)
        rejected_order = rejected_pairs.argsort(stable=True)
        chosen_pairs = chosen_pairs.index_select(0, chosen_order)
        rejected_pairs = rejected_pairs.index_select(0, rejected_order)
        if (
            not torch.equal(chosen_pairs, rejected_pairs)
            or torch.unique(chosen_pairs).numel() != chosen_pairs.numel()
        ):
            raise ValueError(
                "Preference batch must contain exactly one chosen and one rejected "
                "row for every pair_index."
            )
        return (
            tensor.index_select(0, chosen_rows.index_select(0, chosen_order)),
            tensor.index_select(0, rejected_rows.index_select(0, rejected_order)),
        )

    def _preference_loss(
        self,
        rewards: Tensor,
        sample_mask: Tensor,
        global_valid_seqs: Tensor,
        beta: float = 1.0,
        data: Optional[PreferenceLossDataDict] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rewards_chosen, rewards_rejected = self.split_output_tensor(rewards, data)
        mask_chosen, mask_rejected = self.split_output_tensor(sample_mask, data)
        rewards_delta = rewards_chosen - rewards_rejected

        per_sample_loss = (
            -torch.nn.functional.logsigmoid(beta * rewards_delta) * mask_chosen
        )  ## zero out invalid samples

        ## divide by 2 because each preference example corresponds to 2 samples (chosen, rejected)
        return (
            masked_mean(
                per_sample_loss,
                mask_chosen,
                global_normalization_factor=global_valid_seqs / 2,
            ),
            masked_mean(
                rewards_chosen > rewards_rejected,
                mask_chosen,
                global_normalization_factor=global_valid_seqs / 2,
            ),
            masked_mean(
                rewards_chosen,
                mask_chosen,
                global_normalization_factor=global_valid_seqs / 2,
            ),
            masked_mean(
                rewards_rejected,
                mask_rejected,
                global_normalization_factor=global_valid_seqs / 2,
            ),
        )

    def __call__(
        self,
        logits: Tensor,
        data: BatchedDataDict[PreferenceLossDataDict],
        global_valid_seqs: Tensor,
        global_valid_toks: Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sample_mask = data["sample_mask"]

        rewards = logits.squeeze(-1)

        (
            preference_loss,
            accuracy,
            rewards_chosen_mean,
            rewards_rejected_mean,
        ) = self._preference_loss(rewards, sample_mask, global_valid_seqs, data=data)

        ## divide by 2 because we're summing over (chosen, rejected) pairs
        num_valid_samples = sample_mask.sum() / 2

        return preference_loss, {
            "loss": preference_loss.item(),
            "accuracy": accuracy.item(),
            "rewards_chosen_mean": rewards_chosen_mean.item(),
            "rewards_rejected_mean": rewards_rejected_mean.item(),
            "num_valid_samples": num_valid_samples.item(),
        }


class DPOLossConfig(BaseModel, extra="allow"):
    reference_policy_kl_penalty: float = 0.05
    preference_loss_weight: float = 1.0
    sft_loss_weight: float = 0.0
    preference_average_log_probs: bool = False
    sft_average_log_probs: bool = False


class DPOLossDataDict(TypedDict):
    """Required keys for the DPO loss function."""

    input_ids: torch.Tensor
    reference_policy_logprobs: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
    pair_index: NotRequired[torch.Tensor]
    is_chosen: NotRequired[torch.Tensor]


class DPOLossFn(PreferenceLossFn):
    """Direct Preference Optimization (DPO) loss function.

    This loss function implements the DPO algorithm as described in:
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    (https://arxiv.org/abs/2305.18290)

    The loss combines two main components:
    1. Preference Loss: Optimizes the model to prefer chosen responses over rejected ones
    2. SFT Loss (optional): Auxiliary supervised fine-tuning loss on chosen responses

    The total loss is computed as:
    L(θ) = w_p * L_pref(θ) + w_s * L_sft(θ)

    where:
    - w_p is the preference_loss_weight
    - w_s is the sft_loss_weight
    - L_pref(θ) is the preference loss term
    - L_sft(θ) is the supervised fine-tuning loss term

    The preference loss term is computed as:
    L_pref(θ) = -E[log(σ(β * (r_chosen - r_rejected)))]

    where:
    - σ is the sigmoid function
    - β is the reference_policy_kl_penalty
    - r_chosen and r_rejected are the rewards for chosen and rejected responses
    - The rewards are computed as the sum of log probability differences between
      the current policy and reference policy

    If preference_average_log_probs is True, the rewards are averaged over tokens:
    r = (1/n) * Σ_t (log π_θ(a_t|s_t) - log π_ref(a_t|s_t))

    Otherwise, the rewards are summed over tokens.

    The SFT loss term is a standard negative log likelihood loss on the chosen responses.
    If sft_average_log_probs is True, the loss is averaged over tokens.

    Args:
        cfg (DPOLossConfig): Configuration dictionary containing:
            - reference_policy_kl_penalty (float): Strength of the KL penalty term (β)
            - preference_loss_weight (float): Weight for the preference loss term (w_p)
            - sft_loss_weight (float): Weight for the SFT loss term (w_s)
            - preference_average_log_probs (bool): Whether to average log probs across tokens in preference loss
            - sft_average_log_probs (bool): Whether to average log probs across tokens in SFT loss

    Returns:
        tuple[torch.Tensor, dict]: A tuple containing:
            - The total loss value
            - A dictionary with metrics including:
                - loss: Total loss value
                - sft_loss: SFT loss component
                - preference_loss: Preference loss component
                - accuracy: Fraction of examples where chosen response has higher reward
    """

    loss_type = LossType.SEQUENCE_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(self, cfg: DPOLossConfig, use_fused_linear_logprobs: bool = False):
        self.reference_policy_kl_penalty = cfg.reference_policy_kl_penalty
        self.preference_loss_weight = cfg.preference_loss_weight
        self.sft_loss_weight = cfg.sft_loss_weight
        self.preference_average_log_probs = cfg.preference_average_log_probs
        self.sft_average_log_probs = cfg.sft_average_log_probs
        self.use_fused_linear_logprobs = use_fused_linear_logprobs
        self.sft_loss = NLLLossFn(use_fused_linear_logprobs=use_fused_linear_logprobs)
        self.metric_normalizations = {
            "loss": MetricNormalizer.SEQUENCES,
            "sft_loss": MetricNormalizer.SEQUENCES,
            "preference_loss": MetricNormalizer.SEQUENCES,
            "accuracy": MetricNormalizer.SEQUENCES,
            "rewards_chosen_mean": MetricNormalizer.SEQUENCES,
            "rewards_rejected_mean": MetricNormalizer.SEQUENCES,
            "num_valid_samples": MetricNormalizer.NONE,
        }

    def _dpo_loss(
        self,
        next_token_logprobs: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ## TODO(@ashors): there's some duplicate code here with the NLLLossFn function. We should refactor
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]

        ref_logprobs = data["reference_policy_logprobs"][:, :-1]
        diff = (next_token_logprobs - ref_logprobs) * token_mask

        rewards = diff.sum(-1)
        if self.preference_average_log_probs:
            rewards = rewards / token_mask.sum(-1).clamp(min=1)

        return self._preference_loss(
            rewards,
            sample_mask,
            global_valid_seqs,
            self.reference_policy_kl_penalty,
            data=data,
        )

    # TODO a cleaner typing fix would be required (probably that DPOLossFn should not inherit from PreferenceLossFn)
    def __call__(  # type: ignore
        self,
        next_token_logprobs: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: Tensor,
        global_valid_toks: Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sft_loss_chosen = torch.tensor(0.0)
        if self.sft_loss_weight > 0:
            assert global_valid_toks is not None, (
                "global_valid_toks must be provided for SFT loss"
            )
            sft_loss, _ = self.sft_loss(
                next_token_logprobs,
                data,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,  ## unused because sft loss returned is at the sample level
                dpo_loss=True,
                dpo_average_log_probs=self.sft_average_log_probs,
            )
            sft_loss_chosen, _ = self.split_output_tensor(sft_loss, data)
            sample_mask_chosen, _ = self.split_output_tensor(data["sample_mask"], data)
            sft_loss_chosen = masked_mean(
                sft_loss_chosen,
                sample_mask_chosen,
                global_normalization_factor=global_valid_seqs / 2,
            )

        (
            preference_loss,
            accuracy,
            rewards_chosen_mean,
            rewards_rejected_mean,
        ) = self._dpo_loss(next_token_logprobs, data, global_valid_seqs)

        dpo_loss = (
            self.sft_loss_weight * sft_loss_chosen
            + self.preference_loss_weight * preference_loss
        )

        ## divide by 2 because we're summing over (chosen, rejected) pairs
        num_valid_samples = data["sample_mask"].sum() / 2

        return dpo_loss, {
            "loss": dpo_loss.item(),
            "sft_loss": sft_loss_chosen.item(),
            "preference_loss": preference_loss.item(),
            "accuracy": accuracy.item(),
            "rewards_chosen_mean": rewards_chosen_mean.item(),
            "rewards_rejected_mean": rewards_rejected_mean.item(),
            "num_valid_samples": num_valid_samples.item(),
        }


class MPOLossConfig(BaseModel, extra="allow"):
    """Mixed Preference Optimization loss configuration."""

    reference_policy_kl_penalty: float = 0.05
    preference_loss_weight: float = 1.0
    sft_loss_weight: float = 0.0
    bco_loss_weight: float = 1.0
    preference_average_log_probs: bool = False
    sft_average_log_probs: bool = False
    quality_average_log_probs: bool = False
    reward_shift_momentum: float = 0.99
    reward_shift: float = 0.0


class MPOLossFn(PreferenceLossFn):
    """MPO loss with driver-owned BCO reward-shift state.

    The loss combines a DPO preference term, a chosen-response SFT term, and
    a BCO quality term. The BCO shift is intentionally not mutated here:
    loss objects are serialized to distributed workers, so worker-local
    mutation would be lost between optimizer steps and would diverge across
    data-parallel ranks. Instead, the loss returns a raw reward sum/count;
    the algorithm aggregates those values and updates ``reward_shift`` once
    on the driver after each successful training step.
    """

    loss_type = LossType.SEQUENCE_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(
        self, cfg: MPOLossConfig, use_fused_linear_logprobs: bool = False
    ) -> None:
        self.reference_policy_kl_penalty = cfg.reference_policy_kl_penalty
        self.preference_loss_weight = cfg.preference_loss_weight
        self.sft_loss_weight = cfg.sft_loss_weight
        self.bco_loss_weight = cfg.bco_loss_weight
        self.preference_average_log_probs = cfg.preference_average_log_probs
        self.sft_average_log_probs = cfg.sft_average_log_probs
        self.quality_average_log_probs = cfg.quality_average_log_probs
        self.reward_shift_momentum = cfg.reward_shift_momentum
        self.reward_shift = float(cfg.reward_shift)
        self.use_fused_linear_logprobs = use_fused_linear_logprobs

        if not 0.0 <= self.reward_shift_momentum < 1.0:
            raise ValueError("reward_shift_momentum must be in [0, 1)")

        self.metric_normalizations = {
            "loss": MetricNormalizer.SEQUENCES,
            "sft_loss": MetricNormalizer.SEQUENCES,
            "preference_loss": MetricNormalizer.SEQUENCES,
            "bco_loss": MetricNormalizer.SEQUENCES,
            "accuracy": MetricNormalizer.SEQUENCES,
            "rewards_chosen_mean": MetricNormalizer.SEQUENCES,
            "rewards_rejected_mean": MetricNormalizer.SEQUENCES,
            "bco_rewards_chosen_mean": MetricNormalizer.SEQUENCES,
            "bco_rewards_rejected_mean": MetricNormalizer.SEQUENCES,
            "num_valid_samples": MetricNormalizer.NONE,
            # These are sufficient statistics, aggregated by summation before
            # the single driver-side EMA update.
            "bco_reward_sum": MetricNormalizer.NONE,
            "bco_reward_count": MetricNormalizer.NONE,
        }

    def update_reward_shift(self, reward_sum: float, reward_count: float) -> float:
        """Apply one EMA update from globally aggregated BCO rewards."""
        if reward_count <= 0:
            return self.reward_shift
        batch_mean = reward_sum / reward_count
        self.reward_shift = (
            self.reward_shift_momentum * self.reward_shift
            + (1.0 - self.reward_shift_momentum) * batch_mean
        )
        return self.reward_shift

    @staticmethod
    def _sequence_logratios(
        next_token_logprobs: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        *,
        average: bool,
    ) -> Tensor:
        token_mask = data["token_mask"][:, 1:]
        reference_logprobs = data["reference_policy_logprobs"][:, :-1]
        logratios = ((next_token_logprobs - reference_logprobs) * token_mask).sum(
            dim=-1
        )
        if average:
            logratios = logratios / token_mask.sum(dim=-1).clamp(min=1)
        return logratios

    def _sft_loss(
        self,
        next_token_logprobs: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: Tensor,
    ) -> Tensor:
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        per_sequence_loss = -(next_token_logprobs * token_mask).sum(dim=-1)
        if self.sft_average_log_probs:
            per_sequence_loss = per_sequence_loss / token_mask.sum(dim=-1).clamp(min=1)
        chosen_loss, _ = self.split_output_tensor(per_sequence_loss, data)
        chosen_mask, _ = self.split_output_tensor(sample_mask, data)
        return masked_mean(
            chosen_loss,
            chosen_mask,
            global_normalization_factor=global_valid_seqs / 2,
        )

    def _bco_loss(
        self,
        rewards: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        sample_mask = data["sample_mask"]
        chosen_rewards, rejected_rewards = self.split_output_tensor(rewards, data)
        chosen_mask, rejected_mask = self.split_output_tensor(sample_mask, data)
        shift = torch.as_tensor(
            self.reward_shift, dtype=rewards.dtype, device=rewards.device
        )
        chosen_loss = (
            -torch.nn.functional.logsigmoid(chosen_rewards - shift) * chosen_mask
        )
        rejected_loss = (
            -torch.nn.functional.logsigmoid(-(rejected_rewards - shift)) * rejected_mask
        )
        pair_count = global_valid_seqs / 2
        loss = masked_mean(
            chosen_loss,
            chosen_mask,
            global_normalization_factor=pair_count,
        ) + masked_mean(
            rejected_loss,
            rejected_mask,
            global_normalization_factor=pair_count,
        )
        chosen_mean = masked_mean(
            chosen_rewards,
            chosen_mask,
            global_normalization_factor=pair_count,
        )
        rejected_mean = masked_mean(
            rejected_rewards,
            rejected_mask,
            global_normalization_factor=pair_count,
        )
        return loss, chosen_mean, rejected_mean

    def __call__(
        self,
        next_token_logprobs: Tensor,
        data: BatchedDataDict[DPOLossDataDict],
        global_valid_seqs: Tensor,
        global_valid_toks: Tensor | None,
    ) -> tuple[Tensor, dict[str, Any]]:
        preference_logratios = self._sequence_logratios(
            next_token_logprobs,
            data,
            average=self.preference_average_log_probs,
        )
        (
            preference_loss,
            accuracy,
            rewards_chosen_mean,
            rewards_rejected_mean,
        ) = self._preference_loss(
            preference_logratios,
            data["sample_mask"],
            global_valid_seqs,
            beta=self.reference_policy_kl_penalty,
            data=data,
        )

        quality_logratios = self._sequence_logratios(
            next_token_logprobs,
            data,
            average=self.quality_average_log_probs,
        )
        quality_rewards = self.reference_policy_kl_penalty * quality_logratios
        bco_loss, bco_chosen_mean, bco_rejected_mean = self._bco_loss(
            quality_rewards, data, global_valid_seqs
        )

        sft_loss = torch.zeros(
            (), dtype=next_token_logprobs.dtype, device=next_token_logprobs.device
        )
        if self.sft_loss_weight > 0:
            sft_loss = self._sft_loss(next_token_logprobs, data, global_valid_seqs)

        total_loss = (
            self.preference_loss_weight * preference_loss
            + self.sft_loss_weight * sft_loss
            + self.bco_loss_weight * bco_loss
        )
        sample_mask = data["sample_mask"].to(dtype=quality_rewards.dtype)
        bco_reward_sum = (quality_rewards.detach() * sample_mask).sum()
        bco_reward_count = sample_mask.sum()

        return total_loss, {
            "loss": total_loss.item(),
            "sft_loss": sft_loss.item(),
            "preference_loss": preference_loss.item(),
            "bco_loss": bco_loss.item(),
            "accuracy": accuracy.item(),
            "rewards_chosen_mean": rewards_chosen_mean.item(),
            "rewards_rejected_mean": rewards_rejected_mean.item(),
            "bco_rewards_chosen_mean": bco_chosen_mean.item(),
            "bco_rewards_rejected_mean": bco_rejected_mean.item(),
            "num_valid_samples": (data["sample_mask"].sum() / 2).item(),
            "bco_reward_sum": bco_reward_sum.item(),
            "bco_reward_count": bco_reward_count.item(),
        }


class DistillationLossConfig(BaseModel, extra="allow"):
    kl_type: str = "mixed"
    mixed_kl_weight: float = 0.5
    zero_outside_topk: bool = False


class DistillationLossDataDict(TypedDict):
    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor
    teacher_topk_logits: torch.Tensor
    teacher_topk_indices: torch.Tensor


class DistillationLossFn(LossFunction):
    """Distillation loss function."""

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.DISTILLATION

    def __init__(self, cfg: DistillationLossConfig):
        self.kl_type = cfg.kl_type
        self.mixed_kl_weight = cfg.mixed_kl_weight
        self.zero_outside_topk = cfg.zero_outside_topk
        self.log_infinitesimal = -100

        assert self.kl_type in ["forward", "reverse", "mixed"], "Invalid KL type"
        assert self.mixed_kl_weight >= 0 and self.mixed_kl_weight <= 1, (
            "Invalid mixed KL weight"
        )

    def __call__(
        self,
        student_topk_logprobs: torch.Tensor,
        teacher_topk_logprobs: torch.Tensor,
        H_all: torch.Tensor | None,
        data: DistillationLossDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute distillation loss between teacher and student logits."""
        student_probs = student_topk_logprobs.exp()  # [B, S-1, k]
        teacher_probs = teacher_topk_logprobs.exp()  # [B, S-1, k]

        loss_correction_term = torch.zeros_like(student_probs[..., 0])  # [B, S-1]
        if self.zero_outside_topk and self.kl_type != "forward":
            H_rest = H_all - (student_probs * student_topk_logprobs).sum(-1)
            P_rest = 1 - (student_probs.sum(-1))
            # The entropy and prob of the rest of the tokens [B, S-1]
            loss_correction_term = H_rest - self.log_infinitesimal * P_rest  # [B, S-1]
            if self.kl_type == "mixed":
                loss_correction_term = loss_correction_term * (
                    1.0 - self.mixed_kl_weight
                )

        if self.kl_type == "forward":
            per_token_kl = teacher_probs * (
                teacher_topk_logprobs - student_topk_logprobs
            )
        elif self.kl_type == "reverse":
            per_token_kl = student_probs * (
                student_topk_logprobs - teacher_topk_logprobs
            )
        else:
            # mixed KL
            kl_forward = teacher_probs * (teacher_topk_logprobs - student_topk_logprobs)
            kl_reverse = student_probs * (student_topk_logprobs - teacher_topk_logprobs)
            per_token_kl = (
                self.mixed_kl_weight * kl_forward
                + (1.0 - self.mixed_kl_weight) * kl_reverse
            )

        per_token_kl = per_token_kl.sum(dim=-1) + loss_correction_term  # [B, S-1]

        # Masking and reduction
        if "token_mask" in data and "sample_mask" in data:
            token_mask = data["token_mask"][:, 1:]
            sample_mask = data["sample_mask"]
            # Align mask length to current per_token_kl
            max_len = per_token_kl.shape[1]
            token_mask = token_mask[:, :max_len]
            mask = token_mask * sample_mask.unsqueeze(-1)  # [B, S-1]
            # align mask shape to per_token_kl
            kl_loss = masked_mean(
                per_token_kl,
                mask,
                global_normalization_factor=global_valid_toks,
            )
        else:
            kl_loss = per_token_kl.mean()

        metrics = {
            "loss": float(kl_loss.item()) if kl_loss.ndim == 0 else kl_loss,
            "num_valid_samples": data["sample_mask"].sum().item(),
        }

        return kl_loss, metrics


class MseValueLossConfig(BaseModel, extra="forbid"):
    """Config for the MSE value loss used by PPO's value model."""

    # Scaling factor applied to the value loss before it is added to the policy loss.
    scale: float = 1.0
    # Clipping range for value predictions (PPO-style). Set to None to disable clipping.
    cliprange: Optional[float] = None


class MseValueLossFn(LossFunction):
    """Mean Squared Error value loss function with optional clipping (PPO-style).

    When ``cliprange`` is set, value predictions are clipped to
    ``[old_values - cliprange, old_values + cliprange]`` and the loss is
    ``0.5 * max(mse(vpred, returns), mse(vpred_clipped, returns))``.
    This prevents the value function from changing too drastically in a
    single update, mirroring the policy ratio clipping in PPO.
    """

    input_type = LossInputType.LOGIT

    def __init__(self, cfg: MseValueLossConfig):
        self.scale = cfg.scale
        self.cliprange = cfg.cliprange
        self.loss_type = LossType.TOKEN_LEVEL
        self.metric_normalizations = {"num_valid_samples": MetricNormalizer.NONE}

    def __call__(
        self,
        logits: torch.Tensor,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute Mean Squared Error value loss, optionally with clipping."""
        # Squeeze trailing singleton from value head output: [B, S, 1] -> [B, S]
        if logits.ndim > 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        values = logits

        token_mask = data["token_mask"]
        sample_mask = data["sample_mask"]
        returns = data["returns"]
        mask = token_mask * sample_mask.unsqueeze(-1)

        if self.cliprange and self.cliprange > 0:
            old_values = data["values"]
            vpred_clipped = torch.clamp(
                values,
                old_values - self.cliprange,
                old_values + self.cliprange,
            )
            vf_losses_unclipped = (values - returns) ** 2
            vf_losses_clipped = (vpred_clipped - returns) ** 2
            vf_losses = torch.max(vf_losses_unclipped, vf_losses_clipped)
            loss = (
                0.5
                * self.scale
                * masked_mean(
                    vf_losses,
                    mask,
                    global_normalization_factor=global_valid_toks,
                )
            )
            vf_clipfrac = masked_mean(
                (vf_losses_clipped > vf_losses_unclipped).float(),
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()
        else:
            loss = torch.nn.functional.mse_loss(values, returns, reduction="none")
            loss = (
                0.5
                * self.scale
                * masked_mean(
                    loss,
                    mask,
                    global_normalization_factor=global_valid_toks,
                )
            )
            vf_clipfrac = 0.0

        with torch.no_grad():
            # Use global_valid_toks so each MB contributes local_sum/global_total.
            # Summing across MBs in ppo.py then gives the correct global mean.
            returns_mean = masked_mean(
                returns,
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()
            values_mean = masked_mean(
                values,
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()

            # Min/max are per-MB; ppo.py takes min/max across MBs.
            masked_values = values[mask.bool()]
            values_min = (
                masked_values.min().item() if masked_values.numel() > 0 else 0.0
            )
            values_max = (
                masked_values.max().item() if masked_values.numel() > 0 else 0.0
            )

            # Explained variance sufficient statistics.
            # EV = 1 - Var(residual) / Var(returns)
            # We export E[r²] and E[(r-v)²] (both / global_valid_toks so they
            # sum correctly across MBs).  ppo.py combines them with returns_mean
            # and values_mean to compute exact global EV.
            returns_sq_mean = masked_mean(
                returns**2,
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()
            residual_sq_mean = masked_mean(
                (returns - values) ** 2,
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()

        metrics = {
            "loss": float(loss.item()),
            "vf_clipfrac": vf_clipfrac,
            "returns_mean": returns_mean,
            "values_mean": values_mean,
            "values_min": values_min,
            "values_max": values_max,
            "returns_sq_mean": returns_sq_mean,
            "residual_sq_mean": residual_sq_mean,
            "num_valid_samples": sample_mask.sum().item(),
        }

        return loss, metrics
