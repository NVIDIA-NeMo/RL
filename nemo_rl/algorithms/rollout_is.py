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
"""
Rollout Importance Sampling (IS) Helper Module

This module handles importance sampling weight computation for correcting
distribution mismatch between rollout policy (e.g., vLLM BFloat16) and
training policy (e.g., FSDP FP32).

Key Features:
1. Three aggregation levels: token, sequence, geometric
2. Two handling modes: truncate, mask
3. Per-token veto mechanism for catastrophic outliers
4. Memory-efficient computation to prevent CUDA OOM
5. Comprehensive metrics tracking

Usage:
- compute_rollout_importance_weights() computes both IS weights and mismatch metrics
- Used in ClippedPGLossFn to correct for rollout-training distribution mismatch

References:
- When Speed Kills Stability: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda
- Off-policy RL: https://fengyao.notion.site/off-policy-rl
"""

from typing import Any, Optional

import torch

from nemo_rl.algorithms.utils import masked_mean


def compute_rollout_importance_weights(
    train_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    token_mask: torch.Tensor,
    rollout_is_level: str = "token",
    rollout_is_mode: str = "truncate",
    rollout_is_threshold: Optional[float] = None,
    rollout_is_threshold_lower: Optional[float] = None,
    rollout_is_veto_threshold: Optional[float] = None,
    global_valid_tokens: Optional[torch.Tensor] = None,
    global_valid_seqs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compute importance sampling weights and rejection mask for rollout-training mismatch.

    This function computes IS weights to correct for distribution mismatch between rollout
    and training policies, and applies rejection sampling for outliers.

    Key Design: Separation of IS Weights and Rejection Sampling
    - IS weights (rollout_is_weights): Ratios π_train/π_rollout with processing applied:
      * Safety-bounded to prevent overflow:
        - Token level: exp(clamp(log_ratio, -20, 20)) per token
        - Sequence level: exp(clamp(sum(log_ratio), -20, 20)) broadcast to all tokens
        - Geometric level: exp(clamp(mean(log_ratio), -20, 20)) broadcast to all tokens
      * Truncate mode: upper clamped via .clamp(max=upper_threshold)
      * Mask mode: safety-bounded ratios preserved (no threshold clamping)
      * All modes: zeroed at padding positions
      Used for policy gradient calculations
    - Token mask (modified_token_mask): Has rejection applied (mask mode + veto)
      Used for loss aggregation to exclude rejected samples from training

    Reference:
        When Speed Kills Stability: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda

    Memory-efficient implementation:
    - Log-space computation to prevent overflow
    - Safety bounds (exp(±20)) on all exponentiations
    - Metrics computed without large intermediate tensors

    Args:
        train_log_prob: Log probs from training policy (FSDP FP32), shape (batch_size, seq_length)
        rollout_log_prob: Log probs from rollout policy (vLLM BF16), shape (batch_size, seq_length)
        token_mask: Valid token mask (1=valid, 0=padding), shape (batch_size, seq_length)
        rollout_is_level: IS weight aggregation level
            - "token": Per-token ratios ρ_t = π_train(t)/π_rollout(t) (biased but low variance)
            - "sequence": Sequence product ρ_seq = ∏ρ_t (unbiased but high variance)
            - "geometric": Geometric mean ρ_geo = (∏ρ_t)^(1/T) (experimental trade-off)
        rollout_is_mode: Treatment of outlier IS weights
            - "truncate": Clamp weights at upper threshold only. No rejection for outlier ratios,
              but veto can still apply (TIS)
            - "mask": Reject tokens/sequences outside [lower, upper] via token_mask (MIS/rejection sampling)
        rollout_is_threshold: Upper threshold for IS weights (required, e.g., 2.0)
        rollout_is_threshold_lower: Lower threshold for mask mode (if None, defaults to 1/upper)
        rollout_is_veto_threshold: Catastrophic token threshold. If any token has ratio < this,
            reject entire sequence. Applied independently of rollout_is_mode. If None, veto disabled. Default None.

    Returns:
        Tuple of (rollout_is_weights, modified_token_mask, metrics):
            rollout_is_weights: Processed IS weights, shape (batch_size, seq_length). Processing applied:
                - Safety-bounded to [exp(-20), exp(20)] ≈ [2e-9, 5e8]:
                  * Token level: bounds per-token ratios
                  * Sequence/geometric level: bounds aggregated ratio (broadcast to all tokens)
                - Truncate mode: upper clamped via .clamp(max=upper_threshold)
                - Mask mode: safety-bounded ratios preserved (no threshold clamping)
                - All modes: zeroed at padding positions (token_mask == 0)
            modified_token_mask: Token mask with rejection applied:
                - truncate mode: unchanged for outlier ratios, but veto rejection still applied
                - mask mode: tokens outside [lower, upper] masked to 0
                - veto: sequences with catastrophic tokens masked to 0 (applied in both modes)
                Shape (batch_size, seq_length).
            metrics: Dict of IS and mismatch metrics, all scalars
    """
    # Step 0: Detect and mask out sequences with NaN values
    # Check for NaN in either train or rollout log probs
    has_nan_train = torch.isnan(train_log_prob)
    has_nan_rollout = torch.isnan(rollout_log_prob)
    has_nan = has_nan_train | has_nan_rollout

    # Mask out entire sequences if they contain any NaN (sequence-level rejection)
    seq_has_nan = (has_nan & token_mask.bool()).any(dim=-1, keepdim=True)  # (batch_size, 1)
    nan_mask = (~seq_has_nan).float().expand_as(token_mask)  # Broadcast to all tokens

    # Apply NaN mask to token_mask
    token_mask = token_mask * nan_mask

    # Track NaN statistics
    nan_metrics = {}
    nan_metrics["rollout/nan_token_fraction"] = masked_mean(has_nan.float(), token_mask + has_nan.float(), global_normalization_factor=global_valid_tokens) if (token_mask.sum() + has_nan.sum()) > 0 else torch.tensor(0.0)
    nan_metrics["rollout/nan_seq_fraction"] = seq_has_nan.float().mean()
    nan_metrics["rollout/nan_train_token_fraction"] = masked_mean(has_nan_train.float(), token_mask + has_nan_train.float(), global_normalization_factor=global_valid_tokens) if (token_mask.sum() + has_nan_train.sum()) > 0 else torch.tensor(0.0)
    nan_metrics["rollout/nan_rollout_token_fraction"] = masked_mean(has_nan_rollout.float(), token_mask + has_nan_rollout.float(), global_normalization_factor=global_valid_tokens) if (token_mask.sum() + has_nan_rollout.sum()) > 0 else torch.tensor(0.0)

    if rollout_is_threshold is None:
        # Return identity weights and unchanged mask
        return torch.ones_like(train_log_prob), token_mask, nan_metrics

    # Parse thresholds: if lower not specified, use 1/upper (reciprocal)
    upper_threshold = rollout_is_threshold
    if rollout_is_threshold_lower is not None:
        lower_threshold = rollout_is_threshold_lower
    else:
        # Default: lower = 1/upper (reciprocal)
        lower_threshold = 1.0 / upper_threshold

    # Step 1: Compute raw importance weights based on the specified level
    log_ratio = train_log_prob - rollout_log_prob

    # Pre-compute log thresholds
    device = train_log_prob.device
    log_threshold_upper = torch.log(torch.tensor(upper_threshold, device=device))
    log_threshold_lower = torch.log(torch.tensor(lower_threshold, device=device))

    # Safety bound to prevent numerical overflow (exp(20) ≈ 485M)
    SAFETY_BOUND = 20.0

    # Store unclamped values in log-space for accurate metrics
    if rollout_is_level == "token":
        # Token-level IS: π_train(a|s) / π_rollout(a|s) per token
        log_ratio_for_metrics = log_ratio * token_mask

        # Apply safety bound to prevent overflow
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        # rollout_is_weights = torch.exp(log_ratio_safe)
        rollout_is_weights = torch.ones_like(train_log_prob)

    elif rollout_is_level == "sequence":
        # Sequence-level IS: π_train(y|x) / π_rollout(y|x) for entire sequence
        # Product of token ratios: exp(Σ log(π_train/π_rollout))
        log_ratio_sum = (log_ratio * token_mask).sum(dim=-1, keepdim=True)
        log_ratio_for_metrics = log_ratio_sum  # Store for metrics

        # Apply safety bound to prevent overflow
        log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(train_log_prob)

    elif rollout_is_level == "geometric":
        # Geometric mean IS: (∏ π_train/π_rollout)^(1/T)
        # Equivalent to exp(mean(log(π_train/π_rollout)))
        num_tokens = token_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        log_ratio_mean = (log_ratio * token_mask).sum(dim=-1, keepdim=True) / num_tokens
        log_ratio_for_metrics = log_ratio_mean  # Store for metrics

        # Geometric mean rarely explodes due to averaging, but apply safety bound anyway
        log_ratio_mean_safe = torch.clamp(log_ratio_mean, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights = torch.exp(log_ratio_mean_safe).expand_as(train_log_prob)

    else:
        raise ValueError(
            f"Invalid rollout_is_level: {rollout_is_level}. Must be 'token', 'sequence', or 'geometric'."
        )

    # Step 1.5: Apply per-token veto check in log space (memory efficient)
    if rollout_is_veto_threshold is not None:
        log_veto_threshold = torch.log(torch.tensor(rollout_is_veto_threshold, device=device))

        # Check if any token ratio is below veto threshold (in log space)
        # log(π_train/π_rollout) < log(veto_threshold) ⟺ π_train/π_rollout < veto_threshold
        catastrophic_tokens = (log_ratio < log_veto_threshold) & token_mask.bool()

        # For each sequence, check if it has any catastrophic token
        # Use broadcasting instead of expand_as to save memory
        # has_catastrophic = catastrophic_tokens.any(dim=-1, keepdim=True)

        # Create veto mask: 0 if sequence has catastrophic token, 1 otherwise
        veto_mask = (~catastrophic_tokens).float()
    else:
        # No veto mechanism
        catastrophic_tokens = torch.zeros_like(token_mask, dtype=torch.bool)
        # has_catastrophic = torch.zeros((train_log_prob.size(0), 1), dtype=torch.bool, device=device)
        veto_mask = torch.ones((train_log_prob.size(0), 1), dtype=torch.float32, device=device)

    # Step 2: Compute comprehensive metrics
    metrics = compute_is_metrics(
        rollout_is_weights=rollout_is_weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        token_mask=token_mask,
        rollout_is_level=rollout_is_level,
        rollout_is_threshold=upper_threshold,
        rollout_is_threshold_lower=lower_threshold,
        log_threshold_upper=log_threshold_upper,
        log_threshold_lower=log_threshold_lower,
        # has_catastrophic=has_catastrophic,
        catastrophic_tokens=catastrophic_tokens,
        SAFETY_BOUND=SAFETY_BOUND,
        global_valid_tokens=global_valid_tokens,
        global_valid_seqs=global_valid_seqs,
    )

    # Step 3: Apply outlier handling and rejection sampling
    # Key design principle: IS weights and rejection are separate mechanisms
    # - rollout_is_weights: IS weight ratios with mode-specific processing
    #   * Truncate mode: upper clamped to prevent extreme values
    #   * Mask mode: safety-bounded ratios preserved (no threshold clamping, rejection via mask)
    #   Used for policy gradient calculations
    # - modified_token_mask: Has rejection applied (excludes outliers from training)
    #   Used for loss denominator: ensures rejected samples don't dilute gradients

    if rollout_is_mode == "truncate":
        # Truncated IS (TIS): clamp weights to prevent extreme importance ratios
        # Weights are modified by clamping; no rejection via mask for outlier ratios
        # Veto rejection (if enabled) will still be applied to modified_token_mask below
        rollout_is_weights = rollout_is_weights.clamp(max=upper_threshold)
        modified_token_mask = token_mask  # Unchanged for outlier ratios (veto applied later)

    elif rollout_is_mode == "mask":
        # Masked IS (MIS): rejection sampling for outlier IS weights
        # Reject tokens/sequences with IS ratios outside [lower, upper] via token_mask
        # IS weights themselves are NOT threshold-clamped (remain safety-bounded only)
        mask = (rollout_is_weights >= lower_threshold) & (rollout_is_weights <= upper_threshold)
        mask = mask.float()

        # Compute rejection rate metrics
        metrics["rollout/is_masked_fraction"] = masked_mean(1 - mask, token_mask, global_normalization_factor=global_valid_tokens)
        if rollout_is_level in ["sequence", "geometric"]:
            # Sequence-level: all tokens have same weight, check first token
            metrics["rollout/is_seq_masked_fraction"] = (1 - mask[:, 0]).mean()
        else:
            # Token-level: sequence rejected if ANY token is rejected
            seq_has_masked = ((1 - mask) * token_mask).sum(dim=-1) > 0
            metrics["rollout/is_seq_masked_fraction"] = seq_has_masked.float().mean()

        # Apply rejection via token_mask (NOT by clamping IS weights)
        modified_token_mask = token_mask * mask
        # rollout_is_weights kept as safety-bounded ratios (no threshold clamping)

    else:
        raise ValueError(f"Invalid rollout_is_mode: {rollout_is_mode}. Must be 'truncate' or 'mask'.")

    # Apply veto: reject entire sequences with catastrophic tokens (ratio < veto_threshold)
    # Veto is independent of mode - it applies to modified_token_mask after mode-specific handling
    modified_token_mask = modified_token_mask * veto_mask
    # Note: rollout_is_weights unaffected by veto (already clamped in truncate mode, or kept as-is in mask mode)

    # Zero out padding positions in IS weights for correct aggregation
    # This is different from rejection - padding must be zeroed regardless of mode
    rollout_is_weights = rollout_is_weights * token_mask

    # Detach IS weights - they should be treated as constants for gradient computation
    # We don't want to train the policy to increase/decrease the IS weights
    rollout_is_weights = rollout_is_weights.detach()

    # Compute mismatch metrics (KL, PPL, etc.) and merge with IS metrics
    mismatch_metrics = compute_mismatch_metrics(
        train_log_prob=train_log_prob, rollout_log_prob=rollout_log_prob, token_mask=token_mask
    )
    metrics.update(mismatch_metrics)

    # Merge NaN metrics
    metrics.update(nan_metrics)

    # Convert all tensor metrics to scalars for logging
    metrics_scalar = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_scalar[key] = value.item()
        else:
            metrics_scalar[key] = value

    return rollout_is_weights, modified_token_mask, metrics_scalar


def compute_is_metrics(
    rollout_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    token_mask: torch.Tensor,
    rollout_is_level: str,
    rollout_is_threshold: float,
    rollout_is_threshold_lower: float,
    log_threshold_upper: torch.Tensor,
    log_threshold_lower: torch.Tensor,
    # has_catastrophic: torch.Tensor,
    catastrophic_tokens: torch.Tensor,
    SAFETY_BOUND: float,
    global_valid_tokens: Optional[torch.Tensor] = None,
    global_valid_seqs: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Compute comprehensive metrics for importance sampling weights.

    Reference:
        When Speed Kills Stability: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda

    This function computes metrics using a mix of true unclamped values (for max/min/fractions
    in sequence/geometric mode via log-space) and safety-clamped values (for mean/std/ESS)
    to balance accuracy with numerical stability and avoid overflow.
    """
    # Validate that we have at least one valid sample
    # assert token_mask.any(), "Expected at least one valid token in token_mask"

    metrics = {}
    device = rollout_is_weights.device

    # Track veto statistics
    metrics["rollout/is_sequence_has_catastrophic_token_fraction"] = catastrophic_tokens.any(dim=-1, keepdim=True).float().mean()
    metrics["rollout/is_catastrophic_token_fraction"] = masked_mean(catastrophic_tokens.float(), token_mask, global_normalization_factor=global_valid_tokens)

    # Compute metrics based on IS level
    if rollout_is_level in ["sequence", "geometric"]:
        # For sequence/geometric, compute true statistics from log-space
        # This reflects the actual distribution before clamping

        # True max/min in log space
        log_max = log_ratio_for_metrics.max()
        log_min = log_ratio_for_metrics.min()

        # Convert to regular space with safety bound
        metrics["rollout/is_max"] = torch.exp(torch.clamp(log_max, max=SAFETY_BOUND))
        metrics["rollout/is_min"] = torch.exp(log_min)

        # Mean uses clamped weights to avoid overflow
        metrics["rollout/is_mean"] = masked_mean(rollout_is_weights, token_mask, global_normalization_factor=global_valid_tokens)

        # Compute fraction exceeding threshold in log space (accurate)
        exceeds_upper = log_ratio_for_metrics > log_threshold_upper
        below_lower = log_ratio_for_metrics < log_threshold_lower

        if rollout_is_level == "sequence":
            # For sequence level, all tokens in a sequence have the same weight
            metrics["rollout/is_ratio_fraction_high"] = exceeds_upper.float().mean()
            metrics["rollout/is_ratio_fraction_low"] = below_lower.float().mean()
        else:  # geometric
            # Need to expand to match token dimensions
            exceeds_upper_expanded = exceeds_upper.expand_as(token_mask)
            below_lower_expanded = below_lower.expand_as(token_mask)
            metrics["rollout/is_ratio_fraction_high"] = masked_mean(exceeds_upper_expanded.float(), token_mask, global_normalization_factor=global_valid_tokens)
            metrics["rollout/is_ratio_fraction_low"] = masked_mean(below_lower_expanded.float(), token_mask, global_normalization_factor=global_valid_tokens)

    else:
        # Token-level: compute directly from weights
        metrics["rollout/is_mean"] = masked_mean(rollout_is_weights, token_mask, global_normalization_factor=global_valid_tokens)

        # Fraction exceeding thresholds
        rollout_is_above_threshold = rollout_is_weights > rollout_is_threshold
        rollout_is_below_threshold = rollout_is_weights < rollout_is_threshold_lower
        metrics["rollout/is_ratio_fraction_high"] = masked_mean(rollout_is_above_threshold.float(), token_mask, global_normalization_factor=global_valid_tokens)
        metrics["rollout/is_ratio_fraction_low"] = masked_mean(rollout_is_below_threshold.float(), token_mask, global_normalization_factor=global_valid_tokens)

        # Max/min for token level
        mask_bool = token_mask.bool()
        metrics["rollout/is_max"] = rollout_is_weights.masked_fill(~mask_bool, float("-inf")).max()
        metrics["rollout/is_min"] = rollout_is_weights.masked_fill(~mask_bool, float("inf")).min()

    # Compute standard deviation using clamped weights to avoid overflow
    mask_count = token_mask.sum()
    if mask_count > 1:
        # Use clamped weights for variance to avoid squaring huge values
        weights_for_std = rollout_is_weights.clamp(min=rollout_is_threshold_lower, max=rollout_is_threshold)
        # Use mean from clamped weights for consistency
        mean_clamped = masked_mean(weights_for_std, token_mask, global_normalization_factor=global_valid_tokens)
        rollout_is_var = masked_mean(weights_for_std.square(), token_mask) - mean_clamped.square()
        metrics["rollout/is_std"] = torch.sqrt(torch.clamp(rollout_is_var, min=0.0))
    else:
        metrics["rollout/is_std"] = torch.tensor(0.0, device=device)

    # Effective sample size (use clamped weights to avoid overflow)
    weights_for_ess = rollout_is_weights.clamp(min=rollout_is_threshold_lower, max=rollout_is_threshold)
    mean_for_ess = masked_mean(weights_for_ess, token_mask, global_normalization_factor=global_valid_tokens)
    is_weights_normalized = weights_for_ess / (mean_for_ess + 1e-8)
    metrics["rollout/is_eff_sample_size"] = 1.0 / masked_mean(is_weights_normalized.square(), token_mask, global_normalization_factor=global_valid_tokens)

    # Per-sequence breakdown metrics
    if rollout_is_weights.dim() > 1:
        # Compute mean IS weight per sequence
        num_tokens = token_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        seq_mean_weights = (rollout_is_weights * token_mask).sum(dim=-1) / num_tokens.squeeze(-1)

        # Per-sequence statistics
        metrics["rollout/is_seq_mean"] = seq_mean_weights.mean()
        metrics["rollout/is_seq_std"] = (
            seq_mean_weights.std() if seq_mean_weights.numel() > 1 else torch.tensor(0.0, device=device)
        )
        metrics["rollout/is_seq_max"] = seq_mean_weights.max()
        metrics["rollout/is_seq_min"] = seq_mean_weights.min()

        # Identify most problematic sequences
        seq_deviation = (seq_mean_weights - 1.0).abs()
        metrics["rollout/is_seq_max_deviation"] = seq_deviation.max()

        # Fraction of sequences with high IS weights
        metrics["rollout/is_seq_fraction_high"] = (seq_mean_weights > rollout_is_threshold).float().mean()
        metrics["rollout/is_seq_fraction_low"] = (seq_mean_weights < rollout_is_threshold_lower).float().mean()

    return metrics


def compute_mismatch_metrics(
    train_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    token_mask: torch.Tensor,
    global_valid_tokens: Optional[torch.Tensor] = None,
    global_valid_seqs: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Compute training-inference mismatch metrics.

    These metrics help diagnose the mismatch between the rollout policy (e.g., vLLM)
    and the training policy (e.g., FSDP), which can cause training instability.

    Key metrics:
    - mismatch_kl: Direct KL divergence estimator KL(π_rollout || π_training)
    - mismatch_k3_kl: K3 KL estimator for stability (more stable for small KL)
    - training_ppl: Perplexity of training policy
    - rollout_ppl: Perplexity of rollout policy
    - log_ppl_diff: Difference in log perplexities
    - ppl_ratio: Ratio of training PPL to rollout PPL

    Args:
        train_log_prob: Log probabilities from training policy, shape (batch_size, seq_length)
        rollout_log_prob: Log probabilities from rollout policy, shape (batch_size, seq_length)
        token_mask: Mask for valid tokens, shape (batch_size, seq_length)

    Returns:
        Dictionary of mismatch metrics (without prefix)

    Reference:
    - When Speed Kills Stability: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda
    """
    # Validate that we have at least one valid token
    assert token_mask.any(), "Expected at least one valid token in token_mask"

    metrics = {}

    # 1. Training policy perplexity (always available)
    # Formula: exp(-1/|T| * Σ log π_training(y_t|y_<t))
    # where |T| is the number of tokens generated by the model
    num_tokens = token_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    mean_log_prob_training = (train_log_prob * token_mask).sum(dim=-1) / num_tokens.squeeze(-1)  # (batch_size,)
    training_ppl = torch.exp(-mean_log_prob_training).mean()  # Batch mean of per-sequence PPL
    metrics["mismatch/training_ppl"] = training_ppl

    # Also log log-ppl for easier analysis (avoids exponential scale)
    metrics["mismatch/training_log_ppl"] = (-mean_log_prob_training).mean()

    # 2. Compute rollout mismatch metrics
    # 2a. mismatch_kl: Direct estimator for KL(π_rollout || π_training)
    # This is the standard KL divergence: E[log(π_rollout) - log(π_training)]
    # Positive value means rollout policy is more confident than training policy
    metrics["mismatch/kl"] = masked_mean(rollout_log_prob - train_log_prob, token_mask, global_normalization_factor=global_valid_tokens)

    # 2b. mismatch_k3_kl: K3 estimator for KL(π_rollout || π_training)
    # More stable for small KL values using: E[exp(log_ratio) - log_ratio - 1]
    # Formula: KL ≈ E[r - log(r) - 1] where r = π_training/π_rollout
    log_ratio = train_log_prob - rollout_log_prob
    mismatch_k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
    metrics["mismatch/k3_kl"] = masked_mean(mismatch_k3_kl_matrix, token_mask, global_normalization_factor=global_valid_tokens)

    # 2c. Rollout policy perplexity
    mean_log_prob_rollout = (rollout_log_prob * token_mask).sum(dim=-1) / num_tokens.squeeze(-1)  # (batch_size,)
    rollout_ppl = torch.exp(-mean_log_prob_rollout).mean()  # Batch mean of per-sequence PPL
    metrics["mismatch/rollout_ppl"] = rollout_ppl
    metrics["mismatch/rollout_log_ppl"] = (-mean_log_prob_rollout).mean()

    # 2d. Log PPL difference (sequence-level perplexity d_ifference)
    # log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
    # Since ppl = exp(-log_prob), we have:
    #   log(ppl_ratio) = log(training_ppl/rollout_ppl) = log_ppl_diff
    # Positive value means training assigns lower probability (higher PPL) than rollout
    log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
    metrics["mismatch/log_ppl_diff"] = log_ppl_diff.mean()
    metrics["mismatch/log_ppl_abs_diff"] = log_ppl_diff.abs().mean()
    metrics["mismatch/log_ppl_diff_max"] = log_ppl_diff.max()
    metrics["mismatch/log_ppl_diff_min"] = log_ppl_diff.min()

    # 2e. PPL ratio (how much higher is training PPL vs rollout PPL)
    # IMPORTANT: Compute per-sequence ratio first, then average
    # For numerical stability, compute in log space using log_ppl_diff
    # Note: log_ppl_diff = log(ppl_ratio), so ppl_ratio = exp(log_ppl_diff)
    # This is the inverse of geometric IS: ppl_ratio_i = 1 / geometric_is_i for each sequence
    ppl_ratio = torch.exp(log_ppl_diff).mean()  # mean(exp(log_ppl_diff)) = mean(ppl_ratio_i)
    metrics["mismatch/ppl_ratio"] = ppl_ratio

    return metrics
