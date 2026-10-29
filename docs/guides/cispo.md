# An in-depth Walkthrough of CISPO in NeMo RL

This guide covers the [Clipped Importance Sampling Policy Optimization (CISPO)](https://arxiv.org/abs/2506.13585) implementation in NeMo RL.

CISPO is a GRPO-family policy-gradient objective that clips the importance-sampling weight as a detached coefficient instead of using PPO-style hard ratio clipping. This keeps a gradient flowing through `log pi_theta` for *every* token while still bounding the scalar importance weight, whereas standard GRPO/PPO-style clipping can zero out the gradient contribution for tokens whose ratios leave the clip range.

This document focuses on CISPO-specific behavior. For foundational concepts on GRPO including data handling, policy training, generation, and loss functions, see the [NeMo RL GRPO Guide](grpo.md). For a concise algorithm reference, see the [CISPO algorithm page](../about/algorithms/cispo.md).

## Quickstart: Launch a CISPO Run

CISPO reuses the GRPO training path and `ClippedPGLossFn`, so it launches with the same script as GRPO. The nightly recipe validates the objective in a high-off-policy setting with repeated updates per rollout and non-colocated async vLLM generation:

```bash
bash tests/test_suites/llm/grpo-cispo-mm1-async-lag1-highoffpolicy-qwen3-30ba3b-3n8g-megatron-cispo.sh
```

The corresponding config is [examples/configs/recipes/llm/grpo-cispo-mm1-async-lag1-highoffpolicy-qwen3-30ba3b-3n8g-megatron-cispo.yaml](../../examples/configs/recipes/llm/grpo-cispo-mm1-async-lag1-highoffpolicy-qwen3-30ba3b-3n8g-megatron-cispo.yaml). It uses `Qwen/Qwen3-30B-A3B`, Megatron policy training, async GRPO with `max_trajectory_age_steps: 1`, and a separate non-colocated vLLM generation node.

**Reminder**: Don't forget to set your `HF_HOME`, `WANDB_API_KEY`, and `HF_DATASETS_CACHE` (if needed). You'll need to do a `huggingface-cli login` as well for gated models.

## The CISPO Objective

For each generated token, CISPO computes the policy ratio

```text
r_t(theta) = pi_theta(o_t | q, o_<t) / pi_old(o_t | q, o_<t)
```

and uses a clipped, stop-gradient importance weight in the policy loss:

```text
L_CISPO = -A_t * sg(clip(r_t(theta), 1 - eps_low, 1 + eps_high)) * log pi_theta(o_t | q, o_<t)
```

where `sg(.)` is the stop-gradient operator. Because the clipped ratio is detached, the gradient always flows through `log pi_theta(o_t | q, o_<t)`, so no token is dropped from the update — only the magnitude of its contribution is bounded.

## Configuration

CISPO is enabled through the `loss_fn` block:

```yaml
loss_fn:
  use_cispo: true
  normalization: global_token_mean
  sequence_level_importance_ratios: false
  force_on_policy_ratio: false
  ratio_clip_min: 1.0
  ratio_clip_max: 5.0
  ratio_clip_c: null
```

**Key Parameters:**
- **`use_cispo`**: Enables the CISPO objective in the shared `ClippedPGLossFn`.
- **`ratio_clip_min` / `ratio_clip_max`**: Follow the paper's additive epsilon convention. The effective clamp range is `[1 - ratio_clip_min, 1 + ratio_clip_max]`. For example, `ratio_clip_min: 1.0` and `ratio_clip_max: 5.0` clamp ratios to `[0, 6]`; since policy ratios are non-negative, this is effectively an upper-only clamp at `6`.
- **`ratio_clip_c`**: Must be `null` for CISPO — dual clipping is incompatible with the CISPO objective.
- **`normalization`**: use `global_token_mean`, or `group_token_mean` on the supported SingleController path. `sequence_mean` is rejected at construction time. The deprecated `token_level_loss: true` alias remains supported when `normalization` is unset. See [Loss Normalization](grpo.md#loss-normalization).

> [!NOTE]
> When `use_importance_sampling_correction: true`, the shared GRPO loss path additionally multiplies the CISPO token loss by the actor-vs-generation correction `exp(prev_logprobs - generation_logprobs)`. This correction is separate from CISPO's clipped `pi_theta / pi_old` weight.

For single-update training, override these fields in the `loss_fn` example above to reuse the training forward's logprobs. In this mode the CISPO ratio `pi_theta / pi_old` is always 1, so the loss does not read `ratio_clip_min` / `ratio_clip_max`: the TIS clamp on the detached current/rollout ratio `exp(curr_logprobs - generation_logprobs)` is the only bound on the weight. The gradient is the same as GRPO with `force_on_policy_ratio: true` and the same TIS settings; only the reported loss value differs.

```yaml
  force_on_policy_ratio: true
  use_importance_sampling_correction: true
  truncated_importance_sampling_type: tis
  truncated_importance_sampling_ratio_min: 0.0
  truncated_importance_sampling_ratio: 6.0
```

Both the GRPO trainer and SingleController skip the previous-policy forward in this mode unless `grpo.seq_logprob_error_threshold` is set (the GRPO trainer also skips it when `loss_fn.seq_logprob_error_in_loss: true`). This mode needs one optimizer update per rollout batch (`policy.train_global_batch_size == grpo.num_prompts_per_step * grpo.num_generations_per_prompt`). A reference-policy forward is still required when reference KL is enabled.

## References

- **CISPO / MiniMax-M1 Paper**: [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)
- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [CISPO algorithm reference](../about/algorithms/cispo.md)
- [NeMo RL GRPO Guide](grpo.md)
- [NeMo RL DAPO Guide](dapo.md)
