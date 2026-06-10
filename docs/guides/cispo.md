# An in-depth Walkthrough of CISPO in NeMo RL

This guide covers the [Clipped Importance Sampling Policy Optimization (CISPO)](https://arxiv.org/abs/2506.13585) implementation in NeMo RL.

CISPO is a GRPO-family policy-gradient objective that clips the importance-sampling weight as a detached coefficient instead of using PPO-style hard ratio clipping. This keeps a gradient flowing through `log pi_theta` for *every* token while still bounding the scalar importance weight, whereas standard GRPO/PPO-style clipping can zero out the gradient contribution for tokens whose ratios leave the clip range.

This document focuses on CISPO-specific behavior. For foundational concepts on GRPO including data handling, policy training, generation, and loss functions, see the [NeMo RL GRPO Guide](grpo.md). For a concise algorithm reference, see the [CISPO algorithm page](../about/algorithms/cispo.md).

## Quickstart: Launch a CISPO Run

CISPO reuses the GRPO training path and `ClippedPGLossFn`, so it launches with the same script as GRPO. The nightly recipe validates the objective in a high-off-policy setting with repeated updates per rollout and non-colocated async vLLM generation:

```bash
bash tests/test_suites/llm/cispo-mm1-async-lag1-highoffpolicy-qwen3-30ba3b-3n8g-megatron-cispo.sh
```

The corresponding config is [examples/configs/recipes/llm/cispo-mm1-async-lag1-highoffpolicy-qwen3-30ba3b-3n8g-megatron-cispo.yaml](../../examples/configs/recipes/llm/cispo-mm1-async-lag1-highoffpolicy-qwen3-30ba3b-3n8g-megatron-cispo.yaml). It uses `Qwen/Qwen3-30B-A3B`, Megatron policy training, async GRPO with `max_trajectory_age_steps: 1`, and a separate non-colocated vLLM generation node.

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
  token_level_loss: true
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
- **`token_level_loss`**: CISPO is defined per token, so keep this `true`.

> [!NOTE]
> When `use_importance_sampling_correction: true`, the shared GRPO loss path additionally multiplies the CISPO token loss by the actor-vs-generation correction `exp(prev_logprobs - generation_logprobs)`. This correction is separate from CISPO's clipped `pi_theta / pi_old` weight.

## References

- **CISPO / MiniMax-M1 Paper**: [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)
- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [CISPO algorithm reference](../about/algorithms/cispo.md)
- [NeMo RL GRPO Guide](grpo.md)
- [NeMo RL DAPO Guide](dapo.md)
