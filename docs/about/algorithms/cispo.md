# CISPO

[Clipped Importance Sampling Policy Optimization (CISPO)](https://arxiv.org/abs/2506.13585) is a GRPO-family policy-gradient objective that clips the importance-sampling weight as a detached coefficient instead of using PPO-style hard ratio clipping.

For each generated token, CISPO computes the policy ratio

```text
r_t(theta) = pi_theta(o_t | q, o_<t) / pi_old(o_t | q, o_<t)
```

and uses a clipped, stop-gradient importance weight in the policy loss:

```text
L_CISPO = -A_t * sg(clip(r_t(theta), 1 - eps_low, 1 + eps_high)) * log pi_theta(o_t | q, o_<t)
```

This keeps gradients flowing through `log pi_theta` for every token while bounding the scalar importance weight. In contrast, standard GRPO/PPO-style clipping can zero out the gradient contribution for tokens whose ratios leave the clip range.

## Configuration

CISPO uses the same GRPO training path and `ClippedPGLossFn` as GRPO. Enable it in the `loss_fn` block:

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

`ratio_clip_min` and `ratio_clip_max` follow the paper's additive epsilon convention. The effective clamp range is:

```text
[1 - ratio_clip_min, 1 + ratio_clip_max]
```

For example, `ratio_clip_min: 1.0` and `ratio_clip_max: 5.0` clamp ratios to `[0, 6]`. Since policy ratios are non-negative, this is effectively an upper-only clamp at `6`.

When `use_importance_sampling_correction: true`, the shared GRPO loss path additionally multiplies the CISPO token loss by the actor-vs-generation correction `exp(prev_logprobs - generation_logprobs)`. This correction is separate from CISPO's clipped `pi_theta / pi_old` weight.

## Async Lag-1 Recipe

The nightly CISPO recipe validates the objective in a high-off-policy setting with repeated updates per rollout and non-colocated async vLLM generation:

```bash
bash tests/test_suites/llm/cispo-mm1-async-lag1-highoffpolicy-qwen3-30ba3b-3n8g-megatron-cispo.sh
```

The corresponding config is:

```text
examples/configs/recipes/llm/cispo-mm1-async-lag1-highoffpolicy-qwen3-30ba3b-3n8g-megatron-cispo.yaml
```

The recipe uses `Qwen/Qwen3-30B-A3B`, Megatron policy training, async GRPO with `max_trajectory_age_steps: 1`, and a separate non-colocated vLLM generation node.

## Additional Resources

- [In-depth CISPO Guide](../../guides/cispo.md)
- [MiniMax-M1 paper](https://arxiv.org/abs/2506.13585)
- [GRPO documentation](grpo.md)
- [DAPO documentation](dapo.md)
