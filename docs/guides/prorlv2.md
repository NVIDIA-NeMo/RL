# An In-Depth Walkthrough of ProRLv2 in NeMo RL

This guide covers the ProRLv2 configuration pattern in NeMo RL, based on the example config [`examples/configs/prorlv2.yaml`](../../examples/configs/prorlv2.yaml).

ProRLv2 (as used in this repo) is best thought of as **GRPO and a bundle of stability/efficiency techniques** commonly used for long-horizon RL fine-tuning

- **DAPO dynamic sampling**: skip prompt-groups with zero reward variance
- **Decoupled (asymmetric) clipping**: `ratio_clip_max > ratio_clip_min`
- **Token-level policy gradient loss**
- **Importance sampling correction and TIS/CE-POP** (especially helpful for MoE/backend-mismatch scenarios)
- **Reinforce++: Decoupled local / global advantage normalization** (`reinforce_plus_plus`)
- **“Stop properly” penalty** for truncated responses

This document focuses on ProRLv2-specific knobs and gotchas. For foundational concepts on GRPO (data, environments, generation backends, loss/metrics), see the [NeMo RL GRPO Guide](grpo.md). For the original DAPO motivation behind dynamic sampling/overlong shaping, see the [NeMo RL DAPO Guide](dapo.md).

## Quickstart: Launch a ProRLv2 Run

Use the example configuration [`examples/configs/prorlv2.yaml`](../../examples/configs/prorlv2.yaml):

```bash
uv run examples/run_grpo_math.py --config examples/configs/prorlv2.yaml {overrides}
```

`prorlv2.yaml` inherits from [`examples/configs/grpo_math_1B.yaml`](../../examples/configs/grpo_math_1B.yaml) and only overrides a small set of fields under `grpo` and `loss_fn`, plus output directories.

**Reminder**: Don’t forget to set your `HF_HOME`, `WANDB_API_KEY`, and `HF_DATASETS_CACHE` (if needed). You’ll need to do a `huggingface-cli login` as well for gated models.

## DAPO: Dynamic Sampling

Standard GRPO will train on all generated responses, even when a prompt’s `num_generations_per_prompt` responses all receive the same reward (no per-prompt learning signal). **Dynamic sampling** filters to keep only prompt-groups with diverse rewards (`std > 0`), and can accumulate across multiple generation batches until it reaches the target rollout batch size.

- **Config**: enable with `grpo.use_dynamic_sampling: true` and tune:
  - `grpo.batch_multiplier`: how many extra prompts to generate to compensate filtering
  - `grpo.dynamic_sampling_max_gen_batches`: upper bound before raising an error
- **Implementation**: see `dynamic_sampling()` in [`nemo_rl/algorithms/grpo.py`](../../nemo_rl/algorithms/grpo.py).

## Advantage Estimator: Reinforce++

The ProRLv2 recipe uses **Reinforce++** advantage estimation instead of the standard GRPO-style group baseline.

Quick intuition:

- Reinforce++ uses **decoupled local + global normalization**.
- Compared to GRPO-style **local-only normalization**, this decoupling can be **more stable** in longer runs (less sensitivity to per-batch scale/variance shifts).

Computation (as implemented in this repo, with the ProRLv2 example defaults):

```text
Defaults in examples/configs/prorlv2.yaml:
  grpo.adv_estimator.minus_baseline = true
  loss_fn.use_kl_in_reward          = false

Steps:
  1) Per prompt-group, compute mean reward, then subtract it:
       a_i = r_i - mean_{j in same prompt} r_j

  2) Global normalize across *all valid response tokens* in the batch:
       A <- (A - mean(A)) / sqrt(max(var(A), 1e-8))
```

```yaml
grpo:
  adv_estimator:
    name: "reinforce_plus_plus"
    normalize_rewards: true
    use_leave_one_out_baseline: false
    minus_baseline: true
```

- **Config**: `grpo.adv_estimator.name: "reinforce_plus_plus"`
- **Implementation**: the training loop wires this via `ReinforcePlusPlusAdvantageEstimator` in [`nemo_rl/algorithms/grpo.py`](../../nemo_rl/algorithms/grpo.py).
- **Reference**: [REINFORCE++ paper](https://arxiv.org/abs/2501.03262)

## Reward Shaping: “Stop properly” Penalty (Truncation Penalty)

When a generation hits the max length without emitting EOS, many pipelines mark it as **truncated**. The “stop properly” penalty scales the reward for truncated samples:

- `stop_properly_penalty_coef = 0.0`: truncated samples get **zero reward**
- `stop_properly_penalty_coef = 1.0`: **no penalty** (keep original rewards)
- Any value in \([0, 1]\) interpolates between the two.

In the example config:

```yaml
grpo:
  reward_shaping:
    enabled: true
    stop_properly_penalty_coef: 0.0
```

- **Implementation**: `apply_reward_shaping()` in [`nemo_rl/algorithms/reward_functions.py`](../../nemo_rl/algorithms/reward_functions.py).

:::{important}
In the current implementation, if `stop_properly_penalty_coef` is set (not `null`), `apply_reward_shaping()` **returns early** after applying truncation scaling. That means you **cannot** apply DAPO "overlong reward shaping" in the same run unless you set `stop_properly_penalty_coef: null` and provide the DAPO overlong parameters (`overlong_buffer_length`, `overlong_buffer_penalty`, `max_response_length`).
:::

## Loss: Decoupled (Asymmetric) Clipping

ProRLv2 uses DAPO’s “decoupled clipping” idea by setting different lower/upper clip bounds:

```yaml
loss_fn:
  ratio_clip_min: 0.2
  ratio_clip_max: 0.27
```

This keeps PPO/GRPO-style clipping behavior but allows a larger expansion region than the contraction region, which can help exploration and reduce early collapse.

- **Implementation**: `ClippedPGLossFn` documents decoupled clipping in [`nemo_rl/algorithms/loss_functions.py`](../../nemo_rl/algorithms/loss_functions.py).

## Loss: Token-level Policy Gradient

ProRLv2 enables token-level loss:

```yaml
loss_fn:
  token_level_loss: true
```

This computes the policy gradient loss per token (under masking) instead of aggregating per sequence, which is often helpful for long CoT/variable-length rollouts.

## Truncated Importance Sampling

When training and generation backends differ (e.g., numerics, precision, MoE routing, or vLLM vs training framework), you may see a mismatch between:

- `generation_logprobs` (logprobs under the generation backend that produced samples)
- `prev_logprobs` (logprobs under the training framework policy)

NeMo RL supports **importance sampling correction**, and ProRLv2’s example config turns it on together with **truncated importance sampling**.

Quick intuition:

- This is mainly useful for **MoE/backend mismatch** cases, where the generation backend and the training policy can disagree on logprobs.
- We compute an importance weight from `prev_logprobs` (training policy) vs `generation_logprobs` (generator). **ICE-POP** drops outliers by zeroing weights outside \([min, max]\).
- In the common setup of **one policy update per rollout batch** (i.e., minibatch equals the per-step rollout batch; no PPO multi-epoch reuse), the PPO/GRPO likelihood ratio term is effectively **1.0** at update time, so the main stability issue is the MoE/backend-mismatch importance weights.
- “Online ICE-POP” here just means applying that ICE-POP filtering **during loss computation** on the current training batch.

- **Reference**: [The Online IcePop Solution for MoE models](https://hijkzzz.notion.site/online-ice-pop)

```yaml
loss_fn:
  use_importance_sampling_correction: true
  truncated_importance_sampling_ratio: 5.0
  truncated_importance_sampling_ratio_min: 0.5
  truncated_importance_sampling_type: "icepop"
```

- **`use_importance_sampling_correction`**: enable token-level importance weights (must be `true` for truncated IS)
- **`truncated_importance_sampling_ratio`**: upper bound (or upper threshold)
- **`truncated_importance_sampling_ratio_min`**: lower bound used by ICE-POP filtering
- **`truncated_importance_sampling_type`**:
  - `"tis"`: clamp weights to `<= truncated_importance_sampling_ratio`
  - `"icepop"`: set weights outside \([min, max]\) to zero (filter outliers)

- **Implementation**: see `ClippedPGLossFn` init-time checks and logic in [`nemo_rl/algorithms/loss_functions.py`](../../nemo_rl/algorithms/loss_functions.py).

## Full Example Config (Annotated)

The ProRLv2 example config is intentionally small and relies on defaults from `grpo_math_1B.yaml`.

- **Example config**: [`examples/configs/prorlv2.yaml`](../../examples/configs/prorlv2.yaml)
- **Base defaults**: [`examples/configs/grpo_math_1B.yaml`](../../examples/configs/grpo_math_1B.yaml)

## Practical Overrides

A few common overrides when launching:

```bash
uv run examples/run_grpo_math.py \
  --config examples/configs/prorlv2.yaml \
  policy.model_name="Qwen/Qwen2.5-1.5B" \
  logger.wandb_enabled=true \
  logger.wandb.project="prorlv2-dev" \
  checkpointing.checkpoint_dir="results/prorlv2" \
  logger.log_dir="logs/prorlv2"
```

If you want to enable DAPO overlong reward shaping instead of stop-properly:

```bash
uv run examples/run_grpo_math.py \
  --config examples/configs/prorlv2.yaml \
  grpo.reward_shaping.stop_properly_penalty_coef=null \
  grpo.reward_shaping.overlong_buffer_length=4096 \
  grpo.reward_shaping.overlong_buffer_penalty=1.0 \
  grpo.reward_shaping.max_response_length=20480
```

## What to Monitor

In addition to task rewards/accuracy, a few stability signals are particularly useful with ProRLv2-style runs:

- **Dynamic sampling efficiency**: if enabled, watch how often batches need multiple generation rounds (see `dapo.md` for detailed guidance).
- **Training–generation mismatch**: `token_mult_prob_error`, `gen_kl_error`, `policy_kl_error`, `js_divergence_error` are computed in `ClippedPGLossFn` (see the [GRPO metrics section](grpo.md#metrics)).
- **Truncation rate**: if high, either increase `policy.max_total_sequence_length`/`policy.generation.max_model_len` or relax truncation penalty (`stop_properly_penalty_coef`).

## References

- **ProRLv2 blog**: [Scaling LLM Reinforcement Learning with Prolonged Training using ProRL v2](https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/)
- **DAPO**: [Decoupled Clip and Dynamic Sampling Policy Optimization](https://arxiv.org/pdf/2503.14476)
- **GRPO**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- **REINFORCE++**: [REINFORCE++](https://arxiv.org/abs/2501.03262)
- **DLER (stop properly penalty explanation)**: [DLER](https://arxiv.org/pdf/2510.15110)
- **[NeMo RL GRPO Guide](grpo.md)**
- **[NeMo RL DAPO Guide](dapo.md)**
