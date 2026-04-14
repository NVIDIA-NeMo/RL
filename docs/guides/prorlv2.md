# An In-Depth Walkthrough of ProRLv2 in NeMo RL

This guide covers the ProRLv2 configuration pattern in NeMo RL, based on the example config [`examples/configs/prorlv2.yaml`](../../examples/configs/prorlv2.yaml).

ProRLv2 is best thought of as **GRPO plus a bundle of stability/efficiency techniques** commonly used for long-horizon RL fine-tuning:

- **DAPO dynamic sampling**: skip prompt-groups with zero reward variance
- **Decoupled (asymmetric) clipping**: `ratio_clip_max > ratio_clip_min`
- **Token-level policy gradient loss**
- **Importance sampling correction**: Token-level mask (ICE-POP) / Seq-level mask (seq-mask-tis) for backend-mismatch filtering; Pure Online training recommended for MoE
- **Reinforce++-Baseline**: decoupled local/global advantage normalization (`reinforce_plus_plus` + `minus_baseline: true`)
- **"Stop properly" penalty** for truncated responses

This document focuses on ProRLv2-specific knobs and gotchas. For foundational concepts on GRPO (data, environments, generation backends, loss/metrics), see the [NeMo RL GRPO Guide](grpo.md). For the original DAPO motivation behind dynamic sampling/overlong shaping, see the [NeMo RL DAPO Guide](dapo.md).

## Quickstart: Launch a ProRLv2 Run

Use the example configuration [`examples/configs/prorlv2.yaml`](../../examples/configs/prorlv2.yaml):

```bash
uv run examples/run_grpo_math.py --config examples/configs/prorlv2.yaml {overrides}
```

`prorlv2.yaml` inherits from [`examples/configs/grpo_math_1B.yaml`](../../examples/configs/grpo_math_1B.yaml) and only overrides a small set of fields under `grpo` and `loss_fn`, plus output directories.

**Reminder**: Don't forget to set your `HF_HOME`, `WANDB_API_KEY`, and `HF_DATASETS_CACHE` (if needed). You'll need to do a `huggingface-cli login` as well for gated models.

## DAPO: Dynamic Sampling

Standard GRPO will train on all generated responses, even when a prompt's `num_generations_per_prompt` responses all receive the same reward (no per-prompt learning signal). **Dynamic sampling** filters to keep only prompt-groups with diverse rewards (`std > 0`), and can accumulate across multiple generation batches until it reaches the target rollout batch size.

- **Config**: enable with `grpo.use_dynamic_sampling: true` and tune:
  - `grpo.batch_multiplier`: how many extra prompts to generate to compensate filtering
  - `grpo.dynamic_sampling_max_gen_batches`: upper bound before raising an error
- **Implementation**: see `dynamic_sampling()` in [`nemo_rl/algorithms/grpo.py`](../../nemo_rl/algorithms/grpo.py).

## Decoupled local/global advantage normalization

The ProRLv2 recipe uses **Reinforce++-Baseline** advantage estimation instead of the standard GRPO-style group baseline.

Quick intuition:

- Reinforce++-Baseline uses **decoupled local + global normalization**: per-prompt baseline subtraction (local), then batch-wide normalization (global).
- Compared to GRPO-style **local-only normalization**, this decoupling can be **more stable** in longer runs (less sensitivity to per-batch scale/variance shifts).
- Setting `minus_baseline: false` gives plain Reinforce++ (global normalization only, no per-prompt baseline).

Computation (as implemented in this repo, with the ProRLv2 example defaults):

```text
Defaults in examples/configs/prorlv2.yaml:
  grpo.adv_estimator.minus_baseline = true
  loss_fn.use_kl_in_reward          = false

Steps:
  1) Per prompt-group, compute mean reward, then subtract it (minus_baseline=true):
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
    minus_baseline: true    # true = Reinforce++-Baseline; false = plain Reinforce++
```

- **Config**: `grpo.adv_estimator.name: "reinforce_plus_plus"` with `minus_baseline: true`
- **Implementation**: `ReinforcePlusPlusAdvantageEstimator` in [`nemo_rl/algorithms/advantage_estimator.py`](../../nemo_rl/algorithms/advantage_estimator.py).
- **Reference**: [REINFORCE++ paper](https://arxiv.org/abs/2501.03262)

## Reward Shaping: "Stop properly" Penalty (Truncation Penalty)

When a generation hits the max length without emitting EOS, many pipelines mark it as **truncated**. The "stop properly" penalty scales the reward for truncated samples:

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

ProRLv2 uses DAPO's "decoupled clipping" idea by setting different lower/upper clip bounds:

```yaml
loss_fn:
  ratio_clip_min: 0.2
  ratio_clip_max: 0.27
```

This keeps PPO/GRPO-style clipping behavior but allows a larger expansion region than the contraction region, which can help exploration and reduce early collapse.

- **Implementation**: `ClippedPGLossFn` documents decoupled clipping in [`nemo_rl/algorithms/loss/loss_functions.py`](../../nemo_rl/algorithms/loss/loss_functions.py).

## Loss: Token-level Policy Gradient

ProRLv2 enables token-level loss:

```yaml
loss_fn:
  token_level_loss: true
```

This computes the policy gradient loss per token (under masking) instead of aggregating per sequence, which is often helpful for long CoT/variable-length rollouts.

## Importance Sampling Correction and MoE Stability

When training and generation backends differ (e.g., FSDP vs vLLM, BF16 vs FP8), the logprobs they produce can disagree. NeMo RL captures this disagreement as an IS correction weight:

$$\rho_t = \exp(\texttt{prev\_logprobs}_t - \texttt{generation\_logprobs}_t)$$

For dense models $\rho_t$ stays close to 1.0. For **MoE models**, even tiny floating-point differences can flip the router's expert selection, causing $\rho_t$ to spike. This is one of two sources of instability that MoE models face under RL:

| Source | Cause | Solution |
|--------|-------|----------|
| **Router Shift** (algorithmic) | Multi mini-batch updates change the router too fast. | **Pure Online Training** — set `max_num_epochs: 1` (default in `grpo_math_1B.yaml`). |
| **Backend Mismatch** (engineering) | FSDP and vLLM compute different logits; small differences flip expert selection. | **IS Correction + Filtering** — ICE-POP or seq-mask-tis (see below). |

Pure Online training eliminates router shift by ensuring the policy is updated only once per rollout. This leaves backend mismatch as the remaining source of instability, which is handled by filtering.

### Filtering Strategies

All filtering modes require `use_importance_sampling_correction: true`. ICE-POP and seq-mask-tis are both designed for MoE stability; **seq-mask-tis is recommended for reasoning tasks**.

- **Implementation**: `ClippedPGLossFn` in [`nemo_rl/algorithms/loss/loss_functions.py`](../../nemo_rl/algorithms/loss/loss_functions.py).

---

**`"tis"` — Clamp to Max**

Clamp IS weights to `<= truncated_importance_sampling_ratio`. Simple but retains biased signal from router-flipped tokens.

---

**`"icepop"` — Token-Level Masking**

**Zeros out** any token whose IS weight falls outside \([min, max]\). Unlike clamping, this discards noisy tokens entirely — the model only updates on tokens where the two backends agree.

```yaml
loss_fn:
  use_importance_sampling_correction: true
  truncated_importance_sampling_type: "icepop"
  truncated_importance_sampling_ratio: 5.0
  truncated_importance_sampling_ratio_min: 0.5
```

- **Reference**: [Conquering the RL Stability Challenge in MoE with the Icepop Discrepancy Reduction Method](https://hijkzzz.notion.site/online-ice-pop)

---

**`"seq-mask-tis"` — Sequence-Level Masking (Recommended)**

Token-level masking can break the logical coherence of Chain-of-Thought gradients. `seq-mask-tis` operates at the **sequence level** instead:

1. Compute the **geometric mean** of per-token IS ratios for each sequence.
2. **Discard entire sequences** whose geometric mean falls outside a tight tolerance window.
3. For retained sequences, use the **raw, non-truncated** token-level IS ratios — no clamping, no per-token filtering.

```yaml
loss_fn:
  truncated_importance_sampling_type: "seq-mask-tis"
  truncated_importance_sampling_ratio: 1.002
  truncated_importance_sampling_ratio_min: 0.999
```

:::{note}
Under Pure Online training (single update per rollout), the PPO policy ratio is ~1.0, so the effective loss simplifies to $L(\theta) \approx -\mathbb{E}_t[M_{\text{seq}} \cdot \rho_t \cdot A_t]$.
:::

- **Reference**: [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)

---

**Comparison**

| | ICE-POP | seq-mask-tis |
|---|---|---|
| Granularity | per token | per sequence |
| IS weights for retained data | filtered (zeroed outside bounds) | raw / non-truncated |
| Typical bounds | \[0.5, 5.0\] | \[0.999, 1.002\] |
| Best for | General MoE stability | Long-horizon reasoning (preserves CoT coherence) |

:::{tip}
**ProRL v2.1** targets MoE stability with three changes on top of ProRLv2:

- **Pure Online Policy Gradient** (`force_on_policy_ratio: true`) — forces the PPO ratio to 1.0, eliminating clipping entirely. This removes Router Shift as a source of instability.
- **Seq-mask-tis** — sequence-level filtering instead of token-level ICE-POP.
- **No DAPO dynamic sampling** (`use_dynamic_sampling: false`).

Use [`examples/configs/prorlv2_1_moe.yaml`](../../examples/configs/prorlv2_1_moe.yaml) directly. See the [ProRL v2.1 blog post](https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/) for the full motivation.
:::

```bash
# Launch ProRL v2.1 for MoE models
uv run examples/run_grpo_math.py --config examples/configs/prorlv2_1_moe.yaml {overrides}
```

## Full Example Configs

- **ProRLv2** (ICE-POP, DAPO, clipping): [`examples/configs/prorlv2.yaml`](../../examples/configs/prorlv2.yaml) — inherits from [`grpo_math_1B.yaml`](../../examples/configs/grpo_math_1B.yaml)
- **ProRL v2.1** (seq-mask-tis, pure online, MoE): [`examples/configs/prorlv2_1_moe.yaml`](../../examples/configs/prorlv2_1_moe.yaml) — inherits from `prorlv2.yaml`, adds `force_on_policy_ratio`, switches to seq-mask-tis, disables dynamic sampling

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
- **IS out-of-bounds ratio** (`is_oob_ratio`): the fraction of tokens (ICE-POP) or sequences (seq-mask-tis) filtered out by truncated IS. A persistently high value suggests large backend mismatch — check precision settings or relax the bounds.
- **Truncation rate**: if high, either increase `policy.max_total_sequence_length`/`policy.generation.max_model_len` or relax truncation penalty (`stop_properly_penalty_coef`).

## References

- **ProRLv2 blog**: [Scaling LLM Reinforcement Learning with Prolonged Training using ProRL v2](https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/)
- **DAPO**: [Decoupled Clip and Dynamic Sampling Policy Optimization](https://arxiv.org/pdf/2503.14476)
- **GRPO**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- **REINFORCE++**: [REINFORCE++: A Simple and Efficient Approach for RLHF](https://arxiv.org/abs/2501.03262)
- **DLER (stop properly penalty explanation)**: [DLER](https://arxiv.org/pdf/2510.15110)
- **Online ICE-POP**: [Conquering the RL Stability Challenge in MoE with the Icepop Discrepancy Reduction Method](https://hijkzzz.notion.site/online-ice-pop)
- **seq-mask-tis blog**: [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
- **[NeMo RL GRPO Guide](grpo.md)**
- **[NeMo RL DAPO Guide](dapo.md)**
