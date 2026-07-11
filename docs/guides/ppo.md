# An In-Depth Walkthrough of PPO in NeMo RL

This guide details the [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) implementation within NeMo RL. PPO is an actor-critic reinforcement learning algorithm that jointly trains a **policy** (actor) and a **value function** (critic). The value function estimates per-token state values $V(s_t)$, enabling [Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438) — a temporal-difference method that provides lower-variance advantage signals compared to reward-only baselines. PPO was the core RLHF algorithm used in [InstructGPT](https://arxiv.org/abs/2203.02155) and remains widely used for LLM alignment.

## Quickstart: Launch a PPO Run

To get started quickly, use the script [examples/run_ppo.py](../../examples/run_ppo.py), which demonstrates how to train a model on math problems using PPO. You can launch this script locally or through Slurm. For detailed instructions on setting up Ray and launching a job with Slurm, refer to the [cluster documentation](../cluster.md).

We recommend launching the job using `uv`:

```bash
uv run examples/run_ppo.py --config <PATH TO YAML CONFIG> {overrides}
```

If not specified, `config` will default to [examples/configs/ppo_math_1B.yaml](../../examples/configs/ppo_math_1B.yaml).

**Reminder**: Do not forget to set your HF_HOME, WANDB_API_KEY, and HF_DATASETS_CACHE (if needed). You'll need to do a `huggingface-cli login` as well for gated models.

In this guide, we'll walk through how we handle:

* Data, environments, policy, and generation (shared with GRPO)
* Value model (critic)
* Generalized Advantage Estimation (GAE)
* PPO training loop
* Loss

### Data, Environments, Policy, and Generation

PPO uses the same data handling, environments, policy model, and generation infrastructure as GRPO. For details on datasets, data processors, task–environment mapping, the policy interface, vLLM generation, and performance optimizations (sequence packing, dynamic batching), see the [NeMo RL GRPO Guide](grpo.md).

The PPO configuration uses the `ppo:` key instead of `grpo:`, but data, environment, policy, and generation sections remain identical.

## Value Model (Critic)

The value model is the key addition in PPO compared to GRPO. It is a language model with a scalar value head that predicts per-token state values $V(s_t)$, providing the temporal bootstrapping signal needed for GAE.

We define a [ValueInterface](../../nemo_rl/models/value/interfaces.py) that contains everything needed to run a Value model. Similar to the policy, the Value object holds a [RayWorkerGroup](../../nemo_rl/distributed/worker_groups.py) of SPMD (1 proc/GPU) processes coordinated so it appears like 1 GPU.

The value model supports the **Megatron-Core backend** (`value.megatron_cfg.enabled: true`) and the **DTensor backend** (`value.dtensor_cfg.enabled: true`). It uses the same architecture and tokenizer as the policy (configured via `value.model_name`), but is trained with a separate MSE loss on GAE returns.

### Colocated and Non-Colocated Architecture

By default, PPO uses a colocated architecture where the **policy**, **value model**, and **vLLM/SGLang generation engine** share the same set of GPUs. GPU memory is managed by offloading models to CPU between stages: the value model is loaded to GPU only during its inference and training phases, then offloaded to make room for other components.

PPO also supports **non-colocated generation**, where the generation engine runs on a dedicated set of GPUs separate from training (`policy.generation.colocated.enabled: false`, with `policy.generation.colocated.resources.{gpus_per_node,num_nodes}` sizing the dedicated generation resources). In this mode the policy and value model still share a single `train_cluster` (offloaded to CPU between each other as in the colocated case), while generation runs concurrently on its own `inference_cluster`; weights are synchronized via an NCCL broadcast collective spanning both clusters instead of the CUDA-IPC/ZMQ path used when colocated. Non-colocated generation currently requires the vLLM backend — `setup()` asserts this explicitly, since SGLang does not yet implement the collective rendezvous or weight refit needed for non-colocated mode.

### Asynchronous PPO

PPO supports an **asynchronous** training mode (`ppo.async_ppo.enabled: true`) that mirrors [async GRPO](grpo.md): a background `AsyncTrajectoryCollector` continuously generates trajectories into a replay buffer while the driver trains, so generation and training overlap instead of running in lockstep. Async PPO requires non-colocated vLLM generation (`policy.generation.colocated.enabled: false`, `vllm_cfg.async_engine: true`) and importance-sampling correction (`loss_fn.use_importance_sampling_correction: true`) — all enforced by asserts in `async_ppo_train`. Enable it via `examples/run_ppo.py`, which dispatches to `async_ppo_train` when `ppo.async_ppo.enabled` is set.

Each async step: sample a fixed batch from the replay buffer → compute fresh critic **values** at train time → compute fresh policy/reference logprobs → compute GAE advantages → run the `ppo_epochs` inner loop (critic then actor) → perform a **single** weight refit to the generation engine and bump the replay-buffer weight version. Computing values at train time (rather than stashing them when the trajectory was generated) keeps the critic side as fresh as possible.

**Critic-staleness caveat.** Off-policy trajectories are corrected on the *actor* side by importance sampling (`exp(prev_logprobs - generation_logprobs)`), exactly as in async GRPO. PPO's GAE, however, recursively bootstraps value estimates across each trajectory, so bias from a stale trajectory's actions/rewards compounds along the sequence in a way GRPO's memoryless reward-only advantage does not, and there is **no** corresponding critic-side correction. This bias is *bounded* (not eliminated) by keeping `ppo.async_ppo.max_trajectory_age_steps: 1` (at most one policy version stale), which is the recommended and validated setting. Values above 1 are permitted but only warned about; a rigorous fix (folding importance ratios into the GAE recursion, V-trace style) is future work.

**Critic warmup** (`ppo.policy_training_start_step > 0`) is supported: during warmup the policy is frozen, so each step's weight refit skips the actual weight transfer (generation already holds the correct initial weights) but still advances the replay-buffer weight version so the async pipeline keeps making progress.

Because the actor is frozen at its initial policy `π₀` all the way **through** step `W = ppo.policy_training_start_step` (it first trains *during* step `W`, and the refit that publishes `π₁` to generation happens at the *end* of step `W`), **every rollout — whatever its generation-version tag — was produced by the same model `π₀`**. The generation-version counter still increments each step (so the collector's lead can advance), but it does *not* track policy staleness during warmup: gen-version `g ≤ W` ⇒ policy `π₀`, and `g > W` ⇒ policy `π_{g−W}`. This lets the collector bank cheap frozen rollouts far ahead for free: set `ppo.async_ppo.warmup_max_trajectory_age_steps` above `max_trajectory_age_steps` (`A_t`) and it generates that far ahead while the critic pretrains.

The snap-back is governed by **two distinct boundaries** (this is what makes the knob correct and hang-free):

- The **collector's generation-lead** drops to `A_t` at step `W`, so from step `W+1` it stops over-banking `π₀` and regenerates its lead targets against the freshly-trained policy (`π₁`, `π₂`, …).
- The **buffer's eviction age** stays elevated through step `W + A_t`. A frozen (`π₀`) rollout is within `A_t` *policy*-steps of the actor for every step `s ≤ W + A_t` (its policy-age is `s − W`), so those banked rollouts are admitted as legitimate lag-≤`A_t` data — IS-corrected at train time exactly like normal async — and are only evicted once the actor has genuinely moved more than `A_t` steps past `π₀`. (Snapping the eviction age at `W` instead would discard the still-on-policy boundary batch and **deadlock**, since the collector's lead has already advanced past that target and never regenerates it.)

Note this only improves *throughput* when warmup is **generation-bound** (the collector can actually bank ahead). When critic training is the bottleneck it is a throughput no-op — but it is correct and hang-free either way.

**Reading the age metrics.** `avg_trajectory_age` reports the *generation-version* age (`current_weight_version − gen_version`), which **overcounts** off-policyness across the warmup boundary: every gen-version `≤ W` is the same frozen `π₀`, so a rollout banked at gen-version 0 and consumed at step `W+1` shows `avg_trajectory_age ≈ A_w` even though it is only **1 policy step** off-policy. The metric that actually bounds the importance-sampling correction is `avg_trajectory_policy_age` / `max_trajectory_policy_age` = `max(0, s−W) − max(0, g−W)`, which stays **≤ `max_trajectory_age_steps`** at every step (equal to the gen-version age when there is no warmup). A spike in `avg_trajectory_age` at the boundary with `max_trajectory_policy_age` still at the bound is expected and safe; a `max_trajectory_policy_age` *above* the bound would indicate a real regression. Absent/`null` ⇒ same as `max_trajectory_age_steps`.

**v1 limitations.** Async PPO currently requires the vLLM backend (no SGLang/Megatron generation, no colocated inference). DAPO-style dynamic sampling, reward scaling, and reward shaping are also unsupported in async mode (rejected by `examples/run_ppo.py`).

### Value Model Configuration

```yaml
value:
  model_name: ${policy.model_name}       # Same architecture as policy
  train_global_batch_size: 512
  train_micro_batch_size: 4
  max_total_sequence_length: 16384
  precision: "bfloat16"

  megatron_cfg:
    enabled: true
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 1
    context_parallel_size: 1
    activation_checkpointing: false

    optimizer:
      optimizer: "adam"
      lr: 2.0e-6                         # Typically higher than policy LR
      weight_decay: 0.1
      clip_grad: 1.0

    scheduler:
      lr_decay_style: "constant"
      lr_warmup_iters: 10

    distributed_data_parallel_config:
      overlap_grad_reduce: true
      overlap_param_gather: true
      data_parallel_sharding_strategy: "optim_grads_params"
```

For a DTensor PPO recipe, see [ppo-qwen2.5-1.5b-gsm8k-1n8g-automodel-valuetp2sp.yaml](../../examples/configs/recipes/llm/ppo-qwen2.5-1.5b-gsm8k-1n8g-automodel-valuetp2sp.yaml).

## Generalized Advantage Estimation (GAE)

GAE computes advantages using temporal difference (TD) errors and exponentially-weighted averages:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

This is computed recursively backwards:

$$A_t = \delta_t + \gamma \lambda \cdot A_{t+1}$$

The parameter $\lambda$ controls the bias-variance tradeoff: $\lambda = 0$ gives pure TD (low variance, high bias), while $\lambda = 1$ gives Monte Carlo returns (high variance, low bias). The parameter $\gamma$ is the discount factor.

Token-level rewards are constructed as:
- **KL penalty** at every response token: $r_t^{\text{KL}} = -\beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})_t$
- **Terminal reward** at the last response token: the scalar reward from the environment

The implementation uses carry-forward masking: at masked positions (padding, separators in multi-turn) the running accumulators are preserved from the last valid token rather than being zeroed, correctly skipping over non-response tokens without introducing phantom TD errors.

For implementation details, see [GeneralizedAdvantageEstimator](../../nemo_rl/algorithms/advantage_estimator.py).

### GAE Configuration

```yaml
ppo:
  adv_estimator:
    name: "gae"
    gae_lambda: 0.95         # GAE lambda (bias-variance tradeoff)
    gae_gamma: 1             # Discount factor
    normalize_advantages: true
```

### VAPO Decoupled GAE

NeMo RL supports [VAPO](https://arxiv.org/abs/2504.05118)-style decoupled GAE, which uses separate $\lambda$ values for computing value returns vs. policy advantages. This can improve value function accuracy by using MC-like returns ($\lambda_V = 1$) while keeping the policy advantage signal well-tuned.

Additionally, VAPO introduces a length-adaptive $\lambda_{\text{policy}}$ that adjusts based on response length:

$$\lambda_{\text{policy}} = 1 - \frac{1}{\alpha \cdot l}$$

where $l$ is the response length and $\alpha$ controls the adaptation strength.

```yaml
ppo:
  adv_estimator:
    name: "gae"
    gae_lambda: 0.95
    # VAPO decoupled GAE (set to null to disable)
    gae_lambda_value: 1.0    # MC-like returns for value training
    gae_lambda_policy: null  # Use gae_lambda or length-adaptive
    # Length-adaptive lambda_policy = 1 - 1/(alpha * response_length)
    # 0 = disabled
    length_adaptive_alpha: 0.05
```

### Other Advantage Estimators

While GAE is the default for PPO, the implementation also supports running without a value model via `ppo.adv_estimator.name`:

- **`"raw_reward"`**: Raw reward as advantage (no value model, no baselines)

## PPO Training Loop

The PPO training loop, [ppo_train](../../nemo_rl/algorithms/ppo.py), follows this sequence each step:

1. **Generation**: vLLM generates responses from prompts
2. **Environment scoring**: responses are evaluated by task-specific environments (e.g., math verification)
3. **Value inference**: the value model predicts per-token state values
4. **Logprob computation**: the policy computes log probabilities for advantage estimation
5. **Advantage estimation**: GAE computes advantages using value predictions and rewards
6. **Value training**: the critic is updated first (critic-before-actor, following [veRL](https://arxiv.org/abs/2412.09613))
7. **Policy training**: the actor is updated with the clipped surrogate objective

Steps 6–7 repeat `ppo_epochs` times per rollout before generating new responses.

### Multiple Training Steps per Rollout

Unlike GRPO, which performs one training update per rollout, PPO can perform multiple training steps on the same batch of rollout data:

```yaml
ppo:
  ppo_epochs: 4   # Train 4 times on each rollout batch
```

Each step trains both the critic and the actor on the same advantage estimates computed from the initial rollout.

### Critic Warmup

PPO supports training the value model alone for an initial number of steps before starting policy training. This lets the critic establish reasonable value estimates before the actor begins learning, which can improve training stability.

```yaml
ppo:
  policy_training_start_step: 10  # Train critic only for first 10 steps
```

During warmup, generation and environment scoring still run normally — only policy weight updates are skipped.

## Loss

### Policy Loss

PPO uses the same [ClippedPGLossFn](../../nemo_rl/algorithms/loss/loss_functions.py) as GRPO:

$$
L(\theta) = E_{x \sim \pi_{\theta_{\text{old}}}} \Big[ \min \Big(\frac{\pi_\theta(x)}{\pi_{\theta_{\text{old}}}(x)}A_t, \text{clip} \big( \frac{\pi_\theta(x)}{\pi_{\theta_{\text{old}}}(x)}, 1 - \varepsilon, 1 + \varepsilon \big) A_t \Big) \Big] - \beta D_{\text{KL}} (\pi_\theta \| \pi_\text{ref})
$$

The key difference is that $A_t$ comes from GAE (temporal bootstrapping with value function) rather than group-relative baselines. All loss improvements documented in the [GRPO Guide](grpo.md) (dual-clipping, on-policy KL approximation, importance sampling correction, overlong filtering, top-p/top-k filtering) apply equally to PPO.

### Value Loss

The value function is trained with a clipped MSE loss via [MseValueLossFn](../../nemo_rl/algorithms/loss/loss_functions.py):

$$L_V = \frac{1}{2} \max\left((V_\theta - R)^2,\; (V_{\text{clipped}} - R)^2\right)$$

where $V_{\text{clipped}} = \text{clamp}(V_\theta,\; V_{\text{old}} - \epsilon_v,\; V_{\text{old}} + \epsilon_v)$ and $R$ are the GAE returns. This prevents the value function from changing too drastically in a single update, analogous to the policy ratio clipping in the actor loss.

Key parameters:
- **`value_loss_fn.scale`**: Scaling factor for the value loss (default: 1.0; reference recipe overrides to 0.4)
- **`value_loss_fn.cliprange`**: Clip range $\epsilon_v$ for value predictions (default: `null` / disabled; reference recipe overrides to 0.2). Set to `null` to disable clipping.
- **`loss_fn.positive_example_nll_weight`**: VAPO NLL auxiliary loss weight on correct samples (0 = disabled)

## Configuration

```yaml
ppo:
  num_prompts_per_step: 32
  num_generations_per_prompt: 16
  max_rollout_turns: 1
  max_num_epochs: 100000
  max_num_steps: 100000
  ppo_epochs: 4
  policy_training_start_step: 0
  val_period: 20
  val_at_start: true
  val_at_end: false
  seed: 42
  use_dynamic_sampling: false
  overlong_filtering: false

  adv_estimator:
    name: "gae"
    gae_lambda: 0.95
    gae_gamma: 1
    normalize_advantages: true
    gae_lambda_value: null
    gae_lambda_policy: null
    length_adaptive_alpha: 0.0

  reward_scaling:
    enabled: true
    source_min: 0.0
    source_max: 1.0
    target_min: -1.0
    target_max: 1.0

  reward_shaping:
    enabled: true
    overlong_buffer_length: 2048
    overlong_buffer_penalty: 1
    max_response_length: 14336
    stop_properly_penalty_coef: null

loss_fn:
  reference_policy_kl_penalty: 0.0
  ratio_clip_min: 0.2
  ratio_clip_max: 0.28
  ratio_clip_c: 10
  token_level_loss: true
  positive_example_nll_weight: 0.0

value_loss_fn:
  scale: 0.4
  cliprange: 0.2
```

**PPO-specific parameters:**
- **`ppo.ppo_epochs`**: Number of training updates per rollout batch
- **`ppo.policy_training_start_step`**: Number of critic-only warmup steps before policy training begins
- **`ppo.adv_estimator.name`**: Set to `"gae"` for GAE advantage estimation (PPO default)
- **`ppo.adv_estimator.gae_lambda`**: GAE $\lambda$ parameter (bias-variance tradeoff, typically 0.95)
- **`ppo.adv_estimator.gae_gamma`**: Discount factor $\gamma$ (typically 1.0 for outcome-supervised tasks)
- **`value_loss_fn.scale`**: Scaling factor for the value loss
- **`value_loss_fn.cliprange`**: Clip range for value function predictions
- **`loss_fn.positive_example_nll_weight`**: VAPO NLL auxiliary loss weight on correct samples (0 = disabled)

All other parameters (clipping, KL, importance sampling, dynamic sampling, reward shaping, reward scaling) work identically to GRPO. See the [GRPO Guide](grpo.md) for details.

## Metrics

PPO logs all the same metrics as GRPO (see [GRPO Metrics](grpo.md#metrics)). In addition, the following critic-specific metrics are logged:

| Metric | Description |
|--------|-------------|
| `critic/loss` | Value function MSE loss |
| `critic/grad_norm` | Gradient norm of the value model |
| `critic/values_mean` | Mean of predicted values across valid tokens |
| `critic/values_min` | Minimum predicted value |
| `critic/values_max` | Maximum predicted value |
| `critic/returns_mean` | Mean of GAE returns |
| `critic/explained_var` | Explained variance: $1 - \text{Var}(R - V) / \text{Var}(R)$. Higher is better; values near 1.0 indicate the critic accurately predicts returns. |

## Evaluate the Trained Model

Upon completion of the training process, you can refer to our [evaluation guide](eval.md) to assess model capabilities.

## References

- **PPO Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **InstructGPT**: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- **GAE Paper**: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- **VAPO Paper**: [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)
- **veRL**: [veRL: An Efficient and Flexible Library for Post-Training of LLMs](https://arxiv.org/abs/2412.09613)
- **[NeMo RL GRPO Guide](grpo.md)**
- **[NeMo RL DAPO Guide](dapo.md)**
