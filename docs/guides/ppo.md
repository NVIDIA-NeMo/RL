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

### Deployment Architectures

By default, PPO uses a colocated architecture where the **policy**, **value model**, and **generation engine** share one `RayVirtualCluster`. GPU memory is managed by offloading models to CPU between stages: the value model is loaded to GPU only during its inference and training phases, then offloaded to make room for the other components.

PPO also supports non-colocated vLLM generation. In this mode, the policy and value model continue to time-share one training `RayVirtualCluster`, while vLLM runs on a separate inference `RayVirtualCluster` in the same Ray cluster. Updated policy weights are transferred to vLLM through the cross-cluster collective refit path.

```yaml
policy:
  generation:
    backend: vllm
    colocated:
      enabled: false
      resources:
        gpus_per_node: 2
        num_nodes: null
```

When only one node remains for policy and generation after other resources are reserved, `gpus_per_node` reserves that many GPUs for generation and `num_nodes` must be `null` or `1`. When more than one node remains for training and generation, generation uses complete nodes: set `num_nodes` to the number of inference nodes and `gpus_per_node` equal to `cluster.gpus_per_node`. Non-colocated SGLang generation is not currently supported by PPO.

### Asynchronous PPO

Set `ppo.async_ppo.enabled: true` to overlap rollout generation with training. A background collector fills a replay buffer on the non-colocated vLLM GPUs while the policy and value model train on their shared cluster. Values and policy/reference log probabilities are recomputed when a trajectory is sampled, then PPO runs GAE and its normal `ppo_epochs` updates before publishing one new policy version to vLLM.

Async PPO reuses the trajectory collector, replay buffer, and weight-versioning infrastructure described in the [Async GRPO guide](async-grpo.md); this section focuses on PPO-specific behavior and constraints.

Async PPO requires non-colocated vLLM generation with `vllm_cfg.async_engine: true`, `loss_fn.use_importance_sampling_correction: true`, and `loss_fn.force_on_policy_ratio: false`. Dynamic sampling, reward scaling, reward shaping, multiple dataloaders, NeMo Gym, colocated generation, and FP8 KV-scale synchronization are not supported yet.

`max_trajectory_age_steps` is the normal policy-training age limit. The recommended value is `1`; larger values improve overlap but increase off-policy bias in GAE. When `policy_training_start_step > 0`, set `warmup_generation_lead_steps` to a larger value to bank additional rollout batches while the policy is frozen for critic warmup. The collector caps frozen-policy targets at `policy_training_start_step + max_trajectory_age_steps`, so their actual policy-update age remains within the normal limit. The buffer keeps these batches valid through that frontier and then restores the normal age limit. `null` uses `max_trajectory_age_steps` as the generation lead throughout.

Async training stops at `max_num_steps`; the collector cycles the training dataloader as needed. `max_num_epochs` is not supported yet and must be set to `-1`; use `max_num_steps` to control training length. This is a v1-only convention — the SingleController path rejects any non-positive `max_num_epochs`, so set a positive value there (see [single-controller.md](single-controller.md#migrating-a-legacy-async-config)). Async checkpoints save the collector dataloader and replay-buffer state together with policy and value state. By default, incomplete restored targets are retained and gap-filled. Setting `drop_incomplete_targets_on_restore: true` discards their restored rows and fills the target from subsequent dataloader prompts; it does not regenerate the original prompts.

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

### Warm-Starting the Critic

Warmup can be paid once, offline, instead of inside every PPO run: pretrain the value model separately and point a fresh PPO run at that checkpoint.

```yaml
ppo:
  warm_start_value_checkpoint: /path/to/critic_pretrain_run/step_370
  policy_training_start_step: 0  # the critic is already warm
```

`policy_training_start_step: 0` is the natural pairing, but keeping a short online warmup is equally valid: it calibrates the seeded critic on this run's own rollout distribution before the policy starts moving. A nonzero value is also what `async_ppo.warmup_generation_lead_steps` requires (`async_rl.sampler.warmup_lookahead_versions` on the SingleController path) — both must be null when it is 0.

The path is a `step_<n>` directory holding a `value/` subtree — the layout a PPO or critic-pretraining run checkpoints. The critic restores its weights from it, plus optimizer moments and LR-scheduler state whenever the seed carries them; a seed written with `save_optimizer: false` restores weights only and warns. Setup rejects a path with no `value/weights` subtree rather than letting the critic start cold behind the message above. Nothing else is read: the policy starts from the base model, the dataloader from the beginning, and the step counter at 0.

**The seed and this run must agree on the value scheduler.** The seed is restored through the ordinary resume path, so Megatron compares nine scheduler fields against the ones this run builds and raises on the first mismatch — `use_checkpoint_opt_param_scheduler` is off, so `OptimizerParamScheduler._check_and_set` asserts equality. Match all of these across the two runs:

- `value.megatron_cfg.optimizer.lr` and `.min_lr` — they feed `max_lr`/`min_lr` and are the *first* two fields checked. They live in the optimizer block, not the scheduler block.
- `value.megatron_cfg.scheduler`.
- `value.train_global_batch_size` — it multiplies `lr_decay_steps`, `wd_incr_steps` and `lr_warmup_steps`.
- the tick budget `train_iters`. A synchronous run sets it to `min(max_num_steps, max_num_epochs × len(dataloader)) × ppo_epochs`; an async run sets it to `max_num_steps × ppo_epochs`, since async requires `max_num_epochs: -1`. `len(dataloader)` is prompt batches per epoch, so on a synchronous run the dataset size and `num_prompts_per_step` are part of the budget whenever the epoch term is the smaller one — as it is for the shipped recipes that set `max_num_epochs: 15`. Matching `max_num_steps` and `ppo_epochs` alone is not enough there.

A mismatch fails during critic init. Which field is named depends on which input differs: a batch-size difference reports `warmup iterations`, a learning-rate difference reports `learning rate`.

```
AssertionError: OptimizerParamScheduler: class input value <X> and checkpointvalue <Y> for total number of weight decay iterations do not match
```

(`checkpointvalue` runs together in the upstream message; search for it as written.)

Carrying the seed's schedule over is deliberate — it is what lets the post-warmup LR continue instead of restarting. Set `value.megatron_cfg.scheduler.override_opt_param_scheduler: true` if you would rather this run's settings win; the seed's step position is still restored, so the critic resumes at the seed's tick count rather than at step 0. All of this is Megatron-only — a DTensor critic (the default in `ppo_math_1B.yaml`) loads the seed's scheduler state with no comparison and has no override knob.

The warm start applies to a fresh run only: once the run has written a checkpoint of its own, that checkpoint wins. That is what lets the setting stay in the config across resumes — a resubmitted run restores its own critic instead of re-seeding from the pretrained one, with no config edit in between.

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
  warm_start_value_checkpoint: null
  val_period: 20
  val_at_start: true
  val_at_end: false
  seed: 42
  use_dynamic_sampling: false
  overlong_filtering: false
  # null logs mismatch metrics without masking; set a threshold to mask sequences.
  seq_logprob_error_threshold: null

  async_ppo:
    enabled: false
    max_trajectory_age_steps: 1
    warmup_generation_lead_steps: null
    in_flight_weight_updates: false
    recompute_kv_cache_after_weight_updates: false
    drop_incomplete_targets_on_restore: false

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
- **`ppo.warm_start_value_checkpoint`**: Checkpoint step directory whose `value/` seeds the critic on a fresh run. See [Warm-Starting the Critic](#warm-starting-the-critic)
- **`ppo.seq_logprob_error_threshold`**: Nullable sequence-level multiplicative probability-error threshold. PPO always logs sequence-level train/generation mismatch metrics; when this is set, sequences above the threshold are excluded from advantage and loss computation.
- **`ppo.async_ppo`**: Enables replay-buffer-based asynchronous PPO. See [Asynchronous PPO](#asynchronous-ppo) for requirements and staleness controls.
- **`ppo.adv_estimator.name`**: Set to `"gae"` for GAE advantage estimation (PPO default)
- **`ppo.adv_estimator.gae_lambda`**: GAE $\lambda$ parameter (bias-variance tradeoff, typically 0.95)
- **`ppo.adv_estimator.gae_gamma`**: Discount factor $\gamma$ (typically 1.0 for outcome-supervised tasks)
- **`value_loss_fn.scale`**: Scaling factor for the value loss
- **`value_loss_fn.cliprange`**: Clip range for value function predictions
- **`loss_fn.positive_example_nll_weight`**: VAPO NLL auxiliary loss weight on correct samples (0 = disabled)

For synchronous PPO, the remaining clipping, KL, sampling, and reward options work as documented in the [GRPO Guide](grpo.md). Async PPO has the limitations listed above.

## Metrics

PPO logs all the same metrics as GRPO (see [GRPO Metrics](grpo.md#metrics)). It also logs the following PPO-specific metrics:

| Metric | Description |
|--------|-------------|
| `critic/loss` | Value function MSE loss |
| `critic/grad_norm` | Gradient norm of the value model |
| `critic/values_mean` | Mean of predicted values across valid tokens |
| `critic/values_min` | Minimum predicted value |
| `critic/values_max` | Maximum predicted value |
| `critic/returns_mean` | Mean of GAE returns |
| `critic/explained_var` | Explained variance: $1 - \text{Var}(R - V) / \text{Var}(R)$. Higher is better; values near 1.0 indicate the critic accurately predicts returns. |
| `max_seq_mult_prob_error` | Maximum sequence-level multiplicative probability error between generation and training logprobs before optional masking. |
| `mean_seq_mult_prob_error` | Mean sequence-level multiplicative probability error before optional masking. |
| `min_seq_mult_prob_error` | Minimum sequence-level multiplicative probability error before optional masking. |
| `max_seq_mult_prob_error_after_mask` | Maximum sequence-level multiplicative probability error among sequences retained after optional masking. |
| `mean_seq_mult_prob_error_after_mask` | Mean sequence-level multiplicative probability error among sequences retained after optional masking. |
| `min_seq_mult_prob_error_after_mask` | Minimum sequence-level multiplicative probability error among sequences retained after optional masking. |
| `num_masked_seqs_by_logprob_error` | Number of sequences excluded by `ppo.seq_logprob_error_threshold`. |
| `masked_correct_pct` | Fraction of sequences excluded by `ppo.seq_logprob_error_threshold` that received a reward of 1. |

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
