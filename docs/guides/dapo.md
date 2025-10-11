# An in-depth Walkthrough of DAPO in NeMo RL

This guide covers the [Decoupled Clip and Dynamic Sampling for Adaptive Policy Optimization (DAPO)](https://arxiv.org/pdf/2503.14476) implementation in NeMo RL.

DAPO introduces 4 key improvements over GRPO:
1. **Clip-Higher**, which promotes the diversity of the system and avoids entropy collapse
2. **Dynamic Sampling**, which improves training efficiency and stability
3. **Token-Level Policy Gradient Loss**, which is critical in long-CoT RL scenarios
4. **Overlong Reward Shaping**, which reduces reward noise and stabilizes training

This document focuses on DAPO-specific features: Dynamic Sampling and Overlong Reward Shaping. For foundational concepts on GRPO including data handling, policy training, generation, and loss functions, see the [GRPO guide](grpo.md).


## Quickstart: Launch a DAPO Run

To get started quickly, use the example configuration [examples/configs/recipes/llm/dapo-qwen2.5-7b.yaml](../../examples/configs/recipes/llm/dapo-qwen2.5-7b.yaml). You can launch this using the same script as GRPO:

```bash
uv run examples/run_grpo_math.py --config examples/configs/recipes/llm/dapo-qwen2.5-7b.yaml {overrides}
```

**Reminder**: Don't forget to set your HF_HOME, WANDB_API_KEY, and HF_DATASETS_CACHE (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

## Dynamic Sampling

Standard GRPO trains on all generated responses, even when they have identical rewards (zero gradient signal) within a prompt group of generations. Dynamic sampling filters to keep only groups with diverse rewards (`std > 0`), and accumulates them across batches until reaching the target batch size. For implementation details, see the [`dynamic_sampling`](../../nemo_rl/algorithms/grpo.py) function.

**Algorithm**: For each training step:

1. Sample `dapo_batch_multiplier × num_prompts_per_step` prompts from the dataset. The default value of `dapo_batch_multiplier` is 1.
2. Generate `num_generations_per_prompt` responses per prompt and compute rewards
3. Compute the baseline and standard deviation for each prompt group
4. Filter prompt groups where `std > 0`
5. Store these prompts in a cache until reaching the target training batch size of `num_prompts_per_step × num_generations_per_prompt` samples.
6. Samples are accumulated until the maximum number of allowed batches (`max_num_gen_batches`) is reached. If the cache still does not meet the target rollout batch size at that point, an error is raised. To resolve this, consider adjusting parameters such as `num_prompts_per_step` or `num_generations_per_prompt` to increase sample diversity, or revisit the complexity of your data.
7. Perform training on the collected samples with non-zero standard deviation

> [!NOTE]
> `dapo_batch_multiplier` (a float ≥ 1.0) scales the initial prompt pool size by sampling  
> `dapo_batch_multiplier × num_prompts_per_step` prompts before dynamic sampling.  
> Higher values increase memory and compute cost, while very low values (e.g., 1) may slow cache accumulation of prompt groups with non-zero standard deviation. Optimal choice depends on the dataset, model capacity, and training requirements.


## Reward Shaping
DAPO introduces an overlong reward shaping mechanism to reduce reward noise and stabilize training. This approach penalizes responses that exceed a specified length threshold, helping to prevent the model from generating excessively long outputs while maintaining solution quality.

For a detailed explanation of the overlong reward shaping mechanism, please refer to Section 3.4 of the [DAPO paper](https://arxiv.org/pdf/2503.14476). For implementation details, see the [`apply_reward_shaping`](../../nemo_rl/algorithms/reward_functions.py) function.

## Configuration

```yaml
grpo:
  use_dynamic_sampling: true  # Enable DAPO dynamic sampling
  num_prompts_per_step: 512   # Target number of prompts per training step
  num_generations_per_prompt: 16  # Generations per prompt
  dapo_batch_multiplier: 3    # Dataloader batch size = dapo_batch_multiplier × num_prompts_per_step
  max_num_gen_batches: 10     # Maximum number of batches to be used for accumulating non-zero std prompts
  reward_scaling:
    enabled: true
    correct: 1.0
    incorrect: -1.0
  
  reward_shaping:
    enabled: true
    overlong_buffer_length: 4096     # Threshold before penalties apply (paper uses 4096)
    overlong_buffer_penalty: 1.0     # Penalty per excess token
    max_response_length: 20480       # Hard maximum generation length
```

**Key Parameters:**
- **`use_dynamic_sampling`**: When enabled, activates DAPO's dynamic sampling algorithm to filter and accumulate prompt groups with non-zero standard deviation
- **`dapo_batch_multiplier`**: Factor that scales the initial prompt pool size for sampling
- **`max_num_gen_batches`**: Maximum number of batches to be used for accumulating non-zero std prompts
- **`reward_scaling`**: When enabled, maps binary rewards (1.0 for correct, 0.0 for incorrect) to configured values before applying reward shaping
- **`reward_shaping`**: When enabled, applies the overlong penalty mechanism described in the Reward Shaping section above. Responses exceeding `max_response_length - overlong_buffer_length` receive penalties proportional to their excess length, helping to reduce reward noise and stabilize training.

> [!NOTE]
> When dynamic sampling is enabled, monitor the `filtered_reward` metric to track the average reward of the prompts with std > 0.

> [!NOTE]
> **Clip-Higher** and **Token-Level Policy Gradient Loss** are already supported in NeMo RL and can be configured through the `loss_fn` section of your experiment config:
> - Set `ratio_clip_max` to enable Clip-Higher (e.g., `ratio_clip_max: 0.28`)
> - Set `token_level_loss: true` to enable Token-Level Policy Gradient Loss
> 
> See the full [DAPO example config](../../examples/configs/recipes/llm/dapo-qwen2.5-7b.yaml) for reference.

## Example Training Results
Using the [DAPO example config](../../examples/configs/recipes/llm/dapo-qwen2.5-7b.yaml), you can expect to see intermediate plots such as the training reward curve and validation accuracy on AIME24 for Qwen/Qwen2.5-Math-7B. These plots serve as reference outputs to help verify reproducibility. They are not intended to reflect the best accuracy that can be achieved using DAPO for this model.

![DAPO Qwen2.5-7B Training Reward](../assets/dapo_train_reward.png)
![DAPO Qwen2.5-7B Validation Accuracy](../assets/dapo_val_acc.png)

## References

- **DAPO Paper**: [Decoupled Clip and Dynamic Sampling for Adaptive Policy Optimization](https://arxiv.org/pdf/2503.14476)
- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- **NeMo RL GRPO Guide**: [grpo.md](grpo.md)