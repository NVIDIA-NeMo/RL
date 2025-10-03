---
description: "Complete guide to on-policy distillation in NeMo RL for knowledge transfer from larger teacher models to smaller student models"
categories: ["training-algorithms"]
tags: ["distillation", "knowledge-distillation", "model-compression", "teacher-student", "training-execution"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "intermediate"
content_type: "tutorial"
modality: "universal"
---

# Perform On-Policy Distillation

This document explains how to perform on-policy distillation within NeMo RL. On-policy distillation is a knowledge transfer technique where a smaller student model learns from a larger teacher model by generating on-policy sequences and aligning its logits to the teacher via KL divergence. This approach achieves near-larger-model quality at significantly lower computational cost than full RL training.

## What is On-Policy Distillation?

On-policy distillation (Qwen3-style) is a training method that:
- Uses a **student model** (smaller) to generate responses on-policy
- Compares student outputs with a **teacher model** (larger) via KL divergence
- Aligns the student's logit distribution to match the teacher's
- Achieves near-teacher performance at lower inference cost
- Provides an efficient alternative to full RL training for model compression

### When to Use Distillation

**Best for:**
- Model compression and deployment optimization
- Reducing inference costs while maintaining quality
- Creating smaller models for resource-constrained environments
- Transferring capabilities from expensive large models to efficient smaller ones
- Scenarios where full RL training is too computationally expensive

**Key Benefits:**
- Lower computational cost than training large models from scratch
- Maintains much of the teacher model's quality
- Faster inference with smaller student models
- More efficient than traditional offline distillation
- Works well with mathematical reasoning and instruction-following tasks

## Launch a Distillation Run

Use the [examples/run_distillation_math.py](../../../examples/run_distillation_math.py) script to launch a distillation experiment. Launch this script either locally or via Slurm. For details on how to set up Ray and launch a job using Slurm, refer to the [cluster documentation](../../get-started/cluster.md).

Be sure to launch the job using `uv`. The command to launch a distillation job is as follows:

```bash
uv run examples/run_distillation_math.py --config <PATH TO YAML CONFIG> <OVERRIDES>
```

If not specified, `config` will default to `examples/configs/distillation_math.yaml`.

### Single Node Example

To run on-policy distillation on a single GPU using `Qwen/Qwen3-1.7B-Base` as the student and `Qwen/Qwen3-4B` as the teacher:

```bash
uv run python examples/run_distillation_math.py
```

### Customize Parameters

You can customize parameters with command-line overrides. For example:

```bash
uv run python examples/run_distillation_math.py \
  policy.model_name="Qwen/Qwen3-1.7B-Base" \
  teacher.model_name="Qwen/Qwen3-4B" \
  cluster.gpus_per_node=8 \
  distillation.num_prompts_per_step=256 \
  loss_fn.kl_type="mixed"
```

### Multi-Node Example

For distributed distillation training across multiple nodes:

```bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2

COMMAND="uv run ./examples/run_distillation_math.py --config examples/configs/distillation_math.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 checkpointing.checkpoint_dir='results/distill_2nodes' logger.wandb_enabled=True logger.wandb.name='distill-2nodes'" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```

**Reminder**: Don't forget to set your `HF_HOME`, `WANDB_API_KEY`, and `HF_DATASETS_CACHE` (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

## Example Configuration File

NeMo RL allows users to configure distillation experiments using `yaml` config files. An example distillation configuration file can be found at `examples/configs/distillation_math.yaml`.

To override a value in the config, either update the value in the `yaml` file directly, or pass the override via the command line. For example:

```bash
uv run examples/run_distillation_math.py \
    cluster.gpus_per_node=8 \
    logger.wandb.name="distillation-dev-8-gpu"
```

### Key Configuration Parameters

The distillation configuration includes several important sections:

#### Distillation Settings

```yaml
distillation:
    num_prompts_per_step: 128          # Number of prompts to process per training step
    num_generations_per_prompt: 1       # Generations per prompt (typically 1)
    max_rollout_turns: 1                # Max turns for multi-turn rollouts
    max_num_steps: 1000                 # Total training steps
    val_batch_size: 64                  # Validation batch size
    val_period: 20                      # Validate every N steps
    topk_logits_k: 64                   # Top-k logits for efficiency
    seed: 42                            # Random seed
```

#### Loss Function Configuration

```yaml
loss_fn:
    kl_type: "mixed"                    # Options: "forward", "reverse", "mixed"
    mixed_kl_weight: 0.5                # Weight for forward KL when using mixed
    zero_outside_topk: false            # Zero teacher logits outside top-k
```

**KL Divergence Types:**
- **Forward KL**: Minimizes `KL(Teacher || Student)` - student matches teacher
- **Reverse KL**: Minimizes `KL(Student || Teacher)` - mode-seeking behavior
- **Mixed**: Weighted combination of forward and reverse KL

#### Teacher and Student Models

```yaml
policy:  # Student model configuration
    model_name: "Qwen/Qwen3-1.7B-Base"
    train_global_batch_size: 64
    max_total_sequence_length: 8192
    dtensor_cfg:
        enabled: true
        tensor_parallel_size: 2
        context_parallel_size: 2

teacher:  # Teacher model configuration
    model_name: "Qwen/Qwen3-4B"
    generation_batch_size: 64
    vllm_cfg:
        tensor_parallel_size: 2
        max_model_len: 8192
```

## Backend Support

> **Note:** Distillation currently supports the DTensor and vLLM generation backend. Megatron generation/training paths are not supported yet.

The distillation implementation uses:
- **Student training**: DTensor backend with FSDP2, Tensor Parallelism, and Context Parallelism
- **Teacher inference**: vLLM for efficient teacher model inference
- **Generation**: On-policy generation from the student model

## Datasets

Distillation in NeMo RL typically uses prompt datasets where:
- The student generates completions for each prompt
- The teacher provides target logit distributions
- The student learns to match the teacher's distribution via KL loss

The default configuration uses the [DeepScaleR dataset](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview) for mathematical reasoning tasks.

For custom datasets, ensure your data includes:
1. **Prompts**: Input questions or instructions
2. **Context** (optional): Additional context for the task
3. **Task specification**: Dataset name and format requirements

Refer to the [chat dataset document](../../core-design/data-management/chat-datasets.md) for details on data formatting.

## Best Practices

### Model Selection

1. **Teacher-Student Size Ratio**: Typically 2-4x larger teacher works well
   - Example: 4B teacher → 1.7B student
   - Example: 70B teacher → 8B student

2. **Architecture Compatibility**: Use models from the same family when possible
   - Qwen → Qwen
   - Llama → Llama

### Training Tips

1. **Start with Mixed KL**: The `mixed` KL type often works best as a default
2. **Tune Top-K**: Use `topk_logits_k` (32-128) to balance efficiency and quality
3. **Batch Size**: Larger batches generally improve stability
4. **Learning Rate**: Start conservative, similar to SFT learning rates
5. **Validation**: Monitor validation performance to prevent overfitting

### Resource Optimization

1. **Teacher Parallelism**: Use vLLM tensor parallelism for efficient teacher inference
2. **Student Parallelism**: Enable DTensor parallelization for student training
3. **Memory Management**: Use activation checkpointing and gradient checkpointing
4. **Sequence Length**: Match teacher and student max sequence lengths

## Comparison with Other Methods

| Method | Cost | Quality | Use Case |
|--------|------|---------|----------|
| **Full RL Training** | Very High | Highest | Maximum performance needed |
| **On-Policy Distillation** | Medium | High | Efficient model compression |
| **Offline Distillation** | Low | Medium | Pre-computed teacher outputs |
| **SFT** | Low | Variable | Supervised data available |

## Troubleshooting

### Common Issues

**Issue: KL divergence not decreasing**
- Solution: Reduce learning rate or increase batch size
- Check: Verify teacher and student use same tokenizer

**Issue: Out of memory**
- Solution: Reduce `topk_logits_k` or enable more parallelism
- Enable: Activation checkpointing and CPU offload

**Issue: Student quality degrading**
- Solution: Try `mixed` KL instead of reverse KL
- Reduce: `mixed_kl_weight` if using mixed KL

**Issue: Slow training**
- Solution: Increase teacher vLLM tensor parallelism
- Optimize: Teacher batch size and max_model_len

## Evaluate the Trained Model

Upon completion of the training process, you can refer to our [evaluation guide](eval.md) to assess model capabilities.

## Next Steps

- Explore [GRPO training](grpo.md) for full reinforcement learning
- Learn about [SFT](sft.md) for supervised fine-tuning baseline
- Review [DPO](dpo.md) for preference-based alignment
- Check [Model Development](../model-development/index.md) for custom architectures

