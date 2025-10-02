---
description: "Get started with NeMo RL in minutes through progressive examples from basic SFT to advanced multi-GPU training"
categories: ["getting-started"]
tags: ["quickstart", "sft", "grpo", "training-execution", "gpu-accelerated", "distributed", "configuration"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "universal"
---

# Quick Start

Get started with NeMo RL in minutes! This guide will walk you through your first training run with progressive complexity.

## Prerequisites

Before starting, ensure you have:
- [Installed NeMo RL](installation)
- Set up your environment variables
- Authenticated with Hugging Face (if using Llama models)

## Quick Start: SFT on a Small Model

Let's start with a simple supervised fine-tuning (SFT) example using a small model.

### Step 1: Basic SFT Run

Run your first training job:

```bash
cd examples
uv run python run_sft.py \
    cluster.gpus_per_node=1 \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    data.dataset_name=HuggingFaceH4/ultrachat_200k \
    data.dataset_key=train_sft \
    sft.max_num_epochs=1 \
    logger.wandb.name="quickstart-sft"
```

This command:
- Uses 1 GPU
- Trains a 0.5B parameter Qwen model
- Uses the UltraChat dataset
- Runs for 1 epoch
- Logs to Weights and Biases

### Step 2: Monitor Training

Watch the training progress:
- **Console output**: Real-time training metrics
- **WandB dashboard**: Detailed training curves and metrics
- **Logs**: Checkpoint and evaluation results

### Step 3: Verify Results

Check that training completed successfully:
- Look for "Training completed successfully" message
- Verify checkpoints were saved
- Check WandB for training curves

## Intermediate: GRPO Training

Now let's try Group Relative Policy Optimization (GRPO) for preference learning.

### Step 1: GRPO on Math Dataset

```bash
uv run python examples/run_grpo_math.py \
    cluster.gpus_per_node=1 \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    grpo.max_num_epochs=1 \
    logger.wandb.name="quickstart-grpo"
```

This trains the model to solve math problems using RL.

### Step 2: Monitor RL Training

GRPO training shows different metrics:
- **Policy loss**: How well the policy is learning
- **Value loss**: How well the value function estimates rewards
- **Reward**: Average reward per episode
- **Success rate**: Percentage of problems solved correctly

### Step 3: Evaluate Results

After training, evaluate the model:

```bash
uv run python examples/run_eval.py \
    generation.model_name=/path/to/checkpoint \
    data.dataset_name=HuggingFaceH4/MATH-500 \
    data.dataset_key=test \
    eval.num_tests_per_prompt=1
```

## Advanced: Multi-GPU Training

Scale up to multiple GPUs for faster training.

### Step 1: Multi-GPU SFT

```bash
uv run python examples/run_sft.py \
    cluster.gpus_per_node=4 \
    policy.model_name=Qwen/Qwen2.5-1.5B \
    data.dataset_name=HuggingFaceH4/ultrachat_200k \
    data.dataset_key=train_sft \
    sft.max_num_epochs=1 \
    logger.wandb.name="multi-gpu-sft"
```

### Step 2: Monitor Distributed Training

Multi-GPU training shows:
- **GPU utilization**: All GPUs should be active
- **Throughput**: Higher samples/second
- **Memory usage**: Distributed across GPUs

### Step 3: Verify Scaling Performance

Compare single vs multi-GPU performance:
- Training time should scale approximately linearly
- Memory usage per GPU should remain similar
- Final model quality should be equivalent

## Production: Large Model Training

For production training with large models.

### Step 1: Large Model SFT

```bash
uv run python examples/run_sft.py \
    cluster.gpus_per_node=8 \
    policy.model_name=Qwen/Qwen2.5-7B \
    data.dataset_name=HuggingFaceH4/ultrachat_200k \
    data.dataset_key=train_sft \
    sft.max_num_epochs=1 \
    policy.dtensor_cfg.enabled=true \
    policy.dtensor_cfg.tensor_parallel_size=2 \
    logger.wandb.name="large-model-sft"
```

### Step 2: Advanced Configuration

Large models require careful configuration:
- **Tensor parallelism**: Split model across GPUs
- **Gradient checkpointing**: Save memory
- **Mixed precision**: Speed up training
- **Gradient accumulation**: Handle large effective batch sizes

### Step 3: Monitor Resource Usage

Watch for:
- **Memory usage**: Should stay within GPU limits
- **Communication overhead**: NCCL collectives
- **Training stability**: Loss should decrease smoothly

## Customization Examples

### Custom Dataset

Train on your own dataset:

```bash
uv run python examples/run_sft.py \
    cluster.gpus_per_node=1 \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    data.dataset_name=your-username/your-dataset \
    data.dataset_key=train \
    sft.max_num_epochs=3 \
    logger.wandb.name="custom-dataset"
```

### Custom Model

Use a different model architecture:

```bash
uv run python examples/run_sft.py \
    cluster.gpus_per_node=1 \
    policy.model_name=microsoft/DialoGPT-medium \
    data.dataset_name=HuggingFaceH4/ultrachat_200k \
    data.dataset_key=train_sft \
    sft.max_num_epochs=1 \
    logger.wandb.name="custom-model"
```

### Custom Hyperparameters

Tune training parameters:

```bash
uv run python examples/run_sft.py \
    cluster.gpus_per_node=1 \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    data.dataset_name=HuggingFaceH4/ultrachat_200k \
    data.dataset_key=train_sft \
    sft.max_num_epochs=2 \
    sft.learning_rate=1e-4 \
    sft.warmup_steps=100 \
    sft.weight_decay=0.01 \
    logger.wandb.name="custom-hyperparams"
```

## Troubleshooting Quickstart

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size: `algorithm.batch_size=1`
   - Enable gradient accumulation: `algorithm.gradient_accumulation_steps=4`
   - Use smaller model: `policy.model_name=Qwen/Qwen2.5-0.5B`

2. **Model Download Issues**:
   - Check internet connection
   - Verify Hugging Face token: `huggingface-cli login`
   - Use local model path if available

3. **Ray Cluster Issues**:
   - Check Ray status: `ray status`
   - Restart Ray: `ray stop && ray start --head`

### Get Help

- [Installation Guide](installation.md) - Setup and configuration
- [Model Selection](model-selection.md) - Choose appropriate models
- [Troubleshooting](../guides/troubleshooting) - Common issues and solutions
- [Community Support](https://github.com/NVIDIA/NeMo-RL/issues) - GitHub issues

## Next Steps

After completing the quickstart:

1. **Explore Algorithms**:
   - [GRPO Guide](../guides/training-algorithms/grpo) - Advanced preference learning
- [DPO Guide](../guides/training-algorithms/dpo) - Direct preference optimization
- [SFT Guide](../guides/training-algorithms/sft) - Supervised fine-tuning

2. **Scale Up**:
   - [Cluster Setup](cluster.md) - Multi-node training
- [Troubleshooting](../guides/troubleshooting) - Production deployment and support

4. **Evaluate Models**:
   - [Evaluation Guide](../guides/training-algorithms/eval) - Model assessment
- [Troubleshooting](../guides/troubleshooting) - Common issues 