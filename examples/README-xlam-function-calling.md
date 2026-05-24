# xLAM Function Calling -- GRPO Training Example

## Overview

Train a model to generate function calls using the [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) dataset and GRPO. The model learns to produce tool calls in its **native format** (e.g., Qwen3 uses `<tool_call>...</tool_call>` tags) by leveraging `tokenizer.apply_chat_template(tools=...)` for prompt formatting.

**Reward scheme**: 1.0 for exact match (correct function name + arguments), 0.5 for correct name but wrong arguments, 0.0 otherwise.

## Quick Start

### 1. Data

The dataset is auto-downloaded from HuggingFace on first run. No manual setup needed.

### 2. Train (1 GPU)

```bash
uv run python examples/run_grpo.py --config examples/configs/grpo_xlam_function_calling.yaml \
  cluster.num_nodes=1 cluster.gpus_per_node=1 \
  grpo.max_num_steps=50 \
  grpo.num_prompts_per_step=16 \
  grpo.num_generations_per_prompt=8 \
  logger.wandb_enabled=true \
  policy.dtensor_cfg.cpu_offload=false
```

### 3. Train (8 GPU)

```bash
uv run python examples/run_grpo.py --config examples/configs/grpo_xlam_function_calling.yaml \
  cluster.num_nodes=1 cluster.gpus_per_node=8 \
  grpo.num_prompts_per_step=128 \
  grpo.num_generations_per_prompt=16
```

## Results

Training reward trajectory (wandb): *link TBD after training run*

| Step | Mean Reward |
|------|-------------|
| 1    | TBD         |
| 50   | TBD         |
