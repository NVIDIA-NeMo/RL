# xLAM Function Calling -- GRPO Training Example

## Overview

Train a language model to generate correct function calls using GRPO on the
[Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) dataset.
The model sees tool definitions and a user query, then generates a function call in its **native tool-calling format**
(e.g., Qwen3 uses `<tool_call>...</tool_call>`). The environment checks the predicted call against the gold answer.

**Reward scheme:**
- 1.0 -- exact match (correct function name and arguments)
- 0.5 -- correct function name, wrong arguments
- 0.0 -- wrong function name or no valid tool call

## Quick Start

### 1. Data

The dataset is automatically downloaded from HuggingFace on first run.

### 2. Train (1 GPU)

```bash
uv run python examples/run_grpo.py \
  --config examples/configs/grpo_xlam_function_calling.yaml \
  cluster.num_nodes=1 cluster.gpus_per_node=1 \
  grpo.max_num_steps=50 \
  grpo.num_prompts_per_step=16 \
  grpo.num_generations_per_prompt=8 \
  logger.wandb_enabled=true \
  policy.dtensor_cfg.cpu_offload=false
```

### 3. Train (8 GPU)

```bash
uv run python examples/run_grpo.py \
  --config examples/configs/grpo_xlam_function_calling.yaml \
  cluster.num_nodes=1 cluster.gpus_per_node=8 \
  grpo.num_prompts_per_step=128 \
  grpo.num_generations_per_prompt=16 \
  env.function_call.num_workers=8
```

## Results

Training reward trajectory (wandb): *link TBD after training run*

| Step | Mean Reward |
|------|-------------|
| 1    | TBD         |
| 50   | TBD         |
