# NeMo-RL Code Tasks — GRPO Training Example

## Overview

Train a language model to write simple Python utility functions using GRPO. The model receives a coding task description, generates a function, and the environment executes test cases to verify correctness. Reward is 1.0 if all tests pass, 0.0 otherwise.

## Quick Start

### 1. Generate Dataset

```bash
python examples/data/build_nemorl_code_tasks.py
```

This creates `examples/data/nemorl_code_tasks.jsonl` with 85 coding tasks covering string matching, list operations, arithmetic, config validation, and more.

### 2. Train (1 GPU)

```bash
uv run python examples/run_grpo.py --config examples/configs/grpo_nemorl_code.yaml \
  cluster.num_nodes=1 cluster.gpus_per_node=1 \
  grpo.max_num_steps=50 grpo.num_prompts_per_step=16 grpo.num_generations_per_prompt=8 \
  policy.dtensor_cfg.cpu_offload=false logger.wandb_enabled=true
```

### 3. Train (8 GPU)

```bash
uv run python examples/run_grpo.py --config examples/configs/grpo_nemorl_code.yaml \
  cluster.num_nodes=1 cluster.gpus_per_node=8 \
  grpo.num_prompts_per_step=128 grpo.num_generations_per_prompt=16 \
  env.code_task.num_workers=8
```

## Architecture

- **Dataset**: Static JSONL with problem descriptions and test cases (`ResponseDataset`)
- **Processor**: `code_task_data_processor` — extracts problem text as prompt, parses test cases JSON into `extra_env_info`
- **Environment**: `CodeTaskEnvironment` — extracts Python code from model output (```python blocks), executes test cases, returns binary reward
- **Model**: Qwen3-0.6B (small, fast iteration)

## Results

Training reward trajectory (wandb): _link pending_

| Step | Mean Reward |
|------|-------------|
| 1    | TBD         |
| 50   | TBD         |
