# Calculator Tool-Call — GRPO Training Example

## Overview

Multi-step math word problem environment where the model uses a calculator tool via native tool-calling format (`tokenizer.apply_chat_template(tools=...)`) to solve problems. Tests the model's ability to decompose problems, call tools effectively, and combine intermediate results.

Problems span percentages, unit conversions, compound interest, multi-item shopping with tax/discount, currency conversion, and more. Each problem requires 2-4 calculator calls to solve correctly.

## Data Setup

No download needed — problems are procedurally generated. Each training step creates fresh problems with randomized numbers across 10 template categories.

## Key Design Decisions

- **Native tool format**: Uses `apply_chat_template(tools=[CALCULATOR_TOOL])` to embed tool schemas in the model's trained format. The model generates tool calls as `<tool_call>...</tool_call>` (Qwen3's native format) rather than custom tags — no need to learn a new output format through RL.
- **Hard problems**: All problems require 2-4 computation steps with decimal numbers (e.g., $24.99, 8.5% tax, 1.609 km/mile). This avoids trivially high starting accuracy — a 0.6B model starts at ~10-15% and improves through RL.
- **Partial credit**: Reward is 1.0 for exact match (within 0.01 tolerance), 0.5 for answers within 10% of expected, and 0.0 otherwise. This provides denser reward signal for learning.
- **Multi-turn rollout**: `max_rollout_turns: 8` allows up to 3 tool calls + final answer (4 assistant turns = 8 total turns including environment responses).

## Training Commands

### 1-GPU Training (50 steps)

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_grpo_calculator.py \
  --config examples/configs/grpo_calculator.yaml \
  cluster.num_nodes=1 \
  cluster.gpus_per_node=1 \
  grpo.max_num_steps=50 \
  policy.dtensor_cfg.cpu_offload=false \
  policy.max_total_sequence_length=4096 \
  policy.generation.max_new_tokens=512 \
  policy.generation.vllm_cfg.max_model_len=4096 \
  logger.wandb_enabled=true \
  logger.wandb.project=nemo-rl-native-env \
  logger.wandb.name=grpo-calculator-v2-1gpu
```

**Note**: For 1 GPU, `cpu_offload` must be `false` (CPUOffload doesn't work on single GPU with AutoModel).

### 8-GPU Training (50 steps)

```bash
python examples/run_grpo_calculator.py \
  --config examples/configs/grpo_calculator.yaml \
  cluster.num_nodes=1 \
  cluster.gpus_per_node=8 \
  grpo.max_num_steps=50 \
  grpo.num_prompts_per_step=128 \
  env.calculator.num_workers=8 \
  logger.wandb_enabled=true \
  logger.wandb.project=nemo-rl-native-env \
  logger.wandb.name=grpo-calculator-v2-8gpu
```

## Results

| Run | GPUs | Steps | Starting Mean Reward | Final Mean Reward | wandb |
|-----|------|-------|---------------------|-------------------|-------|
| 1-GPU | 1 | 18 | 0.24 | 0.99 | [link](https://wandb.ai/nvidia/nemo-rl-native-env/runs/qx8tp4hi) |
| 8-GPU | 8 | 22 | 0.10 | 0.67 | [link](https://wandb.ai/nvidia/nemo-rl-native-env/runs/nen117kp) |

### Reward Trajectory (1-GPU, exp_006)

| Step | Mean Reward |
|------|------------|
| 1    | 0.24       |
| 5    | 0.48       |
| 10   | 0.93       |
| 15   | 0.97       |
| 18   | 0.99       |

### Reward Trajectory (8-GPU, exp_008)

| Step | Mean Reward |
|------|------------|
| 1    | 0.10       |
| 5    | 0.15       |
| 10   | 0.40       |
| 15   | 0.48       |
| 20   | 0.56       |
| 22   | 0.67       |

The model starts at ~10-24% and rapidly learns to use the calculator tool effectively. The 1-GPU run reaches 99% in just 18 steps.
