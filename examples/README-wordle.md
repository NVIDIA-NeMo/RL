# Wordle Environment

A multi-turn Wordle game environment for GRPO training. The model learns to guess a 5-letter word from a fixed 20-word vocabulary using letter-by-letter feedback.

## Game Rules

- Secret word chosen from 20 common 5-letter words
- Model guesses using `<guess>WORD</guess>` tags
- Feedback per letter: **G** = correct position, **Y** = wrong position, **X** = not in word
- Maximum 6 guesses per game
- Reward: 1.0 for correct guess, 0.1 per green letter on final wrong guess

## Files

| File | Description |
|------|-------------|
| `nemo_rl/environments/games/wordle.py` | Environment (Ray actor) |
| `examples/run_grpo_wordle.py` | Training script |
| `examples/configs/grpo_wordle.yaml` | Config (Qwen3-0.6B, 6 guesses, 20 words) |

## Quick Start

### 1-GPU (dev/debug)

```bash
python examples/run_grpo_wordle.py \
  --config examples/configs/grpo_wordle.yaml \
  cluster.num_nodes=1 cluster.gpus_per_node=1 \
  grpo.max_num_steps=50 grpo.num_prompts_per_step=16 \
  grpo.num_generations_per_prompt=8 \
  policy.dtensor_cfg.cpu_offload=false \
  policy.train_global_batch_size=128
```

### 8-GPU

```bash
python examples/run_grpo_wordle.py \
  --config examples/configs/grpo_wordle.yaml \
  cluster.num_nodes=1 cluster.gpus_per_node=8 \
  grpo.max_num_steps=50 grpo.num_prompts_per_step=128
```

## Training Results

### 1-GPU (50 steps, 16 prompts x 8 gens)

- **W&B**: https://wandb.ai/nvidia/nemo-rl-native-env/runs/rccfzlr2
- Model: Qwen3-0.6B
- Avg reward improved from ~0.06 to peak ~0.64 (steps 35-43), settling around ~0.25-0.32
- Clear learning signal with rewards increasing over training

### 8-GPU (50 steps, 128 prompts x 16 gens)

- **W&B**: https://wandb.ai/nvidia/nemo-rl-native-env/runs/quh1m93c
- Model: Qwen3-0.6B
- Avg reward improved from ~0.10 to ~0.41 over 50 steps
- More stable learning curve due to larger batch size (2048 samples/step)

### Key Observations

- The model learns to use the word list and feedback effectively within 50 steps
- With only 20 words, random chance is 5% (1/20); the model quickly exceeds this
- Partial credit (0.1 per green letter) provides gradient signal even for wrong guesses
- 8-GPU run shows smoother learning due to larger batch statistics
