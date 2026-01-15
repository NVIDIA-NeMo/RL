# NeMo RL Quickstart Tutorial

**Time to complete: 10 minutes**

This tutorial gets you from installation to your first training run in under 10 minutes.
By the end, you'll understand how to use NeMo RL's simple API to train language models with reinforcement learning.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- Access to HuggingFace Hub

## Step 1: Installation (2 minutes)

### Option A: Using pip (recommended for users)

```bash
pip install nemo-rl
```

### Option B: From source (for contributors)

```bash
git clone https://github.com/NVIDIA-NeMo/RL.git nemo-rl
cd nemo-rl
git submodule update --init --recursive
uv venv
uv pip install -e .
```

### Verify Installation

```python
import nemo_rl
print(nemo_rl.__version__)  # Should print version number
```

## Step 2: Your First Training Run (3 minutes)

NeMo RL provides a simple `train()` function that handles all the complexity for you.

### The Simplest Training Script

Create a file called `train_model.py`:

```python
import nemo_rl

# Define your reward function
def length_reward(prompt: str, response: str) -> float:
    """Reward longer responses (simple example)."""
    return min(len(response) / 200, 1.0)

# Train with a single function call
result = nemo_rl.train(
    model="Qwen/Qwen2.5-0.5B",       # Small model for quick testing
    dataset="nvidia/HelpSteer2",     # Built-in dataset
    reward_fn=length_reward,
    max_steps=100,                   # Quick training for demo
)

print(f"Training completed! Final metrics: {result.metrics}")
```

Run it:

```bash
python train_model.py
```

That's it! **5 lines of code** for a complete RL training run.

## Step 3: Custom Reward Functions (2 minutes)

The power of RLHF comes from defining what makes a "good" response.
Here are common reward function patterns:

### Pattern 1: Simple Rule-Based Reward

```python
def correctness_reward(prompt: str, response: str) -> float:
    """Reward responses that contain expected keywords."""
    if "error" in response.lower() or "sorry" in response.lower():
        return 0.0
    if any(word in response.lower() for word in ["answer", "result", "solution"]):
        return 1.0
    return 0.5
```

### Pattern 2: Length-Balanced Reward

```python
def balanced_reward(prompt: str, response: str) -> float:
    """Reward medium-length responses (not too short, not too long)."""
    length = len(response)
    if length < 50:
        return length / 50  # Penalize very short
    elif length < 500:
        return 1.0  # Sweet spot
    else:
        return max(0.3, 1.0 - (length - 500) / 1000)  # Penalize very long
```

### Pattern 3: Multiple Criteria Reward

```python
def multi_criteria_reward(prompt: str, response: str) -> float:
    """Combine multiple reward signals."""
    score = 0.0
    
    # Criterion 1: Not empty
    if len(response.strip()) > 10:
        score += 0.3
    
    # Criterion 2: Proper formatting
    if response.strip().endswith((".", "!", "?")):
        score += 0.2
    
    # Criterion 3: Contains explanation
    if any(word in response.lower() for word in ["because", "therefore", "since"]):
        score += 0.3
    
    # Criterion 4: Reasonable length
    if 100 < len(response) < 1000:
        score += 0.2
    
    return score
```

### Using Your Custom Reward

```python
result = nemo_rl.train(
    model="Qwen/Qwen2.5-0.5B",
    dataset="nvidia/HelpSteer2",
    reward_fn=multi_criteria_reward,  # Your custom function
    max_steps=1000,
)
```

## Step 4: Understanding the Output (2 minutes)

The `train()` function returns a `TrainResult` object with useful information:

```python
result = nemo_rl.train(
    model="Qwen/Qwen2.5-0.5B",
    dataset="nvidia/HelpSteer2",
    reward_fn=length_reward,
    max_steps=100,
)

# Access training results
print(f"Total steps: {result.total_steps}")
print(f"Checkpoint saved at: {result.checkpoint_path}")
print(f"Training metrics: {result.metrics}")

# Access the trained model for further use
trainer = result.trainer
```

### Key Metrics to Watch

- **loss**: Training loss (should decrease)
- **mean_reward**: Average reward across batches (should increase)
- **lr**: Current learning rate

## Step 5: More Control with Trainers (1 minute)

For more control, use trainer classes directly:

```python
from nemo_rl import GRPOTrainer

# Initialize from pretrained model
trainer = GRPOTrainer.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    num_prompts_per_step=32,      # Batch size
    num_generations_per_prompt=8, # Samples per prompt
    learning_rate=1e-6,
)

# Train with your reward function
trainer.fit(
    dataset="nvidia/HelpSteer2",
    reward_fn=length_reward,
    max_steps=1000,
)
```

## Quick Reference: Common Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | *required* | HuggingFace model name or local path |
| `dataset` | *required* | HuggingFace dataset or list of prompts |
| `reward_fn` | *required for GRPO* | Your reward function |
| `algorithm` | `"grpo"` | Training algorithm: "grpo", "sft", "dpo" |
| `max_steps` | `1000` | Maximum training steps |
| `learning_rate` | `1e-6` | Learning rate |
| `batch_size` | `32` | Prompts per training step |

### Model Recommendations

| Use Case | Model | Size |
|----------|-------|------|
| Quick testing | `Qwen/Qwen2.5-0.5B` | 0.5B |
| Development | `Qwen/Qwen2.5-1.5B` | 1.5B |
| Production (small) | `Qwen/Qwen2.5-7B` | 7B |
| Production (large) | `Qwen/Qwen2.5-72B` | 72B |

## Troubleshooting

### Issue: Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Solutions**:
1. Use a smaller model (`Qwen/Qwen2.5-0.5B`)
2. Reduce batch size:
   ```python
   nemo_rl.train(..., batch_size=16)
   ```
3. Reduce generations per prompt:
   ```python
   nemo_rl.train(..., num_generations_per_prompt=4)
   ```

### Issue: Slow Training

**Symptoms**: Training takes much longer than expected

**Solutions**:
1. Check GPU utilization: `nvidia-smi`
2. Ensure CUDA is being used:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```
3. Use a faster reward function (avoid API calls if possible)

### Issue: Reward Not Improving

**Symptoms**: Mean reward stays flat during training

**Solutions**:
1. Verify your reward function:
   ```python
   # Test it manually first
   score = reward_fn("Sample prompt", "Sample response")
   print(f"Score: {score}")  # Should be a number
   ```
2. Check reward scale (should typically be 0-1)
3. Increase learning rate slightly
4. Train for more steps

### Issue: Import Errors

**Symptoms**: `ModuleNotFoundError` when importing nemo_rl

**Solutions**:
1. Verify installation: `pip show nemo-rl`
2. Check Python version: `python --version` (need 3.10+)
3. Reinstall: `pip uninstall nemo-rl && pip install nemo-rl`

## Next Steps

Now that you've completed your first training run, explore these resources:

1. **[API Documentation](api/)** - Complete reference for all functions
2. **[GRPO Guide](guides/grpo.md)** - Deep dive into GRPO training
3. **[SFT Guide](guides/sft.md)** - Supervised fine-tuning
4. **[Custom Environments](guides/environments.md)** - Advanced reward functions
5. **[Architecture Guide](architecture/README.md)** - How NeMo RL works

## Example Scripts

Find more examples in the `examples/` directory:

```bash
# Math reasoning with GRPO
python examples/run_grpo_math.py

# Supervised fine-tuning
python examples/run_sft.py

# Direct preference optimization
python examples/run_dpo.py
```

---

**Congratulations!** You've completed the NeMo RL quickstart tutorial.
You now know how to:

- Install NeMo RL
- Train models with the simple `train()` API
- Define custom reward functions
- Use trainer classes for more control
- Troubleshoot common issues

Happy training! 🚀
