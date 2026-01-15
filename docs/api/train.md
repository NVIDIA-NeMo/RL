# nemo_rl.train() API

The `train()` function is the simplest way to train reinforcement learning models with NeMo RL.
It enables 5-line training scripts with sensible defaults.

## Function Signature

```python
def train(
    model: str,
    dataset: Optional[DataType] = None,
    reward_fn: Optional[RewardFnType] = None,
    algorithm: str = "grpo",
    max_steps: int = 1000,
    max_epochs: int = 1,
    learning_rate: float = 1e-6,
    batch_size: int = 32,
    num_generations_per_prompt: int = 16,
    output_dir: Optional[str] = None,
    callbacks: Optional[Sequence[Callback]] = None,
    **kwargs: Any,
) -> TrainResult:
    """Train a language model with reinforcement learning."""
```

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model identifier. HuggingFace model name (e.g., `"Qwen/Qwen2.5-1.5B"`) or local path to model directory. |
| `dataset` | `str \| list \| DataModule` | Training data. HuggingFace dataset name, list of prompt dictionaries, or DataModule instance. |
| `reward_fn` | `Callable[[str, str], float]` | Reward function (required for GRPO). Signature: `(prompt, response) -> float`. |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm` | `str` | `"grpo"` | Training algorithm: `"grpo"`, `"sft"`, or `"dpo"`. |
| `max_steps` | `int` | `1000` | Maximum number of training steps. |
| `max_epochs` | `int` | `1` | Maximum number of training epochs. |
| `learning_rate` | `float` | `1e-6` | Learning rate for optimizer. |
| `batch_size` | `int` | `32` | Number of prompts per training step. |
| `num_generations_per_prompt` | `int` | `16` | Samples per prompt (GRPO only). |
| `output_dir` | `str \| None` | `None` | Directory for checkpoints/logs. Defaults to `./outputs/{model_name}`. |
| `callbacks` | `list[Callback] \| None` | `None` | Custom callback instances. |

### Algorithm-Specific Parameters

Pass these as keyword arguments:

#### GRPO

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize_rewards` | `bool` | `True` | Normalize rewards across batch. |
| `use_leave_one_out_baseline` | `bool` | `True` | Use leave-one-out baseline. |

#### DPO

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | `float` | `0.1` | KL penalty coefficient. |

## Return Value

Returns a `TrainResult` dataclass:

```python
@dataclass
class TrainResult:
    trainer: BaseTrainer     # The trainer instance
    metrics: Dict[str, Any]  # Final training metrics
    checkpoint_path: str     # Path to best checkpoint
    total_steps: int         # Total steps completed
```

### Accessing Results

```python
result = nemo_rl.train(...)

# Get training metrics
print(f"Final loss: {result.metrics.get('loss', 'N/A')}")
print(f"Total steps: {result.total_steps}")
print(f"Checkpoint: {result.checkpoint_path}")

# Access the trained model
trainer = result.trainer
```

## Usage Examples

### Example 1: Minimal GRPO Training

```python
import nemo_rl

def reward_fn(prompt: str, response: str) -> float:
    """Reward longer, more detailed responses."""
    return min(len(response) / 200, 1.0)

result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    dataset="nvidia/HelpSteer2",
    reward_fn=reward_fn,
)
```

### Example 2: SFT (Supervised Fine-Tuning)

```python
import nemo_rl

# SFT doesn't require a reward function
result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    dataset="nvidia/OpenMathInstruct-2",
    algorithm="sft",
    max_steps=5000,
    learning_rate=2e-5,
)
```

### Example 3: DPO (Direct Preference Optimization)

```python
import nemo_rl

result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    dataset="nvidia/HelpSteer2",
    algorithm="dpo",
    beta=0.1,  # DPO-specific
    max_steps=1000,
)
```

### Example 4: Custom Dataset

```python
import nemo_rl

# Dataset as list of prompts
prompts = [
    {"prompt": "What is 2+2?"},
    {"prompt": "Explain quantum computing"},
    {"prompt": "Write a poem about Python"},
]

result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    dataset=prompts,
    reward_fn=lambda p, r: 1.0,
    max_steps=100,
)
```

### Example 5: With Callbacks

```python
import nemo_rl
from nemo_rl.trainers.callbacks import Callback

class LoggingCallback(Callback):
    def on_step_end(self, trainer, step, metrics):
        print(f"Step {step}: loss={metrics.get('loss', 'N/A')}")

result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    dataset="nvidia/HelpSteer2",
    reward_fn=lambda p, r: 1.0,
    callbacks=[LoggingCallback()],
)
```

### Example 6: Full Configuration

```python
import nemo_rl

result = nemo_rl.train(
    # Model
    model="Qwen/Qwen2.5-7B",
    
    # Data
    dataset="nvidia/OpenMathInstruct-2",
    
    # Reward
    reward_fn=my_complex_reward,
    
    # Algorithm
    algorithm="grpo",
    
    # Training
    max_steps=10000,
    max_epochs=3,
    learning_rate=5e-7,
    batch_size=64,
    num_generations_per_prompt=8,
    
    # Output
    output_dir="./experiments/run_001",
    
    # GRPO-specific
    normalize_rewards=True,
    use_leave_one_out_baseline=True,
    
    # Logging
    tensorboard=True,
    log_level="INFO",
)
```

## Exceptions

| Exception | Cause |
|-----------|-------|
| `ValueError` | Unknown algorithm name |
| `ValueError` | Missing required `dataset` |
| `ValueError` | Missing `reward_fn` for GRPO |
| `TypeError` | Invalid `reward_fn` signature |

## Best Practices

1. **Start simple**: Use minimal parameters first, then add complexity.

2. **Test reward function**: Verify your reward function works before training:
   ```python
   score = reward_fn("test prompt", "test response")
   assert isinstance(score, float), "Reward should be float"
   ```

3. **Use appropriate model size**: Start with smaller models for development.

4. **Monitor training**: Check metrics during training for issues.

5. **Save checkpoints**: Use `output_dir` to persist training progress.

## Related APIs

- [Trainers](trainers.md) - Direct trainer class usage
- [Environments](environments.md) - Custom reward environments
- [Config](config.md) - Advanced configuration options
