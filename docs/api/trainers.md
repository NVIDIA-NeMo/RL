# Trainer Classes

NeMo RL provides specialized trainer classes for different RL algorithms.
Each trainer follows the HuggingFace-style `from_pretrained()` pattern.

## Available Trainers

| Trainer | Algorithm | Use Case |
|---------|-----------|----------|
| `GRPOTrainer` | Group Relative Policy Optimization | RL with reward functions |
| `SFTTrainer` | Supervised Fine-Tuning | Imitation learning |
| `DPOTrainer` | Direct Preference Optimization | Learning from preferences |

## GRPOTrainer

Group Relative Policy Optimization for training with reward signals.

### Class Definition

```python
class GRPOTrainer(BaseTrainer):
    """GRPO trainer for reinforcement learning."""
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        num_prompts_per_step: int = 32,
        num_generations_per_prompt: int = 16,
        learning_rate: float = 1e-6,
        max_sequence_length: int = 512,
        tensor_parallel_size: int = 1,
        **kwargs,
    ) -> "GRPOTrainer":
        """Create trainer from pretrained model."""
    
    def fit(
        self,
        dataset: DataType,
        reward_fn: Optional[RewardFnType] = None,
        environment: Optional[EnvironmentInterface] = None,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> TrainingResult:
        """Train the model."""
```

### from_pretrained() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | `str` | *required* | HuggingFace model or local path |
| `num_prompts_per_step` | `int` | `32` | Batch size (prompts per step) |
| `num_generations_per_prompt` | `int` | `16` | Samples per prompt |
| `learning_rate` | `float` | `1e-6` | Optimizer learning rate |
| `max_sequence_length` | `int` | `512` | Maximum token length |
| `tensor_parallel_size` | `int` | `1` | Tensor parallelism degree |
| `normalize_rewards` | `bool` | `True` | Normalize rewards |
| `use_leave_one_out_baseline` | `bool` | `True` | LOO baseline |

### fit() Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset` | `str \| list \| DataModule` | Training data |
| `reward_fn` | `Callable[[str, str], float]` | Reward function |
| `environment` | `EnvironmentInterface` | Environment instance (alternative to reward_fn) |
| `max_steps` | `int` | Override max training steps |
| `max_epochs` | `int` | Override max epochs |
| `callbacks` | `list[Callback]` | Training callbacks |

### Example Usage

```python
from nemo_rl import GRPOTrainer

# Initialize from pretrained
trainer = GRPOTrainer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    num_prompts_per_step=32,
    num_generations_per_prompt=16,
    learning_rate=1e-6,
)

# Define reward function
def reward_fn(prompt: str, response: str) -> float:
    return 1.0 if "correct" in response else 0.0

# Train
result = trainer.fit(
    dataset="nvidia/OpenMathInstruct-2",
    reward_fn=reward_fn,
    max_steps=1000,
)

print(f"Training completed: {result.metrics}")
```

## SFTTrainer

Supervised Fine-Tuning for imitation learning from demonstrations.

### Class Definition

```python
class SFTTrainer(BaseTrainer):
    """SFT trainer for supervised fine-tuning."""
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        max_sequence_length: int = 512,
        **kwargs,
    ) -> "SFTTrainer":
        """Create trainer from pretrained model."""
    
    def fit(
        self,
        dataset: DataType,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> TrainingResult:
        """Train the model."""
```

### Example Usage

```python
from nemo_rl import SFTTrainer

# Initialize
trainer = SFTTrainer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    batch_size=16,
    learning_rate=2e-5,
)

# Train (no reward function needed)
result = trainer.fit(
    dataset="nvidia/OpenMathInstruct-2",
    max_steps=5000,
)
```

## DPOTrainer

Direct Preference Optimization for learning from preference data.

### Class Definition

```python
class DPOTrainer(BaseTrainer):
    """DPO trainer for preference optimization."""
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        batch_size: int = 32,
        learning_rate: float = 1e-6,
        beta: float = 0.1,
        **kwargs,
    ) -> "DPOTrainer":
        """Create trainer from pretrained model."""
    
    def fit(
        self,
        dataset: DataType,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> TrainingResult:
        """Train the model."""
```

### DPO-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | `float` | `0.1` | KL penalty coefficient |

### Example Usage

```python
from nemo_rl import DPOTrainer

# Initialize
trainer = DPOTrainer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    batch_size=16,
    beta=0.1,
)

# Train with preference data
result = trainer.fit(
    dataset="nvidia/HelpSteer2",  # Preference dataset
    max_steps=1000,
)
```

## BaseTrainer

All trainers inherit from `BaseTrainer`, which provides common functionality.

### Common Methods

```python
class BaseTrainer:
    """Base class for all trainers."""
    
    def fit(
        self,
        dataset: DataType,
        **kwargs,
    ) -> TrainingResult:
        """Train the model."""
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get trainer configuration."""
    
    @property
    def current_step(self) -> int:
        """Get current training step."""
```

## TrainingResult

Returned by `fit()` method:

```python
@dataclass
class TrainingResult:
    metrics: Dict[str, Any]          # Training metrics
    best_checkpoint_path: str        # Path to best checkpoint
    total_steps: int                 # Steps completed
    total_epochs: int                # Epochs completed
```

### Accessing Results

```python
result = trainer.fit(dataset="...", reward_fn=my_fn)

# Access metrics
print(f"Loss: {result.metrics['loss']}")
print(f"Reward: {result.metrics.get('mean_reward', 'N/A')}")

# Use checkpoint
print(f"Best checkpoint: {result.best_checkpoint_path}")
```

## Callbacks

Customize training behavior with callbacks:

```python
from nemo_rl.trainers.callbacks import Callback

class MyCallback(Callback):
    def on_train_begin(self, trainer):
        print("Training started")
    
    def on_step_begin(self, trainer, step):
        pass
    
    def on_step_end(self, trainer, step, metrics):
        if step % 100 == 0:
            print(f"Step {step}: {metrics}")
    
    def on_epoch_end(self, trainer, epoch, metrics):
        print(f"Epoch {epoch} completed")
    
    def on_train_end(self, trainer, result):
        print(f"Training finished: {result.metrics}")

# Use callback
trainer.fit(
    dataset="...",
    reward_fn=my_fn,
    callbacks=[MyCallback()],
)
```

## Advanced Configuration

For advanced use cases, pass configuration directly:

```python
from nemo_rl import GRPOTrainer

# Create trainer with full config
config = {
    "policy": {
        "model_name": "Qwen/Qwen2.5-1.5B",
        "learning_rate": 1e-6,
        "dtensor_cfg": {
            "enabled": True,
            "tensor_parallel_size": 2,
        },
    },
    "grpo": {
        "num_prompts_per_step": 64,
        "num_generations_per_prompt": 8,
    },
}

trainer = GRPOTrainer(config)
trainer.fit(dataset="...", reward_fn=my_fn)
```

## Functional API Helpers

```python
from nemo_rl.api import create_trainer, list_algorithms, get_algorithm

# List available algorithms
print(list_algorithms())  # ['dpo', 'grpo', 'sft']

# Get trainer class
GRPOCls = get_algorithm("grpo")

# Create trainer dynamically
trainer = create_trainer(
    "grpo",
    model="Qwen/Qwen2.5-1.5B",
    learning_rate=1e-6,
)
```

## Related APIs

- [train()](train.md) - Simple training function
- [Environments](environments.md) - Custom reward environments
- [Config](config.md) - Configuration options
