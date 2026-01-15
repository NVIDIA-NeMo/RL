# NeMo RL Architecture Guide

This guide provides a comprehensive overview of the NeMo RL architecture for contributors and developers.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Module Structure](#module-structure)
3. [Data Flow](#data-flow)
4. [Extension Points](#extension-points)
5. [Design Patterns](#design-patterns)
6. [Contributing Guidelines](#contributing-guidelines)

---

## High-Level Architecture

NeMo RL follows a layered architecture designed for modularity and extensibility:

```
┌─────────────────────────────────────────────────────────────┐
│                      User API Layer                         │
│  nemo_rl.train() | Trainer.from_pretrained() | DataModule  │
├─────────────────────────────────────────────────────────────┤
│                     Algorithm Layer                          │
│      GRPOTrainer | SFTTrainer | DPOTrainer | RolloutEngine  │
├─────────────────────────────────────────────────────────────┤
│                     Backend Layer                            │
│   TrainingBackend (DTensor, Megatron)                       │
│   GenerationBackend (vLLM, Megatron)                        │
├─────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                       │
│  ResourceManager | CheckpointManager | LoggerFacade         │
├─────────────────────────────────────────────────────────────┤
│                     Policy Layer                             │
│    TrainingPolicy | GenerationPolicy | PolicyWorkers        │
└─────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Separation of Concerns**: Each layer has distinct responsibilities
2. **Protocol-Based Interfaces**: Use Python Protocols for backend abstraction
3. **Sensible Defaults**: Minimal configuration required for common use cases
4. **Backward Compatibility**: Legacy APIs continue to work with warnings

---

## Module Structure

```
nemo_rl/
├── api/                    # High-level user API
│   ├── train.py           # nemo_rl.train() function
│   └── functional.py      # Functional helpers
│
├── algorithms/             # Training algorithms
│   ├── grpo/              # Group Relative Policy Optimization
│   │   ├── config.py      # GRPOConfig classes
│   │   ├── loss.py        # GRPO loss computation
│   │   ├── data.py        # Data transforms
│   │   └── trainer.py     # GRPOTrainer class
│   ├── sft/               # Supervised Fine-Tuning
│   ├── dpo/               # Direct Preference Optimization
│   └── rollout.py         # RolloutEngine
│
├── backends/               # Backend abstractions
│   ├── factory.py         # BackendFactory
│   ├── training/          # Training backends
│   │   ├── base.py        # TrainingBackend Protocol
│   │   ├── dtensor.py     # DTensorBackend
│   │   └── megatron.py    # MegatronBackend
│   └── generation/        # Generation backends
│       ├── base.py        # GenerationBackend Protocol
│       ├── vllm.py        # VLLMBackend
│       └── megatron.py    # MegatronInferenceBackend
│
├── config/                 # Configuration system
│   ├── base.py            # BaseConfig with Pydantic
│   ├── policy.py          # PolicyConfig
│   ├── training.py        # GRPOConfig, SFTConfig, DPOConfig
│   ├── cluster.py         # ClusterConfig
│   └── validation.py      # Config validation
│
├── data/                   # Data handling
│   ├── module.py          # DataModule interface
│   ├── module_hf.py       # HuggingFace integration
│   └── datasets/          # Built-in datasets
│
├── environments/           # Reward environments
│   ├── interfaces.py      # EnvironmentInterface
│   ├── functional_reward.py # FunctionalRewardWrapper
│   └── math_environment.py # Math reward implementation
│
├── infra/                  # Infrastructure
│   ├── resources.py       # ResourceManager
│   ├── checkpointing.py   # CheckpointManager
│   └── logging.py         # LoggerFacade
│
├── models/                 # Model implementations
│   └── policy/            # Policy models
│       ├── training_policy.py    # TrainingPolicy
│       └── generation_policy.py  # GenerationPolicy
│
└── trainers/               # Trainer base classes
    ├── base.py            # BaseTrainer
    ├── callbacks.py       # Callback system
    └── validation.py      # ValidationRunner
```

### Module Responsibilities

| Module | Responsibility | Max Lines |
|--------|----------------|-----------|
| `api/` | High-level user interface | ~300 |
| `algorithms/{algo}/trainer.py` | Algorithm-specific training logic | ~300 |
| `algorithms/{algo}/loss.py` | Loss computation | ~200 |
| `backends/*/base.py` | Backend protocol definition | ~400 |
| `config/*.py` | Configuration classes | ~200 each |
| `trainers/base.py` | Common trainer functionality | ~500 |

---

## Data Flow

### GRPO Training Flow

```
┌─────────────────┐
│  User Dataset   │  (HuggingFace name, list, DataModule)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   DataModule    │  Handles batching, shuffling
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GRPOTrainer    │  Orchestrates training loop
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌──────────────┐
│Rollout│ │  Training    │
│Engine │ │  Backend     │
└───┬───┘ └──────┬───────┘
    │            │
    ▼            ▼
┌───────────────────────────┐
│   GenerationPolicy        │  Generate responses
├───────────────────────────┤
│   Environment (Reward)    │  Compute rewards
├───────────────────────────┤
│   TrainingPolicy          │  Update model weights
└───────────────────────────┘
```

### Request Flow for train()

```python
nemo_rl.train(model="...", dataset="...", reward_fn=fn)
    │
    ├─► _validate_inputs()          # Check required args
    │
    ├─► _create_trainer_for_algorithm()
    │       │
    │       ├─► _build_grpo_config()  # Build config dict
    │       │
    │       └─► GRPOTrainer(config)   # Create trainer
    │
    └─► trainer.fit(dataset, ...)
            │
            ├─► setup()              # Initialize components
            │       ├─► _setup_logger()
            │       ├─► _setup_checkpointing()
            │       └─► _setup_cluster()
            │
            ├─► _training_loop()     # Main training
            │       ├─► _on_epoch_begin()
            │       ├─► _train_step()
            │       └─► _on_epoch_end()
            │
            └─► TrainResult          # Return results
```

---

## Extension Points

### Adding a New Algorithm

1. **Create algorithm package** in `nemo_rl/algorithms/{name}/`

```
algorithms/myalgo/
├── __init__.py
├── config.py      # MyAlgoConfig
├── loss.py        # Loss function
├── data.py        # Data transforms
└── trainer.py     # MyAlgoTrainer
```

2. **Extend BaseTrainer** in `trainer.py`:

```python
from nemo_rl.trainers.base import BaseTrainer

class MyAlgoTrainer(BaseTrainer):
    """My custom algorithm trainer."""

    @classmethod
    def _build_config_from_pretrained(cls, model, **kwargs):
        config = super()._build_config_from_pretrained(model, **kwargs)
        config["myalgo"] = {
            "my_param": kwargs.pop("my_param", 1.0),
        }
        return config

    def __init__(self, config):
        super().__init__(config)
        self._myalgo_config = config.get("myalgo", {})

    def _train_step(self, batch):
        # Your training logic
        return {"loss": loss_value}

    def _compute_loss(self, batch, outputs):
        # Your loss computation
        return loss_tensor
```

3. **Register in `algorithms/__init__.py`**:

```python
from nemo_rl.algorithms.myalgo import MyAlgoTrainer, MyAlgoConfig

__all__ += ["MyAlgoTrainer", "MyAlgoConfig"]
```

4. **Add to API functional helpers** in `api/functional.py`:

```python
try:
    from nemo_rl.algorithms.myalgo import MyAlgoTrainer
    _ALGORITHM_REGISTRY["myalgo"] = MyAlgoTrainer
except ImportError:
    pass
```

### Adding a New Training Backend

1. **Implement TrainingBackend Protocol**:

```python
from nemo_rl.backends.training.base import TrainingBackend, register_training_backend

@register_training_backend("custom")
class CustomTrainingBackend(TrainingBackend):
    """Custom training backend."""

    def setup(self, config):
        # Initialize backend
        pass

    def train_step(self, batch, loss_fn):
        # Execute training step
        return {"loss": loss_value}

    def get_logprobs(self, batch):
        # Compute log probabilities
        return logprobs

    def save_checkpoint(self, path):
        # Save model state
        pass

    def load_checkpoint(self, path):
        # Load model state
        pass

    def shutdown(self):
        # Clean up resources
        pass
```

2. **Backend is automatically registered** via decorator.

3. **Use via string selection**:

```python
from nemo_rl.backends import get_training_backend

backend = get_training_backend("custom")
```

### Adding a New Generation Backend

Similar to training backend, implement GenerationBackend Protocol:

```python
from nemo_rl.backends.generation.base import GenerationBackend, register_generation_backend

@register_generation_backend("custom")
class CustomGenerationBackend(GenerationBackend):

    def setup(self, config):
        pass

    def generate(self, prompts, sampling_params):
        return responses

    def update_weights(self, state_dict):
        pass

    def shutdown(self):
        pass
```

### Adding a New Environment (Reward Function)

**Option 1: Simple callable** (recommended for simple rewards):

```python
def my_reward(prompt: str, response: str) -> float:
    return 1.0 if "correct" in response else 0.0

trainer.fit(dataset=data, reward_fn=my_reward)
```

**Option 2: FunctionalRewardWrapper** (for more control):

```python
from nemo_rl.environments import FunctionalRewardWrapper

def batch_reward(prompts: list[str], responses: list[str]) -> list[float]:
    return [score(p, r) for p, r in zip(prompts, responses)]

env = FunctionalRewardWrapper(batch_reward, name="batch_scorer")
```

**Option 3: Full EnvironmentInterface** (for complex environments):

```python
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

class MyEnvironment(EnvironmentInterface):
    def step(self, message_log_batch, metadata):
        # Compute rewards
        rewards = self._compute_rewards(message_log_batch)

        return EnvironmentReturn(
            observations=[],
            metadata=[None] * len(message_log_batch),
            next_stop_strings=[None] * len(message_log_batch),
            rewards=torch.tensor(rewards),
            terminateds=torch.ones(len(message_log_batch), dtype=torch.bool),
            answers=None,
        )

    def global_post_process_and_metrics(self, batch):
        return batch, {"mean_reward": batch["rewards"].mean().item()}
```

---

## Design Patterns

### Protocol Pattern (Backend Abstraction)

We use Python's `typing.Protocol` for backend abstraction:

```python
from typing import Protocol

class TrainingBackend(Protocol):
    """Protocol defining training backend interface."""

    def setup(self, config: dict) -> None: ...
    def train_step(self, batch: Any, loss_fn: Callable) -> dict: ...
    def save_checkpoint(self, path: str) -> None: ...
```

**Benefits:**
- Structural subtyping (duck typing with type checking)
- No inheritance required
- Clear interface documentation

### Factory Pattern (Backend Selection)

Backend selection uses a factory with registration:

```python
# Registration
@register_training_backend("custom")
class CustomBackend(TrainingBackend):
    pass

# Usage
backend = get_training_backend("custom")
```

**Benefits:**
- Single string selects backend
- Easy to add custom backends
- Clear error messages for unknown backends

### Adapter Pattern (Functional Rewards)

FunctionalRewardWrapper adapts callables to EnvironmentInterface:

```python
# User provides simple function
def reward(p: str, r: str) -> float:
    return 1.0

# Adapter makes it work with trainer
wrapper = FunctionalRewardWrapper(reward)
# wrapper implements EnvironmentInterface
```

### Callback Pattern (Training Extensibility)

Callbacks provide hooks into the training loop:

```python
from nemo_rl.trainers.callbacks import Callback

class MyCallback(Callback):
    def on_epoch_end(self, trainer, epoch, metrics):
        print(f"Epoch {epoch}: {metrics['loss']}")

trainer.fit(data, callbacks=[MyCallback()])
```

---

## Contributing Guidelines

### Code Style

1. **Type Annotations**: All public functions must have type hints
2. **Docstrings**: Google-style docstrings for all public methods
3. **Line Length**: 88 characters (Black formatter)
4. **Imports**: isort with black profile

```python
def my_function(
    param1: str,
    param2: Optional[int] = None,
) -> dict[str, Any]:
    """Short description.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something is wrong.

    Example:
        >>> result = my_function("test")
    """
    pass
```

### File Size Guidelines

| File Type | Max Lines | Action if Exceeded |
|-----------|-----------|-------------------|
| Trainer | 300 | Split into submodules |
| Config | 200 | Split by category |
| Backend | 500 | Consider abstraction |
| Utility | 200 | Group related functions |

### Testing Requirements

- **Unit Tests**: Required for all new code
- **Coverage**: Minimum 85% for new code
- **Test Location**: Mirror source structure in `tests/unit/`

```python
# tests/unit/algorithms/test_myalgo.py
import pytest
from nemo_rl.algorithms.myalgo import MyAlgoTrainer

class TestMyAlgoTrainer:
    def test_train_step_returns_loss(self):
        trainer = MyAlgoTrainer(config)
        result = trainer._train_step(batch)
        assert "loss" in result
```

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Type hints added for new functions
- [ ] Docstrings added for public APIs
- [ ] Unit tests written and passing
- [ ] No files exceed line limits
- [ ] Backward compatibility maintained
- [ ] Documentation updated if needed

---

## Quick Reference

### Creating a 5-Line Training Script

```python
import nemo_rl

result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    dataset="nvidia/OpenMathInstruct-2",
    reward_fn=lambda p, r: 1.0 if "correct" in r else 0.0,
)
```

### Using from_pretrained()

```python
from nemo_rl import GRPOTrainer

trainer = GRPOTrainer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    num_prompts_per_step=32,
)
trainer.fit(dataset="my-dataset", reward_fn=my_reward)
```

### Selecting Backends

```python
from nemo_rl.backends import get_training_backend, get_generation_backend

training = get_training_backend("dtensor")  # or "megatron"
generation = get_generation_backend("vllm")  # or "megatron"
```

### Custom Callbacks

```python
from nemo_rl.trainers.callbacks import Callback

class LoggingCallback(Callback):
    def on_step_end(self, trainer, step, metrics):
        if step % 100 == 0:
            print(f"Step {step}: loss={metrics['loss']:.4f}")

trainer.fit(data, callbacks=[LoggingCallback()])
```

---

*Last Updated: 2026-01-15*
*Version: 1.0.0*
