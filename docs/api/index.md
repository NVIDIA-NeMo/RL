# NeMo RL API Reference

This section provides complete documentation for all public APIs in NeMo RL.

## Quick Navigation

| Module | Description |
|--------|-------------|
| [train](train.md) | `nemo_rl.train()` - Single-function training API |
| [trainers](trainers.md) | Trainer classes (`GRPOTrainer`, `SFTTrainer`, `DPOTrainer`) |
| [environments](environments.md) | Environment and reward function APIs |
| [config](config.md) | Configuration classes and options |
| [backends](backends.md) | Backend selection and configuration |

## API Hierarchy

```
nemo_rl
├── train()                    # Simplest API - 5 lines of code
├── GRPOTrainer                # GRPO training
│   ├── from_pretrained()      # HuggingFace-style initialization
│   └── fit()                  # Train the model
├── SFTTrainer                 # Supervised fine-tuning
│   ├── from_pretrained()
│   └── fit()
├── DPOTrainer                 # Direct preference optimization
│   ├── from_pretrained()
│   └── fit()
├── Environment                # Base class for reward environments
├── FunctionalRewardWrapper    # Wrap callable as environment
└── api
    ├── create_trainer()       # Dynamic trainer creation
    ├── list_algorithms()      # List available algorithms
    └── get_algorithm()        # Get trainer class by name
```

## Quick Start Examples

### Simplest API: `nemo_rl.train()`

```python
import nemo_rl

result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    dataset="nvidia/HelpSteer2",
    reward_fn=lambda p, r: len(r) / 100,
)
```

### Trainer Classes

```python
from nemo_rl import GRPOTrainer

trainer = GRPOTrainer.from_pretrained("Qwen/Qwen2.5-1.5B")
trainer.fit(dataset="nvidia/HelpSteer2", reward_fn=my_reward)
```

### Custom Environment

```python
from nemo_rl.environments import Environment

class MyReward(Environment):
    def score(self, prompt: str, response: str) -> float:
        return 1.0 if "correct" in response else 0.0

trainer.fit(dataset="...", environment=MyReward())
```

## Import Patterns

### Recommended Imports

```python
# Main training API
import nemo_rl
result = nemo_rl.train(...)

# Trainer classes
from nemo_rl import GRPOTrainer, SFTTrainer, DPOTrainer

# Environments
from nemo_rl.environments import Environment, FunctionalRewardWrapper

# API helpers
from nemo_rl.api import create_trainer, list_algorithms
```

### Full Module Structure

```python
import nemo_rl

# Direct access (lazy loading)
nemo_rl.train              # Main training function
nemo_rl.TrainResult        # Training result class
nemo_rl.GRPOTrainer        # GRPO trainer
nemo_rl.SFTTrainer         # SFT trainer
nemo_rl.DPOTrainer         # DPO trainer
nemo_rl.BaseTrainer        # Base trainer class
nemo_rl.DataModule         # Data loading

# Submodules
nemo_rl.api                # High-level API helpers
nemo_rl.algorithms         # Training algorithms
nemo_rl.environments       # Reward environments
nemo_rl.config             # Configuration classes
nemo_rl.backends           # Training/generation backends
nemo_rl.trainers           # Trainer base classes
nemo_rl.data               # Data handling
nemo_rl.infra              # Infrastructure utilities
```

## Version Information

```python
import nemo_rl

print(nemo_rl.__version__)  # Package version
print(nemo_rl.__license__)  # Apache 2.0
```
