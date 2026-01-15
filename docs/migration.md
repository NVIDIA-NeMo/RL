# Migration Guide: Old API to New API

This guide helps you migrate from the old NeMo RL API to the new simplified API introduced in version 1.0.

## Overview

The new API provides a simpler, more intuitive interface while maintaining full backward compatibility with the old API. Key changes include:

1. **Single-function training**: `nemo_rl.train()` replaces complex config files
2. **HuggingFace-style initialization**: `Trainer.from_pretrained()` for familiar patterns
3. **Simple reward functions**: Any callable works, no Environment class required
4. **Unified configuration**: Programmatic config instead of YAML files

## Quick Comparison

### Before (Old API)

```python
# Old API: Many lines of configuration and setup
from nemo_rl.algorithms.grpo import GRPO
from omegaconf import OmegaConf

# Load complex YAML config
cfg = OmegaConf.load("configs/grpo_math_1B.yaml")

# Modify settings programmatically
cfg.policy.model_name = "Qwen/Qwen2.5-1.5B"
cfg.grpo.max_num_steps = 1000
cfg.policy.optimizer.kwargs.lr = 1e-6

# Create algorithm and run
grpo = GRPO.from_config(cfg)
grpo.fit()
```

### After (New API)

```python
# New API: 5 lines of code
import nemo_rl

result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    dataset="nvidia/OpenMathInstruct-2",
    reward_fn=lambda p, r: 1.0 if "correct" in r else 0.0,
    max_steps=1000,
)
```

## Detailed Migration Guide

### 1. Simple Training Scripts

#### Old Way

```python
from nemo_rl.algorithms.grpo import GRPO
from nemo_rl.environments.math_environment import MathEnvironment
from omegaconf import OmegaConf
import ray

# Initialize Ray cluster
ray.init()

# Load and customize config
cfg = OmegaConf.load("examples/configs/grpo_math_1B.yaml")
cfg.policy.model_name = "Qwen/Qwen2.5-1.5B"
cfg.grpo.num_prompts_per_step = 32
cfg.grpo.num_generations_per_prompt = 16
cfg.grpo.max_num_steps = 1000
cfg.policy.optimizer.kwargs.lr = 1e-6
cfg.checkpointing.checkpoint_dir = "./outputs"

# Create environment
env_cfg = {"num_workers": 8}
env_actor = MathEnvironment.remote(env_cfg)

# Create algorithm and train
grpo = GRPO.from_config(cfg)
grpo.fit(environment=env_actor)
```

#### New Way

```python
import nemo_rl

def math_reward(prompt: str, response: str) -> float:
    """Simple math verification reward."""
    # Your reward logic here
    return 1.0 if "answer" in response else 0.0

result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    dataset="nvidia/OpenMathInstruct-2",
    reward_fn=math_reward,
    max_steps=1000,
    learning_rate=1e-6,
    batch_size=32,
    num_generations_per_prompt=16,
)
```

### 2. Using Trainer Classes

#### Old Way

```python
from nemo_rl.algorithms.grpo import GRPO
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/grpo_math_1B.yaml")
cfg.policy.model_name = "Qwen/Qwen2.5-1.5B"
# ... many more config settings

grpo = GRPO.from_config(cfg)
grpo.fit()
```

#### New Way

```python
from nemo_rl import GRPOTrainer

trainer = GRPOTrainer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    num_prompts_per_step=32,
    num_generations_per_prompt=16,
)

trainer.fit(
    dataset="nvidia/OpenMathInstruct-2",
    reward_fn=my_reward,
    max_steps=1000,
)
```

### 3. Reward Functions

#### Old Way: Environment Class Required

```python
import ray
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

@ray.remote
class MyEnvironment(EnvironmentInterface):
    def __init__(self, cfg):
        self.cfg = cfg
        # Setup workers, etc.
    
    def step(self, message_log_batch, metadata):
        rewards = []
        for msg_log in message_log_batch:
            # Complex processing
            prompt = extract_prompt(msg_log)
            response = extract_response(msg_log)
            reward = self.compute_reward(prompt, response)
            rewards.append(reward)
        
        return EnvironmentReturn(
            observations=[{"role": "assistant", "content": ""}] * len(rewards),
            metadata=metadata,
            next_stop_strings=[None] * len(rewards),
            rewards=torch.tensor(rewards),
            terminateds=torch.ones(len(rewards), dtype=torch.bool),
            answers=None,
        )
    
    def global_post_process_and_metrics(self, batch):
        return batch, {"accuracy": batch["rewards"].mean().item()}
    
    def compute_reward(self, prompt, response):
        # Your reward logic
        return 1.0 if "correct" in response else 0.0
```

#### New Way: Simple Function

```python
# Option 1: Lambda (simplest)
reward_fn = lambda p, r: 1.0 if "correct" in r else 0.0

# Option 2: Function (recommended for complex logic)
def my_reward(prompt: str, response: str) -> float:
    if "error" in response.lower():
        return 0.0
    if "correct" in response.lower():
        return 1.0
    return 0.5

# Option 3: Environment subclass (for state/metrics)
from nemo_rl.environments import Environment

class MyEnvironment(Environment):
    def score(self, prompt: str, response: str) -> float:
        return 1.0 if "correct" in response else 0.0
```

### 4. Configuration Migration

#### Old Way: YAML Config Files

```yaml
# configs/grpo_math_1B.yaml
grpo:
  num_prompts_per_step: 32
  num_generations_per_prompt: 16
  max_num_steps: 1000
  normalize_rewards: true

policy:
  model_name: "Qwen/Qwen2.5-1.5B"
  train_global_batch_size: 512
  max_total_sequence_length: 512
  
  optimizer:
    name: "torch.optim.AdamW"
    kwargs:
      lr: 5.0e-6
      weight_decay: 0.01

checkpointing:
  enabled: true
  checkpoint_dir: "results/grpo"
```

#### New Way: Programmatic Config

```python
import nemo_rl

result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    dataset="nvidia/OpenMathInstruct-2",
    reward_fn=my_reward,
    algorithm="grpo",
    max_steps=1000,
    batch_size=32,
    num_generations_per_prompt=16,
    learning_rate=5e-6,
    output_dir="results/grpo",
)
```

### 5. SFT Training

#### Old Way

```python
from nemo_rl.algorithms.sft import SFT
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/sft.yaml")
cfg.policy.model_name = "Qwen/Qwen2.5-1.5B"

sft = SFT.from_config(cfg)
sft.fit(dataset_path="data/train.jsonl")
```

#### New Way

```python
import nemo_rl

result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    dataset="nvidia/HelpSteer2",
    algorithm="sft",  # No reward_fn needed for SFT
    max_steps=1000,
)
```

### 6. DPO Training

#### Old Way

```python
from nemo_rl.algorithms.dpo import DPO
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/dpo.yaml")
cfg.policy.model_name = "Qwen/Qwen2.5-1.5B"
cfg.dpo.beta = 0.1

dpo = DPO.from_config(cfg)
dpo.fit(dataset_path="data/preferences.jsonl")
```

#### New Way

```python
import nemo_rl

result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    dataset="nvidia/HelpSteer2",
    algorithm="dpo",
    beta=0.1,  # DPO-specific parameter
    max_steps=1000,
)
```

## Automated Config Migration

We provide a tool to automatically convert old YAML configs to new API calls.

### Using the Migration Tool

```bash
# Convert a single config file
python tools/migrate_config.py --input configs/grpo_math_1B.yaml --output train_script.py

# Preview without saving
python tools/migrate_config.py --input configs/grpo_math_1B.yaml --preview
```

### Example Output

Input (`configs/grpo_math_1B.yaml`):
```yaml
grpo:
  num_prompts_per_step: 32
  max_num_steps: 1000
policy:
  model_name: "Qwen/Qwen2.5-1.5B"
  optimizer:
    kwargs:
      lr: 5.0e-6
```

Output (`train_script.py`):
```python
import nemo_rl

result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    algorithm="grpo",
    max_steps=1000,
    batch_size=32,
    learning_rate=5e-6,
    # TODO: Add your reward function
    reward_fn=lambda p, r: 1.0,  # Replace with actual reward logic
)
```

## Deprecation Timeline

| Version | Date | Changes |
|---------|------|---------|
| 0.9 | Current | Old API fully supported, new API available |
| 1.0 | +3 months | Old API deprecated with warnings |
| 1.1 | +6 months | Old API removed from main codebase |
| 2.0 | +12 months | Old API no longer supported |

### Deprecation Warnings

When using old APIs, you'll see warnings like:

```
DeprecationWarning: GRPO.from_config() is deprecated. 
Use nemo_rl.train() or GRPOTrainer.from_pretrained() instead.
See https://docs.nemo-rl.nvidia.com/migration for migration guide.
```

## Backward Compatibility

The old API continues to work for now:

```python
# This still works (with deprecation warning)
from nemo_rl.algorithms.grpo import GRPO
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/grpo_math_1B.yaml")
grpo = GRPO.from_config(cfg)
grpo.fit()
```

To suppress warnings during migration:

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

## Common Migration Patterns

### Pattern 1: Config-based to Function-based

```python
# Before
cfg.grpo.normalize_rewards = True
cfg.grpo.use_leave_one_out_baseline = True

# After
nemo_rl.train(
    ...,
    normalize_rewards=True,
    use_leave_one_out_baseline=True,
)
```

### Pattern 2: Environment to Callable

```python
# Before: Ray actor environment
@ray.remote
class MyEnv(EnvironmentInterface):
    def step(self, logs, meta):
        ...

env = MyEnv.remote(cfg)
grpo.fit(environment=env)

# After: Simple function
def my_reward(prompt, response):
    ...

nemo_rl.train(reward_fn=my_reward, ...)
```

### Pattern 3: Complex Config to from_pretrained

```python
# Before: Manual config building
cfg = OmegaConf.load("base.yaml")
cfg.policy.model_name = "Qwen/Qwen2.5-1.5B"
cfg.policy.dtensor_cfg.tensor_parallel_size = 2
cfg.policy.optimizer.kwargs.lr = 1e-6
grpo = GRPO.from_config(cfg)

# After: from_pretrained with kwargs
trainer = GRPOTrainer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    tensor_parallel_size=2,
    learning_rate=1e-6,
)
```

## Getting Help

If you encounter issues during migration:

1. Check the [API Documentation](api/) for new function signatures
2. See [Quickstart](quickstart.md) for basic examples
3. Open an issue on [GitHub](https://github.com/NVIDIA-NeMo/RL/issues)

## FAQ

### Q: Can I still use YAML config files?

Yes, the old config-based approach still works. However, we recommend migrating to the new API for a simpler experience.

### Q: What if I have custom config options?

Pass them as `**kwargs` to `nemo_rl.train()`:

```python
nemo_rl.train(
    model="...",
    dataset="...",
    reward_fn=my_fn,
    grpo_custom_option=value,  # Prefixed with algorithm name
)
```

### Q: How do I access advanced features?

For advanced features, use the trainer classes directly:

```python
from nemo_rl import GRPOTrainer

trainer = GRPOTrainer.from_pretrained("model", **advanced_kwargs)
trainer.fit(...)
```

### Q: Is the new API as flexible as the old one?

Yes. The new API is a layer on top of the old one. For full control, use trainer classes or pass additional kwargs.
