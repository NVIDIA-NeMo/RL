# Environments

GRPO supports various types of environments for different tasks. Each environment provides a standardized interface for reward computation and evaluation.

## Math Environment

The Math Environment is designed for mathematical reasoning tasks. It evaluates responses to math problems and provides rewards based on correctness.

**Key Features:**
- Evaluates mathematical reasoning
- Supports multiple mathematical domains
- Provides detailed feedback on solution correctness

**Usage:**
```python
from nemo_rl.environments.math_environment import MathEnvironment

env_config = {
    "num_workers": 2,
}

math_env = MathEnvironment.remote(env_config)
```

## Code Environment

The Code Environment is designed for code generation and execution tasks. It provides a sandboxed environment for executing Python code and evaluating the results.

**Usage:**
```python
from nemo_rl.environments.code_environment import CodeEnvironment

env_config = {
    "num_workers": 2,
    "terminate_on_evaluation": True,  # Terminate after code execution
}

code_env = CodeEnvironment.remote(env_config)
```

**Configuration:**
- `num_workers`: Number of parallel workers for code execution
- `terminate_on_evaluation`: Whether to terminate after code execution (True for single-turn, False for multi-turn)

## Reward Model Environment

The Reward Model Environment uses pre-trained reward models to score conversation quality. 

**Usage:**
```python
from nemo_rl.environments.reward_model_environment import RewardModelEnvironment

env_config = {
    "enabled": True,
    "model_name": "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
    "tokenizer": {"name": "Skywork/Skywork-Reward-V2-Qwen3-0.6B"},
    "precision": "bfloat16",
    "batch_size": 32,
    "resources": {"gpus_per_node": 1, "num_nodes": 1},
    "reward_model_cfg": {
        "enabled": True,
        "reward_model_type": "bradley_terry",
    },
}

reward_env = RewardModelEnvironment.remote(env_config)
```
### Complete GRPO Training with Reward Model Environments

See [examples/run_grpo_rm.py](../../examples/run_grpo_rm.py) for a complete example of using the reward model environment with GRPO training.

### Configuration Examples

See [examples/configs/grpo_rm_1B.yaml](../../examples/configs/grpo_rm_1B.yaml) for a complete configuration example.