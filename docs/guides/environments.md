# Environments for GRPO Training

GRPO supports several examples of environments for different tasks. Each environment provides a standardized interface for reward computation and evaluation.

## Math Environment

The Math Environment is designed for mathematical reasoning tasks. It evaluates responses to math problems using `math-verify` and provides rewards based on correctness.

### Key Features
- Evaluates mathematical reasoning
- Supports multiple mathematical domains
- Provides detailed feedback on solution correctness

### Usage
```python
from nemo_rl.environments.math_environment import MathEnvironment

env_config = {
    "num_workers": 2,
}

math_env = MathEnvironment.remote(env_config)
```

## Code Environment

The Code Environment is designed for code generation and execution tasks. It provides a sandboxed environment for executing Python code and evaluating the results.

### Usage
```python
from nemo_rl.environments.code_environment import CodeEnvironment

env_config = {
    "num_workers": 2,
    "terminate_on_evaluation": True,  # Terminate after code execution
}

code_env = CodeEnvironment.remote(env_config)
```

### Configuration
- `num_workers`: Number of parallel workers for code execution
- `terminate_on_evaluation`: Whether to terminate after code execution (True for single-turn, False for multi-turn)

We’re tracking an end-to-end example of this environment in [#858](https://github.com/NVIDIA-NeMo/RL/issues/858). Add a 👍 to show your interest.

## Reward Model Environment

The Reward Model Environment uses pre-trained reward models to score conversation quality. 

### Usage
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

### Resource Allocation in GRPO Training

In GRPO training, resources are allocated across three main components:

- **Policy Actor**: The trained model
- **Generation Actor**: Used for generating responses during rollouts (can be colocated with policy or on separate nodes/GPUs).
- **Reward Model Environment Actor**: Evaluates generated responses and computes rewards

The resource allocation logic works as follows:

#### Single-Node Setup (`num_nodes: 1`)
- All components share the same node
- GPUs are divided between policy training, generation, and reward model
- Example: 
    1. Policy and generation colocated: 8 GPUs total = 4 for colocated policy and generation + 4 for reward model
    2. Policy and generation non-colocated: 8 GPUs total = 2 for policy + 2 for generation + 4 for reward model

#### Multi-Node Setup (`num_nodes > 1`)
- Policy training, generation, and reward model environment can be distributed across different nodes
- Reward model gets dedicated resources as specified in `env.reward_model.resources`
- Generation gets dedicated resources as specified in `policy.generation.colocated.resources`
- Remaining nodes are allocated to policy training

In the future, the resource control part will be refactored to enable fine-grained resource configuration for each actor. For detailed resource management and optimization strategies, see [#1100](https://github.com/NVIDIA-NeMo/RL/issues/1100).

### Complete GRPO Training with Reward Model Environments

See [examples/run_grpo_rm.py](../../examples/run_grpo_rm.py) for a complete example of using the reward model environment with GRPO training.

### Configuration Examples

See [examples/configs/grpo_rm_1B.yaml](../../examples/configs/grpo_rm_1B.yaml) for a complete configuration example.