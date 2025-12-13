# Environments for GRPO Training

GRPO includes multiple environments, each offering a standard interface for reward computation and evaluation.

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
- `terminate_on_evaluation`: Whether to terminate after code execution (True for single-turn, False for multi-turn).

We are tracking an end-to-end example of this environment in [#858](https://github.com/NVIDIA-NeMo/RL/issues/858). Add a 👍 to show your interest.

## Code Jaccard Environment

The Code Jaccard Environment evaluates code (or text) responses by measuring Jaccard-based similarity against ground-truth answers. This is a lightweight, text-similarity reward useful when an execution sandbox is unnecessary or unavailable.

### How It Works
- Extracts the assistant’s response text from each conversation.
- Computes a Jaccard similarity score between the response and ground truth:
  - Tokenizes both texts by whitespace, computes intersection/union, then applies a length ratio penalty.
  - Scores are in [0, 1]. Observations label responses as “aligned/misaligned” using a 0.5 threshold.
- Returns:
  - observations: Environment feedback strings.
  - rewards: Tensor of similarity scores.
  - terminateds: All ones (single-step episodes).
  - answers: The response text when requested (optional).

### Usage
```python
from nemo_rl.environments.code_jaccard_environment import CodeJaccardEnvironment

env_config = {
    "num_workers": 2,
    # Optional default stop strings (unused in scoring but available for consistency)
    "stop_strings": None,
}

code_jaccard_env = CodeJaccardEnvironment.remote(env_config)
```

### Configuration
- `num_workers` (int): Number of parallel verification workers.
- `stop_strings` (list[str] | None): Optional default stop strings (propagated downstream; not required for scoring).

### Sample GRPO Config
```yaml
env:
  code_jaccard:
    num_workers: 2
    stop_strings: null
data:
  env_name: code_jaccard
```

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

- **Policy Actor**: The trained model.
- **Generation Actor**: Used for generating responses during rollouts (can be colocated with policy or on separate nodes/GPUs).
- **Reward Model Environment Actor**: Evaluates generated responses and computes rewards.

The resource allocation logic works as follows:

#### Single-Node Setup (`num_nodes: 1`)
- All components share the same node
- GPUs are divided between policy training, generation, and reward model
- Example: 
    1. Policy and generation colocated: 8 GPUs total = 4 for colocated policy and generation + 4 for reward model
    2. Policy and generation non-colocated: 8 GPUs total = 2 for policy + 2 for generation + 4 for reward model

#### Multi-Node Setup (`num_nodes > 1`)
- Policy training, generation, and reward model environment can be distributed across different nodes.
- Reward model gets dedicated resources as specified in `env.reward_model.resources`.
- Generation gets dedicated resources as specified in `policy.generation.colocated.resources`.
- Remaining nodes are allocated to policy training.

In the future, the resource control part will be refactored to enable fine-grained resource configuration for each actor. For detailed resource management and optimization strategies, see [#1100](https://github.com/NVIDIA-NeMo/RL/issues/1100).

### Complete GRPO Training with Reward Model Environments

See [examples/run_grpo_rm.py](../../examples/run_grpo_rm.py) for a complete example of using the reward model environment with GRPO training.

### Configuration Examples

See [examples/configs/grpo_rm_1B.yaml](../../examples/configs/grpo_rm_1B.yaml) for a complete configuration example.


## Registering Custom Environments

NeMo RL provides a flexible environment registration mechanism that allows you to add custom environments without modifying the source code.

### Using the `register_env` Interface

You can use the `register_env` function to dynamically register new environments without modifying NeMo RL's internal code.

**Function Signature**

```python
from nemo_rl.environments.utils import register_env

register_env(env_name: str, actor_class_fqn: str) -> None
```

**Parameters:**

- `env_name`: Unique identifier name for the environment (string)
- `actor_class_fqn`: Fully Qualified Name of the environment Actor class, in the format `'module.path.ClassName'`

### Example: Registering a Custom Environment

Suppose you've created a custom reinforcement learning environment for code generation tasks:

**1. Create Your Custom Environment Actor Class**

```python
# File: my_custom_envs/code_gen_env.py
import ray
from nemo_rl.environments.interfaces import EnvironmentInterface

@ray.remote
class CodeGenEnvironmentActor(EnvironmentInterface):
    """Custom code generation environment."""
    
    def __init__(self, config):
        self.config = config
        # Initialize your environment
        
    async def reset(self):
        # Reset environment logic
        return initial_state
        
    async def step(self, action):
        # Execute action, return reward, etc.
        return observation, reward, done, info
        
    # Implement other required interface methods...
```

**2. Register the Environment in Your Training Script**

```python
# File: train.py
from nemo_rl.environments.utils import register_env

# Register your custom environment
register_env(
    env_name="code_gen",
    actor_class_fqn="my_custom_envs.code_gen_env.CodeGenEnvironmentActor"
)

# Now you can use "code_gen" in your config
# Training code...
```

**3. Use the Registered Environment in Your Config**

```yaml
# config.yaml
env:
  code_gen:
    num_workers: 2
    max_code_length: 512
    test_cases_per_problem: 5

data:
  env_name: code_gen  # Use your registered environment name
```
