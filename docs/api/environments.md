# Environments and Reward Functions

NeMo RL provides flexible ways to define reward functions, from simple callables to full environment classes.

## Quick Start

Choose the simplest approach for your needs:

| Approach | Complexity | Use When |
|----------|------------|----------|
| Simple callable | Lowest | Basic reward logic |
| `Environment` subclass | Medium | Need state or custom metrics |
| `EnvironmentInterface` | Highest | Multi-turn environments |

## Simple Callable Rewards

The easiest way to define rewards is with a simple function:

```python
import nemo_rl

def reward_fn(prompt: str, response: str) -> float:
    """Reward longer responses."""
    return min(len(response) / 200, 1.0)

result = nemo_rl.train(
    model="Qwen/Qwen2.5-1.5B",
    dataset="nvidia/HelpSteer2",
    reward_fn=reward_fn,  # Just pass the function!
)
```

### Supported Signatures

```python
# Single value
def reward(prompt: str, response: str) -> float:
    return 1.0

# Dict of values (multiple metrics)
def multi_reward(prompt: str, response: str) -> dict[str, float]:
    return {
        "correctness": 1.0,
        "fluency": 0.8,
    }

# Async function
async def async_reward(prompt: str, response: str) -> float:
    result = await api_call(prompt, response)
    return result.score

# Batched function (for efficiency)
def batch_reward(prompts: list[str], responses: list[str]) -> list[float]:
    return [len(r) / 100 for r in responses]
```

## Environment Base Class

For more control, extend the `Environment` class:

```python
from nemo_rl.environments import Environment

class CorrectnessReward(Environment):
    """Reward based on answer correctness."""
    
    def __init__(self, answer_key: dict[str, str]):
        super().__init__(name="correctness")
        self.answer_key = answer_key
    
    def score(self, prompt: str, response: str) -> float:
        """Score a single prompt-response pair."""
        # Extract question ID from prompt
        question_id = prompt.split(":")[0].strip()
        correct_answer = self.answer_key.get(question_id, "")
        
        if correct_answer.lower() in response.lower():
            return 1.0
        return 0.0

# Usage
env = CorrectnessReward({"Q1": "4", "Q2": "Paris"})
trainer.fit(dataset="...", environment=env)
```

### Environment Class API

```python
class Environment:
    """Base class for reward computation environments."""
    
    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        name: Optional[str] = None,
    ):
        """Initialize environment."""
    
    # Required: Override at least one
    def score(self, prompt: str, response: str) -> float:
        """Score a single prompt-response pair."""
        raise NotImplementedError
    
    def score_batch(
        self, prompts: list[str], responses: list[str]
    ) -> list[float]:
        """Score a batch (default: calls score() in parallel)."""
    
    # Optional overrides
    async def score_async(self, prompt: str, response: str) -> float:
        """Async scoring (default: calls sync score())."""
    
    def validate_response(self, prompt: str, response: str) -> bool:
        """Filter invalid responses (default: True)."""
    
    def get_metrics(self, batch) -> dict[str, float]:
        """Custom metrics for logging."""
    
    def setup(self) -> None:
        """One-time initialization."""
    
    def teardown(self) -> None:
        """Cleanup resources."""
```

### EnvironmentConfig

```python
from nemo_rl.environments import EnvironmentConfig

config = EnvironmentConfig(
    name="my_environment",      # For logging
    max_workers=8,              # Thread pool size
    timeout=60.0,               # Async timeout (seconds)
    default_reward=0.0,         # Reward on error
    enable_metrics=True,        # Compute reward stats
)

env = MyEnvironment(config=config)
```

## FunctionalRewardWrapper

Wrap a callable as an environment:

```python
from nemo_rl.environments import FunctionalRewardWrapper

def my_reward(prompt: str, response: str) -> float:
    return len(response) / 100

# Wrap as environment
env = FunctionalRewardWrapper(my_reward, name="length_reward")

# Use directly
rewards = env(["prompt1", "prompt2"], ["response1", "response2"])
print(rewards)  # tensor([0.09, 0.10])
```

### FunctionalRewardConfig

```python
from nemo_rl.environments import FunctionalRewardWrapper, FunctionalRewardConfig

config = FunctionalRewardConfig(
    batch_size=32,           # Parallel batch size
    max_workers=8,           # Thread pool size
    timeout=60.0,            # Computation timeout
    default_reward=0.0,      # Fallback on error
)

env = FunctionalRewardWrapper(my_fn, config=config)
```

## Convenience Classes

### SimpleEnvironment

For environments without metadata:

```python
from nemo_rl.environments import SimpleEnvironment

class LengthReward(SimpleEnvironment):
    def score(self, prompt: str, response: str) -> float:
        return min(len(response) / 200, 1.0)
```

### StatefulEnvironment

For environments with state:

```python
from nemo_rl.environments import StatefulEnvironment

class GameEnvironment(StatefulEnvironment):
    def score(self, prompt: str, response: str) -> float:
        # Access state via metadata
        return self._evaluate(prompt, response)
```

## Decorator API

For quick reward function definition:

```python
from nemo_rl.environments import reward_function, batched_reward

@reward_function("correctness")
def check_correct(prompt: str, response: str) -> float:
    return 1.0 if "correct" in response else 0.0

@batched_reward("batch_scorer")
def score_batch(prompts: list[str], responses: list[str]) -> list[float]:
    return [len(r) / 100 for r in responses]
```

## Helper Functions

### validate_reward_callable

Check if a function can be used as a reward:

```python
from nemo_rl.environments import validate_reward_callable

def good_reward(prompt: str, response: str) -> float:
    return 1.0

def bad_reward(x):  # Wrong signature
    return 1.0

is_valid, error = validate_reward_callable(good_reward)
print(is_valid)  # True

is_valid, error = validate_reward_callable(bad_reward)
print(is_valid, error)  # False, "must accept at least 2 parameters..."
```

### wrap_reward_callable

Wrap any callable as FunctionalRewardWrapper:

```python
from nemo_rl.environments import wrap_reward_callable

# Works with functions
env = wrap_reward_callable(lambda p, r: len(r) / 100)

# Works with existing wrappers (pass-through)
existing = FunctionalRewardWrapper(my_fn)
same = wrap_reward_callable(existing)  # Returns same instance

# Returns None for None input
env = wrap_reward_callable(None)  # None
```

### create_reward_wrapper

Factory function with config options:

```python
from nemo_rl.environments import create_reward_wrapper

env = create_reward_wrapper(
    lambda p, r: len(r) / 100,
    name="length_reward",
    max_workers=4,
    timeout=30.0,
)
```

## EnvironmentInterface (Advanced)

For multi-turn environments or complex scenarios:

```python
from nemo_rl.environments import EnvironmentInterface, EnvironmentReturn

class MultiTurnEnv(EnvironmentInterface):
    """Environment for multi-turn conversations."""
    
    def step(
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[Any],
    ) -> EnvironmentReturn:
        """Process a batch of conversations."""
        rewards = []
        observations = []
        terminateds = []
        
        for msg_log, meta in zip(message_log_batch, metadata):
            # Evaluate conversation
            reward = self._evaluate(msg_log)
            rewards.append(reward)
            
            # Continue or terminate
            if self._should_continue(msg_log):
                observations.append({
                    "role": "user",
                    "content": "Please continue...",
                })
                terminateds.append(False)
            else:
                observations.append({"role": "assistant", "content": ""})
                terminateds.append(True)
        
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(rewards),
            rewards=torch.tensor(rewards),
            terminateds=torch.tensor(terminateds),
            answers=None,
        )
    
    def global_post_process_and_metrics(self, batch):
        """Compute metrics."""
        metrics = {"accuracy": batch["rewards"].mean().item()}
        return batch, metrics
```

## Best Practices

### 1. Start Simple

```python
# Good: Start with a simple function
def reward(prompt, response):
    return 1.0 if "answer" in response else 0.0

# Upgrade to class only if needed
class MyReward(Environment):
    def score(self, prompt, response):
        return 1.0 if "answer" in response else 0.0
```

### 2. Validate Responses

```python
class SafeReward(Environment):
    def validate_response(self, prompt, response):
        if not response.strip():
            return False  # Reject empty
        if len(response) > 10000:
            return False  # Reject very long
        return True
    
    def score(self, prompt, response):
        # Only called for valid responses
        return self._compute_score(prompt, response)
```

### 3. Handle Errors Gracefully

```python
class RobustReward(Environment):
    def score(self, prompt, response):
        try:
            return self._complex_computation(prompt, response)
        except Exception as e:
            logger.warning(f"Score failed: {e}")
            return 0.0  # Or self.config.default_reward
```

### 4. Use Batch Processing for Efficiency

```python
class EfficientReward(Environment):
    def __init__(self):
        super().__init__()
        self.model = load_reward_model()
    
    def score_batch(self, prompts, responses):
        # Batch inference is much faster
        inputs = self.tokenizer(prompts, responses, return_tensors="pt")
        with torch.no_grad():
            scores = self.model(**inputs).logits
        return scores.squeeze().tolist()
```

## Related APIs

- [train()](train.md) - Using reward functions with train()
- [Trainers](trainers.md) - Using environments with trainers
- [Config](config.md) - Configuration options
