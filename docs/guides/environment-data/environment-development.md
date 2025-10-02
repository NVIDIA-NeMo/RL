# Develop Custom Environments

This guide covers how to develop custom environments for reinforcement learning training in NeMo RL.

## Overview

NeMo RL environments support language model training and evaluation. They process conversations between users and AI assistants, assess responses, and provide rewards based on task-specific criteria. Custom environments allow you to define specific problem domains, evaluation metrics, and reward structures for your RL training tasks.

## Environment Interface

All environments must use the `EnvironmentInterface`:

```python
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
import torch

class CustomEnvironment(EnvironmentInterface):
    def __init__(self, config):
        # Initialize environment state and configuration
        self.config = config
    
    def step(
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[Optional[dict]],
    ) -> EnvironmentReturn:
        """Process a batch of conversations and return rewards.
        
        Args:
            message_log_batch: Batch of OpenAI-API-like message logs
            metadata: Batch of environment-specific metadata
            
        Returns:
            EnvironmentReturn: Named tuple with observations, metadata, 
                             next_stop_strings, rewards, and terminateds
        """
        # Process each conversation in the batch
        observations = []
        rewards = []
        terminateds = []
        
        for messages, meta in zip(message_log_batch, metadata):
            # Extract assistant responses
            assistant_responses = [
                msg["content"] for msg in messages 
                if msg["role"] == "assistant"
            ]
            
            # Evaluate the response
            reward = self._evaluate_response(assistant_responses, meta)
            done = self._is_episode_done(messages, meta)
            
            # Create observation
            observation = {
                "role": "environment",
                "content": f"Evaluation: {reward}"
            }
            
            observations.append(observation)
            rewards.append(reward)
            terminateds.append(done)
        
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
        )
    
    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        """Post-processing after all rollouts are complete."""
        return batch, {}
```

## Environment Structure

### Message Processing

NeMo RL environments process conversations in OpenAI-API format:

```python
def _extract_assistant_responses(self, messages: list[dict[str, str]]) -> list[str]:
    """Extract all assistant responses from a conversation."""
    return [
        msg["content"] for msg in messages 
        if msg["role"] == "assistant"
    ]

def _extract_user_prompts(self, messages: list[dict[str, str]]) -> list[str]:
    """Extract all user prompts from a conversation."""
    return [
        msg["content"] for msg in messages 
        if msg["role"] == "user"
    ]
```

### Metadata Management

Environments use metadata to track state across conversations:

```python
class EnvironmentMetadata(TypedDict):
    """Environment-specific metadata structure."""
    task_id: str
    ground_truth: str
    current_step: int
    max_steps: int
    evaluation_criteria: dict
```

### Reward Evaluation

Design reward functions that assess language model responses:

```python
def _evaluate_response(self, responses: list[str], metadata: dict) -> float:
    """Assess assistant responses and return reward.
    
    Args:
        responses: List of assistant response strings
        metadata: Environment metadata with ground truth
        
    Returns:
        float: Reward value (-1.0 to 1.0)
    """
    # Combine all responses
    full_response = " ".join(responses)
    
    # Compare with ground truth
    ground_truth = metadata.get("ground_truth", "")
    
    # Implement your evaluation logic
    if self._is_correct(full_response, ground_truth):
        return 1.0
    elif self._is_partially_correct(full_response, ground_truth):
        return 0.5
    else:
        return -0.1
```

## Best Practices

### 1. Environment Design

- **Clear Evaluation Criteria**: Define specific, measurable evaluation metrics
- **Appropriate Complexity**: Start with simple tasks, add complexity gradually
- **Consistent Interface**: Follow NeMo RL environment patterns
- **Efficient Implementation**: Optimize for batch processing speed

### 2. Message Processing

- **Robust Parsing**: Handle various message formats and edge cases
- **Context Preservation**: Maintain conversation context across turns
- **Error Handling**: Handle malformed messages appropriately
- **Batch Efficiency**: Process many conversations efficiently

### 3. Reward Design

- **Meaningful Evaluation**: Design rewards that reflect task objectives
- **Appropriate Scale**: Use reward values that work well with your training algorithm
- **Consistency**: Maintain consistent evaluation criteria
- **Interpretability**: Make rewards interpretable for debugging

### 4. Metadata Management

- **State Tracking**: Use metadata to track conversation state
- **Ground Truth**: Store reference answers and evaluation criteria
- **Episode Control**: Track episode termination conditions
- **Performance Metrics**: Store metrics for post-processing

## Integration with NeMo RL

### Configuration

Add your environment to the training configuration:

```yaml
environment:
  type: "custom"
  class: "path.to.CustomEnvironment"
  config:
    evaluation_criteria: "accuracy"
    max_steps: 10
    reward_scale: 1.0
```

### Ray Remote Setup

Environments should be Ray actors for distributed processing:

```python
import ray
from nemo_rl.environments.interfaces import EnvironmentInterface

@ray.remote
class CustomEnvironment(EnvironmentInterface):
    def __init__(self, config):
        self.config = config
        # Initialize environment
    
    def step(self, message_log_batch, metadata):
        # Implementation
        pass
```

### Testing

Test your environment thoroughly:

```python
def test_environment():
    # Create test data
    message_log = [
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."}
        ]
    ]
    metadata = [{"ground_truth": "4", "task_id": "test"}]
    
    # Test environment
    env = CustomEnvironment.remote(config)
    result = ray.get(env.step.remote(message_log, metadata))
    
    assert len(result.observations) == 1
    assert result.rewards.shape == (1,)
    assert result.terminateds.shape == (1,)
```

## Debugging

### Common Issues

1. **Message Parsing Errors**: Handle malformed message logs
2. **Reward Scaling**: Check reward magnitude and distribution
3. **Metadata Consistency**: Verify metadata structure and content
4. **Performance**: Monitor batch processing time

### Debugging Tools

```python
import logging
logging.basicConfig(level=logging.DEBUG)

def step(self, message_log_batch, metadata):
    """Add debug information to environment step."""
    print(f"Processing {len(message_log_batch)} conversations")
    
    for i, (messages, meta) in enumerate(zip(message_log_batch, metadata)):
        print(f"Conversation {i}:")
        print(f"  Messages: {len(messages)}")
        print(f"  Metadata: {meta}")
        
        # Extract responses
        responses = self._extract_assistant_responses(messages)
        print(f"  Responses: {responses}")
        
        # Evaluate
        reward = self._evaluate_response(responses, meta)
        print(f"  Reward: {reward}")
    
    # Execute step
    result = self._step(message_log_batch, metadata)
    print(f"Result: {result}")
    return result
```

## Performance Optimization

### Batch Processing

Process multiple conversations efficiently:

```python
def step(self, message_log_batch, metadata):
    """Process batch of conversations efficiently."""
    # Pre-allocate results
    observations = []
    rewards = []
    terminateds = []
    
    # Process in parallel where possible
    for messages, meta in zip(message_log_batch, metadata):
        # Extract responses efficiently
        responses = [
            msg["content"] for msg in messages 
            if msg["role"] == "assistant"
        ]
        
        # Evaluate
        reward = self._evaluate_response(responses, meta)
        done = self._is_episode_done(messages, meta)
        
        observations.append({
            "role": "environment",
            "content": f"Evaluation: {reward}"
        })
        rewards.append(reward)
        terminateds.append(done)
    
    return EnvironmentReturn(
        observations=observations,
        metadata=metadata,
        next_stop_strings=[None] * len(message_log_batch),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        terminateds=torch.tensor(terminateds, dtype=torch.bool),
    )
```

### Caching

Cache expensive evaluation operations:

```python
from functools import lru_cache

class CachedEnvironment(EnvironmentInterface):
    def __init__(self, config):
        self.config = config
    
    @lru_cache(maxsize=1000)
    def _evaluate_response(self, response_hash: str, ground_truth: str) -> float:
        """Cache evaluation results for identical responses."""
        # Implement evaluation logic
        return evaluation_score
```

## Examples

### Math Problem Environment

```python
import ray
import torch
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

@ray.remote
class MathProblemEnvironment(EnvironmentInterface):
    def __init__(self, config):
        self.config = config
    
    def step(self, message_log_batch, metadata):
        observations = []
        rewards = []
        terminateds = []
        
        for messages, meta in zip(message_log_batch, metadata):
            # Extract assistant responses
            responses = [
                msg["content"] for msg in messages 
                if msg["role"] == "assistant"
            ]
            
            # Evaluate mathematical correctness
            reward = self._evaluate_math_response(responses, meta)
            done = self._is_episode_done(messages, meta)
            
            observations.append({
                "role": "environment",
                "content": f"Math evaluation: {reward}"
            })
            rewards.append(reward)
            terminateds.append(done)
        
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
        )
    
    def _evaluate_math_response(self, responses, metadata):
        """Evaluate mathematical problem responses."""
        full_response = " ".join(responses)
        ground_truth = metadata.get("ground_truth", "")
        
        # Simple exact match evaluation
        if full_response.strip() == ground_truth.strip():
            return 1.0
        else:
            return -0.1
    
    def global_post_process_and_metrics(self, batch):
        return batch, {}
```

### Code Generation Environment

```python
@ray.remote
class CodeGenerationEnvironment(EnvironmentInterface):
    def __init__(self, config):
        self.config = config
    
    def step(self, message_log_batch, metadata):
        observations = []
        rewards = []
        terminateds = []
        
        for messages, meta in zip(message_log_batch, metadata):
            # Extract code responses
            code_responses = [
                msg["content"] for msg in messages 
                if msg["role"] == "assistant"
            ]
            
            # Evaluate code quality
            reward = self._evaluate_code_quality(code_responses, meta)
            done = self._is_episode_done(messages, meta)
            
            observations.append({
                "role": "environment",
                "content": f"Code evaluation: {reward}"
            })
            rewards.append(reward)
            terminateds.append(done)
        
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
        )
    
    def _evaluate_code_quality(self, code_responses, metadata):
        """Evaluate code generation quality."""
        # Implement code evaluation logic
        # This could include syntax checking, test execution, etc.
        return 0.5  # Placeholder
```

## Next Steps

- Review the [Environment Interfaces API](../../apidocs/nemo_rl/nemo_rl.environments) for detailed API documentation
- Explore [Math Environment](../../apidocs/nemo_rl/nemo_rl.environments) for a complete implementation example
- Check [Debugging](../development-workflows/debugging) for troubleshooting guidance
- See [Advanced Training](../../advanced/algorithm-development/index) for optimization techniques
- Review [Configuration Reference](../../references/configuration-reference) for environment setup
