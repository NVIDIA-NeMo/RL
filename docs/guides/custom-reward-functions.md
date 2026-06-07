# Custom Reward Functions

This guide provides an end-to-end example for writing a custom reward function and registering it via a custom environment in NeMo RL.

## Overview

While NeMo RL provides built-in environments (Math, Code, etc.) with standard reward functions, you can define your own custom reward logic for domain-specific tasks. The pattern involves three steps:

1. **Implement** a reward function
2. **Register** it via a custom environment class
3. **Configure** it in your training YAML

## Step 1: Implement a Reward Function

A reward function takes a batch of responses and returns a tensor of reward scores. Here is a complete example of a custom reward function that scores responses for conciseness:

```python
# my_rewards.py
import re
import torch
from typing import List, Dict, Any

def conciseness_reward(responses: List[str], **kwargs) -> torch.Tensor:
    """
    Reward function that scores responses based on conciseness.
    Shorter responses that still contain the key answer get higher rewards.

    Args:
        responses: List of generated response strings, one per sample.
        **kwargs: Additional context (e.g., prompts, reference answers).

    Returns:
        Tensor of shape (batch_size,) containing reward scores.
    """
    rewards = []
    for response in responses:
        # Count words
        word_count = len(response.split())
        # Penalize verbosity: reward = 1.0 for < 50 words, decreasing linearly
        if word_count < 50:
            reward = 1.0
        elif word_count < 200:
            reward = 1.0 - (word_count - 50) / 150 * 0.5  # 1.0 → 0.5
        else:
            reward = max(0.0, 1.0 - (word_count - 200) / 800 * 0.5)  # 0.5 → 0.0
        rewards.append(reward)

    return torch.tensor(rewards, dtype=torch.float32)


def keyword_coverage_reward(
    responses: List[str],
    required_keywords: List[str] = None,
    **kwargs
) -> torch.Tensor:
    """
    Reward function that scores responses based on keyword coverage.
    Useful for tasks where specific terms must appear in the answer.

    Args:
        responses: List of generated response strings.
        required_keywords: List of keywords that should appear.

    Returns:
        Tensor of shape (batch_size,) with scores in [0.0, 1.0].
    """
    if required_keywords is None:
        required_keywords = ["conclusion", "result", "analysis"]

    rewards = []
    for response in responses:
        response_lower = response.lower()
        if not required_keywords:
            # No keywords specified — full score
            rewards.append(1.0)
            continue

        matched = sum(1 for kw in required_keywords if kw.lower() in response_lower)
        score = matched / len(required_keywords)
        rewards.append(score)

    return torch.tensor(rewards, dtype=torch.float32)
```

### Reward Function Contract

Every reward function must:

| Requirement | Description |
|-------------|-------------|
| **Signature** | `fn(responses: List[str], **kwargs) -> torch.Tensor` |
| **Output shape** | `(batch_size,)` — one scalar per response |
| **Output range** | Any float; higher = better. Normalization is handled by the trainer. |
| **`kwargs`** | Receives `prompts`, `reference_answers`, and any extra fields configured in YAML. |

## Step 2: Register the Reward via a Custom Environment

Create a custom environment class that wraps your reward function. The environment must subclass `BaseEnvironment` and implement `step()`:

```python
# my_custom_env.py
from nemo_rl.environments.base_environment import BaseEnvironment
from nemo_rl.environments.environment_registry import register_environment
from nemo_rl.environments.environment_return import EnvironmentReturn
from my_rewards import conciseness_reward, keyword_coverage_reward


@register_environment("ConcisenessEnv")
class ConcisenessEnvironment(BaseEnvironment):
    """
    Custom environment that rewards concise, keyword-rich responses.
    Demonstrates how to:
      - Use multiple reward components (GDPO-style).
      - Inject task-specific parameters via the env_config.
    """

    def __init__(self, env_config: dict):
        super().__init__(env_config)
        # Accept optional keyword list from the training config
        self.required_keywords = env_config.get("required_keywords", [])
        self.num_rewards = 2  # Two reward components

    def step(self, responses, prompts=None, reference_answers=None, **kwargs):
        """
        Evaluate responses and return rewards.

        Args:
            responses: List[str] — generated responses.
            prompts: List[str], optional — the prompts.
            reference_answers: List[str], optional — ground-truth answers.

        Returns:
            EnvironmentReturn with rewards tensor of shape (batch_size, num_rewards).
        """
        # Compute per-component rewards
        reward1 = conciseness_reward(responses)
        reward2 = keyword_coverage_reward(
            responses,
            required_keywords=self.required_keywords,
        )

        # Stack into multi-reward tensor: shape (batch_size, num_rewards)
        rewards = torch.stack([reward1, reward2], dim=1)

        # Build metadata (optional)
        metadata = [
            {
                "conciseness_score": float(r1),
                "keyword_score": float(r2),
            }
            for r1, r2 in zip(reward1, reward2)
        ]

        return EnvironmentReturn(
            observations=responses,
            rewards=rewards,          # Multi-reward for GDPO
            terminateds=[False] * len(responses),
            metadata=metadata,
        )
```

### Environment Registration

The `@register_environment("ConcisenessEnv")` decorator makes your environment discoverable by name. Alternatively, register manually:

```python
from nemo_rl.environments.environment_registry import register_environment

register_environment("ConcisenessEnv", ConcisenessEnvironment)
```

## Step 3: Configure the Training Run

Wire your custom environment into the training configuration:

```yaml
# configs/grpo_custom_reward.yaml
trainer:
  env: "ConcisenessEnv"                     # ← must match registration name
  env_config:
    required_keywords:                       # ← passed to env.__init__
      - "result"
      - "analysis"
      - "finding"

  # For GDPO with multiple reward components:
  grpo:
    adv_estimator:
      name: "gdpo"                          # ← use GDPO for multi-reward
    reward_weights: [0.6, 0.4]              # ← conciseness: 60%, keywords: 40%

  # Standard GRPO (single reward from env):
  # grpo:
  #   adv_estimator:
  #     name: "grpo"                       # ← use GRPO for single-reward
```

### Launch

```bash
# Using the custom environment file
uv run examples/run_grpo.py \
  --config configs/grpo_custom_reward.yaml

# With CLI overrides
uv run examples/run_grpo.py \
  --config configs/grpo_custom_reward.yaml \
  trainer.env_config.required_keywords='["result","analysis","finding","conclusion"]'
```

## Complete Example

Here is a self-contained single-file example you can run as a starting point:

```python
"""
single_file_custom_env.py — A complete, runnable example of a custom
reward function and environment in NeMo RL.
"""
import torch
from typing import List

from nemo_rl.environments.base_environment import BaseEnvironment
from nemo_rl.environments.environment_registry import register_environment
from nemo_rl.environments.environment_return import EnvironmentReturn


# --- Reward Function ---
def format_adherence_reward(responses: List[str]) -> torch.Tensor:
    """Reward responses that follow a structured format (JSON-like)."""
    rewards = []
    for resp in responses:
        has_brace = "{" in resp and "}" in resp
        has_key = '"result"' in resp or "'result'" in resp
        score = 1.0 if (has_brace and has_key) else 0.2
        rewards.append(score)
    return torch.tensor(rewards, dtype=torch.float32)


# --- Custom Environment ---
@register_environment("FormatAdherenceEnv")
class FormatAdherenceEnvironment(BaseEnvironment):
    def __init__(self, env_config: dict):
        super().__init__(env_config)
        self.num_rewards = 1

    def step(self, responses, **kwargs):
        rewards = format_adherence_reward(responses)
        return EnvironmentReturn(
            observations=responses,
            rewards=rewards.unsqueeze(1),  # (batch_size, 1)
            terminateds=[False] * len(responses),
        )
```

## Next Steps

- Experiment with different reward formulations and observe how they affect model behavior.
- Combine multiple reward components for complex tasks (e.g., correctness + style + safety).
- See the [Environments Guide](./environments.md) for built-in environment reference.
- See the [GRPO Guide](./grpo.md) for training configuration details.
