---
description: "Learn to build custom training environments for NeMo RL, implementing the EnvironmentInterface and creating specialized reward functions"
categories: ["training-algorithms"]
tags: ["custom-environments", "environment-development", "reward-functions", "advanced", "implementation"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "advanced"
content_type: "tutorial"
modality: "universal"
---

# Build Custom Training Environments

This tutorial teaches you how to build custom training environments for NeMo RL, using the EnvironmentInterface and creating specialized reward functions for your specific use cases.

## What You'll Learn

- **Environment Architecture**: Understand NeMo RL's environment framework
- **Custom Reward Functions**: Design and create specialized reward functions
- **Environment State Management**: Handle complex environment state and metadata
- **Ray Remote Environments**: Scale environments with distributed processing
- **Advanced Environment Patterns**: Learn sophisticated environment design patterns

## Prerequisites

- **NeMo RL**: Installed and configured
- **Ray**: Understanding of Ray distributed computing
- **Python**: Advanced Python programming skills
- **RL Concepts**: Familiarity with reinforcement learning environments

## Tutorial Overview

### **Step 1: Understanding the Environment Interface**

Learn NeMo RL's environment architecture and interface requirements.

### **Step 2: Implementing Basic Environments**

Create simple environments with custom reward functions.

### **Step 3: Advanced Environment Features**

Build environments with complex state management and metadata.

### **Step 4: Distributed Environment Scaling**

Scale environments using Ray remote actors.

### **Step 5: Advanced Environment Patterns**

Learn advanced patterns for complex environment design.

## Step 1: Understanding the Environment Interface

### **NeMo RL Environment Architecture**

NeMo RL provides a flexible environment framework through the `EnvironmentInterface`:

```python
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from typing import List, Optional, Dict, Any
import torch

class EnvironmentInterface(abc.ABC):
    """Base interface for all environments in NeMo RL."""
    
    @abc.abstractmethod
    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[Optional[Dict[str, Any]]],
    ) -> EnvironmentReturn:
        """
        Process a batch of conversations and return rewards.
        
        Args:
            message_log_batch: Batch of conversation message logs
            metadata: Batch of environment metadata
            
        Returns:
            EnvironmentReturn with observations, rewards, and termination flags
        """
        ...
    
    @abc.abstractmethod
    def global_post_process_and_metrics(
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Post-process batch results and compute global metrics.
        
        Args:
            batch: Batch data for post-processing
            
        Returns:
            Tuple of (processed_batch, metrics_dict)
        """
        ...
```

### **EnvironmentReturn Structure**

The `EnvironmentReturn` contains all environment outputs:

```python
from typing import NamedTuple

class EnvironmentReturn(NamedTuple):
    """Return value from environment step."""
    
    observations: List[Dict[str, str]]  # Environment observations
    metadata: List[Optional[Dict[str, Any]]]  # Updated metadata
    next_stop_strings: List[Optional[str]]  # Stop conditions
    rewards: torch.Tensor  # Reward values
    terminateds: torch.Tensor  # Episode termination flags
```

### **Key Components**

1. **Message Processing**: Handle conversation message logs
2. **Reward Computation**: Calculate rewards based on responses
3. **State Management**: Track environment state and metadata
4. **Batch Processing**: Efficiently process many conversations
5. **Termination Logic**: Determine episode termination conditions

## Step 2: Implementing Basic Environments

### **Simple Reward Environment**

Create a basic environment with custom reward functions:

```python
import torch
import torch.nn.functional as F
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

class SimpleRewardEnvironment(EnvironmentInterface):
    """Simple environment with custom reward function."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reward_scale = config.get("reward_scale", 1.0)
        self.max_steps = config.get("max_steps", 10)
        
    def _compute_reward(self, messages: List[Dict[str, str]], metadata: Optional[Dict[str, Any]]) -> float:
        """Compute reward based on conversation quality."""
        if not messages:
            return 0.0
        
        # Extract assistant responses
        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
        
        if not assistant_messages:
            return 0.0
        
        # Simple reward: length-based with quality bonus
        total_length = sum(len(msg["content"]) for msg in assistant_messages)
        base_reward = min(total_length / 100.0, 1.0)  # Normalize to [0, 1]
        
        # Quality bonus for detailed responses
        quality_bonus = 0.0
        for msg in assistant_messages:
            content = msg["content"].lower()
            if any(word in content for word in ["because", "therefore", "however", "furthermore"]):
                quality_bonus += 0.1
        
        total_reward = (base_reward + quality_bonus) * self.reward_scale
        return min(total_reward, 2.0)  # Cap at 2.0
    
    def _is_episode_done(self, messages: List[Dict[str, str]], metadata: Optional[Dict[str, Any]]) -> bool:
        """Determine if episode should terminate."""
        # Terminate after max steps
        if len(messages) >= self.max_steps * 2:  # Each step has user + assistant
            return True
        
        # Terminate if user says goodbye
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if user_messages and any(word in user_messages[-1]["content"].lower() 
                               for word in ["goodbye", "bye", "end", "stop"]):
            return True
        
        return False
    
    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[Optional[Dict[str, Any]]],
    ) -> EnvironmentReturn:
        """Process batch of conversations and return rewards."""
        
        observations = []
        rewards = []
        terminateds = []
        
        for messages, meta in zip(message_log_batch, metadata):
            # Compute reward
            reward = self._compute_reward(messages, meta)
            
            # Check termination
            done = self._is_episode_done(messages, meta)
            
            # Create observation
            observation = {
                "role": "environment",
                "content": f"Reward: {reward:.3f}, Steps: {len(messages)//2}"
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
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Post-process batch and compute global metrics."""
        
        # Extract batch data
        rewards = batch.get("rewards", [])
        terminateds = batch.get("terminateds", [])
        
        # Compute metrics
        metrics = {
            "mean_reward": float(torch.mean(torch.tensor(rewards))) if rewards else 0.0,
            "max_reward": float(torch.max(torch.tensor(rewards))) if rewards else 0.0,
            "completion_rate": float(sum(terminateds) / len(terminateds)) if terminateds else 0.0,
            "total_episodes": len(rewards)
        }
        
        return batch, metrics
```

### **Quality-Based Environment**

Create an environment that rewards response quality:

```python
class QualityBasedEnvironment(EnvironmentInterface):
    """Environment that rewards response quality and relevance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_threshold = config.get("quality_threshold", 0.7)
        self.relevance_weight = config.get("relevance_weight", 0.6)
        self.helpfulness_weight = config.get("helpfulness_weight", 0.4)
        
    def _compute_quality_score(self, response: str, context: str) -> float:
        """Compute quality score based on response characteristics."""
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Relevance score (keyword overlap)
        response_words = set(response_lower.split())
        context_words = set(context_lower.split())
        relevance = len(response_words.intersection(context_words)) / max(len(context_words), 1)
        
        # Helpfulness score (indicator words)
        helpful_indicators = [
            "here's", "let me", "i can", "to help", "solution", "answer",
            "explain", "clarify", "assist", "guide", "show", "demonstrate"
        ]
        helpfulness = sum(1 for indicator in helpful_indicators if indicator in response_lower) / len(helpful_indicators)
        
        # Length appropriateness (not too short, not too long)
        length_score = min(len(response.split()) / 50.0, 1.0)  # Normalize to reasonable length
        
        # Combine scores
        quality_score = (
            self.relevance_weight * relevance +
            self.helpfulness_weight * helpfulness +
            (1 - self.relevance_weight - self.helpfulness_weight) * length_score
        )
        
        return min(quality_score, 1.0)
    
    def _compute_reward(self, messages: List[Dict[str, str]], metadata: Optional[Dict[str, Any]]) -> float:
        """Compute reward based on response quality."""
        if len(messages) < 2:
            return 0.0
        
        # Get the most recent user message and assistant response
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
        
        if not user_messages or not assistant_messages:
            return 0.0
        
        # Get context and response
        context = user_messages[-1]["content"]
        response = assistant_messages[-1]["content"]
        
        # Compute quality score
        quality_score = self._compute_quality_score(response, context)
        
        # Convert to reward
        reward = quality_score * self.config.get("reward_scale", 1.0)
        
        # Bonus for high quality responses
        if quality_score > self.quality_threshold:
            reward *= 1.5
        
        return reward
    
    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[Optional[Dict[str, Any]]],
    ) -> EnvironmentReturn:
        """Process batch and return quality-based rewards."""
        
        observations = []
        rewards = []
        terminateds = []
        
        for messages, meta in zip(message_log_batch, metadata):
            # Compute reward
            reward = self._compute_reward(messages, meta)
            
            # Check termination (simple: after 5 exchanges)
            done = len(messages) >= 10  # 5 user-assistant exchanges
            
            # Create observation
            observation = {
                "role": "environment",
                "content": f"Quality reward: {reward:.3f}"
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
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Post-process and compute quality metrics."""
        
        rewards = batch.get("rewards", [])
        
        metrics = {
            "mean_quality_reward": float(torch.mean(torch.tensor(rewards))) if rewards else 0.0,
            "high_quality_rate": float(sum(1 for r in rewards if r > self.quality_threshold) / len(rewards)) if rewards else 0.0,
            "total_responses": len(rewards)
        }
        
        return batch, metrics
```

## Step 3: Advanced Environment Features

### **Stateful Environment with Metadata**

Create an environment that maintains complex state:

```python
class StatefulEnvironment(EnvironmentInterface):
    """Environment with complex state management and metadata tracking."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conversation_history = {}  # Track conversation history
        self.user_preferences = {}  # Track user preferences
        self.conversation_goals = {}  # Track conversation goals
        self.step_counters = {}  # Track step counts
        
    def _initialize_conversation(self, conversation_id: str, metadata: Optional[Dict[str, Any]]):
        """Initialize conversation state."""
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
            self.user_preferences[conversation_id] = metadata.get("preferences", {}) if metadata else {}
            self.conversation_goals[conversation_id] = metadata.get("goals", []) if metadata else []
            self.step_counters[conversation_id] = 0
    
    def _update_conversation_state(self, conversation_id: str, messages: List[Dict[str, str]]):
        """Update conversation state based on messages."""
        self.conversation_history[conversation_id] = messages
        self.step_counters[conversation_id] += 1
        
        # Extract user preferences from messages
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"].lower()
                if "prefer" in content or "like" in content or "want" in content:
                    # Simple preference extraction
                    self.user_preferences[conversation_id].update({
                        "last_preference": content,
                        "preference_count": self.user_preferences[conversation_id].get("preference_count", 0) + 1
                    })
    
    def _compute_contextual_reward(self, conversation_id: str, messages: List[Dict[str, str]]) -> float:
        """Compute reward based on conversation context and goals."""
        if not messages:
            return 0.0
        
        # Get conversation context
        history = self.conversation_history.get(conversation_id, [])
        preferences = self.user_preferences.get(conversation_id, {})
        goals = self.conversation_goals.get(conversation_id, [])
        
        # Base reward from response quality
        base_reward = self._compute_response_quality(messages[-1]["content"] if messages else "")
        
        # Context relevance bonus
        context_bonus = self._compute_context_relevance(messages, history, preferences)
        
        # Goal progress bonus
        goal_bonus = self._compute_goal_progress(messages, goals)
        
        # Engagement bonus (longer conversations)
        engagement_bonus = min(len(history) / 20.0, 0.5)  # Cap at 0.5
        
        total_reward = base_reward + context_bonus + goal_bonus + engagement_bonus
        return min(total_reward, 2.0)  # Cap at 2.0
    
    def _compute_response_quality(self, response: str) -> float:
        """Compute basic response quality score."""
        if not response:
            return 0.0
        
        # Length appropriateness
        length_score = min(len(response.split()) / 30.0, 1.0)
        
        # Clarity indicators
        clarity_indicators = ["because", "therefore", "however", "specifically", "in other words"]
        clarity_score = sum(1 for indicator in clarity_indicators if indicator in response.lower()) / len(clarity_indicators)
        
        return (length_score + clarity_score) / 2.0
    
    def _compute_context_relevance(self, messages: List[Dict[str, str]], history: List[Dict[str, str]], preferences: Dict[str, Any]) -> float:
        """Compute relevance to conversation context."""
        if not messages or not history:
            return 0.0
        
        # Simple relevance: keyword overlap with history
        current_content = " ".join(msg["content"].lower() for msg in messages[-3:])  # Last 3 messages
        history_content = " ".join(msg["content"].lower() for msg in history[-5:])  # Last 5 history messages
        
        current_words = set(current_content.split())
        history_words = set(history_content.split())
        
        overlap = len(current_words.intersection(history_words)) / max(len(history_words), 1)
        return min(overlap, 0.5)  # Cap at 0.5
    
    def _compute_goal_progress(self, messages: List[Dict[str, str]], goals: List[str]) -> float:
        """Compute progress toward conversation goals."""
        if not goals:
            return 0.0
        
        # Simple goal progress: check if goals are mentioned
        content = " ".join(msg["content"].lower() for msg in messages)
        goal_mentions = sum(1 for goal in goals if goal.lower() in content)
        
        return min(goal_mentions / len(goals), 0.5)  # Cap at 0.5
    
    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[Optional[Dict[str, Any]]],
    ) -> EnvironmentReturn:
        """Process batch with stateful reward computation."""
        
        observations = []
        rewards = []
        terminateds = []
        updated_metadata = []
        
        for i, (messages, meta) in enumerate(zip(message_log_batch, metadata)):
            conversation_id = meta.get("conversation_id", f"conv_{i}") if meta else f"conv_{i}"
            
            # Initialize conversation if needed
            self._initialize_conversation(conversation_id, meta)
            
            # Update conversation state
            self._update_conversation_state(conversation_id, messages)
            
            # Compute contextual reward
            reward = self._compute_contextual_reward(conversation_id, messages)
            
            # Check termination
            done = self.step_counters[conversation_id] >= self.config.get("max_steps", 20)
            
            # Create observation
            observation = {
                "role": "environment",
                "content": f"Contextual reward: {reward:.3f}, Steps: {self.step_counters[conversation_id]}"
            }
            
            # Update metadata
            updated_meta = {
                "conversation_id": conversation_id,
                "step_count": self.step_counters[conversation_id],
                "preferences": self.user_preferences[conversation_id],
                "goals": self.conversation_goals[conversation_id]
            }
            
            observations.append(observation)
            rewards.append(reward)
            terminateds.append(done)
            updated_metadata.append(updated_meta)
        
        return EnvironmentReturn(
            observations=observations,
            metadata=updated_metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
        )
    
    def global_post_process_and_metrics(
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Post-process and compute stateful metrics."""
        
        rewards = batch.get("rewards", [])
        metadata = batch.get("metadata", [])
        
        # Compute conversation statistics
        total_conversations = len(set(meta.get("conversation_id", "") for meta in metadata))
        avg_steps = sum(meta.get("step_count", 0) for meta in metadata) / len(metadata) if metadata else 0
        
        metrics = {
            "mean_contextual_reward": float(torch.mean(torch.tensor(rewards))) if rewards else 0.0,
            "total_conversations": total_conversations,
            "average_steps_per_conversation": avg_steps,
            "conversation_completion_rate": float(sum(1 for meta in metadata if meta.get("step_count", 0) >= 10) / len(metadata)) if metadata else 0.0
        }
        
        return batch, metrics
```

## Step 4: Distributed Environment Scaling

### **Ray Remote Environment**

Scale environments using Ray remote actors:

```python
import ray
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

@ray.remote
class RayRemoteEnvironment(EnvironmentInterface):
    """Ray remote environment for distributed processing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.environment_type = config.get("environment_type", "simple")
        
        # Initialize specific environment
        if self.environment_type == "quality":
            self.inner_env = QualityBasedEnvironment(config)
        elif self.environment_type == "stateful":
            self.inner_env = StatefulEnvironment(config)
        else:
            self.inner_env = SimpleRewardEnvironment(config)
    
    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[Optional[Dict[str, Any]]],
    ) -> EnvironmentReturn:
        """Delegate to inner environment."""
        return self.inner_env.step(message_log_batch, metadata)
    
    def global_post_process_and_metrics(
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Delegate to inner environment."""
        return self.inner_env.global_post_process_and_metrics(batch)

# Environment factory for creating remote environments
class EnvironmentFactory:
    """Factory for creating remote environments."""
    
    @staticmethod
    def create_remote_environment(config: Dict[str, Any], num_workers: int = 4):
        """Create multiple remote environment instances."""
        environments = []
        
        for i in range(num_workers):
            # Create remote environment
            env = RayRemoteEnvironment.remote(config)
            environments.append(env)
        
        return environments
    
    @staticmethod
    def batch_process(environments: List[RayRemoteEnvironment], message_batches: List[List[List[Dict[str, str]]]], metadata_batches: List[List[Optional[Dict[str, Any]]]]):
        """Process batches across multiple remote environments."""
        
        # Distribute batches across environments
        futures = []
        for env, messages, metadata in zip(environments, message_batches, metadata_batches):
            future = env.step.remote(messages, metadata)
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        return results
```

### **Load-Balanced Environment Pool**

Create a pool of environments for load balancing:

```python
class EnvironmentPool:
    """Pool of environments for load balancing."""
    
    def __init__(self, config: Dict[str, Any], pool_size: int = 8):
        self.config = config
        self.pool_size = pool_size
        self.environments = []
        self.current_index = 0
        
        # Initialize environment pool
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the environment pool."""
        for i in range(self.pool_size):
            env = RayRemoteEnvironment.remote(self.config)
            self.environments.append(env)
    
    def get_environment(self) -> RayRemoteEnvironment:
        """Get next environment from pool (round-robin)."""
        env = self.environments[self.current_index]
        self.current_index = (self.current_index + 1) % self.pool_size
        return env
    
    def process_batch(self, message_log_batch: List[List[Dict[str, str]]], metadata: List[Optional[Dict[str, Any]]]) -> EnvironmentReturn:
        """Process batch using load-balanced environment."""
        env = self.get_environment()
        return ray.get(env.step.remote(message_log_batch, metadata))
    
    def process_multiple_batches(self, batches: List[Tuple[List[List[Dict[str, str]]], List[Optional[Dict[str, Any]]]]]) -> List[EnvironmentReturn]:
        """Process multiple batches in parallel."""
        futures = []
        
        for message_batch, metadata_batch in batches:
            env = self.get_environment()
            future = env.step.remote(message_batch, metadata_batch)
            futures.append(future)
        
        return ray.get(futures)
```

## Step 5: Advanced Environment Patterns

### **Composite Environment**

Create environments that combine multiple reward functions:

```python
class CompositeEnvironment(EnvironmentInterface):
    """Environment that combines multiple reward functions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reward_functions = config.get("reward_functions", [])
        self.reward_weights = config.get("reward_weights", [])
        
        # Ensure weights sum to 1
        if self.reward_weights:
            total_weight = sum(self.reward_weights)
            self.reward_weights = [w / total_weight for w in self.reward_weights]
    
    def _compute_composite_reward(self, messages: List[Dict[str, str]], metadata: Optional[Dict[str, Any]]) -> float:
        """Compute composite reward from multiple reward functions."""
        if not self.reward_functions:
            return 0.0
        
        rewards = []
        for reward_fn in self.reward_functions:
            reward = reward_fn(messages, metadata)
            rewards.append(reward)
        
        # Weighted combination
        if self.reward_weights:
            composite_reward = sum(r * w for r, w in zip(rewards, self.reward_weights))
        else:
            composite_reward = sum(rewards) / len(rewards)
        
        return composite_reward
    
    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[Optional[Dict[str, Any]]],
    ) -> EnvironmentReturn:
        """Process batch with composite rewards."""
        
        observations = []
        rewards = []
        terminateds = []
        
        for messages, meta in zip(message_log_batch, metadata):
            # Compute composite reward
            reward = self._compute_composite_reward(messages, meta)
            
            # Check termination
            done = len(messages) >= self.config.get("max_steps", 10) * 2
            
            # Create observation
            observation = {
                "role": "environment",
                "content": f"Composite reward: {reward:.3f}"
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
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Post-process and compute composite metrics."""
        
        rewards = batch.get("rewards", [])
        
        metrics = {
            "mean_composite_reward": float(torch.mean(torch.tensor(rewards))) if rewards else 0.0,
            "reward_components": len(self.reward_functions),
            "total_episodes": len(rewards)
        }
        
        return batch, metrics
```

### **Adaptive Environment**

Create environments that adapt their behavior based on performance:

```python
class AdaptiveEnvironment(EnvironmentInterface):
    """Environment that adapts reward function based on performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adaptation_threshold = config.get("adaptation_threshold", 0.5)
        self.performance_history = []
        self.current_strategy = "balanced"
        
        # Initialize reward strategies
        self.strategies = {
            "balanced": self._balanced_reward,
            "quality_focused": self._quality_focused_reward,
            "engagement_focused": self._engagement_focused_reward
        }
    
    def _balanced_reward(self, messages: List[Dict[str, str]], metadata: Optional[Dict[str, Any]]) -> float:
        """Balanced reward strategy."""
        if not messages:
            return 0.0
        
        # Simple balanced reward
        response_length = len(messages[-1]["content"]) if messages else 0
        return min(response_length / 100.0, 1.0)
    
    def _quality_focused_reward(self, messages: List[Dict[str, str]], metadata: Optional[Dict[str, Any]]) -> float:
        """Quality-focused reward strategy."""
        if not messages:
            return 0.0
        
        # Quality indicators
        response = messages[-1]["content"] if messages else ""
        quality_indicators = ["because", "therefore", "specifically", "in detail"]
        quality_score = sum(1 for indicator in quality_indicators if indicator in response.lower()) / len(quality_indicators)
        
        return quality_score
    
    def _engagement_focused_reward(self, messages: List[Dict[str, str]], metadata: Optional[Dict[str, Any]]) -> float:
        """Engagement-focused reward strategy."""
        if not messages:
            return 0.0
        
        # Engagement indicators
        response = messages[-1]["content"] if messages else ""
        engagement_indicators = ["?", "!", "interesting", "fascinating", "amazing"]
        engagement_score = sum(1 for indicator in engagement_indicators if indicator in response.lower()) / len(engagement_indicators)
        
        return engagement_score
    
    def _adapt_strategy(self, recent_performance: float):
        """Adapt strategy based on recent performance."""
        self.performance_history.append(recent_performance)
        
        if len(self.performance_history) < 10:
            return
        
        # Compute recent average
        recent_avg = sum(self.performance_history[-10:]) / 10
        
        # Adapt strategy based on performance
        if recent_avg < self.adaptation_threshold:
            if self.current_strategy == "balanced":
                self.current_strategy = "quality_focused"
            elif self.current_strategy == "quality_focused":
                self.current_strategy = "engagement_focused"
        else:
            # Good performance, try balanced
            self.current_strategy = "balanced"
    
    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[Optional[Dict[str, Any]]],
    ) -> EnvironmentReturn:
        """Process batch with adaptive rewards."""
        
        observations = []
        rewards = []
        terminateds = []
        
        for messages, meta in zip(message_log_batch, metadata):
            # Compute reward using current strategy
            reward_fn = self.strategies[self.current_strategy]
            reward = reward_fn(messages, meta)
            
            # Adapt strategy
            self._adapt_strategy(reward)
            
            # Check termination
            done = len(messages) >= self.config.get("max_steps", 10) * 2
            
            # Create observation
            observation = {
                "role": "environment",
                "content": f"Adaptive reward ({self.current_strategy}): {reward:.3f}"
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
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Post-process and compute adaptive metrics."""
        
        rewards = batch.get("rewards", [])
        
        metrics = {
            "mean_adaptive_reward": float(torch.mean(torch.tensor(rewards))) if rewards else 0.0,
            "current_strategy": self.current_strategy,
            "performance_history_length": len(self.performance_history),
            "total_episodes": len(rewards)
        }
        
        return batch, metrics
```

## Configuration and Usage

### **Registering Custom Environments**

```python
# In your training script
from nemo_rl.environments.interfaces import EnvironmentInterface

# Register custom environments
def register_custom_environments():
    """Register custom environments for use in training."""
    
    # Register environment types
    environment_registry = {
        "simple_reward": SimpleRewardEnvironment,
        "quality_based": QualityBasedEnvironment,
        "stateful": StatefulEnvironment,
        "composite": CompositeEnvironment,
        "adaptive": AdaptiveEnvironment
    }
    
    return environment_registry

# Use in configuration
config = {
    "environment": {
        "type": "quality_based",
        "reward_scale": 1.0,
        "quality_threshold": 0.7,
        "relevance_weight": 0.6,
        "helpfulness_weight": 0.4
    }
}
```

### **Testing Custom Environments**

```python
def test_custom_environment():
    """Test custom environment implementation."""
    
    # Create test data
    message_batch = [
        [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."}
        ],
        [
            {"role": "user", "content": "Explain neural networks"},
            {"role": "assistant", "content": "Neural networks are computational models inspired by biological neural networks in the brain."}
        ]
    ]
    
    metadata_batch = [
        {"conversation_id": "test_1", "preferences": {"topic": "AI"}},
        {"conversation_id": "test_2", "preferences": {"topic": "ML"}}
    ]
    
    # Test environment
    env = QualityBasedEnvironment({
        "reward_scale": 1.0,
        "quality_threshold": 0.7,
        "relevance_weight": 0.6,
        "helpfulness_weight": 0.4
    })
    
    result = env.step(message_batch, metadata_batch)
    
    print(f"Rewards: {result.rewards}")
    print(f"Terminated: {result.terminateds}")
    print(f"Observations: {result.observations}")
```

## Best Practices

### **1. Environment Design**

- **Modularity**: Design environments as composable components
- **Efficiency**: Optimize for batch processing and GPU utilization
- **State Management**: Handle complex state and metadata efficiently
- **Error Handling**: Gracefully handle edge cases and malformed inputs

### **2. Reward Function Design**

- **Meaningful Rewards**: Design rewards that reflect task objectives
- **Appropriate Scale**: Use reward values that work well with training algorithms
- **Consistency**: Maintain consistent evaluation criteria
- **Interpretability**: Make rewards interpretable for debugging

### **3. Distributed Processing**

- **Load Balancing**: Distribute work evenly across environment instances
- **Fault Tolerance**: Handle environment failures gracefully
- **Resource Management**: Efficiently manage memory and compute resources
- **Monitoring**: Track environment performance and resource usage

## Next Steps

After completing this tutorial:

1. **Experiment with Variants**: Try different reward functions and environment types
2. **Optimize Performance**: Profile and optimize your custom environments
3. **Scale to Production**: Deploy custom environments in production training
4. **Contribute Back**: Share useful environment implementations with the community

## Related Resources

- **[Environment Interfaces API](../../apidocs/nemo_rl/nemo_rl.environments)**: Detailed API documentation
- **[Environment Development Guide](../../guides/environment-data/environment-development)**: Environment development fundamentals
- **[Ray Distributed Computing](../../apidocs/nemo_rl/nemo_rl.distributed)**: Distributed computing with Ray
- **[Advanced Algorithm Development](../../../advanced/algorithm-development/index)**: Advanced algorithm development techniques

## Summary

In this tutorial, you learned:

- ✅ **Environment Architecture**: Understanding NeMo RL's environment framework
- ✅ **Custom Reward Functions**: Designing and creating specialized reward functions
- ✅ **Environment State Management**: Handling complex environment state and metadata
- ✅ **Ray Remote Environments**: Scaling environments with distributed processing
- ✅ **Advanced Environment Patterns**: Learning sophisticated environment design patterns

You now have the skills to build custom training environments that extend NeMo RL's capabilities for your specific use cases.
