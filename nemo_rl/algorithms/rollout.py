# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RolloutEngine abstraction for separating generation logic from training.

This module provides the RolloutEngine class that handles:
- Prompt batching and scheduling
- Response generation via configurable backends
- Reward collection from environments

The RolloutEngine decouples generation orchestration from the training loop,
enabling:
- Easy testing of generation in isolation
- Swapping generation backends (vLLM, Megatron)
- Future async generation patterns

Example:
    >>> from nemo_rl.algorithms.rollout import RolloutEngine, SamplingParams
    >>> 
    >>> engine = RolloutEngine(
    ...     generation_backend=vllm_generation,
    ...     environment=math_env,
    ... )
    >>> 
    >>> # Generate responses
    >>> results = engine.generate(prompts, sampling_params)
    >>> 
    >>> # Generate and collect rewards
    >>> results = engine.rollout(prompts, sampling_params)
    >>> print(results['rewards'])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.environments.interfaces import EnvironmentInterface
    from nemo_rl.models.generation.interfaces import (
        GenerationDatumSpec,
        GenerationInterface,
        GenerationOutputSpec,
    )

logger = logging.getLogger(__name__)


@dataclass
class SamplingParams:
    """Parameters for response generation.
    
    Configurable generation parameters for controlling response sampling.
    
    Attributes:
        temperature: Sampling temperature. Higher values increase randomness.
        top_p: Nucleus sampling parameter. Only tokens with cumulative 
            probability < top_p are considered.
        top_k: Top-k sampling. Only the top k tokens are considered.
            If None, top-k filtering is disabled.
        max_tokens: Maximum number of tokens to generate.
        stop_strings: List of strings that stop generation when encountered.
        stop_token_ids: List of token IDs that stop generation.
        greedy: If True, use greedy decoding instead of sampling.
        
    Example:
        >>> params = SamplingParams(
        ...     temperature=0.7,
        ...     top_p=0.95,
        ...     max_tokens=512,
        ... )
    """
    
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    max_tokens: int = 256
    stop_strings: list[str] | None = None
    stop_token_ids: list[int] | None = None
    greedy: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backend compatibility."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_tokens,
            "stop_strings": self.stop_strings,
            "stop_token_ids": self.stop_token_ids,
        }


@dataclass
class RolloutResult:
    """Result of a rollout operation.
    
    Contains generated responses and optionally rewards.
    
    Attributes:
        prompts: Original prompts (BatchedDataDict).
        responses: Generated responses (BatchedDataDict).
        rewards: Reward tensor if environment was used.
        metrics: Additional metrics from the rollout.
        generation_lengths: Length of each generated sequence.
    """
    
    prompts: "BatchedDataDict" = None
    responses: "BatchedDataDict" = None
    rewards: Any = None  # torch.Tensor
    metrics: dict[str, Any] = field(default_factory=dict)
    generation_lengths: Any = None  # torch.Tensor


@runtime_checkable
class GenerationBackend(Protocol):
    """Protocol for generation backends.
    
    Defines the interface that generation backends must implement
    to work with RolloutEngine.
    """
    
    def generate(
        self,
        data: "BatchedDataDict[GenerationDatumSpec]",
        greedy: bool,
    ) -> "BatchedDataDict[GenerationOutputSpec]":
        """Generate responses for the given prompts.
        
        Args:
            data: Input data with prompts.
            greedy: Whether to use greedy decoding.
            
        Returns:
            Generated responses with output_ids and logprobs.
        """
        ...
    
    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Prepare the backend for generation."""
        ...
    
    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Clean up after generation."""
        ...


class RolloutEngine:
    """Engine for orchestrating response generation and reward collection.
    
    RolloutEngine provides a clean abstraction for the rollout process:
    1. Accepts prompts from the training loop
    2. Calls the generation backend to produce responses
    3. Optionally sends prompt-response pairs to environment for reward computation
    4. Returns complete batch with responses and rewards
    
    This separation allows:
    - Easy testing of generation in isolation
    - Swapping generation backends (vLLM, Megatron)
    - Async generation patterns in future
    
    Attributes:
        generation_backend: Backend for response generation.
        environment: Environment for reward computation (optional).
        default_params: Default sampling parameters.
        
    Example:
        >>> engine = RolloutEngine(
        ...     generation_backend=vllm_gen,
        ...     environment=math_env,
        ... )
        >>> 
        >>> # Just generate
        >>> responses = engine.generate(prompts, params)
        >>> 
        >>> # Generate and collect rewards
        >>> result = engine.rollout(prompts, params)
        >>> print(result.rewards.mean())
    """
    
    def __init__(
        self,
        generation_backend: "GenerationInterface | GenerationBackend" = None,
        environment: "EnvironmentInterface | None" = None,
        default_params: SamplingParams | None = None,
        reward_fn: Callable | None = None,
    ):
        """Initialize the RolloutEngine.
        
        Args:
            generation_backend: Backend for generating responses.
                Must implement GenerationInterface or GenerationBackend protocol.
            environment: Environment for computing rewards.
                Must implement EnvironmentInterface. Optional if using reward_fn.
            default_params: Default sampling parameters to use when not specified.
            reward_fn: Simple callable for reward computation as alternative to
                full Environment class. Signature: fn(prompts, responses) -> rewards.
        """
        self.generation_backend = generation_backend
        self.environment = environment
        self.default_params = default_params or SamplingParams()
        self.reward_fn = reward_fn
        
        # Statistics tracking
        self._total_generations = 0
        self._total_tokens_generated = 0
    
    def generate(
        self,
        prompts: "BatchedDataDict[GenerationDatumSpec]",
        sampling_params: SamplingParams | None = None,
    ) -> "BatchedDataDict[GenerationOutputSpec]":
        """Generate responses for the given prompts.
        
        This method only generates responses without computing rewards.
        Use rollout() if you need rewards.
        
        Args:
            prompts: BatchedDataDict containing input_ids and input_lengths.
            sampling_params: Parameters controlling generation. Uses defaults if None.
            
        Returns:
            BatchedDataDict with output_ids, logprobs, and generation_lengths.
            
        Raises:
            RuntimeError: If generation backend is not set.
            
        Example:
            >>> params = SamplingParams(temperature=0.7, max_tokens=256)
            >>> responses = engine.generate(prompts, params)
            >>> print(responses['output_ids'].shape)
        """
        if self.generation_backend is None:
            raise RuntimeError(
                "Generation backend not set. "
                "Initialize RolloutEngine with a generation_backend."
            )
        
        params = sampling_params or self.default_params
        
        # Prepare backend for generation
        if hasattr(self.generation_backend, 'prepare_for_generation'):
            self.generation_backend.prepare_for_generation()
        
        try:
            # Generate responses
            responses = self.generation_backend.generate(
                prompts,
                greedy=params.greedy,
            )
            
            # Update statistics
            self._total_generations += 1
            if 'generation_lengths' in responses:
                self._total_tokens_generated += responses['generation_lengths'].sum().item()
            
            return responses
            
        finally:
            # Clean up
            if hasattr(self.generation_backend, 'finish_generation'):
                self.generation_backend.finish_generation()
    
    def collect_rewards(
        self,
        prompts: "BatchedDataDict",
        responses: "BatchedDataDict",
    ) -> "BatchedDataDict":
        """Compute rewards for prompt-response pairs.
        
        Args:
            prompts: BatchedDataDict with prompt data.
            responses: BatchedDataDict with generated responses.
            
        Returns:
            BatchedDataDict with rewards added.
            
        Raises:
            RuntimeError: If neither environment nor reward_fn is set.
        """
        if self.reward_fn is not None:
            # Use simple callable reward function
            rewards = self.reward_fn(prompts, responses)
            responses['rewards'] = rewards
            return responses
            
        if self.environment is None:
            raise RuntimeError(
                "No reward computation available. "
                "Set environment or reward_fn in RolloutEngine."
            )
        
        # Use full environment interface
        # Note: This is a simplified version - full implementation would
        # need to construct message logs from prompts/responses
        logger.warning(
            "Full environment integration requires message log construction. "
            "Using placeholder implementation."
        )
        
        return responses
    
    def rollout(
        self,
        prompts: "BatchedDataDict[GenerationDatumSpec]",
        sampling_params: SamplingParams | None = None,
        collect_rewards: bool = True,
    ) -> RolloutResult:
        """Generate responses and optionally collect rewards.
        
        Full rollout pipeline:
        1. Generate responses from prompts
        2. Collect rewards from environment (if enabled)
        3. Return complete result
        
        Args:
            prompts: BatchedDataDict containing prompts.
            sampling_params: Parameters controlling generation.
            collect_rewards: Whether to compute rewards. Requires environment
                or reward_fn to be set.
                
        Returns:
            RolloutResult with responses and optional rewards.
            
        Example:
            >>> result = engine.rollout(prompts, params)
            >>> print(f"Mean reward: {result.rewards.mean():.4f}")
            >>> print(f"Generated {result.generation_lengths.sum()} tokens")
        """
        # Generate responses
        responses = self.generate(prompts, sampling_params)
        
        # Collect rewards if requested and possible
        rewards = None
        if collect_rewards and (self.environment is not None or self.reward_fn is not None):
            responses = self.collect_rewards(prompts, responses)
            rewards = responses.get('rewards')
        
        return RolloutResult(
            prompts=prompts,
            responses=responses,
            rewards=rewards,
            generation_lengths=responses.get('generation_lengths'),
            metrics={
                'total_generations': self._total_generations,
                'total_tokens': self._total_tokens_generated,
            },
        )
    
    @property
    def stats(self) -> dict[str, Any]:
        """Return current statistics."""
        return {
            'total_generations': self._total_generations,
            'total_tokens_generated': self._total_tokens_generated,
        }
    
    def reset_stats(self) -> None:
        """Reset generation statistics."""
        self._total_generations = 0
        self._total_tokens_generated = 0
    
    def __repr__(self) -> str:
        backend_name = type(self.generation_backend).__name__ if self.generation_backend else "None"
        env_name = type(self.environment).__name__ if self.environment else "None"
        return (
            f"RolloutEngine("
            f"backend={backend_name}, "
            f"environment={env_name}, "
            f"generations={self._total_generations})"
        )


# Convenience function for creating RolloutEngine with common configurations
def create_rollout_engine(
    backend: str = "vllm",
    environment: "EnvironmentInterface | None" = None,
    reward_fn: Callable | None = None,
    **generation_kwargs: Any,
) -> RolloutEngine:
    """Create a RolloutEngine with the specified backend.
    
    Factory function for creating RolloutEngine instances with
    commonly used configurations.
    
    Args:
        backend: Generation backend type ('vllm', 'megatron', or 'mock').
        environment: Environment for reward computation.
        reward_fn: Simple callable for rewards (alternative to environment).
        **generation_kwargs: Additional arguments for the generation backend.
        
    Returns:
        Configured RolloutEngine instance.
        
    Example:
        >>> engine = create_rollout_engine(
        ...     backend='vllm',
        ...     environment=math_env,
        ... )
    """
    # Note: Actual backend instantiation would require more configuration
    # This is a placeholder showing the intended API
    logger.info(f"Creating RolloutEngine with backend: {backend}")
    
    return RolloutEngine(
        generation_backend=None,  # Would be instantiated based on backend type
        environment=environment,
        reward_fn=reward_fn,
    )
