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
"""Simplified Environment base class for reward computation.

This module provides an easy-to-use Environment base class that requires
minimal code to implement custom reward functions. For simple use cases,
only the `score()` method needs to be overridden.

Design Goals:
    - Simple: Override only `score()` for basic reward functions
    - Flexible: Override `score_batch()` for optimized batch processing
    - No Ray dependency: Works without Ray actors for simple environments
    - Async support: Override `score_async()` for async reward computation

Example:
    >>> from nemo_rl.environments import Environment
    >>>
    >>> # Simple environment - only override score()
    >>> class LengthReward(Environment):
    ...     def score(self, prompt: str, response: str) -> float:
    ...         return min(len(response) / 100, 1.0)
    >>>
    >>> # Use with trainer
    >>> env = LengthReward()
    >>> trainer.fit(dataset="my_data", environment=env)

For more control, see:
    - `score_batch()`: Override for optimized batch processing
    - `score_async()`: Override for async reward computation
    - `get_metrics()`: Override for custom metric computation
    - `validate_response()`: Override for response validation

Comparison with alternatives:
    - Use `Environment` when you need state or complex logic
    - Use `FunctionalRewardWrapper` when a simple function suffices
    - Use `@reward_function` decorator for the simplest case
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)

import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

# Type variable for environment-specific metadata
MetadataT = TypeVar("MetadataT")


@dataclass
class EnvironmentConfig:
    """Configuration for Environment base class.

    Attributes:
        name: Name of the environment (for logging and metrics).
        max_workers: Maximum threads for parallel score computation.
            Default: 8. Set to 1 for sequential processing.
        timeout: Timeout in seconds for async operations.
            Default: 60.0
        default_reward: Default reward value when score() fails.
            Default: 0.0
        enable_metrics: Whether to compute metrics in global_post_process.
            Default: True
    """

    name: str = "environment"
    max_workers: int = 8
    timeout: float = 60.0
    default_reward: float = 0.0
    enable_metrics: bool = True


class Environment(EnvironmentInterface[MetadataT], ABC, Generic[MetadataT]):
    """Base class for reward computation environments.

    This class provides a simplified interface for creating custom environments.
    For simple use cases, only the `score()` method needs to be overridden.

    The class automatically handles:
        - Batch processing (via thread pool or batched implementation)
        - Message log parsing (extracting prompt/response pairs)
        - Reward tensor creation
        - Basic metrics computation

    Methods to Override:
        Required (choose one):
            - `score(prompt, response)`: Score a single prompt-response pair
            - `score_batch(prompts, responses)`: Score a batch (for efficiency)

        Optional:
            - `score_async(prompt, response)`: Async version of score()
            - `validate_response(prompt, response)`: Validate before scoring
            - `get_metrics(batch)`: Custom metrics computation
            - `setup()`: Called once before first step()
            - `teardown()`: Called when environment is no longer needed

    Example:
        >>> class CorrectnessReward(Environment):
        ...     '''Reward based on answer correctness.'''
        ...
        ...     def __init__(self, answer_key: dict[str, str]):
        ...         super().__init__()
        ...         self.answer_key = answer_key
        ...
        ...     def score(self, prompt: str, response: str) -> float:
        ...         # Extract answer from prompt
        ...         question_id = self._extract_question_id(prompt)
        ...         correct_answer = self.answer_key.get(question_id, "")
        ...
        ...         # Check if response contains correct answer
        ...         if correct_answer.lower() in response.lower():
        ...             return 1.0
        ...         return 0.0
        ...
        ...     def _extract_question_id(self, prompt: str) -> str:
        ...         # Implementation to extract question ID
        ...         return prompt.split()[0]

    Attributes:
        config: Environment configuration.
        name: Name of the environment.
    """

    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        name: Optional[str] = None,
    ):
        """Initialize the Environment.

        Args:
            config: Environment configuration. If None, uses defaults.
            name: Name override for the environment. Takes precedence
                over config.name if provided.
        """
        self.config = config or EnvironmentConfig()

        # Allow name override
        if name is not None:
            self.config.name = name

        self.name = self.config.name

        # Thread pool for parallel processing (lazy init)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._setup_called = False

    # =========================================================================
    # Methods to Override
    # =========================================================================

    def score(self, prompt: str, response: str) -> float:
        """Compute reward score for a single prompt-response pair.

        Override this method to implement your reward function.
        For simple rewards, this is the only method you need to implement.

        Args:
            prompt: The input prompt text.
            response: The model's response text.

        Returns:
            A float reward value. Higher values indicate better responses.

        Raises:
            NotImplementedError: If neither score() nor score_batch() is overridden.

        Example:
            >>> def score(self, prompt: str, response: str) -> float:
            ...     # Reward based on response length
            ...     return min(len(response) / 100, 1.0)
        """
        raise NotImplementedError(
            f"Environment '{self.name}' must implement either score() or score_batch(). "
            f"Override score(prompt, response) -> float for simple reward functions, "
            f"or score_batch(prompts, responses) -> list[float] for batch processing."
        )

    def score_batch(
        self, prompts: List[str], responses: List[str]
    ) -> List[float]:
        """Compute rewards for a batch of prompt-response pairs.

        Default implementation calls score() for each item using a thread pool.
        Override this method for optimized batch processing (e.g., using GPU
        batch inference for reward models).

        Args:
            prompts: List of prompt texts.
            responses: List of response texts.

        Returns:
            List of reward values, one per prompt-response pair.

        Example:
            >>> def score_batch(self, prompts: list[str], responses: list[str]) -> list[float]:
            ...     # Use batch inference for efficiency
            ...     inputs = self.tokenizer(prompts, responses, return_tensors="pt")
            ...     with torch.no_grad():
            ...         scores = self.reward_model(**inputs).logits
            ...     return scores.squeeze().tolist()
        """
        # Default: parallel processing using thread pool
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_workers
            )

        futures = [
            self._executor.submit(self._safe_score, p, r)
            for p, r in zip(prompts, responses)
        ]

        return [f.result(timeout=self.config.timeout) for f in futures]

    async def score_async(self, prompt: str, response: str) -> float:
        """Async version of score(). Default calls sync version.

        Override this method when your reward computation involves
        async operations (e.g., API calls, database queries).

        Args:
            prompt: The input prompt text.
            response: The model's response text.

        Returns:
            A float reward value.

        Example:
            >>> async def score_async(self, prompt: str, response: str) -> float:
            ...     # Call external API for reward
            ...     result = await self.api_client.evaluate(prompt, response)
            ...     return result.score
        """
        return self.score(prompt, response)

    def validate_response(self, prompt: str, response: str) -> bool:
        """Validate a response before scoring.

        Override this method to filter out invalid responses
        before they are scored. Invalid responses receive the
        default_reward value.

        Args:
            prompt: The input prompt text.
            response: The model's response text.

        Returns:
            True if the response is valid and should be scored.
            False if the response is invalid.

        Example:
            >>> def validate_response(self, prompt: str, response: str) -> bool:
            ...     # Reject empty or whitespace-only responses
            ...     if not response.strip():
            ...         return False
            ...     # Reject responses that are too long
            ...     if len(response) > 10000:
            ...         return False
            ...     return True
        """
        return True  # Default: all responses are valid

    def get_metrics(self, batch: BatchedDataDict) -> Dict[str, float]:
        """Compute custom metrics for this environment.

        Override this method to add environment-specific metrics.
        Called during global_post_process_and_metrics().

        Args:
            batch: The processed batch data.

        Returns:
            Dictionary of metric names to values.

        Example:
            >>> def get_metrics(self, batch: BatchedDataDict) -> dict[str, float]:
            ...     rewards = batch.get("rewards", torch.tensor([]))
            ...     return {
            ...         "accuracy": (rewards > 0.5).float().mean().item(),
            ...         "perfect_score_rate": (rewards == 1.0).float().mean().item(),
            ...     }
        """
        return {}

    def setup(self) -> None:
        """Called once before the first step() call.

        Override this method for one-time initialization that
        requires resources (e.g., loading models, connecting to APIs).

        Example:
            >>> def setup(self) -> None:
            ...     self.model = load_reward_model(self.model_path)
            ...     self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        """
        pass

    def teardown(self) -> None:
        """Called when the environment is no longer needed.

        Override this method to clean up resources.

        Example:
            >>> def teardown(self) -> None:
            ...     del self.model
            ...     torch.cuda.empty_cache()
        """
        pass

    # =========================================================================
    # EnvironmentInterface Implementation
    # =========================================================================

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata: List[MetadataT],
    ) -> EnvironmentReturn[MetadataT]:
        """Run a step in the environment, computing rewards.

        This method implements the EnvironmentInterface.step() method.
        It extracts prompts and responses from message logs, computes
        rewards using score() or score_batch(), and returns the results.

        Args:
            message_log_batch: Batch of message logs (OpenAI-style format).
            metadata: Batch of metadata objects.

        Returns:
            EnvironmentReturn with rewards and termination flags.
        """
        # Ensure setup is called
        if not self._setup_called:
            self.setup()
            self._setup_called = True

        batch_size = len(message_log_batch)

        # Extract prompts and responses from message logs
        prompts, responses = self._extract_prompt_response_batch(message_log_batch)

        # Compute rewards
        if asyncio.iscoroutinefunction(self.score_async) and hasattr(
            self, "_use_async"
        ):
            # Use async if explicitly enabled
            rewards = self._compute_rewards_async(prompts, responses)
        else:
            rewards = self.score_batch(prompts, responses)

        # Convert to tensor
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)

        # All episodes terminate after reward computation (single-turn)
        terminateds = torch.ones(batch_size, dtype=torch.bool)

        # Empty observations (no continuation)
        observations = [
            {"role": "assistant", "content": ""} for _ in range(batch_size)
        ]

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * batch_size,
            rewards=reward_tensor,
            terminateds=terminateds,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, Dict[str, Any]]:
        """Post-process batch and compute metrics.

        Args:
            batch: The batch to process.

        Returns:
            Tuple of (processed_batch, metrics).
        """
        metrics: Dict[str, Any] = {
            "environment_name": self.name,
        }

        # Add reward statistics if available
        if "rewards" in batch and self.config.enable_metrics:
            rewards = batch["rewards"]
            if isinstance(rewards, torch.Tensor) and rewards.numel() > 0:
                metrics["mean_reward"] = rewards.float().mean().item()
                metrics["std_reward"] = rewards.float().std().item()
                metrics["min_reward"] = rewards.float().min().item()
                metrics["max_reward"] = rewards.float().max().item()

        # Add custom metrics
        custom_metrics = self.get_metrics(batch)
        metrics.update(custom_metrics)

        return batch, metrics

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_prompt_response_batch(
        self, message_log_batch: List[LLMMessageLogType]
    ) -> tuple[List[str], List[str]]:
        """Extract prompts and responses from message logs.

        Args:
            message_log_batch: Batch of message logs.

        Returns:
            Tuple of (prompts, responses) lists.
        """
        prompts = []
        responses = []

        for message_log in message_log_batch:
            prompt, response = self._extract_prompt_response(message_log)
            prompts.append(prompt)
            responses.append(response)

        return prompts, responses

    def _extract_prompt_response(
        self, message_log: LLMMessageLogType
    ) -> tuple[str, str]:
        """Extract prompt and response from a single message log.

        Args:
            message_log: OpenAI-style message log.

        Returns:
            Tuple of (prompt_text, response_text).
        """
        prompt_parts = []
        response_parts = []

        for message in message_log:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "user":
                prompt_parts.append(str(content))
            elif role == "assistant":
                response_parts.append(str(content))

        prompt = "\n".join(prompt_parts)
        response = "\n".join(response_parts)

        return prompt, response

    def _safe_score(self, prompt: str, response: str) -> float:
        """Safely compute score with validation and error handling.

        Args:
            prompt: The prompt text.
            response: The response text.

        Returns:
            Reward score, or default_reward on error.
        """
        try:
            # Validate first
            if not self.validate_response(prompt, response):
                return self.config.default_reward

            # Compute score
            return self.score(prompt, response)

        except Exception as e:
            logger.warning(
                f"Environment '{self.name}' score() failed: {e}. "
                f"Returning default_reward={self.config.default_reward}"
            )
            return self.config.default_reward

    def _compute_rewards_async(
        self, prompts: List[str], responses: List[str]
    ) -> List[float]:
        """Compute rewards using async score function.

        Args:
            prompts: List of prompts.
            responses: List of responses.

        Returns:
            List of reward values.
        """
        loop = asyncio.new_event_loop()
        try:
            rewards = loop.run_until_complete(
                asyncio.gather(
                    *[
                        self.score_async(p, r)
                        for p, r in zip(prompts, responses)
                    ]
                )
            )
            return list(rewards)
        finally:
            loop.close()

    def shutdown(self) -> None:
        """Clean up resources.

        Called when the environment is no longer needed.
        """
        self.teardown()

        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.shutdown()


# =============================================================================
# Convenience Classes
# =============================================================================


class SimpleEnvironment(Environment[None]):
    """Simplified Environment that uses None for metadata.

    This is a convenience class for environments that don't need
    to track metadata between steps.

    Example:
        >>> class MyReward(SimpleEnvironment):
        ...     def score(self, prompt: str, response: str) -> float:
        ...         return len(response) / 100
    """

    pass


class StatefulEnvironment(Environment[Dict[str, Any]]):
    """Environment that maintains state via dict metadata.

    This is a convenience class for environments that need to
    track state between steps using a dictionary.

    Example:
        >>> class GameEnvironment(StatefulEnvironment):
        ...     def score(self, prompt: str, response: str) -> float:
        ...         # Access state via self.current_metadata
        ...         turn = self.current_metadata.get("turn", 0)
        ...         return self._evaluate_move(prompt, response, turn)
    """

    pass
