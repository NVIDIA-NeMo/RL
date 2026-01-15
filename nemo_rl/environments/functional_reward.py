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
"""Functional reward wrapper for simple callable reward functions.

This module provides a FunctionalRewardWrapper that adapts simple callable
functions to the EnvironmentInterface, allowing users to define rewards
without creating a full Environment class.

Supported function signatures:
    - fn(prompt: str, response: str) -> float
    - fn(prompt: str, response: str) -> dict[str, float]
    - async fn(prompt: str, response: str) -> float
    - fn(prompts: list[str], responses: list[str]) -> list[float] (batched)

Example:
    >>> from nemo_rl.environments.functional_reward import FunctionalRewardWrapper
    >>>
    >>> # Simple reward function
    >>> def length_reward(prompt: str, response: str) -> float:
    ...     return len(response) / 100.0
    >>>
    >>> # Wrap it for use with trainer
    >>> env = FunctionalRewardWrapper(length_reward)
    >>>
    >>> # Or use directly with trainer.fit()
    >>> trainer.fit(dataset="my_data", reward_fn=length_reward)
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

if TYPE_CHECKING:
    from torch import Tensor


# Type aliases for reward functions
RewardValue = Union[float, Dict[str, float]]
SyncRewardFn = Callable[[str, str], RewardValue]
AsyncRewardFn = Callable[[str, str], "asyncio.Future[RewardValue]"]
BatchedRewardFn = Callable[[List[str], List[str]], List[RewardValue]]
RewardFnType = Union[SyncRewardFn, AsyncRewardFn, BatchedRewardFn]


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for reward functions.

    Reward functions can have various signatures:
    - fn(prompt: str, response: str) -> float
    - fn(prompt: str, response: str) -> dict[str, float]
    """

    def __call__(self, prompt: str, response: str) -> RewardValue:
        """Compute reward for a prompt-response pair."""
        ...


@dataclass
class FunctionalRewardConfig:
    """Configuration for FunctionalRewardWrapper.

    Attributes:
        batch_size: Number of samples to process in parallel.
        max_workers: Maximum number of threads for parallel processing.
        timeout: Timeout in seconds for async operations.
        default_reward: Default reward value for failed computations.
    """

    batch_size: int = 32
    max_workers: int = 8
    timeout: float = 60.0
    default_reward: float = 0.0


class FunctionalRewardWrapper(EnvironmentInterface[None]):
    """Adapts simple callable reward functions to EnvironmentInterface.

    This class allows users to use simple functions for reward computation
    without implementing the full EnvironmentInterface. It supports:

    - Single-sample functions: fn(prompt, response) -> reward
    - Batched functions: fn(prompts, responses) -> rewards
    - Sync and async functions
    - Automatic batching for efficiency
    - Multiple reward values (dict[str, float])

    Example:
        >>> # Simple reward function
        >>> def my_reward(prompt: str, response: str) -> float:
        ...     if "correct" in response.lower():
        ...         return 1.0
        ...     return 0.0
        >>>
        >>> # Wrap for environment interface
        >>> env = FunctionalRewardWrapper(my_reward)
        >>>
        >>> # Or use reward that returns multiple values
        >>> def multi_reward(prompt: str, response: str) -> dict[str, float]:
        ...     return {
        ...         "correctness": 1.0 if "correct" in response else 0.0,
        ...         "length": min(len(response) / 100, 1.0),
        ...     }
        >>> env = FunctionalRewardWrapper(multi_reward)
    """

    def __init__(
        self,
        reward_fn: RewardFnType,
        config: Optional[FunctionalRewardConfig] = None,
        name: str = "functional_reward",
        validate_signature: bool = True,
    ):
        """Initialize the FunctionalRewardWrapper.

        Args:
            reward_fn: The reward function to wrap. Can be:
                - fn(prompt, response) -> float
                - fn(prompt, response) -> dict[str, float]
                - async fn(prompt, response) -> float
                - fn(prompts, responses) -> list[float] (batched)
            config: Configuration for reward computation.
            name: Name of this reward wrapper.
            validate_signature: Whether to validate the function signature.

        Raises:
            TypeError: If reward_fn is not callable.
            ValueError: If signature validation fails.
        """
        # Validate callable
        if not callable(reward_fn):
            raise TypeError(
                f"reward_fn must be callable, got {type(reward_fn).__name__}. "
                f"Expected a function with signature: def fn(prompt: str, response: str) -> float"
            )

        self.reward_fn = reward_fn
        self.config = config or FunctionalRewardConfig()
        self.name = name

        # Detect function signature
        self._is_async = asyncio.iscoroutinefunction(reward_fn)
        self._is_batched = self._detect_batched_signature(reward_fn)
        self._returns_dict = False  # Will be detected on first call

        # Validate signature if requested
        if validate_signature:
            self._validate_signature(reward_fn)

        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

    def _validate_signature(self, fn: RewardFnType) -> None:
        """Validate the reward function signature.

        Args:
            fn: The reward function to validate.

        Raises:
            ValueError: If signature is invalid.
        """
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())

            # Check minimum parameter count
            if len(params) < 2:
                raise ValueError(
                    f"Reward function '{fn.__name__}' must accept at least 2 parameters "
                    f"(prompt, response), but has {len(params)}. "
                    f"Expected signature: def {fn.__name__}(prompt: str, response: str) -> float"
                )

            # Check first two parameters accept strings
            for i, param in enumerate(params[:2]):
                param_name = param.name
                hint = param.annotation

                # Skip if no annotation (we'll assume it's correct)
                if hint == inspect.Parameter.empty:
                    continue

                # Check if it's a string type or list of strings (for batched)
                valid_types = (str, List[str], list, "str", "list[str]")
                if hint not in valid_types and not isinstance(hint, str):
                    import logging
                    logging.warning(
                        f"Parameter '{param_name}' has type hint '{hint}', "
                        f"expected 'str' or 'list[str]'. Proceeding anyway."
                    )

            # Check return type if annotated
            return_hint = sig.return_annotation
            if return_hint != inspect.Parameter.empty:
                valid_returns = (
                    float, int, Dict[str, float], dict,
                    List[float], list, "float", "dict[str, float]",
                )
                if return_hint not in valid_returns:
                    import logging
                    logging.warning(
                        f"Return type '{return_hint}' may not be compatible. "
                        f"Expected float, dict[str, float], or list[float]."
                    )

        except (ValueError, TypeError) as e:
            # Re-raise ValueError, convert TypeError
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to validate reward function signature: {e}")

    def _detect_batched_signature(self, fn: RewardFnType) -> bool:
        """Detect if the function expects batched inputs.

        Args:
            fn: The reward function to check.

        Returns:
            True if function expects lists as inputs.
        """
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            if len(params) >= 2:
                # Check type hints
                first_hint = params[0].annotation
                if first_hint in (List[str], list, "list[str]"):
                    return True
        except (ValueError, TypeError):
            pass
        return False

    def _extract_prompt_response(
        self, message_log: LLMMessageLogType
    ) -> tuple[str, str]:
        """Extract prompt and response from message log.

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

    def _compute_single_reward(
        self, prompt: str, response: str
    ) -> Union[float, Dict[str, float]]:
        """Compute reward for a single prompt-response pair.

        Args:
            prompt: The prompt text.
            response: The response text.

        Returns:
            Reward value (float or dict).
        """
        try:
            result = self.reward_fn(prompt, response)
            return result
        except Exception as e:
            import logging

            logging.warning(f"Reward computation failed: {e}")
            return self.config.default_reward

    async def _compute_single_reward_async(
        self, prompt: str, response: str
    ) -> Union[float, Dict[str, float]]:
        """Compute reward asynchronously for a single pair.

        Args:
            prompt: The prompt text.
            response: The response text.

        Returns:
            Reward value.
        """
        try:
            result = await self.reward_fn(prompt, response)
            return result
        except Exception as e:
            import logging

            logging.warning(f"Async reward computation failed: {e}")
            return self.config.default_reward

    def _compute_batched_rewards(
        self, prompts: List[str], responses: List[str]
    ) -> List[Union[float, Dict[str, float]]]:
        """Compute rewards for a batch of prompt-response pairs.

        Args:
            prompts: List of prompts.
            responses: List of responses.

        Returns:
            List of reward values.
        """
        if self._is_batched:
            # Function expects batched inputs
            try:
                return self.reward_fn(prompts, responses)
            except Exception as e:
                import logging

                logging.warning(f"Batched reward computation failed: {e}")
                return [self.config.default_reward] * len(prompts)
        else:
            # Process in parallel using thread pool
            futures = [
                self._executor.submit(self._compute_single_reward, p, r)
                for p, r in zip(prompts, responses)
            ]
            return [f.result(timeout=self.config.timeout) for f in futures]

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata: List[None],
    ) -> EnvironmentReturn[None]:
        """Run a step in the environment, computing rewards.

        Args:
            message_log_batch: Batch of message logs.
            metadata: Batch of metadata (unused for functional rewards).

        Returns:
            EnvironmentReturn with computed rewards.
        """
        batch_size = len(message_log_batch)

        # Extract prompts and responses
        prompts = []
        responses = []
        for msg_log in message_log_batch:
            prompt, response = self._extract_prompt_response(msg_log)
            prompts.append(prompt)
            responses.append(response)

        # Compute rewards
        if self._is_async:
            # Run async reward computation
            loop = asyncio.new_event_loop()
            try:
                rewards = loop.run_until_complete(
                    asyncio.gather(
                        *[
                            self._compute_single_reward_async(p, r)
                            for p, r in zip(prompts, responses)
                        ]
                    )
                )
            finally:
                loop.close()
        else:
            rewards = self._compute_batched_rewards(prompts, responses)

        # Convert rewards to tensor
        if isinstance(rewards[0], dict):
            self._returns_dict = True
            # For dict rewards, use first value as primary reward
            reward_tensor = torch.tensor(
                [list(r.values())[0] if isinstance(r, dict) else r for r in rewards],
                dtype=torch.float32,
            )
        else:
            reward_tensor = torch.tensor(rewards, dtype=torch.float32)

        # All episodes terminate after reward computation
        terminateds = torch.ones(batch_size, dtype=torch.bool)

        # Empty observations (no continuation)
        observations = [{"role": "assistant", "content": ""} for _ in range(batch_size)]

        return EnvironmentReturn(
            observations=observations,
            metadata=[None] * batch_size,
            next_stop_strings=[None] * batch_size,
            rewards=reward_tensor,
            terminateds=terminateds,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        """Post-process batch and return metrics.

        Args:
            batch: The batch to process.

        Returns:
            Tuple of (processed_batch, metrics).
        """
        metrics = {
            "reward_wrapper_name": self.name,
            "is_async": self._is_async,
            "is_batched": self._is_batched,
        }

        if "rewards" in batch:
            rewards = batch["rewards"]
            metrics["mean_reward"] = rewards.mean().item()
            metrics["std_reward"] = rewards.std().item()
            metrics["min_reward"] = rewards.min().item()
            metrics["max_reward"] = rewards.max().item()

        return batch, metrics

    def __call__(
        self,
        prompts: Union[BatchedDataDict, List[str]],
        responses: Union[BatchedDataDict, List[str]],
    ) -> torch.Tensor:
        """Compute rewards for prompts and responses.

        This method provides a simple interface for direct reward computation
        without going through the full environment interface.

        Args:
            prompts: Either a BatchedDataDict or list of prompt strings.
            responses: Either a BatchedDataDict or list of response strings.

        Returns:
            Tensor of rewards.
        """
        # Convert BatchedDataDict to strings if needed
        if isinstance(prompts, BatchedDataDict):
            prompt_strs = self._batch_to_strings(prompts)
        else:
            prompt_strs = prompts

        if isinstance(responses, BatchedDataDict):
            response_strs = self._batch_to_strings(responses)
        else:
            response_strs = responses

        rewards = self._compute_batched_rewards(prompt_strs, response_strs)

        return torch.tensor(
            [r if isinstance(r, (int, float)) else list(r.values())[0] for r in rewards],
            dtype=torch.float32,
        )

    def _batch_to_strings(self, batch: BatchedDataDict) -> List[str]:
        """Convert a BatchedDataDict to list of strings.

        Args:
            batch: The batch to convert.

        Returns:
            List of decoded strings.
        """
        # This would need tokenizer access - placeholder implementation
        if "text" in batch:
            return list(batch["text"])
        return [""] * batch.size

    def shutdown(self) -> None:
        """Shut down the executor."""
        self._executor.shutdown(wait=False)

    def __repr__(self) -> str:
        return (
            f"FunctionalRewardWrapper("
            f"name={self.name!r}, "
            f"is_async={self._is_async}, "
            f"is_batched={self._is_batched})"
        )


def create_reward_wrapper(
    reward_fn: RewardFnType,
    name: str = "custom_reward",
    **config_kwargs: Any,
) -> FunctionalRewardWrapper:
    """Factory function to create a FunctionalRewardWrapper.

    Args:
        reward_fn: The reward function to wrap.
        name: Name for the wrapper.
        **config_kwargs: Additional configuration options.

    Returns:
        Configured FunctionalRewardWrapper.

    Example:
        >>> env = create_reward_wrapper(
        ...     lambda p, r: len(r) / 100,
        ...     name="length_reward",
        ...     max_workers=4,
        ... )
    """
    config = FunctionalRewardConfig(**config_kwargs)
    return FunctionalRewardWrapper(reward_fn, config=config, name=name)


# Convenience decorators for reward functions
def reward_function(name: str = "custom"):
    """Decorator to mark a function as a reward function.

    Args:
        name: Name for the reward function.

    Example:
        >>> @reward_function("correctness")
        ... def check_correct(prompt: str, response: str) -> float:
        ...     return 1.0 if "correct" in response else 0.0
    """

    def decorator(fn: RewardFnType) -> FunctionalRewardWrapper:
        return FunctionalRewardWrapper(fn, name=name)

    return decorator


def batched_reward(name: str = "batched"):
    """Decorator for batched reward functions.

    Args:
        name: Name for the reward function.

    Example:
        >>> @batched_reward("batch_scorer")
        ... def score_batch(prompts: list[str], responses: list[str]) -> list[float]:
        ...     return [len(r) / 100 for r in responses]
    """

    def decorator(fn: BatchedRewardFn) -> FunctionalRewardWrapper:
        wrapper = FunctionalRewardWrapper(fn, name=name)
        wrapper._is_batched = True
        return wrapper

    return decorator


def wrap_reward_callable(
    reward_fn: Union[RewardFnType, FunctionalRewardWrapper, None],
    name: str = "reward",
) -> Optional[FunctionalRewardWrapper]:
    """Wrap a callable as a FunctionalRewardWrapper if needed.

    This is a convenience function that handles various input types:
    - If None, returns None
    - If already a FunctionalRewardWrapper, returns as-is
    - If a callable, wraps it in a FunctionalRewardWrapper

    Args:
        reward_fn: The reward function to wrap. Can be:
            - None (returns None)
            - FunctionalRewardWrapper (returned as-is)
            - callable (wrapped in FunctionalRewardWrapper)
        name: Name for the wrapper (if creating new one).

    Returns:
        FunctionalRewardWrapper or None.

    Raises:
        TypeError: If reward_fn is not callable or None.

    Example:
        >>> # Works with simple functions
        >>> def my_reward(prompt: str, response: str) -> float:
        ...     return len(response) / 100.0
        >>> env = wrap_reward_callable(my_reward)
        >>> 
        >>> # Works with lambdas
        >>> env = wrap_reward_callable(lambda p, r: 1.0 if "correct" in r else 0.0)
        >>> 
        >>> # Pass-through for existing wrappers
        >>> existing = FunctionalRewardWrapper(my_reward)
        >>> same = wrap_reward_callable(existing)
        >>> assert same is existing
    """
    if reward_fn is None:
        return None

    if isinstance(reward_fn, FunctionalRewardWrapper):
        return reward_fn

    if callable(reward_fn):
        return FunctionalRewardWrapper(reward_fn, name=name)

    raise TypeError(
        f"reward_fn must be callable or None, got {type(reward_fn).__name__}. "
        f"Expected: def reward(prompt: str, response: str) -> float"
    )


def validate_reward_callable(
    reward_fn: RewardFnType,
) -> tuple[bool, str]:
    """Validate a reward callable signature without wrapping it.

    This is useful for validating reward functions before training
    without actually creating a wrapper.

    Args:
        reward_fn: The reward function to validate.

    Returns:
        Tuple of (is_valid, error_message).
        If is_valid is True, error_message is empty string.

    Example:
        >>> def good_reward(p: str, r: str) -> float:
        ...     return 1.0
        >>> is_valid, error = validate_reward_callable(good_reward)
        >>> assert is_valid and error == ""
        >>> 
        >>> def bad_reward(x):  # Only 1 parameter
        ...     return 1.0
        >>> is_valid, error = validate_reward_callable(bad_reward)
        >>> assert not is_valid
    """
    if not callable(reward_fn):
        return False, f"Expected callable, got {type(reward_fn).__name__}"

    try:
        sig = inspect.signature(reward_fn)
        params = list(sig.parameters.values())

        if len(params) < 2:
            fn_name = getattr(reward_fn, "__name__", "function")
            return False, (
                f"Reward function '{fn_name}' must accept at least 2 parameters "
                f"(prompt, response), but has {len(params)}. "
                f"Expected: def {fn_name}(prompt: str, response: str) -> float"
            )

        return True, ""

    except (ValueError, TypeError) as e:
        return False, f"Could not inspect signature: {e}"
