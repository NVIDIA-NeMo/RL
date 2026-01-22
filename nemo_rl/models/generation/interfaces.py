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
from abc import ABC, abstractmethod
from typing import Any, NotRequired, TypedDict, Union

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def verify_right_padding(
    data: Union[
        BatchedDataDict["GenerationDatumSpec"], BatchedDataDict["GenerationOutputSpec"]
    ],
    pad_value: int = 0,
    raise_error: bool = True,
) -> tuple[bool, Union[str, None]]:
    """Verify that a tensor is right-padded according to the provided lengths.

    Arguments:
        data: The BatchedDataDict to check, containing either:
            - For GenerationDatumSpec: input_ids and input_lengths
            - For GenerationOutputSpec: output_ids and unpadded_sequence_lengths
        pad_value: The expected padding value (default: 0)
        raise_error: Whether to raise an error if wrong padding is detected

    Returns:
        Tuple of (is_right_padded, error_message)
        - is_right_padded: True if right padding confirmed, False otherwise
        - error_message: None if properly padded, otherwise a description of the issue
    """
    # Extract tensors from the BatchedDataDict
    assert isinstance(data, BatchedDataDict), (
        f"data must be a BatchedDataDict, got type: {type(data)}"
    )

    assert pad_value is not None, (
        "Tokenizer does not have a pad_token_id. \n"
        "Please use the nemo_rl.algorithms.utils.get_tokenizer(...) API which sets pad_token_id if absent."
    )

    # Determine which type of data we're dealing with
    if "input_ids" in data and "input_lengths" in data:
        # GenerationDatumSpec
        tensor = data["input_ids"]
        lengths = data["input_lengths"]
    elif "output_ids" in data and "unpadded_sequence_lengths" in data:
        # GenerationOutputSpec
        tensor = data["output_ids"]
        lengths = data["unpadded_sequence_lengths"]
    else:
        msg = f"Could not find the required pairs of fields. Expected either (input_ids, input_lengths) or (output_ids, unpadded_sequence_lengths). Got keys: {data.keys()}"
        if raise_error:
            raise ValueError(msg)
        return False, msg

    if tensor.ndim != 2:
        msg = f"Expected 2D tensor for padding check, got shape {tensor.shape}"
        if raise_error:
            raise ValueError(msg)
        return False, msg

    batch_size, seq_len = tensor.shape
    if lengths.shape[0] != batch_size:
        msg = f"Mismatch between tensor batch size ({batch_size}) and lengths tensor size ({lengths.shape[0]})"
        if raise_error:
            raise ValueError(msg)
        return False, msg

    # Check each sequence to verify zero padding on the right
    for i in range(batch_size):
        length = lengths[i].item()
        if length > seq_len:
            msg = f"Length {length} at index {i} exceeds tensor sequence dimension {seq_len}"
            if raise_error:
                raise ValueError(msg)
            return False, msg

        # Check that all positions after length are pad_value
        if length < seq_len and not torch.all(tensor[i, length:] == pad_value):
            non_pad_indices = torch.where(tensor[i, length:] != pad_value)[0] + length
            msg = f"Non-padding values found after specified length at index {i}: positions {non_pad_indices.tolist()}"
            if raise_error:
                raise ValueError(msg)
            return False, msg

    return True, None


class ResourcesConfig(TypedDict):
    gpus_per_node: int
    num_nodes: int


class OptionalResourcesConfig(TypedDict):
    # Same as ResourcesConfig, but fields can be null and are validated in grpo.py
    gpus_per_node: int | None
    num_nodes: int | None


class ColocationConfig(TypedDict):
    enabled: bool
    resources: OptionalResourcesConfig


class GenerationConfig(TypedDict):
    """Configuration for generation."""

    backend: str
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int | None
    model_name: NotRequired[str]  # Not Required b/c GRPO writes this
    stop_token_ids: list[int] | None
    stop_strings: list[str] | None
    colocated: NotRequired[ColocationConfig]
    # This isn't meant to be passed by the user, but is populated by nemo_rl.models.generation.__init__.configure_generation_config
    _pad_token_id: NotRequired[int]


class GenerationDatumSpec(TypedDict):
    """Specification for input data required by generation models.

    - input_ids: Tensor of token IDs representing the input sequences (right padded)
    - input_lengths: Tensor containing the actual length of each sequence (without padding)
    - stop_strings: Optional list of strings to stop generation (per sample)
    - __extra__: Additional model-specific data fields

    Example of a batch with 4 entries with different sequence lengths:
    ```
    # Batch of 4 sequences with lengths [3, 5, 2, 4]

    input_ids (padded):
    [
      [101, 2054, 2003,    0,    0],  # Length 3
      [101, 2054, 2003, 2001, 1996],  # Length 5
      [101, 2054,    0,    0,    0],  # Length 2
      [101, 2054, 2003, 2001,    0],  # Length 4
    ]

    input_lengths:
    [3, 5, 2, 4]
    ```

    All functions receiving or returning GenerationDatumSpec should ensure
    right padding is maintained. Use verify_right_padding() to check.
    """

    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    stop_strings: NotRequired[list[str]]
    __extra__: Any


class GenerationOutputSpec(TypedDict):
    """Specification for output data returned by generation models.

    - output_ids: Tensor of token IDs representing the generated sequences (right padded)
    - generation_lengths: Tensor containing the actual length of each generated sequence
    - unpadded_sequence_lengths: Tensor containing the actual length of each input + generated sequence (without padding)
    - logprobs: Tensor of log probabilities for each generated token (right padded with zeros)
    - __extra__: Additional model-specific data fields

    Example of a batch with 2 sequences:
    ```
    # Sample batch with 2 examples
    # - Example 1: Input length 3, generated response length 4
    # - Example 2: Input length 5, generated response length 2

    output_ids (right-padded):
    [
      [101, 2054, 2003, 2023, 2003, 1037, 2200,    0],  # 7 valid tokens (3 input + 4 output)
      [101, 2054, 2003, 2001, 1996, 3014, 2005,    0],  # 7 valid tokens (5 input + 2 output)
    ]

    generation_lengths:
    [4, 2]  # Length of just the generated response part

    unpadded_sequence_lengths:
    [7, 7]  # Length of full valid sequence (input + generated response)

    logprobs (right-padded with zeros):
    [
      [0.0, 0.0, 0.0, -1.2, -0.8, -2.1, -1.5, 0.0],  # First 3 are 0 (input tokens), next 4 are actual logprobs
      [0.0, 0.0, 0.0, 0.0, 0.0, -0.9, -1.7, 0.0],     # First 5 are 0 (input tokens), next 2 are actual logprobs
    ]
    ```

    All functions receiving or returning GenerationOutputSpec should ensure
    right padding is maintained. Use verify_right_padding() to check.
    """

    output_ids: torch.Tensor
    generation_lengths: torch.Tensor  # Length of just the generated response part
    unpadded_sequence_lengths: (
        torch.Tensor
    )  # Length of full valid sequence (input + generated response)
    logprobs: torch.Tensor
    __extra__: Any


class GenerationInterface(ABC):
    """Abstract base class defining the interface for RL policies."""

    @abstractmethod
    def init_collective(
        self, ip: str, port: int, world_size: int
    ) -> list[ray.ObjectRef]:
        """Initialize the collective communication."""
        pass

    @abstractmethod
    def generate(
        self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool
    ) -> BatchedDataDict["GenerationOutputSpec"]:
        pass

    @abstractmethod
    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        pass

    @property
    def requires_kv_scale_sync(self) -> bool:
        """Whether the generation backend requires KV cache scales synchronization."""
        return False

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare the info for refit."""
        raise NotImplementedError

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        """Update the model weights from the given IPC handles."""
        raise NotImplementedError

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        """Update the model weights from collective communication."""
        raise NotImplementedError

    # Optional hook; backends may override to invalidate any reusable caches
    # (e.g., vLLM prefix/KV caches) after weight updates.
    def invalidate_kv_cache(self) -> bool:
        return False

    def clear_logger_metrics(self) -> None:
        """Clear logger metrics for performance reporting.

        This is an optional method that backends can implement to clear
        telemetry metrics. Default implementation does nothing.
        """
        pass

    def get_logger_metrics(self) -> dict[str, Any]:
        """Get logger metrics for performance reporting.

        This is an optional method that backends can implement to collect
        telemetry metrics. Default implementation returns empty dict.

        Returns:
            Dictionary of metrics. Format may vary by backend.
        """
        return {}
