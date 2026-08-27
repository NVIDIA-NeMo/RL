# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import dataclass
from typing import Any, NotRequired, Optional, TypedDict, Union

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# Routed-expert index tensors ([seq, layers, topk]) are carried in the narrowest
# signed dtype that fits ids 0..num_experts-1 plus the -1 missing-route sentinel:
# int8 for <=128 experts (e.g. Qwen3-MoE), int16 for <=32768 (e.g. DeepSeek-V3),
# int32 beyond. This shrinks message logs, transports, replay buffers, and
# checkpoints 2-4x vs int32. The Megatron replay install converts to int64 at the
# gather site, so training math is unaffected. When the expert count cannot be
# determined, fall back to int16.
ROUTED_EXPERTS_FALLBACK_DTYPE = torch.int16

# A routed-expert row whose every top-k slot is this value means "no route was
# captured for this token"; the Megatron replay install falls back to the model's
# own router for those rows. Partially-negative rows are rejected as corruption.
ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL = -1

_ROUTED_EXPERTS_DTYPE_NAMES = {
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
}


def get_num_routed_experts(hf_config: Any) -> Optional[int]:
    """Best-effort read of the routed-expert count from a HF model config.

    Checks the attribute names used by the common MoE architectures (Qwen-MoE,
    DeepSeek, Mixtral), including nested ``text_config`` for VLMs. Returns None
    for dense models or unrecognized configs.
    """
    for owner in (hf_config, getattr(hf_config, "text_config", None)):
        if owner is None:
            continue
        for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
            value = getattr(owner, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def resolve_routed_experts_dtype(num_experts: Optional[int]) -> torch.dtype:
    """Return the narrowest signed dtype that fits expert ids and the -1 sentinel."""
    if num_experts is None:
        return ROUTED_EXPERTS_FALLBACK_DTYPE
    if num_experts - 1 <= torch.iinfo(torch.int8).max:
        return torch.int8
    if num_experts - 1 <= torch.iinfo(torch.int16).max:
        return torch.int16
    return torch.int32


def resolve_routed_experts_dtype_name_for_model(model_name: str) -> str:
    """Resolve the routed-experts carry dtype name ("int8"/"int16"/"int32") for a model.

    Used where only the model name is available (e.g. building the NeMo-Gym env
    config on the driver). Falls back to the default dtype name if the config
    cannot be loaded.
    """
    # Deferred import: transformers config loading is only needed for this sizing.
    from transformers import AutoConfig

    try:
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except (OSError, ValueError):
        return _ROUTED_EXPERTS_DTYPE_NAMES[ROUTED_EXPERTS_FALLBACK_DTYPE]
    return _ROUTED_EXPERTS_DTYPE_NAMES[
        resolve_routed_experts_dtype(get_num_routed_experts(hf_config))
    ]


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


class CheckpointEngineConfig(TypedDict):
    """Normalized internal configuration for checkpoint-engine refit."""

    # "nixl" or a "module:ClassName" path to a CheckpointEngine implementation
    backend: str
    # fraction of total GPU memory used by each transfer bucket
    update_weights_bucket_memory_ratio: float
    # per-backend constructor kwargs, keyed by the configured backend string
    engine_kwargs: dict[str, dict[str, Any]]


class GenerationConfig(TypedDict):
    """Configuration for generation."""

    backend: str
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int | None
    # Validation-only sampling. The exemplar YAMLs default these to the train
    # values above via interpolation (${.temperature}, ...), so validation
    # samples exactly like training unless overridden. Only honored on the
    # NeMo-Gym vLLM rollout path (guarded in grpo.setup()).
    val_temperature: float
    val_top_p: float
    val_top_k: int | None
    model_name: NotRequired[str]  # Not Required b/c GRPO writes this
    stop_token_ids: list[int] | None
    stop_strings: list[str] | None
    bad_words: NotRequired[list[str] | None]
    colocated: NotRequired[ColocationConfig]
    port_range_low: NotRequired[int]
    port_range_high: NotRequired[int]
    use_async_rollouts: NotRequired[bool]
    # This isn't meant to be passed by the user, but is populated by nemo_rl.models.generation.__init__.configure_generation_config
    _pad_token_id: NotRequired[int]
    # MTP draft weights arrive via refit if the trainer trains the MTP layer.
    _mtp_weights_from_refit: NotRequired[bool]
    # Internal debug-only measurement of exact Ray generation arguments.
    # Populated from grpo.debug_payload_metrics; not meant to be set by the user.
    _debug_payload_metrics: NotRequired[bool]


def should_use_async_rollouts(
    generation_config: GenerationConfig | None,
) -> bool:
    """Determine whether a generation backend uses asynchronous rollouts."""
    if generation_config is None:
        return False
    backend = generation_config.get("backend", "")

    if backend == "dynamo":
        return True

    if backend == "sglang":
        return bool(generation_config.get("use_async_rollouts", False))

    if backend == "vllm":
        return bool(generation_config.get("vllm_cfg", {}).get("async_engine", False))

    if backend == "trtllm":
        assert generation_config.get("trtllm_cfg", {}).get("async_engine", False), (
            "TRT-LLM backend requires trtllm_cfg.async_engine=true; the "
            "synchronous engine path (async_engine=false) is no longer supported."
        )
        return True

    if backend == "megatron":
        mcore_cfg = generation_config.get("mcore_generation_config", {})
        assert mcore_cfg.get("async_engine") is None, (
            "Megatron Inference always uses the async engine. The parameter "
            "policy.generation.mcore_generation_config.async_engine was removed."
        )
        return True

    return False


@dataclass
class GenerationSamplingParams:
    """Sampling profile threaded explicitly through rollout entry points.

    Rollout callers construct one from the relevant ``GenerationConfig``
    fields (train or validation) so the sampling used for a rollout is
    visible at the call site instead of flowing through config side-channels.
    Named to distinguish it from ``TrainingSamplingParams`` (train-time logit
    filtering) and vLLM's own ``SamplingParams``.
    """

    temperature: float
    top_p: float
    top_k: int | None

    @classmethod
    def from_generation_config(
        cls, generation_config: "GenerationConfig"
    ) -> "GenerationSamplingParams":
        """Build the train-time sampling profile from a generation config."""
        return cls(
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            top_k=generation_config["top_k"],
        )


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
    - truncated: Boolean tensor indicating if each sequence was truncated (hit max_tokens limit)
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

    truncated:
    [False, True]  # Example 2 was truncated (hit max_tokens limit without EOS)
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
    routed_experts: NotRequired[torch.Tensor]
    r3_routed_experts_missing_routes: NotRequired[torch.Tensor]
    r3_routed_experts_expected_routes: NotRequired[torch.Tensor]
    r3_routed_experts_actual_routes: NotRequired[torch.Tensor]
    truncated: NotRequired[
        torch.Tensor
    ]  # Whether each sequence was truncated and hit max_tokens without stop token
    __extra__: Any


@dataclass(frozen=True)
class CollectiveSenderSpec:
    """Policy-side protocol and packing geometry for NCCL weight transfer."""

    nccl_peer: str = "nemo"
    buffer_size_bytes: int | None = None
    num_buffers: int | None = None


class GenerationInterface(ABC):
    """Abstract base class defining the interface for RL policies."""

    @abstractmethod
    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
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
        """Ready the engine for a generation phase (start or wake it).

        Idempotent wake: calling this on an already-running engine must be safe and cheap.
        """
        pass

    @abstractmethod
    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Wind down after a generation phase.

        Callers may pass `release_gpu` (keyword-only, default True):
        True means the caller needs the GPUs for itself (a training step or a checkpoint save),
        so even a colocated engine must fully stand down;
        False means the phase is merely over, and a colocated engine must keep serving
        usable with no intervening prepare_for_generation.
        Only the colocated Megatron backend honors the flag today; other backends
        ignore it, as do engines on dedicated GPUs.
        """
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        """Shut down generation resources; repeated calls must be safe."""
        pass

    def pause_generation(self, mode: str) -> None:
        """Pause in-flight generation on the backend."""
        raise NotImplementedError

    def continue_generation(self) -> None:
        """Resume previously paused generation on the backend."""
        raise NotImplementedError

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

    def get_collective_sender_spec(self) -> CollectiveSenderSpec:
        """Return policy-side NCCL protocol and packed-buffer requirements."""
        return CollectiveSenderSpec()

    def get_inference_world_size(self) -> int | None:
        """Return a backend-specific collective world size when required."""
        return None

    def prepare_nccl_reshard_refit_info(self, refit_info: dict) -> None:
        """Prepare per-layer param metadata for nccl_reshard-based refit."""
        raise NotImplementedError

    def nccl_reshard_refit(self) -> list[ray.ObjectRef]:
        """Receive weights from training workers via nccl_reshard."""
        raise NotImplementedError

    def attach_fleet_health(self, monitor: Any, selector: Any) -> None:
        """Route this backend's shard selection through fleet health.

        Declared here rather than discovered with ``hasattr`` at the call site, so an
        unsupported backend says so itself and the capability is greppable from the
        interface. Same shape as the refit hooks above.

        Args:
            monitor: ``GenerationFleetHealth`` owning shard eligibility, which the
                backend also reports observed failures and successes to.
            selector: ``HealthyShardSelector`` picking among the serving shards.
        """
        raise NotImplementedError(
            "async_rl.generation_fleet_health.enabled=true is not supported for the "
            f"{type(self).__name__} generation backend"
        )

    # Optional hook; backends may override to invalidate any reusable caches
    # (e.g., vLLM prefix/KV caches) after weight updates.
    def invalidate_kv_cache(self) -> bool:
        return False

    def blocks_training(self) -> bool:
        """Whether this engine must stand down before a training step.

        True when generation shares GPUs with training (colocated): the
        training loop then pauses collection and winds the engine down
        before training. Engines on dedicated GPUs never block training.
        """
        return False

    def wake_carries_weight_updates(self) -> bool:
        """Whether prepare_for_generation alone serves the latest weights.

        True when waking the engine suffices for it to serve weights updated while it slept
        (colocated Megatron: the wake reshards, or the engine shares the training tensors outright).
        The async loop may then defer a wake past a checkpoint save and advance
        the collector's weight version with no explicit transfer.
        Backends whose wake does not reload weights must return False so the loop refits instead.
        """
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
