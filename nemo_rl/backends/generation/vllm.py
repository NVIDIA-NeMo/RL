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
"""vLLM-based generation backend implementation.

This module provides the VLLMBackend class that implements the GenerationBackend
protocol using vLLM for high-throughput inference.

Example:
    >>> from nemo_rl.backends.generation import VLLMBackend, GenerationBackendConfig
    >>>
    >>> backend = VLLMBackend()
    >>> config = GenerationBackendConfig(
    ...     backend_type='vllm',
    ...     model_name='meta-llama/Llama-2-7b-hf',
    ...     max_new_tokens=512,
    ...     temperature=0.7,
    ... )
    >>> backend.setup(config)
    >>> outputs = backend.generate(prompts)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from nemo_rl.backends.generation.base import (
    GenerationBackend,
    GenerationBackendConfig,
    register_generation_backend,
)

if TYPE_CHECKING:
    import ray

    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
    from nemo_rl.models.generation.interfaces import (
        GenerationDatumSpec,
        GenerationOutputSpec,
    )


@register_generation_backend("vllm")
class VLLMBackend:
    """Generation backend using vLLM for high-throughput inference.

    This backend leverages vLLM's optimized inference engine for fast
    text generation. It supports tensor parallelism and various sampling
    strategies.

    Features:
        - High-throughput inference with PagedAttention
        - Continuous batching for efficiency
        - Tensor parallelism support
        - Multiple sampling strategies (greedy, top-k, top-p)
        - KV cache management

    Attributes:
        config: The backend configuration.
        generation_interface: The underlying vLLM generation interface.

    Example:
        >>> backend = VLLMBackend()
        >>> backend.setup(GenerationBackendConfig(
        ...     model_name='meta-llama/Llama-2-7b-hf',
        ...     max_new_tokens=512,
        ...     temperature=0.7,
        ... ))
        >>> outputs = backend.generate(prompts)
    """

    def __init__(
        self,
        cluster: Optional["RayVirtualCluster"] = None,
    ):
        """Initialize VLLMBackend.

        Args:
            cluster: Optional pre-existing Ray virtual cluster.
        """
        self._config: Optional[GenerationBackendConfig] = None
        self._cluster = cluster
        self._generation_interface: Optional[Any] = None
        self._is_initialized = False
        self._vllm_config: Optional[dict[str, Any]] = None

    def setup(self, config: GenerationBackendConfig) -> None:
        """Initialize the vLLM backend with configuration.

        Args:
            config: Generation backend configuration.

        Raises:
            ValueError: If configuration is invalid.
            RuntimeError: If vLLM initialization fails.
        """
        if config.backend_type != "vllm":
            raise ValueError(
                f"VLLMBackend received config with backend_type='{config.backend_type}'. "
                "Expected 'vllm'."
            )

        self._config = config

        # Build vLLM config from backend config
        self._vllm_config = self._build_vllm_config(config)

        self._is_initialized = True

    def _build_vllm_config(self, config: GenerationBackendConfig) -> dict[str, Any]:
        """Build vLLM configuration from backend config.

        Args:
            config: Generation backend configuration.

        Returns:
            vLLM configuration dictionary.
        """
        # Extract vllm-specific kwargs
        vllm_kwargs = config.backend_kwargs.get("vllm_cfg", {})

        return {
            "backend": "vllm",
            "model_name": config.model_name,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "stop_token_ids": config.stop_token_ids,
            "stop_strings": config.stop_strings,
            "vllm_cfg": {
                "tensor_parallel_size": config.tensor_parallel_size,
                "pipeline_parallel_size": config.pipeline_parallel_size,
                "expert_parallel_size": vllm_kwargs.get("expert_parallel_size", 1),
                "async_engine": vllm_kwargs.get("async_engine", False),
                "gpu_memory_utilization": vllm_kwargs.get("gpu_memory_utilization", 0.9),
                "max_model_len": vllm_kwargs.get("max_model_len", None),
                "enforce_eager": vllm_kwargs.get("enforce_eager", False),
                "dtype": vllm_kwargs.get("dtype", "auto"),
                "quantization": vllm_kwargs.get("quantization", None),
                "enable_prefix_caching": vllm_kwargs.get("enable_prefix_caching", False),
                **{k: v for k, v in vllm_kwargs.items() if k not in [
                    "tensor_parallel_size", "pipeline_parallel_size",
                    "expert_parallel_size", "async_engine", "gpu_memory_utilization",
                    "max_model_len", "enforce_eager", "dtype", "quantization",
                    "enable_prefix_caching"
                ]},
            },
            "colocated": config.backend_kwargs.get("colocated", {"enabled": False}),
        }

    def generate(
        self,
        prompts: "BatchedDataDict[GenerationDatumSpec]",
        greedy: bool = False,
        sampling_params: Optional[dict[str, Any]] = None,
    ) -> "BatchedDataDict[GenerationOutputSpec]":
        """Generate text from input prompts using vLLM.

        Args:
            prompts: BatchedDataDict containing input_ids and input_lengths.
            greedy: If True, use greedy decoding.
            sampling_params: Optional override for sampling parameters.

        Returns:
            BatchedDataDict containing generated outputs.

        Raises:
            RuntimeError: If backend is not initialized.
        """
        self._check_initialized()

        if self._generation_interface is None:
            raise RuntimeError(
                "Generation interface not set. Initialize with a cluster first."
            )

        return self._generation_interface.generate(prompts, greedy=greedy)

    def update_weights(self, state_dict: dict[str, Any]) -> None:
        """Update model weights from a state dictionary.

        Args:
            state_dict: Dictionary containing model state.

        Raises:
            RuntimeError: If backend is not initialized.
        """
        self._check_initialized()

        if self._generation_interface is None:
            return

        # Use the appropriate weight update method based on configuration
        if hasattr(self._generation_interface, "update_weights_from_collective"):
            import ray
            ray.get(self._generation_interface.update_weights_from_collective())
        elif hasattr(self._generation_interface, "update_weights_via_ipc_zmq"):
            import ray
            ray.get(self._generation_interface.update_weights_via_ipc_zmq())

    def prepare_for_generation(self) -> None:
        """Prepare the backend for generation mode."""
        self._check_initialized()

        if self._generation_interface is not None:
            self._generation_interface.prepare_for_generation()

    def finish_generation(self) -> None:
        """Clean up after generation."""
        self._check_initialized()

        if self._generation_interface is not None:
            self._generation_interface.finish_generation()

    def invalidate_kv_cache(self) -> bool:
        """Invalidate the KV cache after weight updates.

        Returns:
            True if cache was invalidated successfully.
        """
        self._check_initialized()

        if self._generation_interface is not None:
            return self._generation_interface.invalidate_kv_cache()
        return True

    def shutdown(self) -> None:
        """Clean up resources and shut down the backend."""
        self._generation_interface = None
        self._is_initialized = False

    def _check_initialized(self) -> None:
        """Check if the backend is initialized.

        Raises:
            RuntimeError: If backend is not initialized.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "VLLMBackend is not initialized. Call setup() first."
            )

    @property
    def is_initialized(self) -> bool:
        """Check if the backend has been initialized."""
        return self._is_initialized

    @property
    def backend_type(self) -> str:
        """Return the backend type identifier."""
        return "vllm"

    @property
    def config(self) -> Optional[GenerationBackendConfig]:
        """Return the current configuration."""
        return self._config

    @property
    def vllm_config(self) -> Optional[dict[str, Any]]:
        """Return the vLLM configuration dictionary."""
        return self._vllm_config

    def set_generation_interface(self, interface: Any) -> None:
        """Set the underlying generation interface.

        This is typically called by the trainer to inject the actual
        vLLM generation interface.

        Args:
            interface: The vLLM generation interface.
        """
        self._generation_interface = interface

    @property
    def requires_kv_scale_sync(self) -> bool:
        """Whether the backend requires KV cache scales synchronization."""
        if self._generation_interface is not None:
            return getattr(self._generation_interface, "requires_kv_scale_sync", False)
        return False
