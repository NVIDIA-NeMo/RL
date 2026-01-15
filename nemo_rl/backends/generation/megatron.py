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
"""Megatron-based generation backend implementation.

This module provides the MegatronInferenceBackend class that implements the
GenerationBackend protocol using NVIDIA Megatron-LM for inference.

Example:
    >>> from nemo_rl.backends.generation import MegatronInferenceBackend, GenerationBackendConfig
    >>>
    >>> backend = MegatronInferenceBackend()
    >>> config = GenerationBackendConfig(
    ...     backend_type='megatron',
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


@register_generation_backend("megatron")
class MegatronInferenceBackend:
    """Generation backend using NVIDIA Megatron-LM for inference.

    This backend leverages Megatron-LM's optimized inference capabilities
    for text generation with large language models. It supports tensor,
    pipeline, and expert parallelism.

    Features:
        - Tensor parallelism
        - Pipeline parallelism
        - Expert parallelism for MoE models
        - Integrated with Megatron training
        - Supports colocated inference with training

    Attributes:
        config: The backend configuration.
        generation_interface: The underlying Megatron generation interface.

    Example:
        >>> backend = MegatronInferenceBackend()
        >>> backend.setup(GenerationBackendConfig(
        ...     backend_type='megatron',
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
        """Initialize MegatronInferenceBackend.

        Args:
            cluster: Optional pre-existing Ray virtual cluster.
        """
        self._config: Optional[GenerationBackendConfig] = None
        self._cluster = cluster
        self._generation_interface: Optional[Any] = None
        self._is_initialized = False
        self._megatron_config: Optional[dict[str, Any]] = None

    def setup(self, config: GenerationBackendConfig) -> None:
        """Initialize the Megatron inference backend with configuration.

        Args:
            config: Generation backend configuration.

        Raises:
            ValueError: If configuration is invalid.
            RuntimeError: If Megatron initialization fails.
        """
        if config.backend_type != "megatron":
            raise ValueError(
                f"MegatronInferenceBackend received config with backend_type='{config.backend_type}'. "
                "Expected 'megatron'."
            )

        self._config = config

        # Build Megatron config from backend config
        self._megatron_config = self._build_megatron_config(config)

        self._is_initialized = True

    def _build_megatron_config(self, config: GenerationBackendConfig) -> dict[str, Any]:
        """Build Megatron configuration from backend config.

        Args:
            config: Generation backend configuration.

        Returns:
            Megatron configuration dictionary.
        """
        # Extract megatron-specific kwargs
        megatron_kwargs = config.backend_kwargs.get("megatron_cfg", {})

        return {
            "backend": "megatron",
            "model_name": config.model_name,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "stop_token_ids": config.stop_token_ids,
            "stop_strings": config.stop_strings,
            "megatron_cfg": {
                "tensor_model_parallel_size": config.tensor_parallel_size,
                "pipeline_model_parallel_size": config.pipeline_parallel_size,
                "expert_model_parallel_size": megatron_kwargs.get(
                    "expert_model_parallel_size", 1
                ),
                "context_parallel_size": megatron_kwargs.get("context_parallel_size", 1),
                "sequence_parallel": megatron_kwargs.get("sequence_parallel", True),
                "use_flash_attn": megatron_kwargs.get("use_flash_attn", True),
                **{k: v for k, v in megatron_kwargs.items() if k not in [
                    "tensor_model_parallel_size", "pipeline_model_parallel_size",
                    "expert_model_parallel_size", "context_parallel_size",
                    "sequence_parallel", "use_flash_attn"
                ]},
            },
            "colocated": config.backend_kwargs.get("colocated", {"enabled": True}),
        }

    def generate(
        self,
        prompts: "BatchedDataDict[GenerationDatumSpec]",
        greedy: bool = False,
        sampling_params: Optional[dict[str, Any]] = None,
    ) -> "BatchedDataDict[GenerationOutputSpec]":
        """Generate text from input prompts using Megatron.

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

        For Megatron, weights are typically shared with the training
        backend when colocated, so this may be a no-op.

        Args:
            state_dict: Dictionary containing model state.

        Raises:
            RuntimeError: If backend is not initialized.
        """
        self._check_initialized()

        # Megatron inference typically shares weights with training when colocated
        # Weight updates happen through the training policy
        pass

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
                "MegatronInferenceBackend is not initialized. Call setup() first."
            )

    @property
    def is_initialized(self) -> bool:
        """Check if the backend has been initialized."""
        return self._is_initialized

    @property
    def backend_type(self) -> str:
        """Return the backend type identifier."""
        return "megatron"

    @property
    def config(self) -> Optional[GenerationBackendConfig]:
        """Return the current configuration."""
        return self._config

    @property
    def megatron_config(self) -> Optional[dict[str, Any]]:
        """Return the Megatron configuration dictionary."""
        return self._megatron_config

    def set_generation_interface(self, interface: Any) -> None:
        """Set the underlying generation interface.

        This is typically called by the trainer to inject the actual
        Megatron generation interface.

        Args:
            interface: The Megatron generation interface.
        """
        self._generation_interface = interface

    @property
    def requires_kv_scale_sync(self) -> bool:
        """Whether the backend requires KV cache scales synchronization."""
        return False
