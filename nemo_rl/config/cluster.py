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
"""Cluster configuration for distributed training.

This module provides configuration for cluster resource management,
including GPU allocation, node configuration, and automatic detection.
"""

from __future__ import annotations

import os
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from nemo_rl.config.base import BaseConfig


class ClusterConfig(BaseConfig):
    """Configuration for distributed cluster resources.

    Defines the cluster topology including the number of nodes and GPUs per node.
    Supports automatic detection of available resources.

    Attributes:
        gpus_per_node: Number of GPUs per node (default: 8).
        num_nodes: Number of nodes in the cluster (default: 1).

    Example:
        >>> # Manual configuration
        >>> config = ClusterConfig(num_nodes=4, gpus_per_node=8)

        >>> # Auto-detect available resources
        >>> config = ClusterConfig.auto_detect()
    """

    gpus_per_node: Annotated[int, Field(gt=0, description="Number of GPUs per node")] = 8
    num_nodes: Annotated[int, Field(gt=0, description="Number of nodes in the cluster")] = 1

    @field_validator("gpus_per_node", "num_nodes")
    @classmethod
    def validate_positive(cls, v: int, info) -> int:
        """Ensure values are positive."""
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive, got {v}")
        return v

    @classmethod
    def auto_detect(cls) -> ClusterConfig:
        """Automatically detect cluster configuration.

        Detects the number of available GPUs from CUDA_VISIBLE_DEVICES
        environment variable or defaults to 1 GPU.

        For multi-node setups, uses SLURM environment variables if available.

        Returns:
            ClusterConfig with detected resources.
        """
        # Detect GPUs per node from CUDA_VISIBLE_DEVICES
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_devices:
            gpus_per_node = len(cuda_devices.split(","))
        else:
            # Try to detect using torch if available
            try:
                import torch

                gpus_per_node = torch.cuda.device_count()
            except (ImportError, RuntimeError):
                gpus_per_node = 1

        # Ensure at least 1 GPU
        gpus_per_node = max(1, gpus_per_node)

        # Detect number of nodes from SLURM environment
        num_nodes = 1
        slurm_nnodes = os.environ.get("SLURM_NNODES")
        if slurm_nnodes:
            try:
                num_nodes = int(slurm_nnodes)
            except ValueError:
                pass

        return cls(gpus_per_node=gpus_per_node, num_nodes=num_nodes)

    @property
    def total_gpus(self) -> int:
        """Total number of GPUs in the cluster.

        Returns:
            Total GPU count (num_nodes * gpus_per_node).
        """
        return self.num_nodes * self.gpus_per_node

    @property
    def world_size(self) -> int:
        """World size for distributed training.

        Alias for total_gpus for compatibility with distributed training APIs.

        Returns:
            World size (total number of processes/GPUs).
        """
        return self.total_gpus

    def validate_parallelism(
        self,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int | None = None,
    ) -> bool:
        """Validate that parallelism settings are compatible with cluster.

        Args:
            tensor_parallel_size: Tensor parallel size.
            pipeline_parallel_size: Pipeline parallel size.
            data_parallel_size: Data parallel size (inferred if None).

        Returns:
            True if the configuration is valid.

        Raises:
            ValueError: If parallelism settings exceed available resources.
        """
        model_parallel_size = tensor_parallel_size * pipeline_parallel_size

        if data_parallel_size is None:
            if self.total_gpus % model_parallel_size != 0:
                raise ValueError(
                    f"Total GPUs ({self.total_gpus}) must be divisible by "
                    f"model parallel size ({model_parallel_size} = "
                    f"TP {tensor_parallel_size} × PP {pipeline_parallel_size})"
                )
            data_parallel_size = self.total_gpus // model_parallel_size

        total_required = tensor_parallel_size * pipeline_parallel_size * data_parallel_size

        if total_required > self.total_gpus:
            raise ValueError(
                f"Parallelism requires {total_required} GPUs but only {self.total_gpus} available. "
                f"Reduce tensor_parallel_size ({tensor_parallel_size}), "
                f"pipeline_parallel_size ({pipeline_parallel_size}), "
                f"or data_parallel_size ({data_parallel_size})."
            )

        return True
