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

"""Factory for creating WeightSynchronizer instances.

Selects the appropriate weight synchronizer based on the deployment
topology (colocated vs. non-colocated) and the generation backend
(vLLM uses IPC/ZMQ, SGLang uses HTTP, non-colocated uses NCCL).
"""

from typing import Any, Optional

from nemo_rl.models.generation.constants import (
    MEGATRON_BACKEND,
    SGLANG_BACKEND,
    VLLM_BACKEND,
)
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


def create_weight_synchronizer(
    policy: Any,
    generation: Any,
    generation_backend: str,
    colocated: bool,
    train_cluster: Optional[Any] = None,
    inference_cluster: Optional[Any] = None,
    refit_buffer_size_gb: Optional[int] = None,
) -> WeightSynchronizer:
    """Create the appropriate WeightSynchronizer for the given deployment.

    Args:
        policy: Policy object (ColocatablePolicyInterface).
        generation: Generation object (GenerationInterface).
        generation_backend: Name of the generation backend ("vllm", "sglang", "megatron").
        colocated: Whether policy and generation share the same GPUs.
        train_cluster: RayVirtualCluster for training workers (required for non-colocated).
        inference_cluster: RayVirtualCluster for inference workers (required for non-colocated).
        refit_buffer_size_gb: Optional fixed buffer size for IPC weight staging.

    Returns:
        A WeightSynchronizer instance appropriate for the deployment topology.

    Raises:
        NotImplementedError: If the requested configuration is not supported.
        ValueError: If required arguments are missing.
    """
    _SUPPORTED_BACKENDS = {VLLM_BACKEND, SGLANG_BACKEND, MEGATRON_BACKEND}
    if generation_backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown generation backend {generation_backend!r}. "
            f"Supported backends: {sorted(_SUPPORTED_BACKENDS)}"
        )

    if refit_buffer_size_gb is not None and refit_buffer_size_gb <= 0:
        raise ValueError("refit_buffer_size_gb must be > 0")

    if not colocated:
        if generation_backend == SGLANG_BACKEND:
            raise NotImplementedError(
                "SGLang does not support non-colocated inference mode."
            )
        if train_cluster is None or inference_cluster is None:
            raise ValueError(
                "train_cluster and inference_cluster are required "
                "for non-colocated weight synchronization."
            )

        from nemo_rl.weight_sync.collective_weight_synchronizer import (
            CollectiveWeightSynchronizer,
        )

        return CollectiveWeightSynchronizer(
            policy=policy,
            generation=generation,
            train_cluster=train_cluster,
            inference_cluster=inference_cluster,
        )

    if generation_backend == SGLANG_BACKEND:
        from nemo_rl.weight_sync.http_weight_synchronizer import (
            HTTPWeightSynchronizer,
        )

        return HTTPWeightSynchronizer(
            policy=policy,
            generation=generation,
            refit_buffer_size_gb=refit_buffer_size_gb,
        )

    from nemo_rl.weight_sync.ipc_weight_synchronizer import (
        IPCWeightSynchronizer,
    )

    return IPCWeightSynchronizer(
        policy=policy,
        generation=generation,
        refit_buffer_size_gb=refit_buffer_size_gb,
    )
