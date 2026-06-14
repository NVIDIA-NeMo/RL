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
(vLLM uses IPC/ZMQ, SGLang uses HTTP, non-colocated uses NCCL or a
configured checkpoint-engine backend).
"""

from collections.abc import Callable
from typing import Any, Optional, cast

from nemo_rl.models.generation.constants import (
    MEGATRON_BACKEND,
    SGLANG_BACKEND,
    VLLM_BACKEND,
)
from nemo_rl.models.generation.interfaces import RefitCheckpointEngineConfig
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
        train_cluster: RayVirtualCluster for training workers (required for
            non-colocated collective synchronization).
        inference_cluster: RayVirtualCluster for inference workers (required
            for non-colocated collective synchronization).
        refit_buffer_size_gb: Optional fixed buffer size for IPC weight staging.

    Returns:
        A WeightSynchronizer instance appropriate for the deployment topology.

    Raises:
        NotImplementedError: If the requested configuration is not supported.
        ValueError: If required arguments are missing.
    """
    supported_backends = {VLLM_BACKEND, SGLANG_BACKEND, MEGATRON_BACKEND}
    if generation_backend not in supported_backends:
        raise ValueError(
            f"Unknown generation backend {generation_backend!r}. "
            f"Supported backends: {sorted(supported_backends)}"
        )

    get_checkpoint_engine_config = cast(
        Callable[[], RefitCheckpointEngineConfig | None] | None,
        getattr(generation, "get_checkpoint_engine_config", None),
    )
    checkpoint_engine_config = (
        get_checkpoint_engine_config()
        if get_checkpoint_engine_config is not None
        else None
    )
    if checkpoint_engine_config is not None and checkpoint_engine_config["enabled"]:
        if colocated:
            raise ValueError(
                "checkpoint-engine refit is only supported for non-colocated "
                "generation. Set policy.generation.colocated.enabled=false or "
                "disable policy.generation.checkpoint_engine.enabled."
            )
        if generation_backend == SGLANG_BACKEND:
            raise NotImplementedError(
                "SGLang does not support checkpoint-engine non-colocated refit."
            )

        from nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer import (
            CheckpointEngineWeightSynchronizer,
        )

        return CheckpointEngineWeightSynchronizer(
            policy=policy,
            generation=generation,
            checkpoint_engine_config=checkpoint_engine_config,
        )

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
        )

    if refit_buffer_size_gb is not None and refit_buffer_size_gb <= 0:
        raise ValueError("refit_buffer_size_gb must be > 0")
    from nemo_rl.weight_sync.ipc_weight_synchronizer import (
        IPCWeightSynchronizer,
    )

    return IPCWeightSynchronizer(
        policy=policy,
        generation=generation,
        refit_buffer_size_gb=refit_buffer_size_gb,
    )
