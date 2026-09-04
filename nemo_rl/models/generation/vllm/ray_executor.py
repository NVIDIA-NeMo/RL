# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""vLLM Ray executor that propagates NeMo RL's nested runtime contract."""

from __future__ import annotations

import os
from typing import Any

import ray._private.ray_constants as ray_constants
from vllm.v1.executor.ray_executor import RayDistributedExecutor

from nemo_rl.models.generation.vllm.nested_runtime_env import (
    NestedRuntimeEnvContractError,
    load_nested_runtime_env_contract,
    merge_nested_runtime_env,
)


_RAY_ENABLE_UV_RUN_RUNTIME_ENV = "RAY_ENABLE_UV_RUN_RUNTIME_ENV"


class NemoRayDistributedExecutor(RayDistributedExecutor):
    """Inject the authenticated runtime environment into vLLM Ray workers."""

    def _init_workers_ray(
        self,
        placement_group: Any,
        **ray_remote_kwargs: Any,
    ) -> Any:
        if os.environ.get(_RAY_ENABLE_UV_RUN_RUNTIME_ENV) != "0":
            raise NestedRuntimeEnvContractError(
                f"{_RAY_ENABLE_UV_RUN_RUNTIME_ENV} must equal '0' in "
                "vLLM's EngineCore process"
            )
        if (
            ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV is not False
        ):
            raise NestedRuntimeEnvContractError(
                "Ray imported with RAY_ENABLE_UV_RUN_RUNTIME_ENV enabled; "
                "the nested worker executable cannot be guaranteed"
            )

        desired_runtime_env, _ = load_nested_runtime_env_contract()
        ray_remote_kwargs["runtime_env"] = merge_nested_runtime_env(
            ray_remote_kwargs.get("runtime_env"),
            desired_runtime_env,
        )
        return super()._init_workers_ray(
            placement_group,
            **ray_remote_kwargs,
        )
