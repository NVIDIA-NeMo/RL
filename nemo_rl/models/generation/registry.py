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

"""Generation backend registry."""

from typing import Any, Callable

from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.constants import SGLANG_BACKEND, VLLM_BACKEND
from nemo_rl.models.generation.interfaces import GenerationInterface

_REGISTRY: dict[str, Callable[..., GenerationInterface]] = {}


def register_generation_backend(
    name: str,
    factory: Callable[..., GenerationInterface],
) -> None:
    """Register a generation backend factory."""
    _REGISTRY[name] = factory


def get_registered_backends() -> list[str]:
    """Return names of all registered generation backends."""
    _ensure_builtins_registered()
    return list(_REGISTRY.keys())


def create_generation(
    backend: str,
    cluster: RayVirtualCluster,
    config: dict[str, Any],
) -> GenerationInterface:
    """Instantiate a registered generation backend."""
    _ensure_builtins_registered()

    if backend not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown generation backend {backend!r}. Registered backends: {available}"
        )

    return _REGISTRY[backend](cluster=cluster, config=config)


_BUILTINS_REGISTERED = False


def _ensure_builtins_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True

    def vllm_factory(cluster: RayVirtualCluster, config: Any) -> GenerationInterface:
        from nemo_rl.models.generation.vllm import VllmGeneration

        return VllmGeneration(cluster=cluster, config=config)

    def sglang_factory(cluster: RayVirtualCluster, config: Any) -> GenerationInterface:
        from nemo_rl.models.generation.sglang import SGLangGeneration

        return SGLangGeneration(cluster=cluster, config=config)

    register_generation_backend(VLLM_BACKEND, vllm_factory)
    register_generation_backend(SGLANG_BACKEND, sglang_factory)
