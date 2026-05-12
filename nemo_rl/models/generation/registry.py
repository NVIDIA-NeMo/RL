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

"""Generation backend registry.

Provides a registry/factory that maps backend name strings (e.g.
``"vllm"``, ``"sglang"``) to their concrete
:class:`~nemo_rl.models.generation.interfaces.GenerationInterface`
classes and config types.  Algorithm code uses :func:`create_generation`
instead of importing backend modules directly, keeping it decoupled from
any particular backend implementation.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypedDict

from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.constants import SGLANG_BACKEND, VLLM_BACKEND
from nemo_rl.models.generation.interfaces import GenerationInterface


class _BackendEntry(TypedDict):
    factory: Callable[..., GenerationInterface]
    config_type: Optional[type]


_REGISTRY: dict[str, _BackendEntry] = {}


def register_generation_backend(
    name: str,
    factory: Callable[..., GenerationInterface],
    config_type: Optional[type] = None,
) -> None:
    """Register a generation backend.

    Args:
        name: Backend identifier (e.g. ``"vllm"``).
        factory: Callable that creates a ``GenerationInterface`` instance.
            Must accept at least ``cluster`` and ``config`` keyword
            arguments.
        config_type: Optional ``TypedDict`` subclass used for casting the
            generation config before passing it to *factory*.
    """
    _REGISTRY[name] = _BackendEntry(factory=factory, config_type=config_type)


def get_registered_backends() -> list[str]:
    """Return names of all registered generation backends."""
    return list(_REGISTRY.keys())


def create_generation(
    backend: str,
    cluster: RayVirtualCluster,
    config: dict[str, Any],
) -> GenerationInterface:
    """Instantiate a registered generation backend.

    Args:
        backend: Backend identifier (must have been registered via
            :func:`register_generation_backend`).
        cluster: The :class:`RayVirtualCluster` to host the workers.
        config: Generation configuration dict.  If the backend was
            registered with a *config_type*, the dict is cast to that
            type before being forwarded to the factory.

    Returns:
        A fully-constructed :class:`GenerationInterface` instance.

    Raises:
        ValueError: If *backend* has not been registered.
    """
    _ensure_builtins_registered()

    if backend not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown generation backend {backend!r}. "
            f"Registered backends: {available}"
        )

    entry = _REGISTRY[backend]
    if entry["config_type"] is not None:
        from typing import cast

        config = cast(entry["config_type"], config)

    return entry["factory"](cluster=cluster, config=config)


# ---------------------------------------------------------------------------
# Built-in backends (lazy-registered on first use to avoid heavy imports at
# module load time)
# ---------------------------------------------------------------------------

_BUILTINS_REGISTERED = False


def _ensure_builtins_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True

    def _vllm_factory(
        cluster: RayVirtualCluster, config: Any
    ) -> GenerationInterface:
        from nemo_rl.models.generation.vllm import VllmGeneration

        return VllmGeneration(cluster=cluster, config=config)

    def _sglang_factory(
        cluster: RayVirtualCluster, config: Any
    ) -> GenerationInterface:
        from nemo_rl.models.generation.sglang import SGLangGeneration

        return SGLangGeneration(cluster=cluster, config=config)

    from nemo_rl.models.generation.vllm import VllmConfig
    from nemo_rl.models.generation.sglang import SGLangConfig

    register_generation_backend(VLLM_BACKEND, _vllm_factory, config_type=VllmConfig)
    register_generation_backend(SGLANG_BACKEND, _sglang_factory, config_type=SGLangConfig)
