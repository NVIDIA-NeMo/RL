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
"""N named VllmGeneration replica groups of one policy behind one interface.

Groups get disjoint GPUs (one RayVirtualCluster each) and heterogeneous
``vllm_cfg``/``vllm_kwargs`` overrides; refit joins them into the single
train+inference NCCL broadcast world via per-group ``rank_offset``.
v1 scope: GRPO + NeMo Gym, non-colocated, default NCCL refit.
"""

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, cast

import ray

from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration

_OVERRIDABLE_KEYS = {"vllm_cfg", "vllm_kwargs"}
_FORBIDDEN_VLLM_CFG_KEYS = {"async_engine", "expose_http_server", "max_model_len"}


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        out[key] = (
            _merge(base[key], value)
            if isinstance(base.get(key), dict) and isinstance(value, dict)
            else value
        )
    return out


def build_group_configs(
    generation_config: VllmConfig,
) -> Optional[dict[str, VllmConfig]]:
    """Build one merged VllmConfig per server group; None when unconfigured.

    Call after setup() has finished mutating the generation config
    (model_name, hf_overrides, router replay) so the copies inherit it.
    """
    groups = generation_config.get("server_groups")
    if not groups:
        return None
    names = [g["name"] for g in groups]
    assert len(set(names)) == len(names), f"duplicate server_group names: {names}"
    base = {k: v for k, v in generation_config.items() if k != "server_groups"}
    configs: dict[str, VllmConfig] = {}
    for g in groups:
        overrides = g.get("overrides") or {}
        assert set(overrides) <= _OVERRIDABLE_KEYS, (
            f"server_group '{g['name']}': only {sorted(_OVERRIDABLE_KEYS)} may be "
            f"overridden, got {sorted(overrides)}"
        )
        forbidden = set(overrides.get("vllm_cfg", {})) & _FORBIDDEN_VLLM_CFG_KEYS
        assert not forbidden, (
            f"server_group '{g['name']}': may not override {sorted(forbidden)}"
        )
        assert "max_model_len" not in overrides.get("vllm_kwargs", {}), (
            f"server_group '{g['name']}': max_model_len must be uniform across groups"
        )
        configs[g["name"]] = cast(VllmConfig, _merge(copy.deepcopy(base), overrides))
    return configs


class MultiVllmGeneration(GenerationInterface):
    """Pure fan-out over named VllmGeneration groups; generation goes over HTTP."""

    def __init__(
        self,
        clusters: dict[str, RayVirtualCluster],
        configs: dict[str, VllmConfig],
        defer_model_load: bool = False,
    ):
        self.groups = {
            name: VllmGeneration(
                clusters[name],
                cfg,
                name_prefix=f"vllm_{name}",
                defer_model_load=defer_model_load,
            )
            for name, cfg in configs.items()
        }
        self.cfg = next(iter(configs.values()))

    @property
    def dp_openai_server_base_urls(self) -> list[Optional[str]]:
        return [
            url for g in self.groups.values() for url in g.dp_openai_server_base_urls
        ]

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        futures: list[ray.ObjectRef] = []
        rank_offset = 0
        for g in self.groups.values():
            futures += g.init_collective(
                ip,
                port,
                world_size,
                train_world_size=train_world_size,
                rank_offset=rank_offset,
            )
            rank_offset += len(g.worker_group.workers)
        return futures

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        for g in self.groups.values():
            g.prepare_refit_info(state_dict_info)

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        return [
            f for g in self.groups.values() for f in g.update_weights_from_collective()
        ]

    def load_and_start(self) -> None:
        with ThreadPoolExecutor(max_workers=len(self.groups)) as executor:
            list(executor.map(lambda g: g.load_and_start(), self.groups.values()))

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        return all(
            [g.prepare_for_generation(*args, **kwargs) for g in self.groups.values()]
        )

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        return all([g.finish_generation(*args, **kwargs) for g in self.groups.values()])

    def shutdown(self) -> bool:
        return all([g.shutdown() for g in self.groups.values()])

    def clear_logger_metrics(self) -> None:
        for g in self.groups.values():
            g.clear_logger_metrics()

    def get_logger_metrics(self) -> dict[str, Any]:
        return {
            f"{name}/{k}": v
            for name, g in self.groups.items()
            for k, v in g.get_logger_metrics().items()
        }

    def start_gpu_profiling(self) -> None:
        for g in self.groups.values():
            g.start_gpu_profiling()

    def stop_gpu_profiling(self) -> None:
        for g in self.groups.values():
            g.stop_gpu_profiling()

    @property
    def requires_kv_scale_sync(self) -> bool:
        return any(g.requires_kv_scale_sync for g in self.groups.values())

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "MultiVllmGeneration serves generation over HTTP (NeMo Gym) only"
        )
