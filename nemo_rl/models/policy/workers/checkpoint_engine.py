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

import warnings
from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING, Any, Optional, cast

import torch

from nemo_rl.models.generation.vllm.refit_layout import (
    VllmWeightLayout,
    select_hf_weight_for_vllm_target,
)
from nemo_rl.weight_sync.checkpoint_engine_config import (
    checkpoint_engine_refit_config,
)

if TYPE_CHECKING:
    from nemo_rl.utils.checkpoint_engines.base import CheckpointEngine


def maybe_preinit_nixl_checkpoint_engine(config: dict[str, Any]) -> Any:
    """Preinitialize NIXL when checkpoint-engine refit is configured."""
    generation_config = config.get("generation")
    if generation_config is None:
        return None
    checkpoint_config = checkpoint_engine_refit_config(generation_config)
    if checkpoint_config is None or checkpoint_config["backend"] != "nixl":
        return None

    from nemo_rl.utils.checkpoint_engines.nixl import (
        preinit_nixl_agent,
        resolve_nixl_backend_kwargs,
    )

    backend_name, backend_init_params = resolve_nixl_backend_kwargs(
        checkpoint_config["engine_kwargs"]["nixl"]
    )
    return preinit_nixl_agent(
        backend_name=backend_name, backend_init_params=backend_init_params
    )


class DTensorCheckpointEngineSendMixin:
    """Onload DTensor/FSDP2 policy weights for checkpoint-engine transfer."""

    model: torch.nn.Module

    def _prepare_checkpoint_engine_weight_send(self) -> None:
        if self.cpu_offload:
            warnings.warn(
                "cpu_offload adds an onload/offload cycle during non-colocated "
                "checkpoint-engine refit. Disable it unless GPU memory requires it.",
                stacklevel=2,
            )
            self.model = self.move_to_cuda(self.model)

    def _finalize_checkpoint_engine_weight_send(self) -> None:
        if self.cpu_offload:
            self.model = self.move_to_cpu(self.model)

    def _checkpoint_engine_weight_iterator(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> Iterator[tuple[str, torch.Tensor]]:
        if kv_scales is not None:
            raise NotImplementedError(
                "FP8 kvcache is not currently supported for DTensor path, we will support it in the future."
            )
        return self._checkpoint_engine_params()


class MegatronCheckpointEngineSendMixin:
    """Select destination-local Megatron weights for checkpoint-engine transfer."""

    def _checkpoint_engine_weight_iterator(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> Iterator[tuple[str, torch.Tensor]]:
        weights = self._iter_params_with_optional_kv_scales(kv_scales=kv_scales)
        target_layout = self.checkpoint_engine.get_target_weight_layout()
        if target_layout is None:
            return weights

        target_layout = cast(VllmWeightLayout, target_layout)
        return (
            (name, selected)
            for name, tensor in weights
            if (
                selected := select_hf_weight_for_vllm_target(
                    name, tensor, target_layout=target_layout
                )
            )
            is not None
        )


class PolicyCheckpointEngineMixin:
    """Checkpoint-engine lifecycle shared by policy worker implementations."""

    checkpoint_engine: "CheckpointEngine"
    rank: int

    def _checkpoint_engine_weight_iterator(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support checkpoint-engine refit."
        )

    def _prepare_checkpoint_engine_weight_send(self) -> None:
        pass

    def _finalize_checkpoint_engine_weight_send(self) -> None:
        pass

    async def send_weights_via_checkpoint_engine(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        self._prepare_checkpoint_engine_weight_send()
        try:
            with torch.no_grad():
                await self.checkpoint_engine.send_weights(
                    self._checkpoint_engine_weight_iterator(kv_scales=kv_scales)
                )
        finally:
            self._finalize_checkpoint_engine_weight_send()

    async def checkpoint_engine_rpc(
        self, checkpoint_method: str, method_kwargs: Optional[dict[str, Any]] = None
    ) -> Any:
        kwargs = method_kwargs or {}
        if checkpoint_method == "checkpoint_engine_total_memory_bytes":
            device = torch.cuda.current_device()
            return torch.cuda.get_device_properties(device).total_memory
        if checkpoint_method == "init_checkpoint_engine":
            if getattr(self, "checkpoint_engine", None) is None:
                from nemo_rl.utils.checkpoint_engines.base import (
                    create_checkpoint_engine,
                )

                self.checkpoint_engine = create_checkpoint_engine(
                    kwargs["backend"],
                    bucket_size_bytes=kwargs["bucket_size_bytes"],
                    engine_kwargs=kwargs["engine_kwargs"],
                )
            return
        if checkpoint_method == "prepare_checkpoint_engine":
            metadata = self.checkpoint_engine.prepare()
            if isinstance(metadata, dict):
                return {**metadata, "rank": self.rank}
            return metadata
        if checkpoint_method == "init_checkpoint_engine_process_group":
            return self.checkpoint_engine.init_policy_process_group(
                worker_rank=self.rank, **kwargs
            )
        if checkpoint_method == "send_weights_via_checkpoint_engine":
            return await self.send_weights_via_checkpoint_engine(**kwargs)
        if checkpoint_method == "finalize_checkpoint_engine":
            checkpoint_engine = getattr(self, "checkpoint_engine", None)
            if checkpoint_engine is not None:
                checkpoint_engine.finalize()
            return
        return getattr(self.checkpoint_engine, checkpoint_method)(**kwargs)
