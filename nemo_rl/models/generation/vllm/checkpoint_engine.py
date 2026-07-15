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

import asyncio
import gc
import time
from typing import TYPE_CHECKING, Any

import torch

from nemo_rl.models.generation.vllm.refit_loader import VllmShardedRefitMixin
from nemo_rl.utils.nsys import wrap_with_nvtx_name

if TYPE_CHECKING:
    from nemo_rl.utils.checkpoint_engines.base import CheckpointEngine


def maybe_preinit_nixl_for_vllm_worker(
    worker_wrapper: Any,
    *,
    backend_name: str,
    backend_init_params: dict[str, Any] | None = None,
) -> None:  # pragma: no cover
    from nemo_rl.utils.checkpoint_engines.nixl import preinit_nixl_agent

    vars(worker_wrapper)["_nrl_nixl_preinit_agent"] = preinit_nixl_agent(
        backend_name=backend_name, backend_init_params=backend_init_params
    )


def global_rollout_rank(rank_prefix: int, rollout_world_size: int) -> int:
    rank = torch.distributed.get_rank()
    if torch.distributed.get_world_size() == rollout_world_size:
        return rank
    return rank_prefix + rank


class VllmCheckpointEngineMixin(VllmShardedRefitMixin):
    """Checkpoint-engine lifecycle for vLLM workers."""

    checkpoint_engine: "CheckpointEngine"

    def init_checkpoint_engine(
        self, backend: str, bucket_size_bytes: int, engine_kwargs: dict[str, Any]
    ) -> None:  # pragma: no cover
        if getattr(self, "checkpoint_engine", None) is not None:
            return

        from nemo_rl.utils.checkpoint_engines.base import create_checkpoint_engine

        self.checkpoint_engine = create_checkpoint_engine(
            backend,
            bucket_size_bytes=bucket_size_bytes,
            engine_kwargs=engine_kwargs,
        )

    def prepare_checkpoint_engine(self) -> Any:  # pragma: no cover
        metadata = self.checkpoint_engine.prepare()
        if isinstance(metadata, dict):
            metadata = {**metadata, "rank": torch.distributed.get_rank()}
            if self.checkpoint_engine.shard_hf_weights:
                metadata["weight_layout"] = self._checkpoint_engine_weight_layout()
        return metadata

    def init_checkpoint_engine_process_group(
        self,
        rank_prefix: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[Any],
    ) -> None:  # pragma: no cover
        self.checkpoint_engine.init_rollout_process_group(
            rollout_rank=global_rollout_rank(rank_prefix, rollout_world_size),
            train_world_size=train_world_size,
            rollout_world_size=rollout_world_size,
            metadata=metadata,
        )

    def finalize_checkpoint_engine(self) -> None:  # pragma: no cover
        checkpoint_engine = getattr(self, "checkpoint_engine", None)
        if checkpoint_engine is not None:
            checkpoint_engine.finalize()

    async def _update_weights_from_checkpoint_engine_async(
        self,
    ) -> bool:  # pragma: no cover
        loaded_tensors = 0
        loaded_bytes = 0
        loaded_batches = 0
        load_time = 0.0
        start_time = time.time()

        async for weight_batch in self.checkpoint_engine.receive_weight_batches():
            loaded_batches += 1
            loaded_tensors += len(weight_batch)
            loaded_bytes += sum(weight.nbytes for _name, weight in weight_batch)

            load_start = time.time()
            self._load_weights(weight_batch)
            torch.cuda.current_stream().synchronize()
            load_time += time.time() - load_start
            del weight_batch

        self._maybe_process_fp8_kv_cache()

        if self.checkpoint_engine.cleanup_after_load:
            gc.collect()
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        loaded_gib = loaded_bytes / (1024 * 1024 * 1024)
        print(
            "[vLLM refit] Loaded "
            f"{loaded_tensors} tensors in {loaded_batches} batches via checkpoint "
            f"engine; bytes={loaded_gib:.2f}GiB total={total_time:.2f}s "
            f"receive={max(total_time - load_time, 0.0):.2f}s load={load_time:.2f}s"
        )
        return True

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_checkpoint_engine"
    )
    def update_weights_from_checkpoint_engine(self) -> bool:  # pragma: no cover
        return asyncio.run(self._update_weights_from_checkpoint_engine_async())


class VllmCheckpointEngineRpcMixin:
    """Dispatch checkpoint-engine calls through a synchronous vLLM engine."""

    def checkpoint_engine_rpc(
        self, checkpoint_method: str, method_args: tuple[Any, ...] = ()
    ) -> Any:  # pragma: no cover
        result = self.llm.collective_rpc(checkpoint_method, args=method_args)
        if checkpoint_method == "update_weights_from_checkpoint_engine":
            return all(item for item in result if item is not None)
        return result


class VllmAsyncCheckpointEngineRpcMixin:
    """Dispatch checkpoint-engine calls through an asynchronous vLLM engine."""

    async def checkpoint_engine_rpc_async(
        self, checkpoint_method: str, method_args: tuple[Any, ...] = ()
    ) -> Any:  # pragma: no cover
        from nemo_rl.models.generation.vllm.collective_rpc import (
            resolve_collective_rpc_result,
        )

        result = await self.llm.collective_rpc(checkpoint_method, args=method_args)
        result = await resolve_collective_rpc_result(result)
        if checkpoint_method == "update_weights_from_checkpoint_engine":
            return all(item for item in result if item is not None)
        return result
