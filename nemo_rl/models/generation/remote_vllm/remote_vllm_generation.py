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

"""GenerationInterface facade for a vLLM server outside NeMo-RL's Ray cluster."""

from typing import TYPE_CHECKING, Any

import ray

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    CollectiveSenderSpec,
    GenerationConfig,
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.generation.remote_vllm.client import RemoteVllmClient
from nemo_rl.models.generation.remote_vllm.preflight import (
    preflight_remote_vllm_service,
)
from nemo_rl.models.generation.remote_vllm_config import RemoteVllmServiceConfig

if TYPE_CHECKING:
    from nemo_rl.algorithms.single_controller_utils.config import MasterConfig


class RemoteVllmGeneration(GenerationInterface):
    """Controller-side handle for a stock OpenAI-compatible vLLM server.

    NeMo-Gym sends rollout traffic directly to ``dp_openai_server_base_urls``.
    This object intentionally implements no native Ray generation path; it owns
    only lifecycle and NCCL weight-transfer control calls.
    """

    @classmethod
    def validate_settings(cls, master_config: "MasterConfig") -> None:
        generation = master_config.policy["generation"]
        if not master_config.env.get("should_use_nemo_gym"):
            raise ValueError(
                "policy.generation.backend='remote_vllm' currently requires "
                "env.should_use_nemo_gym=true"
            )
        if generation["colocated"]["enabled"]:
            raise ValueError("remote_vllm generation cannot be colocated")
        RemoteVllmServiceConfig.model_validate(generation["remote_vllm_cfg"])
        if generation.get("refit_transport") is not None:
            raise ValueError(
                "remote_vllm uses native NCCL refit; "
                "policy.generation.refit_transport must be unset"
            )
        if not master_config.policy.get("megatron_cfg", {}).get("enabled"):
            raise ValueError(
                "remote_vllm NCCL refit currently requires a Megatron policy"
            )
        if master_config.token_capture.enabled:
            raise NotImplementedError(
                "remote_vllm does not yet support Single Controller token capture"
            )

    def __init__(self, config: GenerationConfig) -> None:
        self.cfg = config
        self.remote_config = RemoteVllmServiceConfig.model_validate(
            config["remote_vllm_cfg"]
        )
        self.client = RemoteVllmClient(self.remote_config)
        self.dp_openai_server_base_urls: list[str] = [self.remote_config.base_url]
        self.weight_synchronizer = None
        self._paused_for_refit = False

    @property
    def http_server_api_key(self) -> str | None:
        """Give Gym the same credential used for external control requests."""
        return self.client.api_key

    def load_and_start(self) -> None:
        """Match the deferred local-vLLM setup hook; the service is already live."""
        preflight_remote_vllm_service(self.remote_config)

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        del ip, port, world_size, train_world_size
        raise NotImplementedError(
            "remote_vllm builds its NCCL group in RemoteVllmNcclWeightSynchronizer"
        )

    def get_collective_sender_spec(self) -> CollectiveSenderSpec:
        """VLLM's NCCL peer protocol and the packed geometry both sides use."""
        refit = self.remote_config.refit
        return CollectiveSenderSpec(
            nccl_peer="vllm",
            buffer_size_bytes=refit.nccl_packed_buffer_size_bytes,
            num_buffers=refit.nccl_packed_num_buffers,
            use_legacy_packing=True,
        )

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool
    ) -> BatchedDataDict[GenerationOutputSpec]:
        del data, greedy
        raise NotImplementedError(
            "remote_vllm generation is available only through NeMo-Gym's "
            "OpenAI-compatible HTTP path"
        )

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return True

    def shutdown(self) -> bool:
        # The heterogeneous-job launcher, not NeMo-RL, owns the server process.
        return True

    def pause_generation(self, mode: str) -> None:
        self.client.pause(mode=mode, clear_cache=True)
        self._paused_for_refit = True

    def continue_generation(self) -> None:
        self.client.resume()
        self._paused_for_refit = False

    def pause_generation_for_refit(
        self, *, clear_cache: bool, timeout_s: float | None = None
    ) -> bool:
        self.client.pause(
            mode=self.remote_config.refit.inflight_policy,
            clear_cache=clear_cache,
            timeout_s=timeout_s,
        )
        self._paused_for_refit = True
        return True

    def resume_generation_after_refit(self, *, timeout_s: float | None = None) -> bool:
        self.client.resume(timeout_s=timeout_s)
        self._paused_for_refit = False
        return True

    def invalidate_kv_cache(self) -> bool:
        self.client.reset_prefix_cache()
        return True
