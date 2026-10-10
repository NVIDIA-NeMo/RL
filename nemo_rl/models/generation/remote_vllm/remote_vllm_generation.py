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

import secrets
from typing import TYPE_CHECKING, Any

import ray

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.generation.remote_vllm.client import RemoteVllmClient
from nemo_rl.models.generation.remote_vllm.config import RemoteVllmServiceConfig
from nemo_rl.models.generation.remote_vllm.preflight import (
    preflight_remote_vllm_service,
)
from nemo_rl.models.generation.remote_vllm.token_capture_bridge import (
    RemoteVllmTokenCaptureBridge,
    set_remote_token_capture_weight_version,
)

if TYPE_CHECKING:
    from nemo_rl.algorithms.single_controller_utils.config import MasterConfig


class RemoteVllmGeneration(GenerationInterface):
    """Controller-side handle for a stock OpenAI-compatible vLLM server.

    NeMo-Gym sends rollout traffic directly to ``dp_openai_server_base_urls``.
    This object intentionally implements no native Ray generation path; it owns
    only lifecycle and weight-reload control calls.
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
        if not master_config.policy.get("megatron_cfg", {}).get("enabled"):
            raise ValueError(
                "remote_vllm checkpoint export currently requires a Megatron policy"
            )
        RemoteVllmServiceConfig.model_validate(generation["remote_vllm_cfg"])

    def __init__(self, config: dict[str, Any]) -> None:
        self.cfg = config
        self.remote_config = RemoteVllmServiceConfig.model_validate(
            config["remote_vllm_cfg"]
        )
        self.client = RemoteVllmClient(self.remote_config)
        self.dp_openai_server_base_urls: list[str] = [self.remote_config.base_url]
        self.weight_synchronizer = None
        self._paused_for_refit = False
        self._token_capture_bridge: RemoteVllmTokenCaptureBridge | None = None
        self._token_capture_bridge_url: str | None = None
        self._token_capture_auth_token: str | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Serialize only the bridge control coordinates into SingleController."""
        state = self.__dict__.copy()
        # The driver owns the live uvicorn thread, socket, and locks. The
        # SingleController copy only needs the authenticated HTTP coordinates
        # used to rotate the version stamped on subsequently staged calls.
        state["_token_capture_bridge"] = None
        return state

    def load_and_start(self) -> None:
        """Match the deferred local-vLLM setup hook; the service is already live."""
        preflight_remote_vllm_service(self.remote_config)

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        del ip, port, world_size, train_world_size
        raise NotImplementedError("remote_vllm refit uses an HF checkpoint, not NCCL")

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
        if self._token_capture_bridge is not None:
            self._token_capture_bridge.stop()
            self._token_capture_bridge = None
        self._token_capture_bridge_url = None
        self._token_capture_auth_token = None
        return True

    def setup_token_capture(
        self,
        dp_cfg: dict[str, Any],
        staging_partition: str,
        *,
        dp_client: Any,
    ) -> None:
        """Bridge external serving processes to controller-owned token staging."""
        del dp_cfg
        if (
            self._token_capture_bridge is not None
            or self._token_capture_bridge_url is not None
        ):
            raise RuntimeError("remote vLLM token capture is already configured")
        auth_token = secrets.token_hex(32)
        bridge = RemoteVllmTokenCaptureBridge(
            dp_client=dp_client,
            staging_partition=staging_partition,
            auth_token=auth_token,
        )
        bridge.start()
        assert bridge.base_url is not None
        try:
            self.client.configure_token_capture(
                bridge_url=bridge.base_url,
                auth_token=auth_token,
            )
        except RuntimeError:
            bridge.stop()
            raise
        self._token_capture_bridge = bridge
        self._token_capture_bridge_url = bridge.base_url
        self._token_capture_auth_token = auth_token

    def set_rollout_weight_version(self, version: int) -> None:
        """Rotate the version stamped by the controller-side capture bridge."""
        if self._token_capture_bridge is not None:
            self._token_capture_bridge.set_weight_version(version)
            return
        if (
            self._token_capture_bridge_url is None
            or self._token_capture_auth_token is None
        ):
            raise RuntimeError("remote vLLM token capture is not configured")
        set_remote_token_capture_weight_version(
            bridge_url=self._token_capture_bridge_url,
            auth_token=self._token_capture_auth_token,
            version=version,
        )

    def pause_generation(self, mode: str) -> None:
        self.client.pause(mode=mode, clear_cache=True)
        self._paused_for_refit = True

    def continue_generation(self) -> None:
        self.client.resume()
        self._paused_for_refit = False

    def pause_generation_for_refit(self, *, clear_cache: bool) -> bool:
        self.client.pause(
            mode=self.remote_config.refit.inflight_policy,
            clear_cache=clear_cache,
        )
        self._paused_for_refit = True
        return True

    def resume_generation_after_refit(self) -> bool:
        self.client.resume()
        self._paused_for_refit = False
        return True

    def reload_weights(self, weights_path: str) -> None:
        if not self._paused_for_refit:
            raise RuntimeError("External vLLM must be paused before reloading weights")
        self.client.reload_weights(weights_path)

    def invalidate_kv_cache(self) -> bool:
        self.client.reset_prefix_cache()
        return True
