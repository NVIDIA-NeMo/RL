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
"""CPU Ray actors for metadata-only token-capture finalization."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Optional

import ray
import torch

from nemo_rl.data_plane import DataPlaneConfig, build_data_plane_client
from nemo_rl.experience.rollout_reassembler import FinalizedGroup, RolloutReassembler

# Field names whose values are per-token and therefore large, but whose Python
# type is indistinguishable from metadata -- a list[int] of token ids looks just
# like a short list of ids. assert_metadata_only() below already rejects tensors
# and unrecognised types; this list is only for heavy values that would otherwise
# pass it. Add a name here whenever a new per-token field could reach an RPC
# boundary, and update the dataclass inventory test that guards this file.
_FORBIDDEN_RPC_KEYS = frozenset(
    {
        "input_ids",
        "token_ids",
        "token_ids_delta",
        "token_mask",
        "token_mask_delta",
        "generation_logprobs",
        "generation_logprobs_delta",
        "generation_log_probs_delta",
        "logprobs",
        "logprobs_delta",
        "routed_experts",
    }
)


@dataclass(frozen=True)
class ReassemblyRequest:
    """Metadata-only input for one prompt group's finalization."""

    group_id: str
    rollout_ids: tuple[str, ...]
    receipts: tuple[Optional[dict[str, Any]], ...]
    rewards: tuple[float, ...]
    fallback_weight_version: int
    # Stable dataset prompt index; pack_payload stamps it on every row's tag.
    prompt_idx: int
    # Per-rollout advantage-stage flag the receipt path already knows at
    # dispatch time. The finalizer publishes it with the rebuilt rows so the
    # train pump reads the same ``mask_sample`` field as the native path
    # (SingleController reads it unconditionally).
    mask_sample: tuple[bool, ...]
    # Dataset-level loss weight shared by every completion in this prompt group.
    loss_multiplier: float = 1.0


@dataclass(frozen=True)
class RolloutReassemblerActorConfig:
    """Internal constructor values shared by every finalizer actor."""

    partition_id: str
    staging_partition: str
    pad_token_id: int
    router_replay_enabled: bool
    defer_routed_experts_to_policy: bool
    max_seq_len: int


def assert_metadata_only(value: Any, *, path: str = "rpc") -> None:
    """Reject tensors and known heavy row fields reachable from an RPC graph."""
    if isinstance(value, torch.Tensor):
        raise TypeError(
            f"{path} contains a torch.Tensor with shape {tuple(value.shape)}"
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field_info in fields(value):
            assert_metadata_only(
                getattr(value, field_info.name),
                path=f"{path}.{field_info.name}",
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _FORBIDDEN_RPC_KEYS:
                raise TypeError(f"{path} contains forbidden heavy field {key!r}")
            assert_metadata_only(key, path=f"{path}.key")
            assert_metadata_only(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_metadata_only(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported RPC type {type(value).__name__}")


@ray.remote(
    num_cpus=1,
    num_gpus=0,
    max_restarts=0,
    max_task_retries=0,
)
class RolloutReassemblerActor:  # pragma: no cover
    """Own a connect-only TQ client and lightweight finalizer in one process."""

    def __init__(
        self,
        dp_config: DataPlaneConfig,
        config: RolloutReassemblerActorConfig,
    ) -> None:
        dp_client = build_data_plane_client(dp_config, bootstrap=False)
        self._finalizer = RolloutReassembler(
            dp_client,
            partition_id=config.partition_id,
            staging_partition=config.staging_partition,
            pad_token_id=config.pad_token_id,
            router_replay_enabled=config.router_replay_enabled,
            defer_routed_experts_to_policy=config.defer_routed_experts_to_policy,
            max_seq_len=config.max_seq_len,
        )

    def finalize(self, request: ReassemblyRequest) -> FinalizedGroup:
        """Finalize one request without allowing tensor payloads across Ray RPC."""
        assert_metadata_only(request)
        if not (
            len(request.rollout_ids)
            == len(request.receipts)
            == len(request.rewards)
            == len(request.mask_sample)
        ):
            raise ValueError(
                "finalizer request rollout_ids, receipts, rewards, and "
                "mask_sample must be parallel"
            )
        result = self._finalizer.finalize_group(
            request.group_id,
            list(request.rollout_ids),
            list(request.receipts),
            list(request.rewards),
            mask_sample=list(request.mask_sample),
            fallback_weight_version=request.fallback_weight_version,
            prompt_idx=request.prompt_idx,
            loss_multiplier=request.loss_multiplier,
        )
        assert_metadata_only(result)
        return result


def create_rollout_reassembler_actors(
    dp_config: DataPlaneConfig,
    config: RolloutReassemblerActorConfig,
    *,
    num_workers: int,
) -> list[Any]:
    """Construct the fixed validation pool after TQ partitions are registered."""
    if num_workers <= 0:
        raise ValueError(f"num_reassembler_workers must be positive, got {num_workers}")
    return [
        RolloutReassemblerActor.remote(dp_config, config) for _ in range(num_workers)
    ]
