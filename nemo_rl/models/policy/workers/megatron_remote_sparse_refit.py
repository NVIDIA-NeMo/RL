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

"""Canonical Hugging Face sparse-refit state for a Megatron policy worker."""

from collections.abc import Iterator
from typing import Any

import torch

from nemo_rl.models.generation.vllm.config import VllmDeltaCompressionConfig
from nemo_rl.utils.weight_transfer_sparse_codec import DeltaCompressionTracker
from nemo_rl.utils.weight_transfer_stream import (
    init_sparse_delta_baseline_from_iterator,
    stream_sparse_delta_payloads_via_s3_manifest,
)
from nemo_rl.utils.weight_transfer_zmq import stream_sparse_delta_payloads_via_zmq


class MegatronRemoteSparseRefit:
    def __init__(self, worker: Any, delta_config: VllmDeltaCompressionConfig) -> None:
        self._worker = worker
        self._tracker = DeltaCompressionTracker(delta_config.model_dump())

    def _iter_params(self) -> Iterator[tuple[str, torch.Tensor]]:
        return self._worker._iter_params_with_optional_kv_scales()

    def initialize_baseline(
        self,
        *,
        shard_rank: int,
        shard_count: int,
        transport: str,
    ) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        init_sparse_delta_baseline_from_iterator(
            self._iter_params(),
            delta_tracker=self._tracker,
            shard_rank=shard_rank,
            shard_count=shard_count,
            transport=transport,
        )
        return {
            name: (tuple(tensor.shape), tensor.dtype)
            for name, tensor in self._tracker.baseline.items()
        }

    def stream(
        self,
        transport: str,
        targets: list[str],
        *,
        transfer_id: str,
        api_key_env_var: str | None,
        timeout_s: float,
        shard_rank: int,
        shard_count: int,
        overwrite_names: list[str],
    ) -> dict[str, int]:
        streamer = {
            "s3": stream_sparse_delta_payloads_via_s3_manifest,
            "zmq": stream_sparse_delta_payloads_via_zmq,
        }[transport]
        self._tracker.overwrite_names = frozenset(overwrite_names)
        result = streamer(
            self._iter_params(),
            delta_tracker=self._tracker,
            transfer_id=transfer_id,
            refit_targets=targets,
            api_key_env_var=api_key_env_var,
            timeout_s=timeout_s,
            shard_rank=shard_rank,
            shard_count=shard_count,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return result

    def finish(self, succeeded: bool) -> None:
        if succeeded:
            self._tracker.on_sync_succeeded()
        else:
            self._tracker.on_sync_failed()
