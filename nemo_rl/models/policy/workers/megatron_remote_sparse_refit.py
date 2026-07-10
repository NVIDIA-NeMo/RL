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

"""Optional remote sparse-refit state owned by a Megatron policy worker."""

from typing import Any

import torch

from nemo_rl.utils.weight_transfer_remote_sparse import (
    SparseDeltaStreamResult,
    init_sparse_delta_baseline_from_iterator,
    stream_sparse_delta_payloads_via_s3_manifest,
)
from nemo_rl.utils.weight_transfer_sparse_codec import DeltaCompressionTracker
from nemo_rl.utils.weight_transfer_zmq import stream_sparse_delta_payloads_via_zmq


class MegatronRemoteSparseRefit:
    def __init__(self, worker: Any, delta_config: dict[str, Any]) -> None:
        self._worker = worker
        self._tracker = DeltaCompressionTracker(delta_config)

    def initialize_baseline(
        self,
        *,
        shard_rank: int,
        shard_count: int,
        transport: str,
    ) -> None:
        init_sparse_delta_baseline_from_iterator(
            self._worker._iter_params_with_optional_kv_scales(),
            delta_tracker=self._tracker,
            shard_rank=shard_rank,
            shard_count=shard_count,
            transport=transport,
        )

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
    ) -> SparseDeltaStreamResult:
        streamer = {
            "s3": stream_sparse_delta_payloads_via_s3_manifest,
            "zmq": stream_sparse_delta_payloads_via_zmq,
        }.get(transport)
        if streamer is None:
            raise ValueError(
                f"Unsupported remote sparse refit transport {transport!r}."
            )
        result = streamer(
            self._worker._iter_params_with_optional_kv_scales(),
            delta_tracker=self._tracker,
            refit_targets=targets,
            transfer_id=transfer_id,
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
