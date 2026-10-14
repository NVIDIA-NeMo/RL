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

"""NeMo RL's vLLM worker extension as loaded by managed Dynamo workers.

``dynamo.vllm`` loads this through ``--worker-extension-cls``, and the driver calls
its methods through Dynamo's ``/engine/*`` RL routes, which forward an
``engine_rpc`` name to vLLM ``collective_rpc``. Two properties of that transport
shape the additions here:

* Dynamo discards the RPC's return value and reports only whether it raised, so
  every refit entry point raises on failure instead of returning False.
* The HTTP request body is capped at 2 MiB, so large metadata arrives in chunks
  (``nemo_rl.utils.staged_payload``) and is reassembled here.

This module lives under ``vllm/`` rather than ``dynamo/`` so worker processes do not
import the ``dynamo`` package, which pulls in the driver-side client. New methods are
``nrl_``-prefixed: vLLM refuses an extension that shadows a ``Worker`` attribute.
"""

from typing import Any

from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension
from nemo_rl.utils.staged_payload import decode_staged_payload


class DynamoWorkerExtension(VllmInternalWorkerExtension):
    """Refit entry points for a vLLM engine owned by a managed Dynamo worker."""

    # The payload currently being staged. None until the first chunk arrives; a
    # chunk with a new key discards whatever was staged before.
    _nrl_stage_key: str | None = None
    _nrl_stage_total: int = 0
    _nrl_stage_chunks: dict[int, str] | None = None

    def nrl_stage_chunk(self, key: str, index: int, total: int, data: str) -> None:
        """Hold one chunk of a staged payload until the call that consumes it."""
        if self._nrl_stage_key != key:
            self._nrl_stage_key = key
            self._nrl_stage_total = total
            self._nrl_stage_chunks = {}
        if total != self._nrl_stage_total or not 0 <= index < total:
            raise ValueError(
                f"Chunk {index}/{total} does not fit staged payload {key}, "
                f"which has {self._nrl_stage_total} chunks."
            )
        assert self._nrl_stage_chunks is not None
        self._nrl_stage_chunks[index] = data

    def _nrl_take_staged(self, key: str, sha256: str) -> Any:
        """Return the payload staged under ``key`` and clear the staging buffer."""
        chunks = self._nrl_stage_chunks
        if self._nrl_stage_key != key or chunks is None:
            raise RuntimeError(
                f"No payload is staged under {key}; staged key is "
                f"{self._nrl_stage_key}."
            )
        missing = sorted(set(range(self._nrl_stage_total)) - set(chunks))
        if missing:
            raise RuntimeError(f"Staged payload {key} is missing chunks {missing}.")
        ordered = [chunks[index] for index in range(self._nrl_stage_total)]
        self._nrl_stage_key = None
        self._nrl_stage_total = 0
        self._nrl_stage_chunks = None
        return decode_staged_payload(ordered, sha256)

    def nrl_prepare_nccl_reshard_refit_info_from_stage(
        self, key: str, sha256: str
    ) -> None:
        """Run ``prepare_nccl_reshard_refit_info`` on the staged wire-safe metadata."""
        self.prepare_nccl_reshard_refit_info(self._nrl_take_staged(key, sha256))

    def nrl_nccl_reshard_refit(self, refit_timeout_s: float | None) -> bool:
        """Receive one nccl_reshard refit, raising if it did not complete."""
        if not self.nccl_reshard_refit(refit_timeout_s=refit_timeout_s):
            raise RuntimeError("nccl_reshard refit reported failure on this rank.")
        return True
