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

import json
import os
from typing import Any

import torch

from nemo_rl.models.generation.vllm.quantization.fp8_kwargs import FP8_CONFIG_ENV_VAR
from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension
from nemo_rl.utils.staged_payload import decode_staged_payload


def _apply_fp8_patches_from_env() -> None:
    """Apply FP8 rollout patches in this engine worker when the launcher asked for them.

    vLLM imports the worker extension before the worker builds the model, which is
    where the vLLM backend's own hooks would otherwise apply them.
    """
    serialized = os.environ.get(FP8_CONFIG_ENV_VAR)
    if not serialized:
        return
    from nemo_rl.models.generation.vllm.quantization import fp8

    if not fp8.fp8_patches_applied:
        fp8.apply_fp8_patches(None, fp8.FP8Config(**json.loads(serialized)))


_apply_fp8_patches_from_env()


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

    def nrl_prepare_refit_info_from_stage(self, key: str, sha256: str) -> None:
        """Run ``prepare_refit_info`` on staged ``[name, shape, dtype]`` entries.

        The entries keep the trainer's order, which the packed broadcast relies on:
        producer and consumer walk the same sequence to find tensor boundaries.
        """
        state_dict_info = {}
        for name, shape, dtype_name in self._nrl_take_staged(key, sha256):
            dtype = getattr(torch, dtype_name, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(
                    f"Staged tensor {name} has unknown dtype {dtype_name}."
                )
            state_dict_info[name] = (torch.Size(shape), dtype)
        self.prepare_refit_info(state_dict_info)

    def nrl_update_weights_from_collective(
        self, refit_timeout_s: float | None, refit_with_reload_api: bool
    ) -> bool:
        """Receive one collective refit, raising if it did not complete."""
        if not self.update_weights_from_collective(
            refit_timeout_s=refit_timeout_s,
            refit_with_reload_api=refit_with_reload_api,
        ):
            raise RuntimeError("Collective refit reported failure on this rank.")
        return True
