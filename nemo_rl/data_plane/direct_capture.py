# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tensor-only transport for direct native capture, off the HTTP path."""

import json
import uuid
from array import array
from typing import Any

import torch

from nemo_rl.data_plane.tq_token_sink import TQStagingStore
from nemo_rl.experience.direct_capture import NativeCall

DIRECT_STAGING_FIELDS = ["direct_tokens", "direct_logprobs", "direct_metadata"]


class NativeCallStore:
    def __init__(self, dp_client: Any, partition: str, *, router_replay: bool) -> None:
        self._store = TQStagingStore(dp_client, staging_partition=partition)
        self._router_replay = router_replay

    def clear(self, keys: list[str]) -> None:
        self._store.clear(keys)

    def stage(self, call: NativeCall) -> dict[str, Any]:
        """Serialize token/routing arrays once as tensors; JSON holds scalars only."""
        key = f"direct/{uuid.uuid4().hex}"
        meta = {
            "staging_key": key,
            "rollout_id": call.rollout_id,
            "model_call_id": call.call_id,
            "ordinal": call.ordinal,
            "weight_version": call.weight_version,
            "prompt_length": len(call.prompt),
            "generation_length": len(call.tokens),
        }
        # torch.tensor(list-of-Python-ints) holds the GIL for ~0.8 ms at 32K
        # tokens on the profiling Mac. Pack once in C, then let torch retain
        # the buffer; this also avoids an extra concatenated tuple and copy.
        token_buffer = array("q", call.prompt)
        token_buffer.extend(call.tokens)
        fields = {
            "direct_tokens": torch.frombuffer(
                token_buffer, dtype=torch.int64
            ).unsqueeze(0),
            "direct_logprobs": torch.frombuffer(
                array("f", call.logprobs), dtype=torch.float32
            ).unsqueeze(0),
            "direct_metadata": torch.frombuffer(
                bytearray(json.dumps(meta).encode()), dtype=torch.uint8
            ).unsqueeze(0),
        }
        if self._router_replay:
            if call.routed_experts is None:
                raise ValueError(
                    "router replay requested but native routes are missing"
                )
            routes = torch.as_tensor(call.routed_experts)
            size = len(call.prompt) + len(call.tokens)
            if routes.ndim != 3 or routes.shape[0] not in (size - 1, size):
                raise ValueError("native routes do not cover the executed sequence")
            aligned = torch.full((size, *routes.shape[1:]), -1, dtype=torch.int16)
            aligned[:-1] = routes[: size - 1]
            fields["routed_experts"] = aligned.unsqueeze(0)
        self._store.put(key, fields)
        return meta

    def fetch(
        self, manifest: list[dict[str, Any]], rollout_id: str
    ) -> list[NativeCall]:
        if not manifest:
            return []
        fields = DIRECT_STAGING_FIELDS + (
            ["routed_experts"] if self._router_replay else []
        )
        data = self._store.get(
            [entry["staging_key"] for entry in manifest], select_fields=fields
        )
        calls = []
        for i, expected in enumerate(manifest):
            meta = json.loads(bytes(data["direct_metadata"][i].tolist()))
            if meta != expected or meta["rollout_id"] != rollout_id:
                raise ValueError("native staging identity mismatch")
            tokens = tuple(data["direct_tokens"][i].tolist())
            prompt_length = meta["prompt_length"]
            logprobs = tuple(data["direct_logprobs"][i].tolist())
            if (
                len(tokens) != prompt_length + meta["generation_length"]
                or len(logprobs) != meta["generation_length"]
            ):
                raise ValueError("native staging length mismatch")
            calls.append(
                NativeCall(
                    rollout_id,
                    meta["model_call_id"],
                    meta["ordinal"],
                    meta["weight_version"],
                    tokens[:prompt_length],
                    tokens[prompt_length:],
                    logprobs,
                    data["routed_experts"][i] if self._router_replay else None,
                )
            )
        return calls
