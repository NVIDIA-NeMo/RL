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
"""Keep cohort-sized payloads off Ray RPC boundaries.

Actors that exist to take work off the controller only help if the work's
inputs and outputs stay small. Both the token-capture finalizer and the
advantage stage move tensors through DataPlane and return metadata, so this
guard is what makes "metadata-only" checkable rather than aspirational.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import torch

from nemo_rl.data_plane.column_io import TOKEN_ALIGNED_FIELDS
from nemo_rl.data_plane.schema import PER_TOKEN_STAGING_FIELDS

# Field names whose values are per-token and therefore large, but whose Python
# type is indistinguishable from metadata -- a list[int] of token ids looks just
# like a short list of ids. assert_metadata_only() below already rejects tensors
# and unrecognised types; this set is only for heavy values that would otherwise
# pass it.
#
# Built from the lists that own these names so a column added to the codec or
# to the staging sink cannot escape the guard, which is how the hand-kept
# version came to be missing prev_logprobs, advantages and the rest of what the
# advantage actor reads and writes. Only names with no owning list are spelled
# out below. Widening is safe because nothing crossing these RPCs is keyed by a
# column name: row tags carry weight_version, prompt_idx, GROUP_ID_TAG,
# violation counts and row_shapes_key(...) names, and extra_info carries
# padding and micro-batch keys.
FORBIDDEN_RPC_KEYS = (
    TOKEN_ALIGNED_FIELDS
    | frozenset(PER_TOKEN_STAGING_FIELDS)
    | frozenset(
        {
            "token_ids",  # message-log key (payload.py)
            # Gym's spelling of generation_logprobs_delta. nemo_gym is not
            # installed on the driver and cannot be imported here, so
            # test_forbidden_keys_cover_gym_staging_fields pins it in the Gym
            # lane instead.
            "generation_log_probs_delta",
            # The generation backends' spelling (rollouts.py, worker_mixin.py).
            # No DataPlane column owns it, but the value it names is still a
            # per-token float list.
            "logprobs",
        }
    )
)


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
            if key in FORBIDDEN_RPC_KEYS:
                raise TypeError(f"{path} contains forbidden heavy field {key!r}")
            assert_metadata_only(key, path=f"{path}.key")
            assert_metadata_only(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_metadata_only(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported RPC type {type(value).__name__}")
