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
"""Row layout of the privileged value: a per-row token prefix only the critic sees.

A critic row is ``prefix + input_ids[:input_length]`` and its per-token fields
are the policy's shifted right by ``len(prefix)``. Rows name their prefix and
its length in their tags (``PRIVILEGE_KEY_TAG``, ``PRIVILEGE_PREFIX_LEN_TAG``),
so the data plane stores the policy layout only: the value driver plans on
critic lengths (:func:`critic_view_meta`), and the value workers splice the
prefix in after fetch (:func:`prepend_privilege_prefix`) and slice the values
back out (:func:`values_to_policy_layout`).
"""

from dataclasses import replace
from typing import Any, Mapping

import torch

from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.schema import PRIVILEGE_KEY_TAG, PRIVILEGE_PREFIX_LEN_TAG
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _privilege_tags(meta: KVBatchMeta) -> list[dict[str, Any]]:
    tags = meta.tags
    if not tags or any(PRIVILEGE_KEY_TAG not in tag for tag in tags):
        raise ValueError(
            "privileged value: batch rows carry no privilege tag. Every group "
            "must pass through PrivilegePrefixStore.enrich at commit."
        )
    return tags


def row_privilege_keys(meta: KVBatchMeta) -> list[str]:
    """The prefix key of each row of ``meta``, in row order."""
    return [tag[PRIVILEGE_KEY_TAG] for tag in _privilege_tags(meta)]


def _prefix_lengths(meta: KVBatchMeta) -> list[int]:
    return [int(tag[PRIVILEGE_PREFIX_LEN_TAG]) for tag in _privilege_tags(meta)]


def _shift_lengths(meta: KVBatchMeta, sign: int) -> KVBatchMeta:
    if meta.sequence_lengths is None:
        raise ValueError("privileged value: batch carries no sequence lengths")
    lengths = [
        length + sign * prefix_len
        for length, prefix_len in zip(meta.sequence_lengths, _prefix_lengths(meta))
    ]
    return replace(meta, sequence_lengths=lengths)


def critic_view_meta(meta: KVBatchMeta) -> KVBatchMeta:
    """``meta`` with each row's length extended by its prefix (critic layout)."""
    return _shift_lengths(meta, +1)


def policy_view_meta(meta: KVBatchMeta) -> KVBatchMeta:
    """Inverse of :func:`critic_view_meta`: lengths back in the policy layout."""
    return _shift_lengths(meta, -1)


def prepend_privilege_prefix(
    data: BatchedDataDict[Any],
    meta: KVBatchMeta,
    prefixes: Mapping[str, torch.Tensor],
    shift_fields: tuple[str, ...] = (),
) -> BatchedDataDict[Any]:
    """Turn a fetched policy-layout batch into the critic layout, in place.

    Each row becomes ``prefix + input_ids[:L]``. ``token_mask`` and every field in
    ``shift_fields`` (per-token, e.g. returns / old values) are shifted right by
    the prefix length and zero over the prefix, so the value loss and clipping
    see exactly the policy's response positions. The fetched tensors must already
    be padded to at least the longest critic row: the driver mints the forward
    pad target from :func:`critic_view_meta` lengths.
    """
    keys = row_privilege_keys(meta)
    input_ids = data["input_ids"]
    lengths = data["input_lengths"]
    width = input_ids.shape[1]
    new_ids = input_ids.clone()
    new_token_mask = torch.zeros_like(data["token_mask"])
    shifted = {name: torch.zeros_like(data[name]) for name in shift_fields}
    for row, key in enumerate(keys):
        prefix = prefixes[key].to(device=input_ids.device, dtype=input_ids.dtype)
        p, n = int(prefix.numel()), int(lengths[row])
        if p + n > width:
            raise ValueError(
                f"privileged value row {row} needs {p + n} tokens but the batch "
                f"is padded to {width}; the forward pad target was not minted "
                "from critic lengths."
            )
        new_ids[row, p : p + n] = input_ids[row, :n]
        new_ids[row, :p] = prefix
        new_token_mask[row, p : p + n] = data["token_mask"][row, :n]
        for name in shift_fields:
            shifted[name][row, p : p + n] = data[name][row, :n]
    data["input_ids"] = new_ids
    data["token_mask"] = new_token_mask
    data["input_lengths"] = lengths + torch.as_tensor(
        _prefix_lengths(meta), dtype=lengths.dtype, device=lengths.device
    )
    for name, tensor in shifted.items():
        data[name] = tensor
    return data


def values_to_policy_layout(
    values: torch.Tensor,
    meta: KVBatchMeta,
    policy_lengths: torch.Tensor,
) -> torch.Tensor:
    """Slice critic-layout values back onto the policy's token positions."""
    out = torch.zeros_like(values)
    for row, p in enumerate(_prefix_lengths(meta)):
        n = int(policy_lengths[row])
        out[row, :n] = values[row, p : p + n]
    return out
