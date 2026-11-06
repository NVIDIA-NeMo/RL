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
"""Prompt-group identity shared by TQ advantage consumers."""

import torch

from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.schema import GROUP_ID_TAG

# Forest rows, execution padding included, name their group here instead
# (``LogicalRowTag.dispatch_group_id``).
_FOREST_GROUP_TAG = "dispatch_group_id"


def row_group_ids(meta: KVBatchMeta) -> list[str]:
    """Return each row's explicit prompt-group id in metadata order.

    Missing tags raise instead of falling back to tokens, which can be equal
    for distinct prompts or change during a rollout.
    """
    if meta.tags is None:
        raise ValueError(
            f"{len(meta.sample_ids)} row(s) of partition {meta.partition_id!r} "
            f"carry no tags, so the {GROUP_ID_TAG!r} baseline key is unavailable"
        )
    untagged = [
        meta.sample_ids[i] for i, tag in enumerate(meta.tags) if GROUP_ID_TAG not in tag
    ]
    if untagged:
        raise ValueError(
            f"{len(untagged)} row(s) carry no {GROUP_ID_TAG!r} tag "
            f"(first: {untagged[0]!r}); rollout producers must stamp every row"
        )
    return [tag[GROUP_ID_TAG] for tag in meta.tags]


def prompt_group_ids(meta: KVBatchMeta) -> list[str]:
    """Return the distinct prompt-group ids of ``meta``'s rows, in row order.

    Reads the row tags rather than parsing the ``_g{i}`` suffix back off the
    sample ids, so every consumer agrees with the advantage stage about what
    a group is. Forest rows carry ``dispatch_group_id``; others carry
    ``GROUP_ID_TAG``.
    """
    if any(_FOREST_GROUP_TAG in tag for tag in meta.tags or []):
        assert meta.tags is not None
        return list(dict.fromkeys(tag[_FOREST_GROUP_TAG] for tag in meta.tags))
    return list(dict.fromkeys(row_group_ids(meta)))


def group_index_column(group_ids: list[str]) -> torch.Tensor:
    """Factorize prompt groups into a ``[N, 1]`` integer equality key.

    Indices are local to this call. Preserve the original group ids across
    filtering and concatenation, then factorize the final selected batch;
    concatenating independently factorized columns would merge distinct groups.
    """
    index = {group_id: i for i, group_id in enumerate(dict.fromkeys(group_ids))}
    return torch.tensor(
        [index[group_id] for group_id in group_ids], dtype=torch.long
    ).reshape(-1, 1)
