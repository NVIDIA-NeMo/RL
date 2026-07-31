# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Env-gated dump of canonical training rows at their TQ publish sites.

Set ``NRL_SC_DUMP_TRAIN_ROWS=<dir>`` to append one JSON line per training row
whenever a group is published to the ``rollout_data`` partition — from the
legacy ``TQReplayBuffer.commit`` path and from the token-capture
``BlackboxFinalizer`` publish. Off (no I/O, no imports of the payload) unless
the env var is set. Used by the S5 legacy-vs-capture offline row diff.
"""

import json
import os
import threading
from collections.abc import Mapping
from typing import Any, Optional

import torch

_DUMP_ENV_VAR = "NRL_SC_DUMP_TRAIN_ROWS"
# The finalizer publishes from a worker thread (asyncio.to_thread) while the
# legacy path publishes from the SC event loop; serialize appends.
_G_WRITE_LOCK = threading.Lock()


def _row_value(tensor: torch.Tensor, row: int) -> Any:
    value = tensor[row]
    if value.dim() == 0:
        return value.item()
    return value.tolist()


def maybe_dump_train_rows(
    *,
    source: str,
    group_id: str,
    sample_ids: list[str],
    train_batch: Mapping[str, torch.Tensor],
    weight_version: Optional[int],
) -> None:
    """Append each row of a published group to the dump file, if enabled.

    Args:
        source: Publish site tag (``"legacy_commit"`` or ``"finalizer"``).
        group_id: Prompt-group id the rows belong to.
        sample_ids: Canonical per-row sample ids (``{group_id}_g{i}``).
        train_batch: Column tensors as passed to ``pack_payload``.
        weight_version: Weight version stamped on the rows' tags.
    """
    dump_dir = os.environ.get(_DUMP_ENV_VAR)
    if not dump_dir:
        return
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"train_rows_{source}.jsonl")
    lines = []
    for i, sample_id in enumerate(sample_ids):
        record = {
            "source": source,
            "group_id": group_id,
            "sample_id": sample_id,
            "weight_version": weight_version,
            **{name: _row_value(tensor, i) for name, tensor in train_batch.items()},
        }
        lines.append(json.dumps(record))
    with _G_WRITE_LOCK, open(path, "a") as f:
        f.write("\n".join(lines) + "\n")
