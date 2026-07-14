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
"""Adapt verified TQ staging rows to NeMo Gym's ``TokenSource`` protocol.

In direct black-box mode the worker-side staging write is the token store:
Gym's capture holds token-free ``ModelCallRecord``s and the sealed call
manifest, while the per-call token/mask/logprob deltas live only in the
``rollout_staging`` partition under ``<rollout_id>/<call_id>``. This module
rebuilds Gym ``TokenEntry`` records from an immutable, already-verified
snapshot of those rows so Gym's pure trajectory builder can assemble chains
without performing any remote TQ I/O itself (and without NeMo Gym depending
on NeMo-RL internals).

The finalizer owns fetching and integrity verification (hash chain, digest,
mask layout); this adapter owns only the delta -> per-call token
reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StagedCallSnapshot:
    """One verified staging row, in admission order.

    ``prev_len`` is the length of the committed sequence this delta extends
    (0 for the first call and for future ``new_root`` full-prompt rows); the
    delta is ``rendered_prompt[prev_len:] + generated``, with mask 0.0 on the
    prompt carry and 1.0 on generated tokens.
    """

    call_id: str
    prev_len: int
    token_ids_delta: list[int]
    token_mask_delta: list[float]
    logprobs_delta: list[float]
    weight_version: int | None = None
    model: str = ""
    # Forest rows carry an explicit storage parent; a rootless row (first call
    # or new_root compaction) has none and is self-contained (prev_len == 0).
    # Parentless rows with prev_len > 0 are legacy linear turns and extend the
    # running sequential sequence.
    parent_call_id: str | None = None


def fetch_staged_snapshots(
    *,
    dp_client: Any,
    staging_partition: str,
    rollout_id: str,
    call_ids: list[str],
    turn_meta: dict[str, Any],
) -> list[StagedCallSnapshot]:
    """Read the manifest-named staging rows into an immutable snapshot.

    ``turn_meta`` maps ``call_id`` to the cursor's committed turn record (for
    ``prev_len``/``weight_version``); the caller has already reconciled the
    sealed manifest against those records.
    """
    snapshots: list[StagedCallSnapshot] = []
    for call_id in call_ids:
        record = turn_meta[call_id]
        row = dp_client.get_samples(
            sample_ids=[f"{rollout_id}/{call_id}"],
            partition_id=staging_partition,
            select_fields=[
                "token_ids_delta",
                "token_mask_delta",
                "generation_logprobs_delta",
            ],
        )
        snapshots.append(
            StagedCallSnapshot(
                call_id=call_id,
                prev_len=int(record.prev_len),
                token_ids_delta=[int(t) for t in row["token_ids_delta"][0]],
                token_mask_delta=[float(m) for m in row["token_mask_delta"][0]],
                logprobs_delta=[float(p) for p in row["generation_logprobs_delta"][0]],
                weight_version=record.weight_version,
            )
        )
    return snapshots


class StagedSnapshotTokenSource:
    """Implements Gym's ``TokenSource`` protocol over a verified snapshot.

    Reconstruction: walking rows in admission order, each row's full prompt is
    the running committed sequence truncated to ``prev_len`` plus the row's
    prompt-carry tokens (mask 0.0 prefix); its generation is the mask 1.0
    suffix. The running sequence then advances to prompt + generation. This is
    the exact inverse of the worker's ``build_staging_delta``.
    """

    def __init__(
        self, rollout_id: str, snapshots: list[StagedCallSnapshot], model: str = ""
    ):
        self.rollout_id = rollout_id
        self.snapshots = list(snapshots)
        self.model = model

    def entries(self, rollout_id: str) -> list[Any]:
        from nemo_gym.observability.records import TokenEntry

        if rollout_id != self.rollout_id:
            raise ValueError(
                f"snapshot holds rollout {self.rollout_id!r}, asked for {rollout_id!r}"
            )
        entries: list[Any] = []
        cumulative: list[int] = []
        cumulative_by_call: dict[str, list[int]] = {}
        for seq, snapshot in enumerate(self.snapshots):
            if snapshot.parent_call_id is not None:
                # Forest row: the base is the storage parent's cumulative
                # sequence, which the registry committed at exactly prev_len.
                if snapshot.parent_call_id not in cumulative_by_call:
                    raise ValueError(
                        f"call {snapshot.call_id}: parent "
                        f"{snapshot.parent_call_id} precedes it in no snapshot"
                    )
                base = cumulative_by_call[snapshot.parent_call_id]
                if len(base) != snapshot.prev_len:
                    raise ValueError(
                        f"call {snapshot.call_id}: prev_len={snapshot.prev_len} "
                        f"does not equal parent length {len(base)}"
                    )
            else:
                base = cumulative
            if snapshot.prev_len > len(base):
                raise ValueError(
                    f"call {snapshot.call_id}: prev_len={snapshot.prev_len} exceeds "
                    f"committed sequence length {len(base)}"
                )
            if not (
                len(snapshot.token_ids_delta)
                == len(snapshot.token_mask_delta)
                == len(snapshot.logprobs_delta)
            ):
                raise ValueError(f"call {snapshot.call_id}: misaligned delta arrays")
            # The prompt carry is a contiguous 0.0 prefix of the mask; any 0
            # after a 1 means the delta was not produced by build_staging_delta.
            boundary = 0
            for mask_value in snapshot.token_mask_delta:
                if mask_value == 0.0:
                    boundary += 1
                else:
                    break
            if any(m == 0.0 for m in snapshot.token_mask_delta[boundary:]):
                raise ValueError(
                    f"call {snapshot.call_id}: token_mask_delta is not a "
                    "prompt-carry prefix followed by generated tokens"
                )
            prompt_token_ids = (
                base[: snapshot.prev_len] + snapshot.token_ids_delta[:boundary]
            )
            generation_token_ids = snapshot.token_ids_delta[boundary:]
            generation_log_probs = snapshot.logprobs_delta[boundary:]
            entries.append(
                TokenEntry(
                    rollout_id=self.rollout_id,
                    request_id=snapshot.call_id,
                    prompt_token_ids=prompt_token_ids,
                    generation_token_ids=generation_token_ids,
                    generation_log_probs=generation_log_probs,
                    model=snapshot.model or self.model,
                    weight_version=snapshot.weight_version,
                    seq=seq,
                )
            )
            cumulative_by_call[snapshot.call_id] = (
                prompt_token_ids + generation_token_ids
            )
            if snapshot.parent_call_id is None:
                cumulative = cumulative_by_call[snapshot.call_id]
        return entries
