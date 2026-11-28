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

"""Build and write the rows of a generation-prefix cut.

The engine freezes each live call and hands over what its row is built from,
a ``GenerationCutRow``. Building Gym's records from it costs time linear in
the call's whole prompt and generated prefix, so it runs where the rows are
written: in the prefix writer actor when there is one, otherwise in the
caller's writer thread.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

# token_ids_delta (int64) plus token_mask_delta and generation_logprobs_delta
# (float32), as the sink encodes them.
_TOKEN_BYTES = 8 + 2 * 4
# Rough per-row allowance for the scalar, digest and offset columns.
_ROW_BYTES = 512


@dataclass(frozen=True)
class GenerationCutRow:
    """One frozen live call: what its chunk row and prefix digest are built from.

    The first chunk of a call is its whole prefix so far, prompt delta
    included; a later chunk holds only the tokens generated since the last
    staged chunk.
    """

    # Gym's ActiveCall: identity, the resolved prompt prefix and any restored cut.
    call: Any
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_logprobs: list[float]
    chunk_token_ids: list[int]
    chunk_logprobs: list[float]
    first_chunk: bool
    chunk_sequence: int
    call_extras: dict[str, Any] | None = None
    route_extras: dict[str, Any] | None = None
    attachments: dict[str, torch.Tensor] | None = None

    @property
    def delta_len(self) -> int:
        """Tokens the chunk row carries."""
        if not self.first_chunk:
            return len(self.chunk_token_ids)
        restored = self.call.generation_cut
        if restored is None:
            prompt_delta = len(self.prompt_token_ids) - self.call.admission.prev_len
        else:
            prompt_delta = len(restored.token_ids_delta)
        return prompt_delta + len(self.generated_token_ids)

    @property
    def chunk_extras(self) -> dict[str, Any] | None:
        """The extras the chunk row carries: a first chunk's also has the call's."""
        if not self.first_chunk:
            return self.route_extras
        if self.route_extras is None:
            return self.call_extras
        return {**(self.call_extras or {}), **self.route_extras}

    def estimated_bytes(self) -> int:
        """Estimated encoded size of the chunk row, for the write pipeline's bounds."""
        size = self.delta_len * _TOKEN_BYTES + _ROW_BYTES
        admission = self.call.admission
        for text in (
            admission.capture_key,
            admission.model_call_id,
            admission.parent_call_id,
        ):
            if text is not None:
                size += len(text.encode("utf-8"))
        extras = self.chunk_extras
        if extras is not None:
            size += len(json.dumps(extras, ensure_ascii=False).encode("utf-8"))
        if self.attachments:
            size += sum(
                value.numel() * value.element_size()
                for value in self.attachments.values()
            )
        return size


@dataclass(frozen=True)
class StagedGenerationCut:
    """One written cut row, and the digest of its call's whole prefix."""

    staging_key: str
    prefix_digest: str


class GenerationPrefixBatchSink(Protocol):
    """Writes built cut records; ``TQTokenSink`` is the production one."""

    def stage_generation_prefix_batch(
        self,
        records: Sequence[Any],
        *,
        checkpoint_id: str,
        chunk_sequences: Sequence[int],
        attachments: Sequence[dict[str, Any] | None] | None = None,
    ) -> list[str]: ...


def _record_builder(sink: GenerationPrefixBatchSink) -> Any:
    """A Gym capture used only for its record builders."""
    # Deferred: nemo_gym is an optional extra absent in non-Gym runs.
    from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture

    # The builders stamp versions from the call, never from this callback.
    return RolloutTokenCapture(sink=sink, weight_version_fn=lambda: 0)


def generation_cut_prefix_record(row: GenerationCutRow, builder: Any) -> Any:
    """The record of the call's whole prefix so far; its digest is the cut's.

    Raises:
        ValueError: Gym refuses the prompt or the generated tokens.
    """
    return builder.build_prefix_record(
        row.call,
        prompt_token_ids=row.prompt_token_ids,
        generated_token_ids=row.generated_token_ids,
        generated_logprobs=row.generated_logprobs,
        extras=row.call_extras,
    )


def build_generation_cut_rows(
    rows: Sequence[GenerationCutRow], builder: Any
) -> list[tuple[Any, str]]:
    """Each row's chunk record and its call's prefix digest.

    The advertised prefix digest covers tokens and call extras, not routes:
    each chunk's own digest covers its routes, and a restore checks every
    chunk's routes against it.
    """
    built = []
    for row in rows:
        prefix = generation_cut_prefix_record(row, builder)
        if not row.first_chunk:
            chunk = builder.build_generation_chunk_record(
                row.call,
                generated_token_ids=row.chunk_token_ids,
                generated_logprobs=row.chunk_logprobs,
                extras=row.route_extras,
            )
        elif row.route_extras is None:
            chunk = prefix
        else:
            chunk = builder.build_prefix_record(
                row.call,
                prompt_token_ids=row.prompt_token_ids,
                generated_token_ids=row.generated_token_ids,
                generated_logprobs=row.generated_logprobs,
                extras=row.chunk_extras,
            )
        built.append((chunk, prefix.digest))
    return built


def stage_generation_cut_rows(
    sink: GenerationPrefixBatchSink,
    rows: Sequence[GenerationCutRow],
    *,
    checkpoint_id: str,
) -> list[StagedGenerationCut]:
    """Build every row's records, then write the chunk rows in one batch.

    Raises:
        ValueError: Gym refuses a row; nothing is written.
        Exception: the batch write failed; the sink has discarded its keys.
    """
    built = build_generation_cut_rows(rows, _record_builder(sink))
    keys = sink.stage_generation_prefix_batch(
        [chunk for chunk, _ in built],
        checkpoint_id=checkpoint_id,
        chunk_sequences=[row.chunk_sequence for row in rows],
        attachments=[row.attachments for row in rows],
    )
    return [
        StagedGenerationCut(staging_key=key, prefix_digest=digest)
        for key, (_, digest) in zip(keys, built, strict=True)
    ]
