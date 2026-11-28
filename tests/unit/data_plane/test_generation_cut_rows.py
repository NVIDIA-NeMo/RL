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
"""A cut row's batching figures match the record built from it."""

from dataclasses import replace
from typing import Any

import pytest
import torch

from nemo_rl.data_plane.generation_cut_rows import (
    GenerationCutRow,
    StagedGenerationCut,
    stage_generation_cut_rows,
)

pytestmark = pytest.mark.nemo_gym


class _RecordingSink:
    def __init__(self) -> None:
        self.records: list[Any] = []

    def stage_generation_prefix_batch(
        self, records, *, checkpoint_id, chunk_sequences, attachments=None
    ) -> list[str]:
        self.records.extend(records)
        return [f"{checkpoint_id}/{sequence}" for sequence in chunk_sequences]


def _token_in_row(*, first_chunk: bool, restored: bool) -> GenerationCutRow:
    from nemo_gym.token_id_capture.staging.capture import (
        ActiveCall,
        RolloutTokenCapture,
    )
    from nemo_gym.token_id_capture.staging.records import (
        CaptureAdmission,
        StagedCallBaseSnapshot,
    )

    prefix = [1, 2, 3]
    admission = CaptureAdmission(
        capture_key="rollout-0:0",
        model_call_id="call-1",
        parent_call_id="call-0",
        parent_chain_hash="0" * 64,
        prev_len=len(prefix),
        mode="token_in",
        required_prefix_token_ids=prefix,
    )
    generation_cut = None
    prompt = [*prefix, 4]
    if restored:
        # An earlier cut of the same call kept one generated token.
        earlier = RolloutTokenCapture(
            sink=_RecordingSink(), weight_version_fn=lambda: 0
        ).build_prefix_record(
            ActiveCall(admission=admission, weight_version=1, prefix_token_ids=prefix),
            prompt_token_ids=prompt,
            generated_token_ids=[10],
            generated_logprobs=[-0.1],
        )
        generation_cut = StagedCallBaseSnapshot.model_validate(
            {
                name: value
                for name, value in earlier.model_dump().items()
                if name in StagedCallBaseSnapshot.model_fields
            }
        )
        prompt = [*prefix, 4, 10]
    generated = [20, 21, 22]
    logprobs = [-0.2, -0.3, -0.4]
    return GenerationCutRow(
        call=ActiveCall(
            admission=admission,
            weight_version=2,
            prefix_token_ids=prefix,
            generation_cut=generation_cut,
        ),
        prompt_token_ids=prompt,
        generated_token_ids=generated,
        generated_logprobs=logprobs,
        chunk_token_ids=generated if first_chunk else generated[-1:],
        chunk_logprobs=logprobs if first_chunk else logprobs[-1:],
        first_chunk=first_chunk,
        chunk_sequence=0 if first_chunk else 1,
        call_extras={"media": "spans"},
    )


@pytest.mark.parametrize("routes", [False, True], ids=["text", "routes"])
@pytest.mark.parametrize("restored", [False, True], ids=["fresh", "restored"])
@pytest.mark.parametrize("first_chunk", [True, False], ids=["first", "later"])
def test_row_figures_match_the_chunk_record(
    first_chunk: bool, restored: bool, routes: bool
) -> None:
    row = _token_in_row(first_chunk=first_chunk, restored=restored)
    if routes:
        row = replace(
            row,
            route_extras={"routed_experts": {"shape": [1]}},
            attachments={"routed_experts": torch.zeros((row.delta_len, 2, 2))},
        )
    sink = _RecordingSink()
    [staged] = stage_generation_cut_rows(sink, [row], checkpoint_id="cut")
    [chunk] = sink.records
    assert row.delta_len == chunk.delta_len == len(chunk.token_ids_delta)
    assert row.chunk_extras == chunk.extras
    if first_chunk and not routes:
        # The first chunk without routes is the prefix record itself.
        assert staged == StagedGenerationCut("cut/0", chunk.digest)
    else:
        assert staged.staging_key == f"cut/{row.chunk_sequence}"
        assert staged.prefix_digest != chunk.digest
