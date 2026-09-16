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

from nemo_rl.algorithms.x_token.mopd import (
    compute_offsets_manual,
    generated_think_tool_marker_error,
    proven_template_only_teacher_token_indices,
    split_sampled_assistant_eot,
    trim_outer_whitespace_with_offsets_for_alignment,
)


class _QwenEOTTokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 99


class _GroupedDecodeTokenizer:
    all_special_ids = []

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "x" if [int(token_id) for token_id in ids] == [10, 11] else ""


class _NoNativeOffsetsTokenizer:
    all_special_ids = []

    def __call__(
        self,
        text,
        *,
        return_tensors=None,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        del return_tensors, add_special_tokens
        if return_offsets_mapping:
            raise NotImplementedError("offsets are unavailable")
        return {"input_ids": [10, 11] if text == " x " else []}

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        mapping = {(10,): " ", (11,): "x ", (10, 11): " x "}
        return mapping.get(tuple(int(token_id) for token_id in ids), "")


def test_sampled_assistant_eot_is_removed_only_when_present():
    tokenizer = _QwenEOTTokenizer()

    assert split_sampled_assistant_eot(tokenizer, [1, 2, 99]) == ([1, 2], 2)
    assert split_sampled_assistant_eot(tokenizer, [1, 2]) == ([1, 2], None)


def test_think_and_tool_structure_accepts_proven_open_tool_transcript():
    transcript = (
        "reasoning</think>answer<tool_call>"
        '{"name":"weather","arguments":{"city":"SF"}}'
        "</tool_call>"
    )

    assert generated_think_tool_marker_error(transcript, initial_state="open") is None
    assert (
        generated_think_tool_marker_error(transcript, initial_state="unknown")
        == "unknown_thinking_state"
    )


def test_template_provenance_distinguishes_template_only_and_mixed_tokens():
    transformed = "r\n</think>\n\nx"
    # Token 2 contains sampled </think> plus an inserted newline. Tokens 1 and
    # 3 contain only template-inserted whitespace and can be excluded safely.
    offsets = [(0, 1), (1, 2), (2, 11), (11, 12), (12, 13)]

    assert proven_template_only_teacher_token_indices(
        "r</think>x", transformed, offsets
    ) == ({1, 3}, {2})


def test_manual_offsets_support_tokens_that_decode_only_as_a_group():
    assert compute_offsets_manual(_GroupedDecodeTokenizer(), [10, 11], "x") == [
        (0, 1),
        (0, 1),
    ]


def test_trim_outer_whitespace_reconstructs_offsets_when_backend_has_none():
    assert trim_outer_whitespace_with_offsets_for_alignment(
        _NoNativeOffsetsTokenizer(), [10, 11], " x "
    ) == ([11], 1, 0, "x", [(0, 1)])
