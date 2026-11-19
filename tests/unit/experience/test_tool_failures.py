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
import pytest

from nemo_rl.experience.tool_failures import (
    CATEGORIES,
    EDIT_FAILED,
    PERMISSION_DENIED,
    READ_MISSING,
    SCHEMA_ERROR,
    SEARCH_TOOL_ERROR,
    TOOL_ERROR,
    ToolTagIds,
    classify_tool_response,
    failed_tool_turns,
)

# A toy vocabulary: every piece is one token, decoding concatenates the pieces.
_TAGS = [
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<|im_start|>",
    "<|im_end|>",
    "</think>",
]
IDS = ToolTagIds(
    tool_call_open=0, tool_response_open=2, tool_response_close=3, think_close=6
)


class _Row:
    """Builds a row from (pieces, generated) spans over a growing vocabulary."""

    def __init__(self):
        self.vocab = list(_TAGS)
        self.token_ids: list[int] = []
        self.token_mask: list[float] = []

    def add(self, pieces: list[str], *, generated: bool) -> "_Row":
        for piece in pieces:
            if piece not in self.vocab:
                self.vocab.append(piece)
            self.token_ids.append(self.vocab.index(piece))
            self.token_mask.append(1.0 if generated else 0.0)
        return self

    def turn(self, *calls: str, text: str = "thinking") -> "_Row":
        pieces = [text]
        for tool in calls:
            pieces += ["<tool_call>", f"\n<function={tool}>\n", "args", "</tool_call>"]
        return self.add(pieces + ["<|im_end|>"], generated=True)

    def replies(self, *responses: str) -> "_Row":
        pieces = ["<|im_start|>", "user\n"]
        for response in responses:
            pieces += ["<tool_response>", "\n" + response + "\n", "</tool_response>"]
        return self.add(
            pieces + ["<|im_end|>", "<|im_start|>", "assistant\n"], generated=False
        )

    def decode_many(self, seqs: list[list[int]]) -> list[str]:
        return ["".join(self.vocab[t] for t in seq) for seq in seqs]

    def run(self, categories=CATEGORIES):
        return failed_tool_turns(
            self.token_ids,
            self.token_mask,
            ids=IDS,
            decode_many=self.decode_many,
            categories=categories,
        )

    def turn_spans(self) -> list[tuple[int, int]]:
        spans, start = [], None
        for k, m in enumerate(self.token_mask + [0.0]):
            if m and start is None:
                start = k
            elif not m and start is not None:
                spans.append((start, k))
                start = None
        return spans


@pytest.mark.parametrize(
    "tool, response, expected",
    [
        (
            "edit",
            "Could not find oldString in the file. It must match exactly",
            EDIT_FAILED,
        ),
        (
            "edit",
            "Found multiple matches for oldString. Provide more context",
            EDIT_FAILED,
        ),
        (
            "str_replace_editor",
            "ERROR:\nNo replacement was performed, old_str `x` did not appear verbatim",
            EDIT_FAILED,
        ),
        (
            "str_replace_editor",
            "ERROR:\nInvalid `path` parameter: /app/missing.py does not exist",
            SCHEMA_ERROR,
        ),
        (
            "write",
            "You must read the file /repo/a.py before overwriting it",
            EDIT_FAILED,
        ),
        ("read", "Error: File not found: /repo/missing.py", READ_MISSING),
        ("read", "Offset 900 is out of range for this file (120 lines)", READ_MISSING),
        ("glob", "Error: ENOENT", READ_MISSING),
        (
            "edit",
            "The edit tool was called with invalid arguments: [expected string]",
            SCHEMA_ERROR,
        ),
        (
            "invalid",
            "Error: Model tried to call unavailable tool 'apply_patch'",
            SCHEMA_ERROR,
        ),
        (
            "bash",
            "The user has specified a rule which prevents you from using this command",
            PERMISSION_DENIED,
        ),
        (
            "bash_command",
            "[blocked: command contains a network call]",
            PERMISSION_DENIED,
        ),
        ("bash_command", "[blocked: $(...) command substitution]", PERMISSION_DENIED),
        ("bash", "NotFound: FileSystem.access (/workspace/x)", READ_MISSING),
        ("grep", "ripgrep execution failed: exit status 2", SEARCH_TOOL_ERROR),
        ("webfetch", "Error: Request failed with status code 404", TOOL_ERROR),
    ],
)
def test_classify_harness_error_replies(tool, response, expected):
    assert classify_tool_response(tool, response) == expected


@pytest.mark.parametrize(
    "tool, response",
    [
        # A shell command's own error output is the environment answering, not a failed call.
        ("bash", "ERROR: test_foo.py::test_bar - AssertionError"),
        ("execute_bash", "Error: cannot import name 'x'"),
        # Without the call's tool name only the harness's specific replies count.
        (None, "Error: something went wrong"),
        ("edit", "Edit applied successfully."),
        ("read", "<file>\n00001| import os\n</file>"),
        ("grep", "Found 3 matches\nsrc/a.py:"),
    ],
)
def test_classify_successes_and_shell_errors(tool, response):
    assert classify_tool_response(tool, response) is None


def test_shell_output_mimicking_a_tool_error_is_not_a_failure():
    # Programs print these themselves; only the harness's own replies count for a shell tool.
    assert classify_tool_response("bash", "rg: regex parse error:\n    (?<x)") is None
    assert classify_tool_response("bash", "File not found: config.yaml") is None
    assert classify_tool_response(None, "Could not find oldString in the file") is None


def test_marks_only_the_turn_whose_call_failed():
    row = _Row().add(["<|im_start|>", "system prompt", "<|im_end|>"], generated=False)
    row.turn("read", "edit").replies(
        "<file>ok</file>", "Could not find oldString in the file"
    )
    row.turn("edit").replies("Edit applied successfully.")
    row.turn(text="final answer")
    found = row.run()
    spans = row.turn_spans()
    assert found.turns == 3 and found.failed_turns == 1 and found.tool_responses == 3
    assert found.failed_calls == {EDIT_FAILED: 1}
    expected = [False] * len(row.token_ids)
    for k in range(*spans[0]):
        expected[k] = True
    assert found.mask.tolist() == expected


def test_kth_response_answers_kth_call():
    # The read succeeds, the bash call's output starts with "Error" -- a shell error, not a failed call.
    # Matching the responses the other way round would classify "Error ..." under the read tool.
    row = (
        _Row()
        .turn("read", "bash")
        .replies("<file>ok</file>", "Error: tests failed")
        .turn(text="done")
    )
    assert row.run().failed_turns == 0
    row = (
        _Row()
        .turn("bash", "read")
        .replies("ok", "Error: no such file")
        .turn(text="done")
    )
    found = row.run()
    assert found.failed_turns == 1 and found.failed_calls == {READ_MISSING: 1}


def test_tool_call_tag_inside_thinking_is_not_a_call():
    # The thinking drafts an edit call; the call the harness ran is bash, whose output starts with "Error".
    row = _Row().add(
        [
            "plan ",
            "<tool_call>",
            "\n<function=edit>\n",
            "</tool_call>",
            "</think>",
            "<tool_call>",
        ]
        + ["\n<function=bash>\n", "args", "</tool_call>", "<|im_end|>"],
        generated=True,
    )
    row.replies("Error: tests failed").turn(text="done")
    found = row.run()
    assert (
        found.tool_responses == 1
        and found.failed_turns == 0
        and found.failed_calls == {}
    )


def test_calls_start_after_the_first_think_close():
    # vLLM ends the thinking at the first </think>; a later one inside a call's arguments is text, and the
    # <tool_call> after it is not the call the harness ran first.
    row = _Row().add(
        ["plan ", "</think>", "<tool_call>", "\n<function=bash>\n", "x ", "</think>"]
        + ["<tool_call>", "\n<function=edit>\n", "</tool_call>", "<|im_end|>"],
        generated=True,
    )
    row.replies("Error: tests failed").turn(text="done")
    assert row.run().failed_turns == 0


def test_categories_filter_marks_but_still_counts():
    row = (
        _Row()
        .turn("edit")
        .replies("Could not find oldString")
        .turn("read")
        .replies("Error: File not found: x")
    )
    found = row.run(categories=(READ_MISSING,))
    spans = row.turn_spans()
    assert found.failed_turns == 1
    assert found.failed_calls == {EDIT_FAILED: 1, READ_MISSING: 1}
    assert not found.mask[spans[0][0] : spans[0][1]].any()
    assert found.mask[spans[1][0] : spans[1][1]].all()


def test_tag_inside_a_response_body_is_not_a_new_response():
    # The read's output quotes a <tool_response> tag; the edit's reply is still the second response.
    row = _Row().turn("read", "edit")
    row.add(
        [
            "<|im_start|>",
            "user\n",
            "<tool_response>",
            "file says ",
            "<tool_response>",
            " text",
        ],
        generated=False,
    )
    row.add(
        [
            "</tool_response>",
            "<tool_response>",
            "Could not find oldString",
            "</tool_response>",
        ],
        generated=False,
    )
    row.add(["<|im_end|>"], generated=False)
    row.turn(text="done")
    found = row.run()
    assert found.tool_responses == 2 and found.failed_calls == {EDIT_FAILED: 1}


def test_unterminated_response_and_turn_without_replies():
    # A response the row ends inside is still read; a turn with no reply after it is never marked.
    row = (
        _Row()
        .turn("edit")
        .add(["<tool_response>", "Could not find oldString"], generated=False)
    )
    assert row.run().failed_turns == 1
    row = _Row().turn("edit")
    found = row.run()
    assert found.turns == 1 and found.failed_turns == 0 and found.tool_responses == 0


def test_no_generated_tokens():
    row = _Row().add(["<|im_start|>", "prompt"], generated=False)
    found = row.run()
    assert found.turns == 0 and not found.mask.any() and found.mask.shape == (2,)


def test_resolve_tool_tag_ids_requires_single_tool_tags():
    from nemo_rl.experience.tool_failures import resolve_tool_tag_ids

    vocab = {
        "<tool_call>": 14,
        "<tool_response>": 16,
        "</tool_response>": 17,
        "</think>": 13,
    }

    class _Tokenizer:
        unk_token_id = 0

        def __init__(self, split=(), missing=()):
            self.split, self.missing = split, missing

        def convert_tokens_to_ids(self, tag):
            return 0 if tag in self.missing else vocab.get(tag, 0)

        def encode(self, tag, add_special_tokens):
            return [1, 2] if tag in self.split else [vocab[tag]]

    assert resolve_tool_tag_ids(_Tokenizer()) == ToolTagIds(14, 16, 17, think_close=13)
    # No single-token </think>: no thinking span, calls counted from the turn start.
    assert resolve_tool_tag_ids(_Tokenizer(missing=("</think>",))).think_close is None
    with pytest.raises(ValueError, match="</tool_response>"):
        resolve_tool_tag_ids(_Tokenizer(split=("</tool_response>",)))
