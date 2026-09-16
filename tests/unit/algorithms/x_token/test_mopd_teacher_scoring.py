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

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from nemo_rl.algorithms.x_token.token_aligner import AlignmentPair


class _CharTokenizer:
    chat_template = "plain"
    pad_token_id = 0
    eos_token = None
    eos_token_id = None
    all_special_ids = []

    def __call__(
        self,
        text,
        return_tensors=None,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        del return_tensors, add_special_tokens
        result = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id))

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
    ):
        assert tokenize is False
        assert add_generation_prompt is False
        return "".join(str(message.get("content") or "") for message in messages)


class _Sharding:
    def get_axis_size(self, axis):
        assert axis == "data_parallel"
        return 1


class _TeacherGroup:
    cfg = {"max_total_sequence_length": 512}
    sharding_annotations = _Sharding()

    def __init__(self):
        self.calls = 0

    def get_logprobs(self, batch):
        self.calls += 1
        self.last_batch = batch
        return {
            "reference_logprobs": torch.full(
                batch["input_ids"].shape, -2.0, dtype=torch.float32
            )
        }


def _cross_tokenizer_config():
    return {
        "tokenizer": {
            "name": "teacher-tokenizer",
            "chat_template": "default",
            "chat_template_kwargs": {},
            "tokenizer_kwargs": {},
        },
        "alignment_method": "offset_cluster_decode_fix",
        "mask_first_teacher_prefix_chunk": False,
        "exclude_proven_template_only_teacher_tokens": False,
        "missing_think_close_policy": "mask",
    }


def test_cross_token_scorer_projects_only_generated_assistant_turns_in_tool_loop(
    monkeypatch,
):
    from nemo_rl.algorithms.x_token import mopd_teacher_scoring as scoring

    alignment_calls = []

    def exact_alignment(
        student_ids,
        teacher_ids,
        *,
        aligner,
        method,
        student_offsets,
        teacher_offsets,
    ):
        del aligner
        alignment_calls.append(
            (student_ids, teacher_ids, method, student_offsets, teacher_offsets)
        )
        return [
            AlignmentPair(
                s_tokens=[],
                t_tokens=[],
                s_start=0,
                s_end=len(student_ids),
                t_start=0,
                t_end=len(teacher_ids),
                is_correct=True,
            )
        ]

    monkeypatch.setattr(scoring, "align_token_ids", exact_alignment)
    student_tokenizer = _CharTokenizer()
    teacher_tokenizer = _CharTokenizer()
    teacher_group = _TeacherGroup()
    scorer = scoring.build_mopd_teacher_scorer(
        student_tokenizer=student_tokenizer,
        teacher_group=teacher_group,
        cross_tokenizer_config=_cross_tokenizer_config(),
        teacher_tokenizer=teacher_tokenizer,
        aligner=SimpleNamespace(
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
        ),
    )
    tool_call = (
        'answer<tool_call>{"name":"weather","arguments":{"city":"SF"}}</tool_call>'
    )
    tool_result = "sunny"
    final_answer = "It is sunny."
    prompt_ids = student_tokenizer("q")["input_ids"]
    tool_call_ids = student_tokenizer(tool_call)["input_ids"]
    tool_result_ids = student_tokenizer(tool_result)["input_ids"]
    final_answer_ids = student_tokenizer(final_answer)["input_ids"]
    input_ids = torch.tensor(
        [prompt_ids + tool_call_ids + tool_result_ids + final_answer_ids],
        dtype=torch.long,
    )

    result = scorer.score(
        input_ids=input_ids,
        input_lengths=torch.tensor([input_ids.shape[1]]),
        message_logs=[
            [
                {"role": "user", "content": "q", "token_ids": prompt_ids},
                {
                    "role": "assistant",
                    "content": tool_call,
                    "token_ids": tool_call_ids,
                    "generation_logprobs": [0.0] * len(tool_call_ids),
                },
                {
                    "role": "tool",
                    "content": tool_result,
                    "token_ids": tool_result_ids,
                },
                {
                    "role": "assistant",
                    "content": final_answer,
                    "token_ids": final_answer_ids,
                    "generation_logprobs": [0.0] * len(final_answer_ids),
                },
            ]
        ],
    )

    expected_scores = torch.tensor(
        [
            [0.0] * len(prompt_ids)
            + [-2.0] * len(tool_call_ids)
            + [0.0] * len(tool_result_ids)
            + [-2.0] * len(final_answer_ids)
        ]
    )
    expected_mask = torch.tensor(
        [
            [False] * len(prompt_ids)
            + [True] * len(tool_call_ids)
            + [False] * len(tool_result_ids)
            + [True] * len(final_answer_ids)
        ]
    )
    torch.testing.assert_close(result.logprobs, expected_scores)
    assert torch.equal(result.valid_mask, expected_mask)
    assert teacher_group.calls == 1
    assert len(alignment_calls) == 2
    assert all(call[2] == "offset_cluster_decode_fix" for call in alignment_calls)
    assert result.metrics["mopd/turns_aligned"] == 2.0


def test_alignment_tokenizer_uses_plain_transformers_construction(monkeypatch):
    from nemo_rl.algorithms.x_token import mopd_teacher_scoring as scoring

    tokenizer = MagicMock()
    base_apply_chat_template = MagicMock(return_value="rendered")
    tokenizer.apply_chat_template = base_apply_chat_template
    calls = []

    def from_pretrained(*args, **kwargs):
        calls.append(("load", args, kwargs))
        return tokenizer

    monkeypatch.setattr(scoring.AutoTokenizer, "from_pretrained", from_pretrained)
    fake_fastokens = SimpleNamespace(
        _patched=True,
        unpatch_transformers=lambda: calls.append(("unpatch",)),
        patch_transformers=lambda: calls.append(("repatch",)),
    )
    monkeypatch.setitem(sys.modules, "fastokens", fake_fastokens)
    config = _cross_tokenizer_config()
    config["tokenizer"] = {
        "name": "teacher-tokenizer",
        "chat_template": "custom-template",
        "chat_template_kwargs": {"enable_thinking": False},
        "tokenizer_kwargs": {"revision": "stable"},
    }

    loaded = scoring.load_mopd_alignment_tokenizer(config)
    rendered = loaded.apply_chat_template([], tokenize=False)

    assert loaded is tokenizer
    assert calls == [
        ("unpatch",),
        (
            "load",
            ("teacher-tokenizer",),
            {"revision": "stable", "trust_remote_code": True},
        ),
        ("repatch",),
    ]
    assert tokenizer.chat_template == "custom-template"
    base_apply_chat_template.assert_called_once_with(
        [], tokenize=False, enable_thinking=False
    )
    assert rendered == "rendered"


def test_student_alignment_tokenizer_ignores_active_fastokens(monkeypatch):
    from nemo_rl.algorithms.x_token import mopd_teacher_scoring as scoring

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 1
    calls = []

    def from_pretrained(*args, **kwargs):
        calls.append(("load", args, kwargs))
        return tokenizer

    monkeypatch.setattr(scoring.AutoTokenizer, "from_pretrained", from_pretrained)
    fake_fastokens = SimpleNamespace(
        _patched=True,
        unpatch_transformers=lambda: calls.append(("unpatch",)),
        patch_transformers=lambda: calls.append(("repatch",)),
    )
    monkeypatch.setitem(sys.modules, "fastokens", fake_fastokens)

    loaded = scoring.load_mopd_student_alignment_tokenizer(
        {
            "name": "student-tokenizer",
            "chat_template": "default",
            "tokenizer_kwargs": {"revision": "stable"},
            "use_fastokens": True,
        }
    )

    assert loaded is tokenizer
    assert calls == [
        ("unpatch",),
        (
            "load",
            ("student-tokenizer",),
            {"revision": "stable", "trust_remote_code": True},
        ),
        ("repatch",),
    ]
