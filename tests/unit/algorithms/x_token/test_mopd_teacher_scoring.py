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

import os
import sys
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.x_token import mopd_teacher_scoring as scoring
from nemo_rl.algorithms.x_token.token_aligner import AlignmentPair, TokenAligner


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


class _PieceTokenizer(_CharTokenizer):
    def __init__(self, pieces: list[str]) -> None:
        self.pieces = dict(enumerate(pieces, start=1))
        self.ids = {piece: token_id for token_id, piece in self.pieces.items()}

    def __call__(
        self, text: str, return_offsets_mapping: bool = False, **kwargs
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        del kwargs
        ids = []
        offsets = []
        cursor = 0
        while cursor < len(text):
            piece = max(
                (piece for piece in self.ids if text.startswith(piece, cursor)),
                key=len,
            )
            ids.append(self.ids[piece])
            offsets.append((cursor, cursor + len(piece)))
            cursor += len(piece)
        result: dict[str, list[int] | list[tuple[int, int]]] = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode(
        self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = False
    ) -> str:
        del skip_special_tokens
        return "".join(self.pieces[int(token_id)] for token_id in ids if int(token_id))

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.pieces[int(token_id)] for token_id in ids]


class _Sharding:
    def get_axis_size(self, axis):
        assert axis == "data_parallel"
        return 1


class _TeacherGroup:
    cfg = {"max_total_sequence_length": 512}
    sharding_annotations = _Sharding()

    def __init__(self, logprobs: torch.Tensor | None = None) -> None:
        self.calls = 0
        self.logprobs = logprobs

    def get_logprobs(self, batch):
        self.calls += 1
        self.last_batch = batch
        if self.logprobs is not None:
            assert self.logprobs.shape == batch["input_ids"].shape
            return {"reference_logprobs": self.logprobs.clone()}
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


def test_cross_token_scorer_projects_nonconstant_scores_with_real_aligner() -> None:
    from nemo_rl.algorithms.x_token import mopd_teacher_scoring as scoring

    student_tokenizer = _PieceTokenizer(["q", "a", "b", "c", "d"])
    teacher_tokenizer = _PieceTokenizer(["q", "ab", "cd"])
    # The prompt score must be excluded; the two answer scores differ.
    teacher_group = _TeacherGroup(logprobs=torch.tensor([[-7.0, -2.0, -3.0]]))
    scorer = scoring.build_mopd_teacher_scorer(
        student_tokenizer=student_tokenizer,
        teacher_group=teacher_group,
        cross_tokenizer_config=_cross_tokenizer_config(),
        teacher_tokenizer=teacher_tokenizer,
    )

    result = scorer.score(
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        input_lengths=torch.tensor([5]),
        message_logs=[
            [
                {"role": "user", "content": "q", "token_ids": [1]},
                {
                    "role": "assistant",
                    "content": "abcd",
                    "token_ids": [2, 3, 4, 5],
                    "generation_logprobs": [0.0] * 4,
                },
            ]
        ],
    )

    # The real aligner maps [a, b] to [ab] and [c, d] to [cd].
    torch.testing.assert_close(
        result.logprobs, torch.tensor([[0.0, -1.0, -1.0, -1.5, -1.5]])
    )
    assert torch.equal(
        result.valid_mask, torch.tensor([[False, True, True, True, True]])
    )
    assert teacher_group.calls == 1
    torch.testing.assert_close(
        teacher_group.last_batch["input_ids"], torch.tensor([[1, 2, 3]])
    )
    torch.testing.assert_close(
        teacher_group.last_batch["input_lengths"], torch.tensor([3])
    )


@pytest.mark.parametrize(
    (
        "only_unmask_final",
        "supervise_tool_call",
        "expected_alignment_calls",
        "final_answer",
    ),
    [
        (False, True, 2, "It is sunny."),
        (True, False, 1, "It is sunny."),
        # Select the logical final generated assistant before checking whether
        # it is alignable. An empty final turn must not fall back to the prior
        # tool-call turn.
        (True, False, 0, ""),
    ],
)
def test_cross_token_scorer_applies_generated_turn_mask_in_tool_loop(
    monkeypatch,
    only_unmask_final,
    supervise_tool_call,
    expected_alignment_calls,
    final_answer,
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
        only_unmask_final=only_unmask_final,
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
    trailing_tool_result = "logged"
    prompt_ids = student_tokenizer("q")["input_ids"]
    tool_call_ids = student_tokenizer(tool_call)["input_ids"]
    tool_result_ids = student_tokenizer(tool_result)["input_ids"]
    final_answer_ids = student_tokenizer(final_answer)["input_ids"]
    trailing_tool_result_ids = student_tokenizer(trailing_tool_result)["input_ids"]
    input_ids = torch.tensor(
        [
            prompt_ids
            + tool_call_ids
            + tool_result_ids
            + final_answer_ids
            + trailing_tool_result_ids
        ],
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
                {
                    "role": "tool",
                    "content": trailing_tool_result,
                    "token_ids": trailing_tool_result_ids,
                },
            ]
        ],
    )

    expected_scores = torch.tensor(
        [
            [0.0] * len(prompt_ids)
            + ([-2.0] if supervise_tool_call else [0.0]) * len(tool_call_ids)
            + [0.0] * len(tool_result_ids)
            + [-2.0] * len(final_answer_ids)
            + [0.0] * len(trailing_tool_result_ids)
        ]
    )
    expected_mask = torch.tensor(
        [
            [False] * len(prompt_ids)
            + [supervise_tool_call] * len(tool_call_ids)
            + [False] * len(tool_result_ids)
            + [True] * len(final_answer_ids)
            + [False] * len(trailing_tool_result_ids)
        ]
    )
    torch.testing.assert_close(result.logprobs, expected_scores)
    assert torch.equal(result.valid_mask, expected_mask)
    assert teacher_group.calls == 1
    assert len(alignment_calls) == expected_alignment_calls
    assert all(call[2] == "offset_cluster_decode_fix" for call in alignment_calls)
    assert result.metrics["mopd/turns_aligned"] == float(expected_alignment_calls)
    assert result.metrics["mopd/turns_excluded_by_loss_mask"] == float(
        int(only_unmask_final)
    )
    teacher_length = int(teacher_group.last_batch["input_lengths"][0].item())
    assert teacher_tokenizer.decode(
        teacher_group.last_batch["input_ids"][0, :teacher_length]
    ) == ("q" + tool_call + tool_result + final_answer + trailing_tool_result)


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


@pytest.fixture(scope="module")
def native_qwen3_tokenizer() -> PreTrainedTokenizerBase:
    """Load only the pinned, unmodified Qwen3 tokenizer and native template.

    NRL_TEST_QWEN3_TOKENIZER may point to a local copy of these tokenizer assets
    for offline execution. No model configuration or weights are loaded.
    """
    return AutoTokenizer.from_pretrained(
        os.environ.get("NRL_TEST_QWEN3_TOKENIZER", "Qwen/Qwen3-0.6B"),
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
    )


class _PositionTeacherGroup(_TeacherGroup):
    def get_logprobs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.calls += 1
        self.last_batch = batch
        positions = torch.arange(1, batch["input_ids"].shape[1] + 1).float()
        return {
            "reference_logprobs": -positions.unsqueeze(0).expand_as(batch["input_ids"])
        }


def _native_scoring_case(
    tokenizer: PreTrainedTokenizerBase,
    response: str,
    *,
    prompt_suffix: str = "",
    enable_thinking: bool = True,
    split_opening: bool = False,
    missing_close_policy: str = "mask",
    mask_prefix: bool = False,
    teacher_tokenizer: PreTrainedTokenizerBase | None = None,
) -> tuple[
    scoring.MOPDTeacherScorer,
    _PositionTeacherGroup,
    list[int],
    list[int],
    list[dict[str, Any]],
]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "q"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if enable_thinking:
        assert prompt.endswith("<|im_start|>assistant\n")
    else:
        assert prompt.endswith("<think>\n\n</think>\n\n")
    prompt_ids = tokenizer(prompt + prompt_suffix, add_special_tokens=False)[
        "input_ids"
    ]
    if split_opening:
        assert response.startswith("<think>")
        # A valid rollout need not use the canonical single marker token.
        response_ids = [
            token_id
            for fragment in ("<", "think", ">", response[len("<think>") :])
            for token_id in tokenizer(fragment, add_special_tokens=False)["input_ids"]
        ]
        assert len(tokenizer("<think>", add_special_tokens=False)["input_ids"]) == 1
        assert tokenizer.decode(response_ids, skip_special_tokens=False) == response
    else:
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    messages = [
        {"role": "user", "content": "q", "token_ids": prompt_ids},
        {
            "role": "assistant",
            "content": response,
            "token_ids": response_ids,
            "generation_logprobs": [0.0] * len(response_ids),
        },
    ]
    config = _cross_tokenizer_config()
    config["missing_think_close_policy"] = missing_close_policy
    config["mask_first_teacher_prefix_chunk"] = mask_prefix
    config["exclude_proven_template_only_teacher_tokens"] = True
    teacher = _PositionTeacherGroup()
    scorer = scoring.build_mopd_teacher_scorer(
        student_tokenizer=tokenizer,
        teacher_tokenizer=teacher_tokenizer or tokenizer,
        teacher_group=teacher,
        cross_tokenizer_config=config,
    )
    assert isinstance(scorer.context.aligner, TokenAligner)
    return scorer, teacher, prompt_ids, response_ids, messages


def _assert_native_scores(
    *,
    tokenizer: PreTrainedTokenizerBase,
    scorer: scoring.MOPDTeacherScorer,
    teacher: _PositionTeacherGroup,
    prompt_ids: list[int],
    response_ids: list[int],
    messages: list[dict[str, Any]],
    teacher_text: str,
    dropped_prefix_tokens: int,
    mask_prefix: bool = False,
) -> scoring.MOPDTeacherScoreResult:
    student_ids = prompt_ids + response_ids
    prepared = scorer._prepare_sample(
        student_row_ids=student_ids,
        student_length=len(student_ids),
        message_log=messages,
        metrics=Counter(),
    )
    assert (
        tokenizer.decode(prepared.teacher_ids, skip_special_tokens=False)
        == teacher_text
    )
    assert len(prepared.turns) == 1
    turn = prepared.turns[0]
    assert turn.student_global_start == len(prompt_ids) + dropped_prefix_tokens
    # Assert absolute positions on the original rollout grid, including sampled EOS.
    assert sorted(
        turn.student_global_start + position
        for pair in turn.pairs
        if pair.s_start >= 0 and pair.is_correct
        for position in range(pair.s_start, pair.s_end)
    ) == list(range(len(prompt_ids) + dropped_prefix_tokens, len(student_ids)))

    result = scorer.score(
        input_ids=torch.tensor([student_ids]),
        input_lengths=torch.tensor([len(student_ids)]),
        message_logs=[messages],
    )
    assert teacher.calls == 1
    teacher_ids = teacher.last_batch["input_ids"][0].tolist()
    assert teacher_ids == list(prepared.teacher_ids)
    expected_mask = torch.zeros((1, len(student_ids)), dtype=torch.bool)
    expected_scores = torch.zeros((1, len(student_ids)), dtype=torch.float32)
    # These cases deliberately use distinct body token IDs. Locate their actual
    # teacher positions independently of the aligner's pairs and projection code.
    teacher_cursor = turn.teacher_global_start
    for local_position in range(dropped_prefix_tokens, len(response_ids)):
        teacher_position = teacher_ids.index(
            response_ids[local_position], teacher_cursor
        )
        teacher_cursor = teacher_position + 1
        if mask_prefix and local_position == dropped_prefix_tokens:
            continue
        student_position = len(prompt_ids) + local_position
        expected_mask[0, student_position] = True
        expected_scores[0, student_position] = -(teacher_position + 1)
    torch.testing.assert_close(result.logprobs, expected_scores)
    assert torch.equal(result.valid_mask, expected_mask)
    assert result.metrics["mopd/teacher_prefix_chunks_masked"] == float(mask_prefix)
    assert result.metrics["mopd/student_offset_failures"] == 0
    return result


@pytest.mark.parametrize("split_opening", [False, True])
@pytest.mark.parametrize("mask_prefix", [False, True])
@pytest.mark.parametrize("reasoning", ["reasoning", ""])
def test_native_qwen3_generated_thinking_scores_original_positions(
    native_qwen3_tokenizer: PreTrainedTokenizerBase,
    split_opening: bool,
    mask_prefix: bool,
    reasoning: str,
) -> None:
    tokenizer = native_qwen3_tokenizer
    scorer, teacher, prompt_ids, response_ids, messages = _native_scoring_case(
        tokenizer,
        f"<think>{reasoning}</think>answer<|im_end|>",
        split_opening=split_opening,
        mask_prefix=mask_prefix,
    )
    result = _assert_native_scores(
        tokenizer=tokenizer,
        scorer=scorer,
        teacher=teacher,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        messages=messages,
        teacher_text=(
            "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
            f"<think>\n{reasoning}\n</think>\n\nanswer<|im_end|>\n"
        ),
        dropped_prefix_tokens=3 if split_opening else 1,
        mask_prefix=mask_prefix,
    )
    assert result.metrics.get("mopd/missing_think_close_detected", 0) == 0
    if not mask_prefix or reasoning:
        assert result.metrics["mopd/template_only_teacher_tokens_excluded"] > 0


@pytest.mark.parametrize("split_opening", [False, True])
@pytest.mark.parametrize("sample_eot", [False, True])
@pytest.mark.parametrize("missing_close_policy", ["mask", "preserve_open_if_proven"])
def test_native_qwen3_unfinished_generated_thinking_policy(
    native_qwen3_tokenizer: PreTrainedTokenizerBase,
    split_opening: bool,
    sample_eot: bool,
    missing_close_policy: str,
) -> None:
    tokenizer = native_qwen3_tokenizer
    eos = "<|im_end|>" if sample_eot else ""
    scorer, teacher, prompt_ids, response_ids, messages = _native_scoring_case(
        tokenizer,
        "<think>\nreasoning" + eos,
        split_opening=split_opening,
        missing_close_policy=missing_close_policy,
    )
    if missing_close_policy == "preserve_open_if_proven":
        result = _assert_native_scores(
            tokenizer=tokenizer,
            scorer=scorer,
            teacher=teacher,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            messages=messages,
            teacher_text=(
                "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
                "<think>\nreasoning" + eos
            ),
            dropped_prefix_tokens=4 if split_opening else 2,
        )
        assert result.metrics["mopd/open_state_preserved"] == 1
    else:
        student_ids = prompt_ids + response_ids
        prepared = scorer._prepare_sample(
            student_row_ids=student_ids,
            student_length=len(student_ids),
            message_log=messages,
            metrics=Counter(),
        )
        assert tokenizer.decode(prepared.teacher_ids, skip_special_tokens=False) == (
            "<|im_start|>user\nq<|im_end|>\n"
        )
        assert not prepared.turns
        result = scorer.score(
            input_ids=torch.tensor([student_ids]), message_logs=[messages]
        )
        assert not result.valid_mask.any()
        assert not result.logprobs.any()
        assert teacher.calls == 1
        assert teacher.last_batch["input_ids"][0].tolist() == list(prepared.teacher_ids)
        assert result.metrics["mopd/malformed_structure_masked"] == 1
    assert result.metrics["mopd/missing_think_close_detected"] == 1


@pytest.mark.parametrize(
    "response",
    [
        "answer",
        "<think><think>reasoning</think>answer",
        "answer<think>reasoning</think>",
        "</think><think>reasoning",
        "<think>reasoning</think><think>again</think>",
        "<think>reasoning</think></think>answer",
        "</think>answer",
        "<think>reasoning</think>answer</tool_call>",
    ],
)
def test_native_qwen3_malformed_generated_markers_mask_turn(
    native_qwen3_tokenizer: PreTrainedTokenizerBase,
    response: str,
) -> None:
    scorer, teacher, prompt_ids, response_ids, messages = _native_scoring_case(
        native_qwen3_tokenizer,
        response,
        missing_close_policy="preserve_open_if_proven",
    )
    result = scorer.score(
        input_ids=torch.tensor([prompt_ids + response_ids]), message_logs=[messages]
    )
    assert not result.valid_mask.any()
    assert not result.logprobs.any()
    assert teacher.calls == 1
    assert (
        native_qwen3_tokenizer.decode(
            teacher.last_batch["input_ids"][0], skip_special_tokens=False
        )
        == "<|im_start|>user\nq<|im_end|>\n"
    )
    assert result.metrics["mopd/malformed_structure_masked"] == 1


@pytest.mark.parametrize("failure", ["student_reconstruction", "teacher_render"])
def test_native_qwen3_unproven_preservation_masks_turn(
    native_qwen3_tokenizer: PreTrainedTokenizerBase,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    tokenizer = native_qwen3_tokenizer
    teacher_tokenizer = None
    if failure == "teacher_render":
        teacher_tokenizer = deepcopy(tokenizer)
        # This teacher closes thinking even when asked for an open prefix.
        teacher_tokenizer.chat_template = (
            "{% for message in messages %}{{ '<|im_start|>' + message.role + '\\n' "
            "+ message.content + '<|im_end|>\\n' }}{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}{% endif %}"
        )
    scorer, teacher, prompt_ids, response_ids, messages = _native_scoring_case(
        tokenizer,
        "<think>reasoning",
        missing_close_policy="preserve_open_if_proven",
        teacher_tokenizer=teacher_tokenizer,
    )
    if failure == "student_reconstruction":
        recover = scorer._recover_transcript

        def unproven_recovery(
            **kwargs: Any,
        ) -> tuple[list[dict[str, str]], dict[int, int]]:
            recovered_messages, mapping = recover(**kwargs)
            recovered_messages[-1]["content"] = "reasoning"
            return recovered_messages, mapping

        monkeypatch.setattr(scorer, "_recover_transcript", unproven_recovery)
    student_ids = prompt_ids + response_ids
    prepared = scorer._prepare_sample(
        student_row_ids=student_ids,
        student_length=len(student_ids),
        message_log=messages,
        metrics=Counter(),
    )
    assert tokenizer.decode(prepared.teacher_ids, skip_special_tokens=False) == (
        "<|im_start|>user\nq<|im_end|>\n"
    )
    assert not prepared.turns
    result = scorer.score(
        input_ids=torch.tensor([student_ids]), message_logs=[messages]
    )
    assert not result.valid_mask.any()
    assert not result.logprobs.any()
    assert teacher.calls == 1
    assert teacher.last_batch["input_ids"][0].tolist() == list(prepared.teacher_ids)
    assert result.metrics["mopd/missing_think_close_detected"] == 1
    assert result.metrics["mopd/malformed_structure_masked"] == 1
    if failure == "teacher_render":
        assert result.metrics["mopd/teacher_open_render_failures"] == 1


@pytest.mark.parametrize("unfinished", [False, True])
def test_nano_style_prompt_prefilled_thinking_keeps_positions(
    native_qwen3_tokenizer: PreTrainedTokenizerBase,
    unfinished: bool,
) -> None:
    tokenizer = native_qwen3_tokenizer
    response = "reasoning" if unfinished else "reasoning</think>answer<|im_end|>"
    scorer, teacher, prompt_ids, response_ids, messages = _native_scoring_case(
        tokenizer,
        response,
        prompt_suffix="<think>\n",
        missing_close_policy="preserve_open_if_proven",
    )
    _assert_native_scores(
        tokenizer=tokenizer,
        scorer=scorer,
        teacher=teacher,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        messages=messages,
        teacher_text=(
            "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\nreasoning"
            + ("" if unfinished else "\n</think>\n\nanswer<|im_end|>\n")
        ),
        dropped_prefix_tokens=0,
    )


def test_native_qwen3_closed_prompt_keeps_plain_answer_scores(
    native_qwen3_tokenizer: PreTrainedTokenizerBase,
) -> None:
    tokenizer = native_qwen3_tokenizer
    scorer, teacher, prompt_ids, response_ids, messages = _native_scoring_case(
        tokenizer, "answer<|im_end|>", enable_thinking=False
    )
    _assert_native_scores(
        tokenizer=tokenizer,
        scorer=scorer,
        teacher=teacher,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        messages=messages,
        teacher_text=(
            "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\nanswer<|im_end|>\n"
        ),
        dropped_prefix_tokens=0,
    )


@pytest.mark.parametrize("policy", ["mask", "preserve_open_if_proven"])
def test_unfinished_native_thinking_masks_later_dependent_turns(
    native_qwen3_tokenizer: PreTrainedTokenizerBase,
    policy: str,
) -> None:
    tokenizer = native_qwen3_tokenizer
    scorer, teacher, prompt_ids, response_ids, messages = _native_scoring_case(
        tokenizer, "<think>reasoning<|im_end|>", missing_close_policy=policy
    )
    next_prompt_ids = tokenizer(
        "\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )["input_ids"]
    next_response_ids = tokenizer(
        "<think>other</think>done<|im_end|>", add_special_tokens=False
    )["input_ids"]
    messages.extend(
        [
            {"role": "user", "content": "next", "token_ids": next_prompt_ids},
            {
                "role": "assistant",
                "content": "<think>other</think>done",
                "token_ids": next_response_ids,
                "generation_logprobs": [0.0] * len(next_response_ids),
            },
        ]
    )
    first_turn_end = len(prompt_ids) + len(response_ids)
    student_ids = prompt_ids + response_ids + next_prompt_ids + next_response_ids
    prepared = scorer._prepare_sample(
        student_row_ids=student_ids,
        student_length=len(student_ids),
        message_log=messages,
        metrics=Counter(),
    )
    expected_teacher_text = "<|im_start|>user\nq<|im_end|>\n"
    if policy == "preserve_open_if_proven":
        expected_teacher_text += "<|im_start|>assistant\n<think>\nreasoning<|im_end|>"
    assert tokenizer.decode(prepared.teacher_ids, skip_special_tokens=False) == (
        expected_teacher_text
    )
    result = scorer.score(
        input_ids=torch.tensor([student_ids]), message_logs=[messages]
    )
    expected_mask = torch.zeros_like(result.valid_mask)
    expected_scores = torch.zeros_like(result.logprobs)
    if policy == "preserve_open_if_proven":
        expected_mask[0, len(prompt_ids) + 1 : first_turn_end] = True
        teacher_cursor = prepared.turns[0].teacher_global_start
        for local_position, token_id in enumerate(response_ids[1:], start=1):
            teacher_position = prepared.teacher_ids.index(token_id, teacher_cursor)
            teacher_cursor = teacher_position + 1
            expected_scores[0, len(prompt_ids) + local_position] = -(
                teacher_position + 1
            )
    assert torch.equal(result.valid_mask, expected_mask)
    torch.testing.assert_close(result.logprobs, expected_scores)
    assert teacher.calls == 1
    assert teacher.last_batch["input_ids"][0].tolist() == list(prepared.teacher_ids)
    assert result.metrics["mopd/missing_think_close_detected"] == 1
    assert result.metrics["mopd/causal_suffix_masked"] == 1


@pytest.mark.parametrize("response", ["<think>", "<think>\n", "<|im_end|>"])
@pytest.mark.parametrize("policy", ["mask", "preserve_open_if_proven"])
def test_native_qwen3_rollout_without_body_is_fully_masked(
    native_qwen3_tokenizer: PreTrainedTokenizerBase,
    response: str,
    policy: str,
) -> None:
    scorer, teacher, prompt_ids, response_ids, messages = _native_scoring_case(
        native_qwen3_tokenizer, response, missing_close_policy=policy
    )
    result = scorer.score(
        input_ids=torch.tensor([prompt_ids + response_ids]), message_logs=[messages]
    )
    assert not result.valid_mask.any()
    assert not result.logprobs.any()
    assert teacher.calls == 1
    assert result.metrics.get("mopd/sample_prepare_failures", 0) == 0
    expected_teacher_text = "<|im_start|>user\nq<|im_end|>\n"
    if policy == "preserve_open_if_proven" and response.startswith("<think>"):
        expected_teacher_text += "<|im_start|>assistant\n<think>\n"
    assert (
        native_qwen3_tokenizer.decode(
            teacher.last_batch["input_ids"][0], skip_special_tokens=False
        )
        == expected_teacher_text
    )


def test_scorer_emits_every_counter_on_clean_batch() -> None:
    """Rollout metrics are averaged per key over only the groups that report it."""
    import ast
    import inspect

    scorer = scoring.build_mopd_teacher_scorer(
        student_tokenizer=_PieceTokenizer(["q", "a", "b", "c", "d"]),
        teacher_group=_TeacherGroup(logprobs=torch.tensor([[-7.0, -2.0, -3.0]])),
        cross_tokenizer_config=_cross_tokenizer_config(),
        teacher_tokenizer=_PieceTokenizer(["q", "ab", "cd"]),
    )
    result = scorer.score(
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        message_logs=[
            [
                {"role": "user", "content": "q", "token_ids": [1]},
                {
                    "role": "assistant",
                    "content": "abcd",
                    "token_ids": [2, 3, 4, 5],
                    "generation_logprobs": [0.0] * 4,
                },
            ]
        ],
    )
    counters = {
        node.slice.value
        for node in ast.walk(ast.parse(inspect.getsource(scoring)))
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "metrics"
        and isinstance(node.slice, ast.Constant)
    }
    assert result.valid_mask.any()
    missing = sorted(name for name in counters if f"mopd/{name}" not in result.metrics)
    assert not missing, f"clean batch omits counters: {missing}"

