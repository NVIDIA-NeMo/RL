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

"""Regression tests for strict offset matching and native chat regions."""

from __future__ import annotations

import pytest
import torch

from nemo_rl.algorithms.x_token import token_aligner as ta
from nemo_rl.algorithms.x_token.token_aligner import (
    AlignmentBatch,
    NativeAlignmentPart,
    NativeAlignmentRegions,
    TokenAligner,
)


@pytest.mark.parametrize("structured_parts", [False, True])
def test_native_part_coalesces_nfc_offsets_without_changing_token_ids(structured_parts):
    student = _OffsetTokenizer({1: "e", 2: "́"})
    teacher = _OffsetTokenizer({7: "é"})
    aligner = TokenAligner(student, teacher, None)
    part = NativeAlignmentPart(
        name="prose", span=(0, 2), text="é", allow_native_difference=False
    )
    pairs = aligner.align_one_offset_per_asst(
        [1, 2],
        [(0, 1), (1, 2)],
        [(0, 2)],
        [7],
        [(0, 1)],
        [(0, 2)],
        student_alignment_regions=[NativeAlignmentRegions(answer=(0, 2))],
        teacher_alignment_regions=[NativeAlignmentRegions(answer=(0, 2))],
        student_answer_parts=[[part]] if structured_parts else None,
        teacher_answer_parts=[[part]] if structured_parts else None,
        student_rendered_text="é",
        teacher_rendered_text="é",
        included_regions=["answer"],
    )
    assert [(p.s_start, p.s_end, p.t_start, p.t_end) for p in pairs] == [(0, 2, 0, 1)]
    assert all((p.is_correct for p in pairs))


def test_native_inline_content_coalesces_nested_checkmark_offsets():
    class ByteTokenizer(_OffsetTokenizer):
        def __init__(self, chunks):
            super().__init__({i: chunk.hex() for i, chunk in enumerate(chunks)})
            self.chunks = chunks

        def decode(self, ids, **kwargs):
            return b"".join((self.chunks[i] for i in ids)).decode(
                "utf-8", errors="replace"
            )

    student = ByteTokenizer([b" \xe2\x9c", b"\x93"])
    teacher = ByteTokenizer([" ✓".encode()])
    aligner = TokenAligner(student, teacher, None)
    pairs = aligner.align_one_offset_per_asst(
        [0, 1],
        [(0, 2), (1, 2)],
        [(0, 2)],
        [0],
        [(0, 2)],
        [(0, 2)],
        student_alignment_regions=[NativeAlignmentRegions(answer=(0, 2))],
        teacher_alignment_regions=[NativeAlignmentRegions(answer=(0, 2))],
        student_rendered_text=" ✓",
        teacher_rendered_text=" ✓",
        included_regions=["answer"],
    )
    assert [(p.s_start, p.s_end, p.t_start, p.t_end) for p in pairs] == [(0, 2, 0, 1)]
    assert all((p.is_correct for p in pairs))


@pytest.mark.parametrize("text", ["👩", "�"])
def test_native_part_consumes_all_unicode_byte_tokens(text):
    encoded = text.encode("utf-8")

    class ByteTokenizer(_OffsetTokenizer):
        def __init__(self, chunks):
            super().__init__({i: str(i) for i in range(len(chunks))})
            self.chunks = chunks

        def decode(self, ids, **kwargs):
            return b"".join((self.chunks[i] for i in ids)).decode(
                "utf-8", errors="replace"
            )

    student = ByteTokenizer([bytes([x]) for x in encoded])
    teacher = ByteTokenizer([encoded[:-1], encoded[-1:]])
    aligner = TokenAligner(student, teacher, None)
    part = NativeAlignmentPart(
        name="prose", span=(0, 1), text=text, allow_native_difference=False
    )
    pairs = aligner.align_one_offset_per_asst(
        list(range(len(encoded))),
        [(0, 1)] * len(encoded),
        [(0, 1)],
        [0, 1],
        [(0, 1)] * 2,
        [(0, 1)],
        student_alignment_regions=[NativeAlignmentRegions(answer=(0, 1))],
        teacher_alignment_regions=[NativeAlignmentRegions(answer=(0, 1))],
        student_answer_parts=[[part]],
        teacher_answer_parts=[[part]],
        included_regions=["answer"],
    )
    assert [(p.s_start, p.s_end, p.t_start, p.t_end) for p in pairs] == [
        (0, len(encoded), 0, 2)
    ]
    assert all((p.is_correct for p in pairs))


class _FakeTokenizer:
    """Map id -> token string via a precomputed dict; minimal HF stand-in."""

    def __init__(self, id_to_tok: dict[int, str]):
        self._id_to_tok = id_to_tok

    def convert_ids_to_tokens(self, ids):
        return [self._id_to_tok.get(int(i), f"<unk{i}>") for i in ids]


class _OffsetTokenizer(_FakeTokenizer):
    bos_token_id = None
    eos_token_id = None
    pad_token_id = None
    unk_token_id = None
    sep_token_id = None
    cls_token_id = None
    mask_token_id = None
    all_special_ids: list[int] = []

    def decode(
        self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join((self._id_to_tok[int(i)] for i in ids))


class _CountingOffsetTokenizer(_OffsetTokenizer):
    def __init__(self, id_to_tok: dict[int, str]):
        super().__init__(id_to_tok)
        self.convert_calls = 0

    def convert_ids_to_tokens(self, ids):
        self.convert_calls += 1
        return super().convert_ids_to_tokens(ids)


def test_offset_cluster_decode_fix_preserves_multitoken_spans():
    student = _OffsetTokenizer({1: "he", 2: "llo"})
    teacher = _OffsetTokenizer({7: "hello"})
    aligner = TokenAligner(
        student_tokenizer=student,
        teacher_tokenizer=teacher,
        projection_matrix_path="/unused",
    )
    result = aligner.align(
        torch.tensor([[1, 2]]),
        torch.tensor([[7]]),
        student_offsets=torch.tensor([[(0, 2), (2, 5)]]),
        teacher_offsets=torch.tensor([[(0, 5)]]),
    )
    assert result.pair_valid.tolist() == [[True]]
    assert result.pair_is_correct.tolist() == [[True]]
    assert result.student_chunk_id.tolist() == [[0, 0]]
    assert result.teacher_chunk_id.tolist() == [[0]]


def test_offset_cluster_reuses_tokens_converted_for_canonical_merge_scan():
    student = _CountingOffsetTokenizer({1: "he", 2: "llo"})
    teacher = _CountingOffsetTokenizer({7: "hello"})
    aligner = TokenAligner(
        student_tokenizer=student,
        teacher_tokenizer=teacher,
        projection_matrix_path="/unused",
    )
    aligner.align(
        torch.tensor([[1, 2]]),
        torch.tensor([[7]]),
        student_offsets=torch.tensor([[(0, 2), (2, 5)]]),
        teacher_offsets=torch.tensor([[(0, 5)]]),
    )
    assert student.convert_calls == 1
    assert teacher.convert_calls == 1


def test_offset_cluster_decode_fix_merges_nested_byte_artifact():
    student = _OffsetTokenizer({1: "A", 17194: "Ġâī", 1160: "ł", 2: "B"})
    teacher = _OffsetTokenizer({1: "A", 91050: "Ġâīł", 2: "B"})
    aligner = TokenAligner(
        student_tokenizer=student,
        teacher_tokenizer=teacher,
        projection_matrix_path="/unused",
    )
    result = aligner.align(
        torch.tensor([[1, 17194, 1160, 2]]),
        torch.tensor([[1, 91050, 2]]),
        student_offsets=torch.tensor([[(0, 1), (1, 3), (2, 3), (3, 4)]]),
        teacher_offsets=torch.tensor([[(0, 1), (1, 3), (3, 4)]]),
    )
    assert result.pair_valid.tolist() == [[True, True, True]]
    assert result.pair_is_correct.tolist() == [[True, True, True]]
    assert result.student_chunk_id.tolist() == [[0, 1, 1, 2]]
    assert result.teacher_chunk_id.tolist() == [[0, 1, 2]]


def test_offset_cluster_decode_fix_merges_nested_teacher_byte_artifact():
    student = _OffsetTokenizer({91050: "Ġâīł"})
    teacher = _OffsetTokenizer({17194: "Ġâī", 1160: "ł"})
    aligner = TokenAligner(
        student_tokenizer=student,
        teacher_tokenizer=teacher,
        projection_matrix_path="/unused",
    )
    result = aligner.align(
        torch.tensor([[91050]]),
        torch.tensor([[17194, 1160]]),
        student_offsets=torch.tensor([[(0, 2)]]),
        teacher_offsets=torch.tensor([[(0, 2), (1, 2)]]),
    )
    assert result.pair_valid.tolist() == [[True]]
    assert result.pair_is_correct.tolist() == [[True]]
    assert result.student_chunk_id.tolist() == [[0]]
    assert result.teacher_chunk_id.tolist() == [[0, 0]]


def test_offset_cluster_decode_fix_does_not_merge_unrecognized_overlap():
    student = _OffsetTokenizer({1: "A", 2: "outer", 3: "nested", 4: "B"})
    teacher = _OffsetTokenizer({1: "A", 2: "outer", 4: "B"})
    aligner = TokenAligner(
        student_tokenizer=student,
        teacher_tokenizer=teacher,
        projection_matrix_path="/unused",
    )
    result = aligner.align(
        torch.tensor([[1, 2, 3, 4]]),
        torch.tensor([[1, 2, 4]]),
        student_offsets=torch.tensor([[(0, 1), (1, 3), (2, 3), (3, 4)]]),
        teacher_offsets=torch.tensor([[(0, 1), (1, 3), (3, 4)]]),
    )
    assert result.student_chunk_id.tolist() == [[0, 1, 2, 3]]
    assert result.teacher_chunk_id.tolist() == [[0, 1, 3]]
    assert result.pair_is_correct.tolist() == [[True, True, False, True]]


@pytest.mark.parametrize(
    ("offsets", "special_token_ids"),
    [
        ([(1, 2), (3, 4)], []),
        ([(2, 3), (1, 3)], []),
        ([(1, 3), (0, 0)], []),
        ([(1, 3), (2, 3)], [1160]),
    ],
    ids=["gap", "non_monotonic", "zero_width", "special"],
)
def test_canonical_merge_offsets_leave_unsafe_ranges_unchanged(
    offsets, special_token_ids
):
    assert (
        ta._normalize_canonical_merge_offsets(
            ["Ġâī", "ł"],
            offsets,
            token_ids=[17194, 1160],
            special_token_ids=special_token_ids,
        )
        == offsets
    )


def test_offset_cluster_byte_artifact_preserves_native_assistant_indices():
    student = _OffsetTokenizer({9: "P", 17194: "Ġâī", 1160: "ł", 8: "E"})
    teacher = _OffsetTokenizer({9: "P", 91050: "Ġâīł", 8: "E"})
    aligner = TokenAligner(
        student_tokenizer=student,
        teacher_tokenizer=teacher,
        projection_matrix_path="/unused",
    )
    pairs = aligner.align_one_offset_per_asst(
        [9, 17194, 1160, 8],
        [(0, 1), (10, 12), (11, 12), (12, 13)],
        [(10, 12)],
        [9, 91050, 8],
        [(0, 1), (20, 22), (22, 23)],
        [(20, 22)],
        student_asst_mask=[0, 1, 1, 0],
        teacher_asst_mask=[0, 1, 0],
        student_eot_indices=[-1],
        teacher_eot_indices=[-1],
    )
    assert len(pairs) == 1
    assert pairs[0].s_start == 1
    assert pairs[0].s_end == 3
    assert pairs[0].t_start == 1
    assert pairs[0].t_end == 2
    assert pairs[0].is_correct


def test_offset_cluster_byte_artifact_preserves_second_turn_semantic_region():
    student = _OffsetTokenizer({9: "P", 3: "X", 17194: "Ġâī", 1160: "ł", 8: "E"})
    teacher = _OffsetTokenizer({9: "P", 3: "X", 91050: "Ġâīł", 8: "E"})
    aligner = TokenAligner(
        student_tokenizer=student,
        teacher_tokenizer=teacher,
        projection_matrix_path="/unused",
    )
    pairs = aligner.align_one_offset_per_asst(
        [9, 3, 9, 17194, 1160, 8],
        [(0, 1), (10, 11), (12, 13), (30, 32), (31, 32), (33, 34)],
        [(10, 11), (30, 32)],
        [9, 3, 9, 91050, 8],
        [(0, 1), (20, 21), (22, 23), (40, 42), (43, 44)],
        [(20, 21), (40, 42)],
        student_asst_mask=[0, 1, 0, 1, 1, 0],
        teacher_asst_mask=[0, 1, 0, 1, 0],
        student_alignment_regions=[
            NativeAlignmentRegions(answer=(10, 11)),
            NativeAlignmentRegions(answer=(30, 32)),
        ],
        teacher_alignment_regions=[
            NativeAlignmentRegions(answer=(20, 21)),
            NativeAlignmentRegions(answer=(40, 42)),
        ],
        student_eot_indices=[-1, -1],
        teacher_eot_indices=[-1, -1],
        included_regions=["answer"],
    )
    assert [(pair.s_start, pair.s_end) for pair in pairs] == [(1, 2), (3, 5)]
    assert [(pair.t_start, pair.t_end) for pair in pairs] == [(1, 2), (3, 4)]
    assert all((pair.is_correct for pair in pairs))


@pytest.mark.parametrize(
    ("student_decoded", "teacher_decoded", "expected"),
    [
        ("same", "same", True),
        (" same", "same", False),
        ("\n", " ", False),
        (" ", "  ", False),
        ("e\u0301", "é", False),
        ("中", "中", True),
        ("", "", True),
    ],
)
def test_text_and_chat_use_identical_strict_decoded_matching(
    student_decoded, teacher_decoded, expected
):
    class DecodingTokenizer(_OffsetTokenizer):
        def __init__(self, surface, decoded):
            super().__init__({1: surface})
            self.decoded = decoded

        def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            assert skip_special_tokens is False
            assert clean_up_tokenization_spaces is False
            return self.decoded

    # Deliberately different vocab spellings with equal decodes still match.
    student = DecodingTokenizer("student_spelling", student_decoded)
    teacher = DecodingTokenizer("teacher_spelling", teacher_decoded)
    aligner = TokenAligner(student, teacher, "/unused")
    ids = torch.tensor([[1]])
    offsets = torch.tensor([[(0, 1)]])
    plain = aligner.align(ids, ids, student_offsets=offsets, teacher_offsets=offsets)
    chat = aligner.align_chat(
        ids,
        ids,
        student_offsets=offsets + 10,
        teacher_offsets=offsets + 20,
        student_asst_char_spans=[[(10, 11)]],
        teacher_asst_char_spans=[[(20, 21)]],
        student_eot_indices=[[-1]],
        teacher_eot_indices=[[-1]],
    )
    assert plain.pair_is_correct.tolist() == [[expected]]
    assert torch.equal(plain.pair_is_correct, chat.pair_is_correct)


def test_decoding_error_is_reported_without_fallback():
    class BrokenTokenizer(_OffsetTokenizer):
        def decode(self, ids, **kwargs):
            raise ValueError("invalid tokenizer decoder")

    aligner = TokenAligner(
        BrokenTokenizer({1: "same"}), _OffsetTokenizer({1: "same"}), "/unused"
    )
    with pytest.raises(ValueError, match="invalid tokenizer decoder"):
        aligner._align_one_offset([1], [1], [(0, 1)], [(0, 1)])


@pytest.mark.parametrize("source", ["x", "é", "中", " "])
def test_native_unicode_repair_rejects_equal_replacement_decodes(source):
    tokenizer = _OffsetTokenizer({1: "\ufffd"})
    aligner = TokenAligner(tokenizer, tokenizer, "/unused")
    with pytest.raises(ValueError, match="unequal decoded token spans"):
        aligner.align_chat(
            torch.tensor([[1]]),
            torch.tensor([[1]]),
            student_offsets=torch.tensor([[(0, 1)]]),
            teacher_offsets=torch.tensor([[(0, 1)]]),
            student_asst_char_spans=[[(0, 1)]],
            teacher_asst_char_spans=[[(0, 1)]],
            student_alignment_regions=[[NativeAlignmentRegions(answer=(0, 1))]],
            teacher_alignment_regions=[[NativeAlignmentRegions(answer=(0, 1))]],
            student_rendered_texts=[source],
            teacher_rendered_texts=[source],
            included_regions=["answer"],
        )


def test_native_unicode_repair_preserves_whitespace():
    student = _OffsetTokenizer({1: " x"})
    teacher = _OffsetTokenizer({1: "x"})
    aligner = TokenAligner(student, teacher, "/unused")
    with pytest.raises(ValueError, match="unequal decoded token spans"):
        aligner.align_one_offset_per_asst(
            [1],
            [(0, 2)],
            [(0, 2)],
            [1],
            [(0, 2)],
            [(0, 2)],
            student_alignment_regions=[NativeAlignmentRegions(answer=(0, 2))],
            teacher_alignment_regions=[NativeAlignmentRegions(answer=(0, 2))],
            student_rendered_text=" x",
            teacher_rendered_text=" x",
            included_regions=["answer"],
        )


def test_public_native_api_repairs_unicode_and_preserves_original_indices():
    student = _OffsetTokenizer({0: "header", 1: "e", 2: "\u0301", 3: "EOS"})
    teacher = _OffsetTokenizer({0: "header", 1: "é", 3: "EOS"})
    aligner = TokenAligner(student, teacher, "/unused")
    batch = aligner.align_chat(
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[0, 1, 3]]),
        student_offsets=torch.tensor([[(0, 2), (2, 3), (3, 4), (4, 7)]]),
        teacher_offsets=torch.tensor([[(0, 4), (4, 5), (6, 9)]]),
        student_asst_char_spans=[[(2, 4)]],
        teacher_asst_char_spans=[[(4, 6)]],
        student_alignment_regions=[[NativeAlignmentRegions(answer=(2, 4))]],
        teacher_alignment_regions=[[NativeAlignmentRegions(answer=(4, 6))]],
        student_answer_parts=[
            [[NativeAlignmentPart(name="prose", span=(2, 4), text="e\u0301")]]
        ],
        teacher_answer_parts=[
            [[NativeAlignmentPart(name="prose", span=(4, 6), text="e\u0301")]]
        ],
        student_rendered_texts=["##e\u0301EOS"],
        teacher_rendered_texts=["####e\u0301EOS"],
        student_eot_indices=[[3]],
        teacher_eot_indices=[[2]],
    )
    assert isinstance(batch, AlignmentBatch)
    assert batch.student_chunk_id.tolist() == [[-1, 0, 0, 1]]
    assert batch.teacher_chunk_id.tolist() == [[-1, 0, 1]]
    assert batch.pair_is_correct.tolist() == [[True, True]]


@pytest.mark.parametrize(
    "field",
    [
        "student_answer_parts",
        "teacher_answer_parts",
        "student_rendered_texts",
        "teacher_rendered_texts",
        "student_alignment_regions",
        "teacher_alignment_regions",
        "student_eot_indices",
        "teacher_eot_indices",
    ],
)
def test_public_native_api_validates_every_metadata_batch_size(field):
    tokenizer = _OffsetTokenizer({1: "x"})
    aligner = TokenAligner(tokenizer, tokenizer, "/unused")
    with pytest.raises(ValueError, match=f"{field} must have one entry"):
        aligner.align_chat(
            torch.tensor([[1]]),
            torch.tensor([[1]]),
            student_offsets=torch.tensor([[(0, 1)]]),
            teacher_offsets=torch.tensor([[(0, 1)]]),
            student_asst_char_spans=[[(0, 1)]],
            teacher_asst_char_spans=[[(0, 1)]],
            **{field: []},
        )


@pytest.mark.parametrize(
    ("student_eos", "teacher_eos", "expected"),
    [
        (1, 1, True),
        (None, 1, False),
        (1, None, False),
        (None, None, False),
    ],
)
def test_synthetic_eot_requires_surface_equality_or_both_eos(
    student_eos, teacher_eos, expected
):
    student = _OffsetTokenizer({1: "student_EOT"})
    teacher = _OffsetTokenizer({1: "teacher_EOT"})
    student.eos_token_id = student_eos
    teacher.eos_token_id = teacher_eos
    aligner = TokenAligner(student, teacher, "/unused")
    pairs = aligner.align_one_offset_per_asst(
        [1],
        [(0, 1)],
        [(0, 0)],
        [1],
        [(0, 1)],
        [(0, 0)],
        student_eot_indices=[0],
        teacher_eot_indices=[0],
    )
    assert len(pairs) == 1
    assert pairs[0].is_correct is expected


def test_orphan_empty_decodes_are_incorrect():
    student = _OffsetTokenizer({1: ""})
    teacher = _OffsetTokenizer({1: ""})
    pairs = TokenAligner(student, teacher, "/unused")._align_one_offset(
        [1], [1], [(0, 1)], [(0, 2)]
    )
    assert len(pairs) == 2
    assert not any(pair.is_correct for pair in pairs)


def test_kernel_rejects_inconsistent_preconverted_token_strings():
    tokenizer = _OffsetTokenizer({1: "x"})
    with pytest.raises(ValueError, match="student token strings"):
        ta.align_by_offsets_cluster(
            [1], [(0, 1)], tokenizer, [1], [(0, 1)], tokenizer, student_tokens_str=[]
        )


@pytest.mark.parametrize(
    ("student_text", "teacher_text", "allowed"),
    [
        ("True", "true", True),
        ("False", "false", True),
        ("None", "null", True),
        ("True", "false", False),
        ("True", " true", False),
        ("foo", "bar", False),
    ],
)
def test_native_scalar_differences_are_limited_to_same_boolean_or_null(
    student_text, teacher_text, allowed
):
    student = _OffsetTokenizer({1: student_text})
    teacher = _OffsetTokenizer({1: teacher_text})
    aligner = TokenAligner(student, teacher, "/unused")
    kwargs = dict(
        student_ids=[1],
        student_offsets=[(0, len(student_text))],
        student_asst_char_spans=[(0, len(student_text))],
        teacher_ids=[1],
        teacher_offsets=[(0, len(teacher_text))],
        teacher_asst_char_spans=[(0, len(teacher_text))],
        student_alignment_regions=[
            NativeAlignmentRegions(answer=(0, len(student_text)))
        ],
        teacher_alignment_regions=[
            NativeAlignmentRegions(answer=(0, len(teacher_text)))
        ],
        student_answer_parts=[
            [
                NativeAlignmentPart(
                    name="scalar",
                    span=(0, len(student_text)),
                    text=student_text,
                    allow_native_difference=True,
                )
            ]
        ],
        teacher_answer_parts=[
            [
                NativeAlignmentPart(
                    name="scalar",
                    span=(0, len(teacher_text)),
                    text=teacher_text,
                    allow_native_difference=True,
                )
            ]
        ],
        student_rendered_text=student_text,
        teacher_rendered_text=teacher_text,
        included_regions=["answer"],
    )
    if allowed:
        assert aligner.align_one_offset_per_asst(**kwargs) == []
    else:
        with pytest.raises(ValueError, match="refusing offset-based KD"):
            aligner.align_one_offset_per_asst(**kwargs)


def test_native_answer_parts_must_match_rendered_source():
    tokenizer = _OffsetTokenizer({1: "x"})
    aligner = TokenAligner(tokenizer, tokenizer, "/unused")
    part = NativeAlignmentPart(name="prose", span=(0, 1), text="x")
    with pytest.raises(
        ValueError, match="student turn 0.*differs from rendered source"
    ):
        aligner.align_one_offset_per_asst(
            [1],
            [(0, 1)],
            [(0, 1)],
            [1],
            [(0, 1)],
            [(0, 1)],
            student_alignment_regions=[NativeAlignmentRegions(answer=(0, 1))],
            teacher_alignment_regions=[NativeAlignmentRegions(answer=(0, 1))],
            student_answer_parts=[[part]],
            teacher_answer_parts=[[part]],
            student_rendered_text="z",
            teacher_rendered_text="x",
            included_regions=["answer"],
        )


@pytest.mark.parametrize("offsets", [[(0, 0), (1, 1)], [(3, 2), (2, 3)]])
def test_canonical_merge_offsets_preserve_empty_and_reversed_ranges(offsets):
    assert (
        ta._normalize_canonical_merge_offsets(
            ["Ġâī", "ł"], offsets, token_ids=[1, 2], special_token_ids=[]
        )
        == offsets
    )


@pytest.mark.parametrize("native", [False, True])
def test_ordinary_canonical_boundary_tokens_retain_strict_decode_labels(native):
    student = _OffsetTokenizer({1: " Hello"})
    teacher = _OffsetTokenizer({1: "Hello"})
    aligner = TokenAligner(student, teacher, "/unused")
    kwargs = dict(
        student_asst_mask=[1],
        teacher_asst_mask=[1],
        student_eot_indices=[-1],
        teacher_eot_indices=[-1],
    )
    if native:
        kwargs.update(
            student_alignment_regions=[NativeAlignmentRegions(answer=(1, 6))],
            teacher_alignment_regions=[NativeAlignmentRegions(answer=(0, 5))],
        )
    pairs = aligner.align_one_offset_per_asst(
        [1], [(0, 6)], [(1, 6)], [1], [(0, 5)], [(0, 5)], **kwargs
    )
    if native:
        # Native boundaries deliberately omit tokens crossing formatting.
        assert pairs == []
    else:
        assert len(pairs) == 1
        assert pairs[0].s_start == pairs[0].t_start == 0
        assert pairs[0].is_correct is False


def test_cjk_byte_artifact_normalization_keeps_full_original_token_range():
    student = _OffsetTokenizer({1: "prefix", 2: "Ġä¸", 3: "Ń", 4: "suffix"})
    teacher = _OffsetTokenizer({1: "prefix", 2: "Ġä¸Ń", 4: "suffix"})
    aligner = TokenAligner(student, teacher, "/unused")
    result = aligner.align(
        torch.tensor([[1, 2, 3, 4]]),
        torch.tensor([[1, 2, 4]]),
        student_offsets=torch.tensor([[(0, 6), (6, 8), (7, 8), (8, 14)]]),
        teacher_offsets=torch.tensor([[(0, 6), (6, 8), (8, 14)]]),
    )
    assert result.student_chunk_id.tolist() == [[0, 1, 1, 2]]
    assert result.teacher_chunk_id.tolist() == [[0, 1, 2]]
    assert result.pair_is_correct.tolist() == [[True, True, True]]
