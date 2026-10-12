"""CPU regression tests for CSD table preparation and validation."""

import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tools.csd.build_subtoks_table import (
    build_subtoks_table,
    load_tokenizer,
    tokenizer_fingerprint,
)
from tools.csd.build_subtoks_table import (
    main as build_main,
)
from tools.csd.validate_subtoks_tables import main as validate_main
from tools.csd.validate_subtoks_tables import validate_table


def _byte_tokenizer(
    extra_tokens: list[str], merges: list[tuple[str, str]]
) -> PreTrainedTokenizerFast:
    surfaces = sorted(pre_tokenizers.ByteLevel.alphabet()) + extra_tokens
    backend = Tokenizer(
        models.BPE(
            vocab={surface: i for i, surface in enumerate(surfaces)}, merges=merges
        )
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def _save_table(
    path: Path, source: PreTrainedTokenizerFast, destination: PreTrainedTokenizerFast
) -> dict:
    subtoks, lengths = build_subtoks_table(source, destination, max_subtokens=8)
    table = {
        "subtoks": subtoks,
        "lengths": lengths,
        "N_max": 8,
        "V_src": len(source),
        "V_dst": len(destination),
        "source_fingerprint": tokenizer_fingerprint(source),
        "destination_fingerprint": tokenizer_fingerprint(destination),
    }
    torch.save(table, path)
    return table


def test_forward_and_reverse_tables_include_complete_chains():
    student = _byte_tokenizer(["ab"], [("a", "b")])
    teacher = _byte_tokenizer(["xy"], [("x", "y")])
    for source, destination, word in (
        (student, teacher, "ab"),
        (teacher, student, "xy"),
    ):
        subtoks, lengths = build_subtoks_table(source, destination, max_subtokens=8)
        source_id = source.convert_tokens_to_ids(word)
        assert lengths[source_id].item() == 2
        assert subtoks[source_id, :2].tolist() == destination.convert_tokens_to_ids(
            list(word)
        )
        assert subtoks[source_id, 2:].tolist() == [-1] * 6


def test_partial_utf8_is_preserved_as_bytes():
    source = _byte_tokenizer(["âĢ"], [("â", "Ģ")])
    destination = _byte_tokenizer(["ï¿", "ï¿½"], [("ï", "¿"), ("ï¿", "½")])
    source_id = source.convert_tokens_to_ids("âĢ")
    assert source.decode([source_id]) == "\ufffd"
    subtoks, lengths = build_subtoks_table(source, destination, max_subtokens=8)
    assert lengths[source_id].item() == 2
    assert subtoks[source_id, :2].tolist() == destination.convert_tokens_to_ids(
        ["â", "Ģ"]
    )


def test_overflow_is_unmapped_instead_of_truncating_the_event():
    source = _byte_tokenizer(["ab"], [("a", "b")])
    destination = _byte_tokenizer([], [])
    subtoks, lengths = build_subtoks_table(source, destination, max_subtokens=1)
    source_id = source.convert_tokens_to_ids("ab")
    assert lengths[source_id].item() == 0
    assert subtoks[source_id].tolist() == [-1]


def test_nonpositive_chain_limit_is_rejected():
    tokenizer = _byte_tokenizer([], [])
    with pytest.raises(ValueError, match="must be positive"):
        build_subtoks_table(tokenizer, tokenizer, max_subtokens=0)


@pytest.mark.parametrize("unsupported_side", ["source", "destination"])
@pytest.mark.parametrize("decoder_family", ["metaspace", "wordpiece"])
def test_context_dependent_decoders_are_rejected_before_table_io(
    tmp_path: Path, unsupported_side: str, decoder_family: str
):
    if decoder_family == "metaspace":
        backend = Tokenizer(
            models.Unigram([("x", 0.0), ("▁the", -1.0), ("the", -2.0)], unk_id=0)
        )
        backend.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    else:
        backend = Tokenizer(models.WordPiece({"[UNK]": 0, "x": 1, "##the": 2}))
        backend.decoder = decoders.WordPiece(prefix="##")
    unsupported = PreTrainedTokenizerFast(tokenizer_object=backend)
    if decoder_family == "metaspace":
        # Isolated decoding loses the leading space, so a decoded-text-only
        # table would falsely equate this token with the byte-BPE token "the".
        assert unsupported.decode([1]) == "the"
        assert unsupported.decode([0, 1]) == "x the"
    byte_tokenizer = _byte_tokenizer([], [])
    source, destination = (
        (unsupported, byte_tokenizer)
        if unsupported_side == "source"
        else (byte_tokenizer, unsupported)
    )
    error = "BPE model and ByteLevel decoder"
    with pytest.raises(ValueError, match=error):
        build_subtoks_table(source, destination, max_subtokens=8)
    with pytest.raises(ValueError, match=error):
        validate_table(tmp_path / "table.pt", source=source, destination=destination)

    snapshot = tmp_path / "unsupported_tokenizer"
    unsupported.save_pretrained(snapshot)
    with pytest.raises(ValueError, match=error):
        load_tokenizer(str(snapshot), revision=None)


def test_semantic_roles_precede_shared_spelling_and_missing_roles_are_unmapped():
    source = _byte_tokenizer([], [])
    source.add_special_tokens({"eos_token": "<shared>", "bos_token": "<begin>"})
    destination = _byte_tokenizer([], [])
    destination.add_special_tokens({"pad_token": "<shared>", "eos_token": "<end>"})
    subtoks, lengths = build_subtoks_table(source, destination, max_subtokens=8)
    assert lengths[source.eos_token_id].item() == 1
    assert subtoks[source.eos_token_id, 0].item() == destination.eos_token_id
    assert lengths[source.bos_token_id].item() == 0


def test_ambiguous_role_alias_uses_compatible_exact_special_spelling():
    source = _byte_tokenizer([], [])
    source.add_special_tokens({"eos_token": "<end>", "pad_token": "<end>"})
    destination = _byte_tokenizer([], [])
    destination.add_special_tokens({"eos_token": "<end>", "pad_token": "<pad>"})
    subtoks, lengths = build_subtoks_table(source, destination, max_subtokens=8)
    assert lengths[source.eos_token_id].item() == 1
    assert subtoks[source.eos_token_id, 0].item() == destination.eos_token_id


def test_ordinary_added_token_cannot_become_a_semantic_special():
    source = _byte_tokenizer([], [])
    source.add_tokens(["<end>"])
    destination = _byte_tokenizer([], [])
    destination.add_special_tokens({"eos_token": "<end>"})
    _, lengths = build_subtoks_table(source, destination, max_subtokens=8)
    assert lengths[source.convert_tokens_to_ids("<end>")].item() == 0


@pytest.mark.parametrize(
    "corruption", ["event", "range", "padding", "fingerprint", "role"]
)
def test_validator_rejects_invalid_tables(tmp_path: Path, corruption: str):
    source = _byte_tokenizer(["ab"], [("a", "b")])
    destination = _byte_tokenizer([], [])
    source.add_special_tokens({"eos_token": "<source-end>"})
    destination.add_special_tokens({"eos_token": "<destination-end>"})
    path = tmp_path / "table.pt"
    table = _save_table(path, source, destination)
    validate_table(path, source=source, destination=destination)
    row = source.convert_tokens_to_ids("ab")
    if corruption == "event":
        table["subtoks"][row, 1] = destination.convert_tokens_to_ids("x")
    elif corruption == "range":
        table["subtoks"][row, 0] = len(destination)
    elif corruption == "padding":
        table["subtoks"][row, 7] = 0
    elif corruption == "fingerprint":
        table["source_fingerprint"] = "wrong-tokenizer"
    elif corruption == "role":
        table["subtoks"][source.eos_token_id, 0] = 0
    torch.save(table, path)
    with pytest.raises(ValueError):
        validate_table(path, source=source, destination=destination)


def test_public_build_and_validate_cli_workflow(tmp_path: Path, monkeypatch):
    student = _byte_tokenizer(["ab"], [("a", "b")])
    teacher = _byte_tokenizer(["xy"], [("x", "y")])
    student_dir, teacher_dir = tmp_path / "student", tmp_path / "teacher"
    student.save_pretrained(student_dir)
    teacher.save_pretrained(teacher_dir)
    forward, reverse = (
        tmp_path / "student_to_teacher.pt",
        tmp_path / "teacher_to_student.pt",
    )
    for source, destination, output in (
        (student_dir, teacher_dir, forward),
        (teacher_dir, student_dir, reverse),
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "build_subtoks_table",
                "--source-tokenizer",
                str(source),
                "--destination-tokenizer",
                str(destination),
                "--max-subtokens",
                "8",
                "--output",
                str(output),
            ],
        )
        build_main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_subtoks_tables",
            "--student-tokenizer",
            str(student_dir),
            "--teacher-tokenizer",
            str(teacher_dir),
            "--student-to-teacher",
            str(forward),
            "--teacher-to-student",
            str(reverse),
        ],
    )
    validate_main()
