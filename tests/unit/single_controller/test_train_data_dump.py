# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from nemo_rl.utils.train_data_dump import TrainDataDump


def test_chunks_keep_masked_rows_and_long_sequences_until_commit(tmp_path):
    writer = TrainDataDump(str(tmp_path))
    size = 262144
    tokens = torch.arange(size).reshape(1, -1)
    for sid, length, mask in [("long", size, 1), ("masked", 3, 0)]:
        writer.add_chunk(
            step=4,
            sample_ids=[sid],
            tags=[{"prompt_idx": 12}],
            input_lengths=torch.tensor([length]),
            sequences={"token_ids": tokens, "advantages": tokens.float()},
            scalars={"sample_loss_mask": torch.tensor([mask])},
        )
    final = tmp_path / "train_data_step5.jsonl"
    assert not final.exists()
    writer.finish_step(4)
    rows = [json.loads(line) for line in final.read_text().splitlines()]
    assert [r["idx"] for r in rows] == [0, 1]
    assert rows[0]["token_ids"][0] == list(range(size))
    assert rows[1]["token_ids"] == [[0, 1, 2]]
    assert rows[1]["sample_loss_mask"] == [0]
    assert rows[1]["metadata"] == [{"prompt_idx": 12}]
    assert not final.with_suffix(".jsonl.partial").exists()
    with pytest.raises(RuntimeError, match="no training dump"):
        writer.finish_step(5)


def test_resume_replaces_partial_and_keeps_steps_apart(tmp_path: Path) -> None:
    # An interrupted publish leaves a .partial behind; the next one overwrites
    # it rather than appending to a half-written file.
    stale = tmp_path / "train_data_step8.jsonl.partial"
    stale.write_text("interrupted attempt\n")
    writer = TrainDataDump(str(tmp_path))
    args = dict(
        sample_ids=["resumed"],
        tags=None,
        input_lengths=torch.tensor([1]),
        sequences={"token_ids": torch.tensor([[9]])},
        scalars={},
    )
    writer.add_chunk(step=7, **args)
    part = tmp_path / "train_data_step8.jsonl.part-0"
    assert json.loads(part.read_text())["token_ids"] == [[9]]
    # Each step owns its own part file, so moving on neither publishes nor
    # corrupts the previous one.
    writer.add_chunk(step=8, **args)
    assert (tmp_path / "train_data_step9.jsonl.part-0").exists()
    assert part.exists()
    with pytest.raises(RuntimeError, match="no training dump"):
        writer.finish_step(42)
    writer.finish_step(7)
    published = tmp_path / "train_data_step8.jsonl"
    assert json.loads(published.read_text())["idx"] == 0
    assert not part.exists()
    assert not stale.exists()


def test_shards_write_separate_parts_and_merge_in_order(tmp_path: Path) -> None:
    """The pool runs a step's shards concurrently, so each needs its own file.

    idx has to be assigned by the merge: no single writer sees every row, and
    a per-writer counter would hand the same index to every shard.
    """
    args: dict[str, Any] = dict(
        tags=None,
        input_lengths=torch.tensor([1]),
        sequences={"token_ids": torch.tensor([[1]])},
        scalars={},
    )
    shards = [TrainDataDump(str(tmp_path), shard_id=str(i)) for i in range(3)]
    for i, shard in enumerate(shards):
        shard.add_chunk(step=0, sample_ids=[f"s{i}"], **args)
    assert sorted(p.name for p in tmp_path.glob("*.part-*")) == [
        f"train_data_step1.jsonl.part-{i}" for i in range(3)
    ]
    # Any shard can publish: the merge reads the parts off disk rather than
    # whatever that writer happens to have buffered.
    shards[1].finish_step(0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "train_data_step1.jsonl").read_text().splitlines()
    ]
    assert [row["idx"] for row in rows] == [0, 1, 2]
    assert [row["sample_id"] for row in rows] == [["s0"], ["s1"], ["s2"]]
    assert not list(tmp_path.glob("*.part-*"))


def test_unreachable_shard_fails_instead_of_publishing_a_short_dump(
    tmp_path: Path,
) -> None:
    """A pool off the controller's filesystem writes parts it cannot see.

    Those shards still run and still report their row counts, so without the
    count check the merge would publish whatever happened to be local and the
    dump would look complete while silently missing a shard.
    """
    reachable = TrainDataDump(str(tmp_path), shard_id="0")
    reachable.add_chunk(
        step=0,
        sample_ids=["here"],
        tags=None,
        input_lengths=torch.tensor([1]),
        sequences={"token_ids": torch.tensor([[1]])},
        scalars={},
    )
    controller = TrainDataDump(str(tmp_path))
    # Two shards reported a row each; only one part landed on this filesystem.
    with pytest.raises(RuntimeError, match="must share this filesystem"):
        controller.finish_step(0, 2)
    assert not list(tmp_path.glob("*.jsonl"))
    assert not list(tmp_path.glob("*.partial"))
    # The honest count still publishes.
    controller.finish_step(0, 1)
    assert (tmp_path / "train_data_step1.jsonl").exists()


def test_controller_publishes_a_step_it_never_wrote(tmp_path: Path) -> None:
    """With a pool the controller holds a dump that only ever merges.

    It calls finish_step without a single add_chunk of its own, which the
    pre-pool writer rejected because it keyed off its own row counter.
    """
    worker = TrainDataDump(str(tmp_path), shard_id="7")
    worker.add_chunk(
        step=2,
        sample_ids=["only"],
        tags=None,
        input_lengths=torch.tensor([1]),
        sequences={"token_ids": torch.tensor([[3]])},
        scalars={},
    )
    controller = TrainDataDump(str(tmp_path))
    controller.finish_step(2)
    rows = (tmp_path / "train_data_step3.jsonl").read_text().splitlines()
    assert [json.loads(line)["sample_id"] for line in rows] == [["only"]]


@pytest.fixture
def chunk() -> dict[str, Any]:
    return {
        "sample_ids": ["a", "b"],
        "tags": [{"source": "gym", "nested": {"attempt": 2}}, {}],
        "input_lengths": torch.tensor([2, 1]),
        "sequences": {
            "token_ids": torch.tensor([[10, 11, 0], [20, 0, 0]]),
            "token_loss_mask": torch.tensor([[0, 1, 0], [1, 0, 0]]),
            "advantages": torch.tensor(
                [[-0.5, 0.25, 99.0], [0.5, 99.0, 99.0]], requires_grad=True
            ),
            "generation_logprobs": torch.full((2, 3), -0.5),
            "prev_logprobs": torch.full((2, 3), -0.25),
            "teacher_logprobs": torch.full((2, 3), -0.125),
        },
        "scalars": {
            "sample_loss_mask": torch.tensor([1, 0]),
            "pre_seq_error_sample_loss_mask": torch.tensor([1, 1]),
            "rewards": torch.tensor([0.5, -1.0]),
            "prompt_ids": torch.tensor([[7, 8], [7, 8]]),
        },
    }


def test_dump_preserves_singleton_batch_schema(
    tmp_path: Path, chunk: dict[str, Any]
) -> None:
    original_advantages = chunk["sequences"]["advantages"].detach().clone()
    writer = TrainDataDump(str(tmp_path / "nested" / "logs"))
    writer.add_chunk(step=0, **chunk)
    writer.finish_step(0)
    rows = [
        json.loads(line)
        for line in (writer.log_dir / "train_data_step1.jsonl").read_text().splitlines()
    ]
    expected_keys = {
        "idx",
        "step",
        "sample_id",
        "input_lengths",
        "metadata",
        *chunk["sequences"],
        *chunk["scalars"],
    }
    assert len(rows) == len(chunk["sample_ids"])
    for i, row in enumerate(rows):
        length = chunk["input_lengths"][i].item()
        assert set(row) == expected_keys
        assert row["idx"] == i
        assert row["step"] == 1
        assert row["sample_id"] == chunk["sample_ids"][i : i + 1]
        assert row["input_lengths"] == [length]
        assert row["metadata"] == chunk["tags"][i : i + 1]
        for key, values in chunk["sequences"].items():
            assert row[key] == values[i : i + 1, :length].tolist(), key
        for key, values in chunk["scalars"].items():
            assert row[key] == values[i : i + 1].tolist(), key
    assert chunk["sequences"]["advantages"].requires_grad
    torch.testing.assert_close(chunk["sequences"]["advantages"], original_advantages)


@pytest.mark.parametrize(
    ("invalid_column", "existing_chunk"),
    [
        ("input_lengths", False),
        ("tags", False),
        ("token_ids", False),
        ("rewards", False),
        ("negative", False),
        ("too_long", False),
        ("short_advantages", False),
        ("too_long", True),
    ],
)
def test_invalid_chunk_leaves_dump_unchanged(
    tmp_path: Path,
    chunk: dict[str, Any],
    invalid_column: str,
    existing_chunk: bool,
) -> None:
    writer = TrainDataDump(str(tmp_path))
    if existing_chunk:
        writer.add_chunk(step=0, **chunk)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    if invalid_column in ("input_lengths", "tags"):
        chunk[invalid_column] = chunk[invalid_column][:1]
    elif invalid_column == "token_ids":
        chunk["sequences"]["token_ids"] = chunk["sequences"]["token_ids"][:1]
    elif invalid_column == "rewards":
        chunk["scalars"]["rewards"] = chunk["scalars"]["rewards"][:1]
    elif invalid_column == "short_advantages":
        # Token IDs fit; validation must also check later sequence columns.
        chunk["sequences"]["advantages"] = chunk["sequences"]["advantages"][:, :1]
    else:
        # Invalid second row must be caught before even the valid first row is written.
        chunk["input_lengths"] = torch.tensor(
            [2, -1 if invalid_column == "negative" else 4]
        )
    with pytest.raises(ValueError, match="column lengths|input length"):
        writer.add_chunk(step=0, **chunk)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert writer.rows == (2 if existing_chunk else 0)
    assert writer.step == (0 if existing_chunk else None)


def test_jagged_columns_dump_real_rows_without_padding(tmp_path: Path) -> None:
    # Two prompt groups with different prompt lengths: the data plane returns
    # prompt_ids jagged, and padding them would append fake 0 tokens.
    def jagged(rows: list[list[int]]) -> torch.Tensor:
        return torch.nested.as_nested_tensor(
            [torch.tensor(row) for row in rows], layout=torch.jagged
        )

    prompts = [[5, 6, 7], [5, 6, 7], [9, 8], [9, 8]]
    tokens = [[5, 6, 7, 1, 2], [5, 6, 7, 3], [9, 8, 4], [9, 8, 5, 6]]
    args = dict(
        sample_ids=["p0_g0", "p0_g1", "p1_g0", "p1_g1"],
        tags=None,
        sequences={"token_ids": jagged(tokens)},
        scalars={"prompt_ids": jagged(prompts)},
    )
    writer = TrainDataDump(str(tmp_path))
    with pytest.raises(ValueError, match="input length"):
        # Validation is per row: 5 fits row 0 but exceeds jagged row 3.
        writer.add_chunk(step=0, input_lengths=torch.tensor([5, 4, 3, 5]), **args)
    writer.add_chunk(step=0, input_lengths=torch.tensor([5, 4, 3, 4]), **args)
    writer.finish_step(0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "train_data_step1.jsonl").read_text().splitlines()
    ]
    assert [row["prompt_ids"] for row in rows] == [[p] for p in prompts]
    assert [row["token_ids"] for row in rows] == [[t] for t in tokens]


def test_zero_length_and_missing_optional_columns(
    tmp_path: Path, chunk: dict[str, Any]
) -> None:
    writer = TrainDataDump(str(tmp_path))
    chunk["input_lengths"] = torch.tensor([[0], [1]])
    chunk["tags"] = None
    chunk["scalars"] = {}
    chunk["sequences"] = {"token_ids": chunk["sequences"]["token_ids"]}
    writer.add_chunk(step=0, **chunk)
    writer.finish_step(0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "train_data_step1.jsonl").read_text().splitlines()
    ]
    assert rows[0]["token_ids"] == [[]]
    assert rows[0]["input_lengths"] == [0]
    assert rows[1]["token_ids"] == [[20]]
    assert all(row["metadata"] == [{}] for row in rows)
    assert all("teacher_logprobs" not in row for row in rows)


def test_consecutive_steps_reset_row_indices_and_preserve_prior_dump(
    tmp_path: Path, chunk: dict[str, Any]
) -> None:
    writer = TrainDataDump(str(tmp_path))
    writer.add_chunk(step=0, **chunk)
    writer.finish_step(0)
    first = tmp_path / "train_data_step1.jsonl"
    original = first.read_bytes()
    writer.add_chunk(step=1, **chunk)
    writer.add_chunk(step=1, **chunk)
    writer.finish_step(1)
    rows = [
        json.loads(line)
        for line in (tmp_path / "train_data_step2.jsonl").read_text().splitlines()
    ]
    assert [row["idx"] for row in rows] == [0, 1, 2, 3]
    assert [row["sample_id"] for row in rows] == [["a"], ["b"], ["a"], ["b"]]
    assert all(row["step"] == 2 for row in rows)
    assert first.read_bytes() == original
    assert not list(tmp_path.glob("*.partial"))


def test_empty_chunk_cannot_publish_completed_step(tmp_path: Path) -> None:
    writer = TrainDataDump(str(tmp_path))
    writer.add_chunk(
        step=0,
        sample_ids=[],
        tags=None,
        input_lengths=torch.empty(0, dtype=torch.long),
        sequences={},
        scalars={},
    )
    with pytest.raises(RuntimeError, match="no training dump"):
        writer.finish_step(0)
    assert not list(tmp_path.glob("*.jsonl"))
