from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
import json

import pyarrow.parquet as pq
import pytest
import torch

from nemo_rl.utils.train_data_dump import TrainDataDump


def write(writer: TrainDataDump, *, step: int = 0, sample_id: str = "s") -> str:
    return writer.add_chunk(
        step=step,
        sample_ids=[sample_id],
        tags=[{"group_id": "g", "nested": {"attempt": 2}}],
        input_lengths=torch.tensor([2]),
        sequences={
            "token_ids": torch.tensor([[7, 8, 0]]),
            "advantages": torch.tensor([[0.25, -0.5, 99.0]], requires_grad=True),
            "teacher_logprobs": torch.tensor([[-0.5, -torch.inf, 0.0]]),
        },
        scalars={"sample_loss_mask": torch.tensor([0])},
    )


def test_concurrent_shards_write_readable_chunks(tmp_path: Path) -> None:
    writers = [TrainDataDump(str(tmp_path), shard_id=str(i)) for i in range(3)]
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(lambda i: write(writers[i], sample_id=f"s{i}"), range(3)))
    root = tmp_path / "train_data_parquet"
    assert len(list((root / "step-00000001").glob("*.parquet"))) == 3
    rows = pq.read_table(root).to_pylist()
    assert sorted(row["sample_id"] for row in rows) == ["s0", "s1", "s2"]
    for row in rows:
        assert row["token_ids"] == [7, 8]
        assert row["teacher_logprobs"] == [-0.5, -float("inf")]
        assert row["prev_logprobs"] is None
    write(writers[0], step=1)
    assert pq.read_table(root).num_rows == 4


def test_long_jagged_rows_and_null_schema(tmp_path: Path) -> None:
    writer = TrainDataDump(str(tmp_path))
    length = 262144
    tokens = torch.nested.as_nested_tensor(
        [torch.arange(length), torch.tensor([5])], layout=torch.jagged
    )
    writer.add_chunk(
        step=0,
        sample_ids=["long", "short"],
        tags=None,
        input_lengths=torch.tensor([length, 0]),
        sequences={"token_ids": tokens},
        scalars={
            "prompt_ids": torch.nested.as_nested_tensor(
                [torch.tensor([1, 2]), torch.tensor([3])], layout=torch.jagged
            )
        },
    )
    rows = pq.read_table(tmp_path / "train_data_parquet").to_pylist()
    assert rows[0]["token_ids"] == list(range(length))
    assert rows[1]["token_ids"] == []
    assert [row["prompt_ids"] for row in rows] == [[1, 2], [3]]
    assert all(
        row["group_id"] is None and row["teacher_logprobs"] is None for row in rows
    )


def test_failed_write_is_not_discoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(table, path, **kwargs):
        path.write_bytes(b"incomplete parquet")
        raise OSError("disk full")

    write(TrainDataDump(str(tmp_path)), sample_id="complete")
    monkeypatch.setattr(pq, "write_table", fail)
    with pytest.raises(OSError, match="disk full"):
        write(TrainDataDump(str(tmp_path)))
    assert len(list(tmp_path.rglob("*.parquet"))) == 1
    assert pq.read_table(tmp_path / "train_data_parquet")["sample_id"].to_pylist() == [
        "complete"
    ]
    assert list(tmp_path.rglob("*.partial"))


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
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
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
    assert {
        str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    } == before


def test_training_fields_survive_parquet(tmp_path: Path, chunk: dict[str, Any]) -> None:
    chunk["sequences"].update(
        values=torch.tensor([[0.125, 0.5, 99.0], [0.75, 99.0, 99.0]]),
        returns=torch.tensor([[1.25, 1.5, 99.0], [1.75, 99.0, 99.0]]),
        reference_logprobs=torch.full((2, 3), -0.75),
    )
    writer = TrainDataDump(str(tmp_path))
    writer.add_chunk(step=0, **chunk)
    table = pq.read_table(tmp_path / "train_data_parquet")
    rows = table.to_pylist()
    assert [row["sample_id"] for row in rows] == ["a", "b"]
    assert [row["sample_loss_mask"] for row in rows] == [1, 0]
    assert [row["rewards"] for row in rows] == [0.5, -1.0]
    assert [row["input_lengths"] for row in rows] == [2, 1]
    assert json.loads(rows[0]["metadata"]) == chunk["tags"][0]
    for key in (
        "token_ids",
        "token_loss_mask",
        "advantages",
        "generation_logprobs",
        "prev_logprobs",
        "teacher_logprobs",
        "reference_logprobs",
        "values",
        "returns",
    ):
        assert rows[0][key] == chunk["sequences"][key][0, :2].tolist()
        assert rows[1][key] == chunk["sequences"][key][1, :1].tolist()
    assert str(table.schema.field("values").type) == "list<element: float>"
    assert chunk["sequences"]["advantages"].requires_grad


def test_manifest_lists_only_reported_chunks(tmp_path, monkeypatch):
    writer = TrainDataDump(str(tmp_path))
    write(writer, sample_id="abandoned")
    name = write(writer)
    directory = tmp_path / "train_data_parquet" / "step-00000001"
    original = (directory / name).read_bytes()

    def no_footer_reads(*args, **kwargs):
        pytest.fail("Finalization must not read Parquet footers")

    monkeypatch.setattr(pq, "read_metadata", no_footer_reads)
    writer.finish_step(0, files=[name])
    manifest = json.loads((directory / "_manifest.json").read_text())
    assert manifest == {"step": 1, "files": [name]}
    assert (directory / name).read_bytes() == original
    assert pq.read_table([directory / f for f in manifest["files"]]).num_rows == 1
    with pytest.raises(FileExistsError):
        writer.finish_step(0, files=[name])


@pytest.mark.parametrize("failure", ["missing", "empty", "publish"])
def test_failed_finalization_has_no_manifest(tmp_path, monkeypatch, failure):
    writer = TrainDataDump(str(tmp_path))
    name = write(writer)
    directory = tmp_path / "train_data_parquet" / "step-00000001"
    if failure == "missing":
        (directory / name).unlink()
    elif failure == "empty":
        (directory / name).write_bytes(b"")
    else:

        def fail(*args):
            raise OSError("publication failed")

        monkeypatch.setattr("nemo_rl.utils.train_data_dump.os.replace", fail)
    with pytest.raises((RuntimeError, OSError)):
        writer.finish_step(0, files=[name])
    assert not (directory / "_manifest.json").exists()


def test_bfloat16_and_singleton_scalar_columns(tmp_path, chunk):
    chunk["sequences"]["advantages"] = chunk["sequences"]["advantages"].to(
        torch.bfloat16
    )
    chunk["scalars"]["rewards"] = chunk["scalars"]["rewards"].reshape(2, 1)
    TrainDataDump(str(tmp_path)).add_chunk(step=0, **chunk)
    rows = pq.read_table(tmp_path / "train_data_parquet").to_pylist()
    assert [row["rewards"] for row in rows] == [0.5, -1.0]
    assert [row["prompt_ids"] for row in rows] == [[7, 8], [7, 8]]
    assert [row["advantages"] for row in rows] == [[-0.5, 0.25], [0.5]]
