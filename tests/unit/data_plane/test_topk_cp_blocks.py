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
"""The trainer's fetch of the captured top-k: unpadded, one context-parallel block per rank.

The reference is what the fetch did before: ``materialize`` pads every row to the forward pad
target and every rank of the replica group gets the whole padded field. Each rank must now hold
exactly its block of that field, and gathering the blocks over the context-parallel group must
give it back bit for bit. Multi-rank cases run on CPU with Gloo.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from nemo_rl.data_plane import KVBatchMeta, materialize
from nemo_rl.data_plane import worker_mixin
from nemo_rl.data_plane.schema import (
    GENERATION_TOPK_CP_BLOCK_FIELDS,
    GENERATION_TOPK_FIELDS,
    GLOBAL_FORWARD_PAD_SEQLEN,
)
from nemo_rl.data_plane.worker_mixin import (
    TQWorkerMixin,
    _topk_blocks,
    gather_topk_cp_blocks,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_K = 3
# Rows of very different lengths: 12 is longer than the test chunk, 1 is shorter than a block.
_LENGTHS = (5, 1, 12, 7, 2)
_PAD_TO = 15
_PADS = {"input_ids": 7, "generation_topk_ids": -1}
# 4 tokens of int32/float32 top-k (16 bytes each) per broadcast chunk.
_CHUNK_BYTES = 4 * _K * 4


def _wire(lengths=_LENGTHS, *, dense: bool = False, seed: int = 0) -> TensorDict:
    """Rows as the data plane hands them to a worker: jagged, dense when equally long."""
    generator = torch.Generator().manual_seed(seed)
    columns = {
        "input_ids": [
            torch.randint(0, 100, (length,), generator=generator) for length in lengths
        ],
        "generation_topk_ids": [
            torch.randint(0, 100, (length, _K), generator=generator, dtype=torch.int32)
            for length in lengths
        ],
        "generation_topk_logprobs": [
            torch.randn(length, _K, generator=generator) for length in lengths
        ],
    }
    pack = (
        torch.stack
        if dense
        else (lambda rows: torch.nested.as_nested_tensor(rows, layout=torch.jagged))
    )
    return TensorDict(
        {
            **{name: pack(rows) for name, rows in columns.items()},
            "input_lengths": torch.tensor(lengths, dtype=torch.int32),
        },
        batch_size=len(lengths),
    )


def _expected_block(padded: torch.Tensor, index: int, count: int, pad) -> torch.Tensor:
    """Block ``index`` of the padded field, padded on to ``count`` equal blocks."""
    size = -(-padded.shape[1] // count)
    tail = count * size - padded.shape[1]
    full = torch.nn.functional.pad(padded, (0, 0, 0, tail), value=pad)
    return full[:, index * size : (index + 1) * size]


@pytest.mark.parametrize("count", [1, 2, 3, 4])
@pytest.mark.parametrize("pad_to_seqlen", [0, _PAD_TO])
@pytest.mark.parametrize("dense", [False, True])
def test_blocks_are_blocks_of_the_materialized_field(count, pad_to_seqlen, dense):
    lengths = (6, 6, 6) if dense else _LENGTHS
    td = _wire(lengths, dense=dense)
    padded = materialize(
        td, layout="padded", pad_value_dict=_PADS, pad_to_seqlen=pad_to_seqlen
    )
    values = {field: td.get(field) for field in GENERATION_TOPK_FIELDS}
    for index in range(count):
        out = _topk_blocks(
            values,
            split=(index, count),
            pad_value_dict=_PADS,
            pad_to_seqlen=pad_to_seqlen,
        )
        for field in GENERATION_TOPK_FIELDS:
            key = field if count == 1 else GENERATION_TOPK_CP_BLOCK_FIELDS[field]
            assert set(out) >= {key}
            expected = _expected_block(padded[field], index, count, _PADS.get(field, 0))
            assert out[key].dtype == expected.dtype
            assert torch.equal(out[key], expected), (field, index)
        if count == 1:
            assert set(out) == set(GENERATION_TOPK_FIELDS)


def _run(body, world_size: int, tmp_path, *args, backend: str = "gloo") -> None:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    init_file = str(tmp_path / "init")
    procs = [
        ctx.Process(
            target=_in_group,
            args=(body, rank, world_size, init_file, queue, args, backend),
        )
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    try:
        for proc in procs:
            proc.join(timeout=120)
        results = sorted(queue.get(timeout=5) for _ in range(world_size))
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
    assert results == [(rank, "ok") for rank in range(world_size)], results


def _in_group(body, rank, world_size, init_file, queue, args, backend) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    torch.set_num_threads(1)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend, init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    worker_mixin._TOPK_CHUNK_BYTES = _CHUNK_BYTES
    try:
        body(rank, world_size, *args)
        queue.put((rank, "ok"))
    except Exception as error:  # pragma: no cover - surface failures to the parent
        queue.put((rank, f"{type(error).__name__}: {error}"))
    finally:
        dist.destroy_process_group()


def _broadcast_body(rank: int, world_size: int, tp_size: int) -> None:
    """The source broadcasts unpadded chunks; each rank keeps its context-parallel block."""
    td = _wire()
    cp_size = world_size // tp_size
    cp_rank = rank // tp_size
    padded = materialize(
        td, layout="padded", pad_value_dict=_PADS, pad_to_seqlen=_PAD_TO
    )
    out = _topk_blocks(
        {field: td.get(field) for field in GENERATION_TOPK_FIELDS} if rank == 0 else {},
        split=(cp_rank, cp_size),
        pad_value_dict=_PADS,
        pad_to_seqlen=_PAD_TO,
        is_leader=rank == 0,
        src=0,
        group=dist.group.WORLD,
    )
    assert set(out) == set(GENERATION_TOPK_CP_BLOCK_FIELDS.values())
    for field, key in GENERATION_TOPK_CP_BLOCK_FIELDS.items():
        expected = _expected_block(padded[field], cp_rank, cp_size, _PADS.get(field, 0))
        assert torch.equal(out[key], expected), (rank, field)

    # A source without the fields (an empty batch) leaves every rank without them.
    assert (
        _topk_blocks(
            {},
            split=(cp_rank, cp_size),
            pad_value_dict=_PADS,
            pad_to_seqlen=_PAD_TO,
            is_leader=rank == 0,
            src=0,
            group=dist.group.WORLD,
        )
        == {}
    )


@pytest.mark.parametrize(
    ("world_size", "tp_size"), [(2, 1), (4, 1), (4, 2)], ids=["cp2", "cp4", "tp2cp2"]
)
def test_broadcast_leaves_each_rank_its_block(tmp_path, world_size, tp_size):
    _run(_broadcast_body, world_size, tmp_path, tp_size)


class _Client:
    """Hands out ``td`` and records what each fetch selected."""

    def __init__(self, td: TensorDict) -> None:
        self.td = td
        self.selected: list[list[str]] = []

    def get_samples(self, sample_ids, partition_id, select_fields):
        self.selected.append(list(select_fields))
        return self.td.select(*select_fields)


class _Worker(TQWorkerMixin):
    def __init__(self, td, *, rank: int, split, group) -> None:
        self._dp_client = _Client(td)
        self._rank = rank
        self._split = split
        self._group = group

    def _get_replica_group(self):
        return self._group

    def _is_replica_leader(self) -> bool:
        return self._rank == 0

    def _topk_split(self):
        return self._split

    def _pad_value_dict(self):
        return dict(_PADS)


def _meta(td: TensorDict) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"s{i}" for i in range(td.batch_size[0])],
        fields=list(td.keys()),
        extra_info={GLOBAL_FORWARD_PAD_SEQLEN: _PAD_TO},
    )


def _fetch_body(rank: int, world_size: int, policy: str) -> None:
    """``_fetch`` keeps every other field as before and each rank's top-k block."""
    td = _wire()
    meta = _meta(td)
    before = materialize(
        td, layout="padded", pad_value_dict=_PADS, pad_to_seqlen=_PAD_TO
    )
    group = dist.group.WORLD
    for split in [(rank, world_size), (0, 1), None]:
        worker = _Worker(td, rank=rank, split=split, group=group)
        data = worker._fetch(meta, fetch_policy=policy)
        fetched = worker._dp_client.selected
        if policy == "leader_broadcast" and rank != 0:
            assert fetched == []
        else:
            assert fetched == [
                [
                    field
                    for field in meta.fields
                    if split is not None or field not in GENERATION_TOPK_FIELDS
                ]
            ]
        for field in ("input_ids", "input_lengths"):
            assert torch.equal(data[field], before[field]), (split, field)
        if split is None:
            topk = set(GENERATION_TOPK_FIELDS) | set(
                GENERATION_TOPK_CP_BLOCK_FIELDS.values()
            )
            assert not topk & set(data), sorted(data)
        elif split[1] == 1:
            for field in GENERATION_TOPK_FIELDS:
                assert torch.equal(data[field], before[field]), field
        else:
            assert not set(GENERATION_TOPK_FIELDS) & set(data)
            for field, key in GENERATION_TOPK_CP_BLOCK_FIELDS.items():
                expected = _expected_block(
                    before[field], rank, world_size, _PADS.get(field, 0)
                )
                assert torch.equal(data[key], expected), (split, field)
            gather_topk_cp_blocks(data, group)
            for field in GENERATION_TOPK_FIELDS:
                assert torch.equal(data[field], before[field]), (split, field)


@pytest.mark.parametrize("policy", ["leader_broadcast", "independent"])
def test_fetch_gives_each_rank_its_block(tmp_path, policy):
    _run(_fetch_body, 4, tmp_path, policy)


def _microbatch_body(rank: int, world_size: int) -> None:
    """Each microbatch gathers its rows back, also cut to a dynamic-batching width."""
    td = _wire()
    before = materialize(
        td, layout="padded", pad_value_dict=_PADS, pad_to_seqlen=_PAD_TO
    )
    data = materialize(
        td.exclude(*GENERATION_TOPK_FIELDS),
        layout="padded",
        pad_value_dict=_PADS,
        pad_to_seqlen=_PAD_TO,
    )
    data.update(
        _topk_blocks(
            {field: td.get(field) for field in GENERATION_TOPK_FIELDS},
            split=(rank, world_size),
            pad_value_dict=_PADS,
            pad_to_seqlen=_PAD_TO,
        )
    )
    for (start, end), width in (((0, 2), _PAD_TO), ((2, 5), 13), ((4, 5), 3)):
        microbatch = data.slice(start, end)
        expected = before.slice(start, end)
        microbatch.truncate_tensors(dim=1, truncated_len=width)
        expected.truncate_tensors(dim=1, truncated_len=width)
        gather_topk_cp_blocks(microbatch, dist.group.WORLD)
        assert set(microbatch) == set(expected)
        for field in GENERATION_TOPK_FIELDS:
            assert torch.equal(microbatch[field], expected[field]), (start, field)

    too_narrow = data.slice(0, 1)
    too_narrow["input_ids"] = torch.zeros(1, world_size * 100, dtype=torch.long)
    with pytest.raises(ValueError, match="positions per row"):
        gather_topk_cp_blocks(too_narrow, dist.group.WORLD)


@pytest.mark.parametrize("world_size", [2, 4])
def test_microbatch_gathers_the_rows_back(tmp_path, world_size):
    _run(_microbatch_body, world_size, tmp_path)


def _nccl_body(rank: int, world_size: int) -> None:
    """Rows cross the GPU in chunks, the blocks land on the host and no device copy stays."""
    td = _wire()
    padded = materialize(
        td, layout="padded", pad_value_dict=_PADS, pad_to_seqlen=_PAD_TO
    )
    allocated = torch.cuda.memory_allocated()
    out = _topk_blocks(
        {field: td.get(field) for field in GENERATION_TOPK_FIELDS} if rank == 0 else {},
        split=(rank, world_size),
        pad_value_dict=_PADS,
        pad_to_seqlen=_PAD_TO,
        is_leader=rank == 0,
        src=0,
        group=dist.group.WORLD,
    )
    assert torch.cuda.memory_allocated() == allocated
    data = BatchedDataDict(
        {"input_ids": padded["input_ids"].cuda()}
        | {key: blocks.cuda() for key, blocks in out.items()}
    )
    for field, key in GENERATION_TOPK_CP_BLOCK_FIELDS.items():
        assert out[key].device.type == "cpu"
        expected = _expected_block(padded[field], rank, world_size, _PADS.get(field, 0))
        assert torch.equal(out[key], expected), field
    gather_topk_cp_blocks(data, dist.group.WORLD)
    for field in GENERATION_TOPK_FIELDS:
        assert torch.equal(data[field].cpu(), padded[field]), field


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="two CUDA devices are required for NCCL",
)
def test_nccl_blocks_and_gather(tmp_path):
    _run(_nccl_body, min(4, torch.cuda.device_count()), tmp_path, backend="nccl")


def test_batch_without_blocks_is_left_alone():
    data = BatchedDataDict({"input_ids": torch.zeros(2, 3, dtype=torch.long)})
    gather_topk_cp_blocks(data, None)
    assert set(data) == {"input_ids"}


def test_topk_split_default_keeps_whole_rows():
    class _Stub(TQWorkerMixin):
        pass

    assert _Stub()._topk_split() == (0, 1)
