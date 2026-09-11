# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Tests for ``_broadcast_batched_data_dict`` on two-rank process groups.

Exercises the helper that backs ``_fetch(fetch_policy="leader_broadcast")``.
Most cases run on CPU with Gloo; one optional NCCL case covers device staging.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data_plane.worker_mixin import _broadcast_batched_data_dict
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _in_group(body, rank: int, world_size: int, tmp_init_file: str, q, backend):
    """Run ``body(rank)`` in a process group, reporting the outcome via ``q``."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{tmp_init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        body(rank)
        q.put((rank, "ok"))
    except Exception as e:  # pragma: no cover — surface failures to parent
        q.put((rank, f"err: {type(e).__name__}: {e}"))
    finally:
        dist.destroy_process_group()


def _collect_two_rank_results(body, tmp_init_file: str, backend: str = "gloo"):
    """Spawn two ranks over ``body`` and collect both outcomes."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_in_group, args=(body, rank, 2, tmp_init_file, q, backend))
        for rank in range(2)
    ]
    for p in procs:
        p.start()
    try:
        for p in procs:
            p.join(timeout=30)
        assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
        return sorted([q.get(timeout=5) for _ in range(2)])
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)


def _run_two_ranks(body, tmp_init_file: str, backend: str = "gloo"):
    """Spawn two ranks over ``body`` and require both to report ok."""
    results = _collect_two_rank_results(body, tmp_init_file, backend)
    assert results == [(0, "ok"), (1, "ok")], results


def _pixel_rows():
    return [
        torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4),
        torch.arange(1 * 5 * 4, dtype=torch.float32).reshape(1, 5, 4) + 100,
        None,
    ]


def _packed(rows):
    return PackedTensor(
        [r.clone() if r is not None else None for r in rows],
        dim_to_pack=0,
        preprocess_mode="pad_to_max_shape",
    )


def _round_trip_body(rank: int):
    # ``pixel_values`` is the case that mattered: a PackedTensor is not a
    # torch.Tensor, so before the ``packed_wire`` branch it rode the object
    # list and ``broadcast_object_list`` pickled the pixels into device memory.
    # Rows differ in their trailing dims and one sample has no media, which is
    # what the format exists for.
    rows = _pixel_rows()
    data = (
        BatchedDataDict(
            {
                "input_ids": torch.arange(12, dtype=torch.long).reshape(3, 4),
                "input_lengths": torch.tensor([4, 3, 2], dtype=torch.int32),
                "scalar_meta": "step_42",
                "pixel_values": _packed(rows),
            }
        )
        if rank == 0
        else None
    )

    out = _broadcast_batched_data_dict(
        data, is_leader=(rank == 0), src=0, group=dist.group.WORLD
    )

    assert torch.equal(
        out["input_ids"], torch.arange(12, dtype=torch.long).reshape(3, 4)
    )
    assert torch.equal(out["input_lengths"], torch.tensor([4, 3, 2], dtype=torch.int32))
    assert out["scalar_meta"] == "step_42"

    packed = out["pixel_values"]
    assert isinstance(packed, PackedTensor), type(packed).__name__
    # Compare on logical rows, not ``.tensors``: ``from_wire`` returns segments
    # flat with a CSR row map, so an empty row contributes no entry there.
    expected = _packed(rows)
    assert (
        packed.logical_segment_counts_by_row()
        == expected.logical_segment_counts_by_row()
        == [1, 1, 0]
    )
    assert torch.equal(packed.as_tensor(), expected.as_tensor())


def _deduplicated_round_trip_body(rank: int):
    first_physical_row = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    third_physical_row = torch.arange(3, dtype=torch.float32).reshape(1, 3)
    physical_rows = [
        first_physical_row,
        None,
        third_physical_row,
    ]
    data = (
        BatchedDataDict(
            {
                "pixel_values": PackedTensor(
                    physical_rows,
                    dim_to_pack=0,
                    preprocess_mode="patchify",
                    preprocess_kwargs={"patch_dim": 2},
                    _row_offsets=[0, 1, 2, 4],
                    _segment_indices=[0, 1, 2, 0],
                    _segment_provenance=[b"a", b"b", b"c"],
                )
            }
        )
        if rank == 0
        else None
    )

    out = _broadcast_batched_data_dict(
        data, is_leader=(rank == 0), src=0, group=dist.group.WORLD
    )

    packed = out["pixel_values"]
    assert isinstance(packed, PackedTensor), type(packed).__name__
    assert packed.preprocess_mode == "patchify"
    assert packed.preprocess_kwargs == {"patch_dim": 2}
    assert packed._row_offsets == [0, 1, 2, 4]
    assert packed._segment_indices == [0, 1, 2, 0]
    assert packed._segment_provenance == [b"a", b"b", b"c"]
    assert packed.tensors[1] is None
    assert packed.tensors[0] is not None
    assert packed.tensors[2] is not None
    assert torch.equal(packed.tensors[0], first_physical_row)
    assert torch.equal(packed.tensors[2], third_physical_row)


def _all_empty_body(rank: int):
    # One DP shard of a mixed image/text batch can hold only media-free
    # samples. ``pixel_values`` is still in ``meta.fields``, so the shard
    # rebuilds an empty PackedTensor -- and the key must survive the broadcast,
    # since consumers branch on the key set.
    data = (
        BatchedDataDict(
            {
                "input_ids": torch.arange(8, dtype=torch.long).reshape(2, 4),
                "pixel_values": PackedTensor(
                    [None, None],
                    dim_to_pack=0,
                    preprocess_mode="pad_to_max_shape",
                ),
            }
        )
        if rank == 0
        else None
    )

    out = _broadcast_batched_data_dict(
        data, is_leader=(rank == 0), src=0, group=dist.group.WORLD
    )

    assert set(out.keys()) == {"input_ids", "pixel_values"}, sorted(out.keys())
    packed = out["pixel_values"]
    assert isinstance(packed, PackedTensor), type(packed).__name__
    assert packed.logical_segment_counts_by_row() == [0, 0]
    assert packed.as_tensor() is None
    assert packed.preprocess_mode == "pad_to_max_shape"
    assert packed.preprocess_kwargs == {}


def _nccl_cpu_packed_round_trip_body(rank: int):
    first = torch.arange(6, dtype=torch.int16).reshape(2, 3)
    second = torch.arange(3, dtype=torch.int16).reshape(1, 3) + 10
    data = (
        BatchedDataDict(
            {
                "pixel_values": PackedTensor(
                    [first, None, second],
                    dim_to_pack=0,
                    _row_offsets=[0, 1, 2, 4],
                    _segment_indices=[0, 1, 2, 0],
                    _segment_provenance=[b"a", b"b", b"c"],
                )
            }
        )
        if rank == 0
        else None
    )

    out = _broadcast_batched_data_dict(
        data, is_leader=(rank == 0), src=0, group=dist.group.WORLD
    )

    packed = out["pixel_values"]
    assert packed._row_offsets == [0, 1, 2, 4]
    assert packed._segment_indices == [0, 1, 2, 0]
    assert packed._segment_provenance == [b"a", b"b", b"c"]
    assert packed.tensors[1] is None
    assert packed.tensors[0] is not None
    assert packed.tensors[2] is not None
    assert packed.tensors[0].device.type == "cpu"
    assert packed.tensors[2].device.type == "cpu"
    assert packed.tensors[0].dtype == torch.int16
    assert packed.tensors[2].dtype == torch.int16
    assert torch.equal(packed.tensors[0], first)
    assert torch.equal(packed.tensors[2], second)


def _unsupported_type_body(rank: int):
    data = BatchedDataDict({"source_ids": ["a", "b"]}) if rank == 0 else None
    _broadcast_batched_data_dict(
        data, is_leader=(rank == 0), src=0, group=dist.group.WORLD
    )


def test_leader_broadcast_round_trip(tmp_path):
    _run_two_ranks(_round_trip_body, str(tmp_path / "init"))


def test_leader_broadcast_preserves_packed_tensor_deduplication(tmp_path):
    _run_two_ranks(_deduplicated_round_trip_body, str(tmp_path / "init_dedup"))


def test_leader_broadcast_keeps_media_free_packed_key(tmp_path):
    """An all-empty packed field keeps its key on both sides of the broadcast.

    ``to_wire`` answers "is there payload", not "is there a field". Deriving
    the broadcast key set from it made a media-free shard emit a different key
    set than the same shard on the independent-fetch path.
    """
    _run_two_ranks(_all_empty_body, str(tmp_path / "init_empty"))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="two CUDA devices are required for NCCL broadcast",
)
def test_leader_broadcast_restores_cpu_packed_tensor_after_nccl(tmp_path):
    _run_two_ranks(
        _nccl_cpu_packed_round_trip_body,
        str(tmp_path / "init_nccl"),
        backend="nccl",
    )


def test_packed_tensor_broadcast_rejects_tensor_header_state():
    packed = _packed(_pixel_rows())
    packed.__dict__["cached_tensor"] = torch.ones(1)

    with pytest.raises(TypeError, match="header must be tensor-free"):
        packed.broadcast_parts()


def test_packed_tensor_broadcast_rejects_oversized_payload():
    packed = PackedTensor([torch.ones(2)], dim_to_pack=0)
    header, shapes, _, _, _ = packed.broadcast_parts()

    with pytest.raises(ValueError, match="shapes describe 2 elements"):
        header.rebuild_from_broadcast_parts(shapes, torch.ones(3))


def test_leader_broadcast_reports_descriptor_error_to_all_ranks(tmp_path):
    results = _collect_two_rank_results(
        _unsupported_type_body, str(tmp_path / "init_error")
    )

    assert results[0][0] == 0
    assert results[0][1].startswith("err: TypeError:")
    assert results[1][0] == 1
    assert results[1][1].startswith("err: RuntimeError:")
    assert all("source_ids" in outcome for _, outcome in results)


def test_get_replica_group_default_is_none():
    """TQWorkerMixin._get_replica_group must default to None.

    The base default lets ``_fetch(fetch_policy="leader_broadcast")``
    fall back to the independent path when no backend override exists
    (Phase 1 / FSDP2 with TP=CP=PP=1).
    """
    from nemo_rl.data_plane.worker_mixin import TQWorkerMixin

    class _Stub(TQWorkerMixin):
        pass

    assert _Stub()._get_replica_group() is None
