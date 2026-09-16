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
"""Unit test for ``_broadcast_batched_data_dict`` on a 2-rank gloo group.

Exercises the helper that backs ``_fetch(fetch_policy="leader_broadcast")``.
Runs on CPU (gloo) so it stays in the no-GPU Tier 1 lane.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data.packed_rollouts import (
    TREE_ATTENTION_EDGE_LENGTHS,
    TREE_ATTENTION_EDGE_SOURCE_INDICES,
    TREE_ATTENTION_EDGE_TARGET_IDS,
    TREE_ATTENTION_FRAGMENT_SELECTIONS,
    TREE_ATTENTION_LAYOUTS,
    TreeAttentionLayout,
    materialize_tree_attention_fragments,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.data_plane.schema import MICRO_BATCH_INDICES, MICRO_BATCH_LENGTHS
from nemo_rl.data_plane.worker_mixin import (
    TQWorkerMixin,
    _broadcast_batched_data_dict,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _in_gloo_group(body, rank: int, world_size: int, tmp_init_file: str, q):
    """Run ``body(rank)`` in a gloo group, reporting the outcome via ``q``."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo",
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


def _collect_two_rank_results(body, tmp_init_file: str):
    """Spawn two ranks over ``body`` and collect both outcomes."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_in_gloo_group, args=(body, rank, 2, tmp_init_file, q))
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


def _run_two_ranks(body, tmp_init_file: str):
    """Spawn two ranks over ``body`` and require both to report ok."""
    results = _collect_two_rank_results(body, tmp_init_file)
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
        pad_to_max_shape=True,
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
                    [None, None], dim_to_pack=0, pad_to_max_shape=True
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
    assert packed.pad_to_max_shape is True


def _unsupported_type_body(rank: int):
    data = BatchedDataDict({"source_ids": ["a", "b"]}) if rank == 0 else None
    _broadcast_batched_data_dict(
        data, is_leader=(rank == 0), src=0, group=dist.group.WORLD
    )


def test_leader_broadcast_round_trip(tmp_path):
    _run_two_ranks(_round_trip_body, str(tmp_path / "init"))


def test_leader_broadcast_keeps_media_free_packed_key(tmp_path):
    """An all-empty packed field keeps its key on both sides of the broadcast.

    ``to_wire`` answers "is there payload", not "is there a field". Deriving
    the broadcast key set from it made a media-free shard emit a different key
    set than the same shard on the independent-fetch path.
    """
    _run_two_ranks(_all_empty_body, str(tmp_path / "init_empty"))


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


def _tree_fetch_body(rank: int, *, mode: str) -> None:
    class Client(NoOpDataPlaneClient):
        fetch_count = 0

        def get_samples(self, *args: Any, **kwargs: Any) -> TensorDict:
            assert rank == 0, "only the replica leader may fetch tree payloads"
            self.fetch_count += 1
            return super().get_samples(*args, **kwargs)

    class Worker(TQWorkerMixin):
        def __init__(self, client: Client) -> None:
            self._dp_client = client

        def _get_replica_group(self) -> dist.ProcessGroup:
            return dist.group.WORLD

        def _is_replica_leader(self) -> bool:
            return rank == 0

    layouts = [
        TreeAttentionLayout((3, 2, 2), (-1, 0, 0), (0, 3, 3), (1, 3, 5), 9),
        TreeAttentionLayout((4,), (-1,), (0,), (1, 2), 4),
    ]
    logical = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [[10, 11, 12, 20, 21, 30, 31], [40, 41, 42, 43, 0, 0, 0]]
            ),
            "input_lengths": torch.tensor([7, 4]),
            "routed_experts": torch.arange(28).reshape(2, 7, 2, 1),
            "generation_logprobs": torch.tensor(
                [[0.0, 1.0, 2.0, 3.0], [0.0, 4.0, 5.0, 0.0]]
            ),
            "token_mask": torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]]),
            TREE_ATTENTION_EDGE_SOURCE_INDICES: torch.tensor([[1, 3, 5], [1, 2, -1]]),
            TREE_ATTENTION_EDGE_TARGET_IDS: torch.tensor(
                [[101, 102, 103], [201, 202, 0]]
            ),
            TREE_ATTENTION_EDGE_LENGTHS: torch.tensor([3, 2]),
        }
    )
    meta = KVBatchMeta(
        partition_id="trees",
        task_name="prev_lp",
        sample_ids=["s0", "s1"],
        fields=list(logical),
        sequence_lengths=[7, 4],
        extra_info={TREE_ATTENTION_LAYOUTS: layouts},
    )
    client = Client()
    if rank == 0:
        client.register_partition(
            partition_id="trees",
            fields=list(logical),
            num_samples=2,
            consumer_tasks=["prev_lp"],
        )
        client.put_samples(
            sample_ids=meta.sample_ids,
            partition_id="trees",
            fields=TensorDict(dict(logical), batch_size=(2,)),
        )

    fragmented = mode != "unfragmented"
    if fragmented:
        rank_metas, _ = shard_meta_for_dp(
            meta,
            dp_world=1,
            sequence_packing_args={
                "algorithm": "modified_first_fit_decreasing",
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "max_tokens_per_microbatch": 5,
                "sequence_length_pad_multiple": 1,
            },
        )
        meta = rank_metas[0]
    else:
        meta.extra_info.update(
            {MICRO_BATCH_INDICES: [[[0, 1], [1, 2]]], MICRO_BATCH_LENGTHS: [[7, 4]]}
        )

    worker = Worker(client)
    seen_targets = []

    def tensor_only_broadcast(
        data: BatchedDataDict[Any] | None, **kwargs: Any
    ) -> BatchedDataDict[Any]:
        if data is not None:
            assert TREE_ATTENTION_LAYOUTS not in data
        return _broadcast_batched_data_dict(data, **kwargs)

    with patch(
        "nemo_rl.data_plane.worker_mixin._broadcast_batched_data_dict",
        side_effect=tensor_only_broadcast,
    ) as broadcast:
        batches = (
            [(worker._fetch_presharded(meta), meta)]
            if mode == "fallback"
            else worker._iter_fetch_presharded_microbatches(meta)
        )
        for data, micro_meta in batches:
            parents = [int(sample_id[1:]) for sample_id in micro_meta.sample_ids]
            if fragmented:
                selected = micro_meta.extra_info[TREE_ATTENTION_FRAGMENT_SELECTIONS]
                expected = materialize_tree_attention_fragments(
                    logical,
                    fragments=[fragment for _, fragment in selected],
                    parent_indices=[parents[parent] for parent, _ in selected],
                )
                assert data[TREE_ATTENTION_LAYOUTS] == expected[TREE_ATTENTION_LAYOUTS]
                if mode != "fallback":
                    assert data["input_lengths"].sum().item() <= 5
            else:
                expected = BatchedDataDict(
                    {key: value[parents] for key, value in logical.items()}
                )
                assert data[TREE_ATTENTION_LAYOUTS] == [
                    layouts[parent] for parent in parents
                ]
            for key in logical:
                # Unfragmented source rows may contain trailing padding.
                want = expected[key]
                if want.ndim > 1:
                    want = want[:, : data[key].shape[1]]
                torch.testing.assert_close(data[key], want)
            for row, length in zip(
                data[TREE_ATTENTION_EDGE_TARGET_IDS],
                data[TREE_ATTENTION_EDGE_LENGTHS],
                strict=True,
            ):
                seen_targets.extend(row[:length].tolist())
        assert broadcast.call_count > 0

    assert sorted(seen_targets) == [101, 102, 103, 201, 202]
    assert client.fetch_count == ((1 if fragmented else 2) if rank == 0 else 0)


@pytest.mark.parametrize("mode", ["unfragmented", "fragmented", "fallback"])
def test_tree_fetch_restores_layouts_on_all_replicas(tmp_path: Path, mode: str) -> None:
    _run_two_ranks(partial(_tree_fetch_body, mode=mode), str(tmp_path / "init_tree"))


@pytest.mark.parametrize("mode", ["unfragmented", "fragmented", "fallback"])
def test_tree_fetch_metadata_flow_without_collectives(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Check both replica branches independently of distributed startup."""
    payloads: deque[BatchedDataDict[Any]] = deque()
    group = SimpleNamespace(size=lambda: 2)
    monkeypatch.setattr(dist, "group", SimpleNamespace(WORLD=group))
    monkeypatch.setattr(dist, "get_global_rank", lambda _group, rank: rank)

    def record_or_receive(
        data: BatchedDataDict[Any] | None, *, is_leader: bool, src: int, group: Any
    ) -> BatchedDataDict[Any]:
        if is_leader:
            assert data is not None
            assert TREE_ATTENTION_LAYOUTS not in data
            payloads.append(deepcopy(data))
            return data
        assert data is None
        return payloads.popleft()

    # _tree_fetch_body wraps the actual mixin boundary and compares both ranks'
    # materialized fields against the original logical rows.
    monkeypatch.setattr(
        sys.modules[__name__], "_broadcast_batched_data_dict", record_or_receive
    )
    _tree_fetch_body(0, mode=mode)
    _tree_fetch_body(1, mode=mode)
    assert not payloads
