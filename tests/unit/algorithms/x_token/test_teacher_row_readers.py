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
"""CPU coverage for the sparse and dense teacher row readers."""

import copy

import pytest
import torch

from nemo_rl.algorithms.x_token.loss_utils import (
    DenseTeacherRowReader,
    SparseTeacherRowReader,
    gather_sparse_teacher_rows_from_ipc,
)
from nemo_rl.distributed.model_utils import (
    _get_tokens_on_this_cp_rank,
    cp_load_balanced_segments,
    distributed_vocab_topk_logz_force,
)
from nemo_rl.models.policy.utils import DENSE_TEACHER_IPC_FLAT_LAYOUT

CPU = torch.device("cpu")
BATCH, SEQ, VOCAB, K, TEMPERATURE = 2, 8, 12, 3, 0.7


def _canonical_order(row_logits: torch.Tensor) -> list[int]:
    """Token ids by descending logit, ties broken by the lowest id."""
    return sorted(range(row_logits.numel()), key=lambda v: (-float(row_logits[v]), v))


def _reference():
    torch.manual_seed(7)
    logits = torch.randn(BATCH, SEQ, VOCAB)
    # Row (1, 5) has an exact cutoff tie (ids 3 and 9 at rank K), so the
    # evicted slot is ambiguous by value and must follow the recorded tail id.
    logits[1, 5, :] = torch.linspace(-1.0, 1.0, VOCAB)
    logits[1, 5, 3] = logits[1, 5, -3]
    force = torch.full((BATCH, SEQ, 2), -1)
    for b in range(BATCH):
        for t in range(SEQ):
            order = _canonical_order(logits[b, t])
            force[b, t, 0] = order[0]  # a natural top-k label
            force[b, t, 1] = order[-1]  # a label outside the natural top-k
    return logits, force


def _sparse_payload(cp_size: int = 2):
    """Export the reference with the real primitive and split it like Megatron."""
    logits, force = _reference()
    fields = distributed_vocab_topk_logz_force(
        logits,
        force,
        K,
        None,
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        real_vocab_size=VOCAB,
        temperature=TEMPERATURE,
    )
    names = (
        "topk_logits",
        "topk_indices",
        "log_z",
        "natural_tail_indices",
        "forced_logits",
        "forced_indices",
        "forced_in_topk",
    )
    full = dict(zip(names, fields))
    full["log_z"] = full["log_z"].unsqueeze(-1)
    entries = [{"teacher_shards": []} for _ in range(BATCH)]
    for cp_rank in range(cp_size):
        local = {
            name: _get_tokens_on_this_cp_rank(value, cp_rank, cp_size, seq_dim=1)
            .unsqueeze(0)
            .contiguous()
            for name, value in full.items()
        }
        local_seq = SEQ // cp_size
        for b in range(BATCH):
            entries[b]["teacher_shards"].append(
                {
                    "transport": "sparse_topk",
                    **{f"{name}_ipc": tensor for name, tensor in local.items()},
                    "buf_idx": 0,
                    "sample_index_in_buf": b,
                    "topk_shape": (local_seq, K),
                    "forced_shape": (local_seq, 2),
                    "scalar_shape": (local_seq,),
                    "k": K,
                    "force_width": 2,
                    "temperature": TEMPERATURE,
                    "real_vocab_size": VOCAB,
                    "forced_in_topk_k": K,
                    "cp_rank": cp_rank,
                    "cp_size": cp_size,
                    "full_seq_len": SEQ,
                    "sequence_segments": cp_load_balanced_segments(
                        SEQ, cp_rank, cp_size
                    ),
                }
            )
    return entries, logits, force


def _expected_support(row_logits: torch.Tensor, label: int):
    natural = _canonical_order(row_logits)[:K]
    if label not in natural:
        natural = natural[: K - 1] + [label]
    ids = sorted(natural)
    return ids, row_logits[ids]


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_sparse_reader_materializes_natural_or_forced_support(cp_size):
    entries, logits, force = _sparse_payload(cp_size)
    reader = SparseTeacherRowReader(entries, device=CPU)
    requests = [
        (b, t, slot) for b in range(BATCH) for t in range(SEQ) for slot in (0, 1)
    ]
    batch = torch.tensor([b for b, _, _ in requests])
    positions = torch.tensor([t for _, t, _ in requests])
    labels = torch.stack([force[b, t, slot] for b, t, slot in requests])

    values, ids, log_z, in_topk = reader.gather_rows(batch, positions, labels)

    assert (reader.k, reader.full_seq_len, reader.force_width) == (K, SEQ, 2)
    assert values.dtype == torch.float32 and ids.dtype == torch.int32
    for row, (b, t, slot) in enumerate(requests):
        expected_ids, expected_values = _expected_support(
            logits[b, t], int(force[b, t, slot])
        )
        assert ids[row].tolist() == expected_ids
        torch.testing.assert_close(values[row], expected_values)
        assert bool(in_topk[row]) == (slot == 0)
    torch.testing.assert_close(
        log_z, torch.logsumexp(logits[batch, positions] / TEMPERATURE, dim=-1)
    )


def test_sparse_reader_reuses_routes_and_backings_across_bounded_gathers(monkeypatch):
    # A tensor-backed CPU payload must not consult CUDA.
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA was consulted")),
    )
    entries, _, force = _sparse_payload()
    reader = SparseTeacherRowReader(entries, device=CPU)

    reader.gather_rows(
        torch.tensor([1, 0]), torch.tensor([5, 7]), force[[1, 0], [5, 7], 0]
    )
    cached = {key: tensor.data_ptr() for key, tensor in reader._backing_tensors.items()}
    reader.gather_rows(
        torch.tensor([1, 0]), torch.tensor([3, 6]), force[[1, 0], [3, 6], 1]
    )

    # Seven fields per CP rank's slab, shared by every sample of that rank.
    assert len(cached) == 7 * 2
    assert cached == {
        key: tensor.data_ptr() for key, tensor in reader._backing_tensors.items()
    }
    reader.close()
    assert reader._backing_tensors == {}


def test_one_shot_helper_matches_reader():
    entries, _, force = _sparse_payload()
    batch = torch.tensor([1, 0, 1, 0])
    positions = torch.tensor([5, 7, 0, 3])
    labels = force[batch, positions, 1]
    expected = SparseTeacherRowReader(entries, device=CPU).gather_rows(
        batch, positions, labels
    )
    actual = gather_sparse_teacher_rows_from_ipc(
        entries, batch, positions, labels, device=CPU
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@pytest.mark.parametrize("failure", ["overlap", "gap", "local_gap"])
def test_sparse_reader_rejects_invalid_segment_coverage(failure):
    entries, _, _ = _sparse_payload()
    broken = copy.deepcopy(entries)
    shards = broken[0]["teacher_shards"]
    if failure == "overlap":
        shards[1]["sequence_segments"] = [(0, 1, 2), (2, 4, 2)]
    elif failure == "gap":
        shards[1]["sequence_segments"] = [(0, 3, 1), (1, 4, 3)]
    else:
        shards[0]["sequence_segments"] = [(0, 0, 2), (3, 6, 1)]

    with pytest.raises(ValueError, match="cover"):
        SparseTeacherRowReader(broken, device=CPU)


def test_sparse_reader_rejects_unknown_transport_and_unforced_labels():
    entries, logits, force = _sparse_payload()
    legacy = copy.deepcopy(entries)
    legacy[0]["teacher_shards"][0]["transport"] = "dense"
    with pytest.raises(ValueError, match="Unsupported sparse teacher transport"):
        SparseTeacherRowReader(legacy, device=CPU)

    reader = SparseTeacherRowReader(entries, device=CPU)
    unforced = int(next(v for v in range(VOCAB) if v not in force[0, 2].tolist()))
    with pytest.raises(ValueError, match="exactly once in the forced sidecar"):
        reader.gather_rows(
            torch.tensor([0]), torch.tensor([2]), torch.tensor([unforced])
        )
    with pytest.raises(ValueError, match="real-vocabulary ids"):
        reader.gather_rows(torch.tensor([0]), torch.tensor([2]), torch.tensor([VOCAB]))
    with pytest.raises(ValueError, match="exactly once"):
        reader.gather_rows(torch.tensor([0]), torch.tensor([SEQ]), force[0, 0, :1])


def test_sparse_reader_rejects_inconsistent_forced_copy():
    entries, _, force = _sparse_payload(cp_size=1)
    shard = entries[0]["teacher_shards"][0]
    # Row 0's slot-0 label is a natural id; corrupt its forced logit.
    shard["forced_logits_ipc"] = shard["forced_logits_ipc"].clone()
    shard["forced_logits_ipc"][0, 0, 0, 0] += 1.0
    reader = SparseTeacherRowReader(entries, device=CPU)
    with pytest.raises(ValueError, match="different raw logits"):
        reader.gather_rows(torch.tensor([0]), torch.tensor([0]), force[0, 0, :1])


def _dense_payload(*, tp_size: int, cp_size: int, compact_valid_len=None):
    """Dense teacher records in the Megatron/Automodel sender formats."""
    torch.manual_seed(3)
    logits = torch.randn(BATCH, SEQ, VOCAB)
    local_seq, local_vocab = SEQ // cp_size, VOCAB // tp_size
    entries = [{"teacher_shards": []} for _ in range(BATCH)]
    storages = []
    for cp_rank in range(cp_size):
        for tp_rank in range(tp_size):
            seq = slice(cp_rank * local_seq, (cp_rank + 1) * local_seq)
            vocab = slice(tp_rank * local_vocab, (tp_rank + 1) * local_vocab)
            common = {
                "tp_rank": tp_rank,
                "tp_size": tp_size,
                "cp_rank": cp_rank,
                "cp_size": cp_size,
                "vocab_start_index": vocab.start,
                "vocab_end_index": vocab.stop,
                "global_seq_start": seq.start,
                "full_seq_len": SEQ,
                "full_vocab_size": VOCAB,
                "actual_shape": (local_seq, local_vocab),
            }
            if compact_valid_len is None:
                storage = logits[:, seq, vocab].unsqueeze(0).contiguous()
                for b in range(BATCH):
                    entries[b]["teacher_shards"].append(
                        {
                            **common,
                            "payload_ipc": ("rect", cp_rank, tp_rank),
                            "buf_idx": 0,
                            "sample_index_in_buf": b,
                        }
                    )
            else:
                stored = [
                    max(0, min(valid, seq.stop) - seq.start)
                    for valid in compact_valid_len
                ]
                storage = torch.cat(
                    [
                        logits[b, seq.start : seq.start + stored[b], vocab]
                        for b in range(BATCH)
                    ]
                ).contiguous()
                offset = 0
                for b in range(BATCH):
                    entries[b]["teacher_shards"].append(
                        {
                            **common,
                            "payload_ipc": ("compact", cp_rank, tp_rank),
                            "buf_idx": 0,
                            "sample_index_in_buf": b,
                            "ipc_layout": DENSE_TEACHER_IPC_FLAT_LAYOUT,
                            "storage_shape": tuple(storage.shape),
                            "storage_token_offset": offset,
                            "storage_used_tokens": int(storage.shape[0]),
                            "storage_capacity_tokens": int(storage.shape[0]),
                            "stored_seq_len": stored[b],
                            "valid_seq_len": compact_valid_len[b],
                            "dtype": storage.dtype,
                        }
                    )
                    offset += stored[b]
            storages.append(((cp_rank, tp_rank), storage))
    return entries, logits, dict(storages)


def _patch_rebuild(monkeypatch, storages):
    opened = []

    def rebuild(handle, _device):
        opened.append(handle)
        return storages[(handle[1], handle[2])]

    monkeypatch.setattr(
        "nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc", rebuild
    )
    return opened


@pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 1), (2, 2), (3, 4)])
def test_dense_reader_stitches_requested_rows_across_tp_cp(
    monkeypatch, tp_size, cp_size
):
    entries, logits, storages = _dense_payload(tp_size=tp_size, cp_size=cp_size)
    opened = _patch_rebuild(monkeypatch, storages)
    reader = DenseTeacherRowReader(entries, device=torch.device("cuda", 0))
    reader.device = CPU  # CPU stand-in for the consumer GPU
    batch = torch.tensor([1, 0, 1, 1, 0])
    positions = torch.tensor([7, 0, 3, 3, 5])

    rows = reader.gather_rows(batch, positions)
    torch.testing.assert_close(rows, logits[batch, positions])
    assert (reader.full_seq_len, reader.full_vocab_size) == (SEQ, VOCAB)

    reader.gather_rows(torch.tensor([0, 1]), torch.tensor([6, 2]))
    # Each producer storage is opened at most once per reader.
    assert len(opened) == len(set(opened)) <= tp_size * cp_size
    reader.close()
    assert reader._storage == {}


def test_dense_reader_reads_compact_suffix_as_zero(monkeypatch):
    entries, logits, storages = _dense_payload(
        tp_size=2, cp_size=2, compact_valid_len=[5, 2]
    )
    _patch_rebuild(monkeypatch, storages)
    reader = DenseTeacherRowReader(entries, device=torch.device("cuda", 0))
    reader.device = CPU
    batch = torch.tensor([0, 0, 0, 1, 1])
    positions = torch.tensor([4, 5, 7, 1, 6])

    rows = reader.gather_rows(batch, positions)

    torch.testing.assert_close(rows[0], logits[0, 4])
    torch.testing.assert_close(rows[3], logits[1, 1])
    for zero_row in (1, 2, 4):
        assert torch.equal(rows[zero_row], torch.zeros(VOCAB))


def test_dense_reader_validates_grid_and_positions(monkeypatch):
    entries, _, storages = _dense_payload(tp_size=2, cp_size=2)
    _patch_rebuild(monkeypatch, storages)
    incomplete = copy.deepcopy(entries)
    incomplete[0]["teacher_shards"].pop()
    with pytest.raises(ValueError, match="incomplete TP/CP shard coverage"):
        DenseTeacherRowReader(incomplete, device=CPU)

    reader = DenseTeacherRowReader(entries, device=CPU)
    with pytest.raises(IndexError, match="position"):
        reader.gather_rows(torch.tensor([0]), torch.tensor([SEQ]))
    with pytest.raises(IndexError, match="batch index"):
        reader.gather_rows(torch.tensor([BATCH]), torch.tensor([0]))
