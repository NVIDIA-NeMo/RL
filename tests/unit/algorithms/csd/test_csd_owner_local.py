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
"""Owner-local CSD evaluation against the replicated reference.

The student logits stay in their load-balanced CP shard and TP vocabulary
shard. Summed over CP ranks, the owner-local loss and gradient must equal the
replicated TP1/CP1 loss with no ``1/CP`` factor, at every TP/CP size.
"""

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.algorithms.loss.csd import (
    CSDLossConfig,
    CSDLossFn,
    owner_local_config_violations,
)
from nemo_rl.algorithms.x_token import loss_utils
from nemo_rl.algorithms.x_token.loss_utils import (
    LocalizedAlignment,
    SparseTeacherRowReader,
    _chunk_ids_to_spans,
    prepare_csd_owner_local_loss_input,
)
from nemo_rl.algorithms.xtoken_off_policy_distillation import (
    _build_teacher_force_include_token_id_slots,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    _get_tokens_on_this_cp_rank,
    cp_load_balanced_segments,
    distributed_vocab_topk_logz_force,
)
from tests.unit.algorithms.csd.test_csd_loss_parity import _write_subtok_tables

V = 16  # student and teacher vocabulary (ids 0..9 are 1-to-1 common)
SEQ = 8
TEMPERATURE = 1.3
SPARSE_K = 4

# (student_start, student_end, teacher_start, teacher_end) per chunk:
#   c0 common at position 0 (unshifted) and c1 1-to-2 shifted share teacher
#   row 0, which therefore needs two forced labels; c2 2-to-1; c3 common;
#   c4 2-to-2 whose student prefix row 5 and final row 6 sit on different CP
#   ranks at CP=2.
CHUNKS = [(0, 1, 0, 1), (1, 2, 1, 3), (3, 5, 4, 5), (5, 6, 5, 6), (6, 8, 6, 8)]
STUDENT_IDS = [[0, 5, 0, 9, 5, 2, 3, 1], [1, 5, 0, 9, 6, 4, 7, 2]]
TEACHER_IDS = [[0, 8, 3, 0, 5, 2, 3, 1], [1, 8, 4, 0, 6, 4, 7, 2]]


def _fixture(tmpdir: str, *, seed: int = 11) -> dict:
    os.makedirs(tmpdir, exist_ok=True)
    torch.manual_seed(seed)
    fwd_path, rev_path = _write_subtok_tables(tmpdir, V, V)
    student_chunk_id = torch.full((2, SEQ), -1, dtype=torch.long)
    teacher_chunk_id = torch.full((2, SEQ), -1, dtype=torch.long)
    for k, (ss, se, ts, te) in enumerate(CHUNKS):
        student_chunk_id[:, ss:se] = k
        teacher_chunk_id[:, ts:te] = k
    return {
        "fwd_path": fwd_path,
        "rev_path": rev_path,
        "student_logits": torch.randn(2, SEQ, V),
        "teacher_logits": torch.randn(2, SEQ, V) * 2.0,
        # A same-tokenizer teacher for the multi-teacher case.
        "same_vocab_teacher_logits": torch.randn(2, SEQ, V),
        "student_ids": torch.tensor(STUDENT_IDS),
        "teacher_ids": torch.tensor(TEACHER_IDS),
        "student_chunk_id": student_chunk_id,
        "teacher_chunk_id": teacher_chunk_id,
        "pair_valid": torch.ones((2, len(CHUNKS)), dtype=torch.bool),
        "sample_mask": torch.ones(2, dtype=torch.long),
    }


def _loss_config(
    fx: dict,
    *,
    owner_local: bool,
    k: int,
    realized_only: bool = False,
    noise_filter_topk: int = 0,
    dynamic_loss_scaling: bool = True,
    same_vocab_teacher: bool = False,
    keep_realized: bool = True,
) -> dict:
    num_teachers = 2 if same_vocab_teacher else 1
    return {
        "temperature": TEMPERATURE,
        "vocab_topk": 8,
        "reverse_kl": False,
        "kl_loss_weight": 1.0,
        "ce_loss_scale": 0.3,
        "dynamic_loss_scaling": dynamic_loss_scaling,
        "kd_loss_mode": "sum",
        "student_vocab_size": V,
        "teacher_vocab_sizes": [V] * num_teachers,
        "projection_matrix_paths": [None] * num_teachers,
        "teacher_weights": [1.0, 0.5][:num_teachers],
        "common_indices_from_subtoks": True,
        "student_to_teacher_table_paths": [fx["fwd_path"], None][:num_teachers],
        "teacher_to_student_table_paths": [fx["rev_path"], None][:num_teachers],
        "kl_chunk_shift": True,
        "csd_loss_fn": "jsd",
        "csd_last_pos_loss_fn": "kl",
        "csd_jsd_beta": 0.4,
        "csd_mismatch_loss_beta": 2.0,
        "csd_noise_filter_topk": noise_filter_topk,
        "csd_realized_only_mismatch": realized_only,
        "teacher_topk_ipc_k": k,
        "teacher_topk_ipc_keep_realized": keep_realized,
        # Small tiles so the batching loops run more than once.
        "csd_selected_row_chunk_size": 3,
        "csd_support_chunk_batch_size": 2,
        "owner_local": owner_local,
    }


def _batch(fx: dict) -> BatchedDataDict:
    token_mask = torch.ones((2, SEQ), dtype=torch.long)
    return BatchedDataDict(
        {
            "input_ids": fx["student_ids"],
            "input_lengths": torch.full((2,), SEQ),
            "token_mask": token_mask,
            "kd_token_mask": token_mask,
            "sample_mask": fx["sample_mask"],
            "teacher_0_input_ids": fx["teacher_ids"],
            "teacher_0_token_mask": token_mask,
            "alignment_0_pair_valid": fx["pair_valid"],
            "alignment_0_pair_is_correct": fx["pair_valid"],
            "alignment_0_student_chunk_id": fx["student_chunk_id"],
            "alignment_0_teacher_chunk_id": fx["teacher_chunk_id"],
        }
    )


def _normalizers(fx: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_mask = fx["sample_mask"].float()
    valid_toks = torch.tensor(float((SEQ - 1) * sample_mask.sum()))
    valid_chunks = (fx["pair_valid"].float() * sample_mask.unsqueeze(-1)).sum()
    return sample_mask.sum(), valid_toks, valid_chunks


def _sparse_records(
    fx: dict, *, k: int, noise_filter_topk: int, teacher_cp: int
) -> list[dict]:
    """Sparse teacher records as the Megatron sender lays them out."""
    force = _build_teacher_force_include_token_id_slots(
        _batch(fx), teacher_idx=0, kl_chunk_shift=True
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
    full = dict(
        zip(
            names,
            distributed_vocab_topk_logz_force(
                fx["teacher_logits"],
                force,
                k,
                None,
                vocab_start_index=0,
                vocab_end_index=V,
                real_vocab_size=V,
                temperature=TEMPERATURE,
                noise_filter_k=noise_filter_topk,
            ),
        )
    )
    full["log_z"] = full["log_z"].unsqueeze(-1)
    entries: list[dict] = [{"teacher_shards": []} for _ in range(2)]
    local_seq = SEQ // teacher_cp
    for cp_rank in range(teacher_cp):
        local = {
            name: _get_tokens_on_this_cp_rank(value, cp_rank, teacher_cp, seq_dim=1)
            .unsqueeze(0)
            .contiguous()
            for name, value in full.items()
        }
        for b in range(2):
            entries[b]["teacher_shards"].append(
                {
                    "transport": "sparse_topk",
                    **{f"{name}_ipc": value for name, value in local.items()},
                    "buf_idx": 0,
                    "sample_index_in_buf": b,
                    "topk_shape": (local_seq, k),
                    "forced_shape": (local_seq, 2),
                    "scalar_shape": (local_seq,),
                    "k": k,
                    "force_width": 2,
                    "temperature": TEMPERATURE,
                    "real_vocab_size": V,
                    "forced_in_topk_k": noise_filter_topk or k,
                    "full_seq_len": SEQ,
                    "sequence_segments": cp_load_balanced_segments(
                        SEQ, cp_rank, teacher_cp
                    ),
                }
            )
    return entries


def _dense_records(teacher_logits: torch.Tensor, *, tp: int, cp: int) -> list[dict]:
    """Rectangular dense teacher records with contiguous CP windows."""
    local_seq, local_vocab = SEQ // cp, V // tp
    entries: list[dict] = [{"teacher_shards": []} for _ in range(2)]
    for cp_rank in range(cp):
        for tp_rank in range(tp):
            seq = slice(cp_rank * local_seq, (cp_rank + 1) * local_seq)
            vocab = slice(tp_rank * local_vocab, (tp_rank + 1) * local_vocab)
            storage = teacher_logits[:, seq, vocab].unsqueeze(0).contiguous()
            for b in range(2):
                entries[b]["teacher_shards"].append(
                    {
                        "payload_ipc": storage,
                        "buf_idx": 0,
                        "sample_index_in_buf": b,
                        "actual_shape": (local_seq, local_vocab),
                        "tp_rank": tp_rank,
                        "tp_size": tp,
                        "cp_rank": cp_rank,
                        "cp_size": cp,
                        "vocab_start_index": vocab.start,
                        "vocab_end_index": vocab.stop,
                        "global_seq_start": seq.start,
                        "full_seq_len": SEQ,
                        "full_vocab_size": V,
                    }
                )
    return entries


def _replicated_reference(
    fx: dict, *, same_vocab_teacher: bool = False, **cfg_kwargs
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """TP1/CP1 replicated CSD loss, gradient and metrics with a dense teacher."""
    loss_fn = CSDLossFn(
        _loss_config(
            fx,
            owner_local=False,
            k=0,
            same_vocab_teacher=same_vocab_teacher,
            **cfg_kwargs,
        )
    )
    batch = _batch(fx)
    logits = fx["student_logits"].clone().requires_grad_(True)
    max_pairs = int(fx["pair_valid"].shape[1])
    aligns = {
        0: LocalizedAlignment(
            sample_mask=fx["sample_mask"],
            pair_valid=fx["pair_valid"],
            pair_is_correct=fx["pair_valid"],
            student_input_ids=fx["student_ids"],
            student_token_mask=batch["token_mask"],
            student_kd_token_mask=batch["kd_token_mask"],
            teacher_input_ids=fx["teacher_ids"],
            student_spans=_chunk_ids_to_spans(fx["student_chunk_id"], max_pairs),
            teacher_spans=_chunk_ids_to_spans(fx["teacher_chunk_id"], max_pairs),
            num_chunks=torch.full((2,), max_pairs),
        )
    }
    teachers = {0: fx["teacher_logits"]}
    if same_vocab_teacher:
        aligns[1] = LocalizedAlignment(
            sample_mask=fx["sample_mask"],
            student_input_ids=fx["student_ids"],
            student_token_mask=batch["token_mask"],
            student_kd_token_mask=batch["kd_token_mask"],
        )
        teachers[1] = fx["same_vocab_teacher_logits"]
    valid_seqs, valid_toks, valid_chunks = _normalizers(fx)
    loss, metrics = loss_fn(
        batch,
        valid_seqs,
        valid_toks,
        logits,
        logits,
        teachers,
        aligns,
        global_valid_chunks_by_idx={0: valid_chunks},
        global_valid_kd_toks=valid_toks,
    )
    loss.backward()
    return loss.detach(), logits.grad.detach(), metrics


def _groups(rank: int, tp_size: int, cp_size: int):
    """Return ``(tp_rank, cp_rank, tp_group, cp_group)`` on a [TP, CP] grid."""
    tp_rank, cp_rank = rank // cp_size, rank % cp_size
    tp_group = cp_group = None
    for cp_index in range(cp_size):
        group = dist.new_group(
            ranks=[tp_index * cp_size + cp_index for tp_index in range(tp_size)]
        )
        if cp_index == cp_rank:
            tp_group = group
    for tp_index in range(tp_size):
        group = dist.new_group(
            ranks=[tp_index * cp_size + cp_index for cp_index in range(cp_size)]
        )
        if tp_index == tp_rank:
            cp_group = group
    return (
        tp_rank,
        cp_rank,
        tp_group if tp_size > 1 else None,
        cp_group if cp_size > 1 else None,
    )


def _owner_local_step(
    fx: dict,
    *,
    transport: str,
    tp_rank: int,
    cp_rank: int,
    tp_size: int,
    cp_size: int,
    tp_group,
    cp_group,
    same_vocab_teacher: bool = False,
    records: list[dict] | None = None,
    sparse_k: int = SPARSE_K,
    **cfg_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict, torch.Tensor]:
    """One rank's owner-local loss, local gradient and metrics."""
    k = sparse_k if transport == "sparse" else 0
    loss_fn = CSDLossFn(
        _loss_config(
            fx,
            owner_local=True,
            k=k,
            same_vocab_teacher=same_vocab_teacher,
            **cfg_kwargs,
        )
    )
    positions = _get_tokens_on_this_cp_rank(
        torch.arange(SEQ), cp_rank, cp_size, seq_dim=0
    )
    vocab_width = V // tp_size
    local_logits = (
        fx["student_logits"][
            :, positions, tp_rank * vocab_width : (tp_rank + 1) * vocab_width
        ]
        .clone()
        .requires_grad_(True)
    )
    batch = _batch(fx)
    if records is None:
        records = (
            _sparse_records(
                fx,
                k=k,
                noise_filter_topk=cfg_kwargs.get("noise_filter_topk", 0),
                teacher_cp=2,
            )
            if transport == "sparse"
            else _dense_records(fx["teacher_logits"], tp=2, cp=2)
        )
    batch[f"teacher_0_{transport if transport == 'sparse' else 'full'}_logits_ipc"] = (
        records
    )
    if same_vocab_teacher:
        batch["teacher_1_full_logits_ipc"] = [{"teacher_shards": []}] * 2
    (
        student_logits_contig,
        teacher_full,
        row_payloads,
        aligns,
        tp_g,
        cp_g,
        dp_cp_group,
    ) = prepare_csd_owner_local_loss_input(
        local_logits,
        batch,
        cross_tokenizer_teacher_mask=[True, False][: 2 if same_vocab_teacher else 1],
        vocab_parallel_group=tp_group,
        context_parallel_group=cp_group,
    )
    valid_seqs, valid_toks, valid_chunks = _normalizers(fx)
    loss, metrics = loss_fn(
        batch,
        valid_seqs,
        valid_toks,
        local_logits,
        student_logits_contig,
        teacher_full,
        aligns,
        teacher_row_payloads_by_idx=row_payloads,
        global_valid_chunks_by_idx={0: valid_chunks},
        global_valid_kd_toks=valid_toks,
        tp_group=tp_g,
        cp_group=cp_g,
        dp_cp_group=dp_cp_group,
    )
    loss.backward()
    return loss.detach(), local_logits.grad.detach(), metrics, positions


def _reassemble_gradient(
    local_grad: torch.Tensor, *, world_size: int, tp_size: int, cp_size: int
) -> torch.Tensor:
    gathered = [torch.empty_like(local_grad) for _ in range(world_size)]
    dist.all_gather(gathered, local_grad.contiguous())
    full = torch.zeros(2, SEQ, V)
    vocab_width = V // tp_size
    for rank, shard in enumerate(gathered):
        tp_rank, cp_rank = rank // cp_size, rank % cp_size
        positions = _get_tokens_on_this_cp_rank(
            torch.arange(SEQ), cp_rank, cp_size, seq_dim=0
        )
        full[:, positions, tp_rank * vocab_width : (tp_rank + 1) * vocab_width] = shard
    return full


def _init_gloo(rank: int, world_size: int, init_file: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _run_topology(
    rank: int,
    world_size: int,
    init_file: str,
    tmpdir: str,
    tp_size: int,
    cp_size: int,
    transport: str,
    case: str,
) -> None:
    _init_gloo(rank, world_size, init_file)
    try:
        tp_rank, cp_rank, tp_group, cp_group = _groups(rank, tp_size, cp_size)
        fx = _fixture(os.path.join(tmpdir, f"rank{rank}"))
        cfg_kwargs: dict = {}
        same_vocab_teacher = False
        if case == "realized_only":
            cfg_kwargs["realized_only"] = True
        elif case == "noise_filter":
            cfg_kwargs["noise_filter_topk"] = 6
        elif case == "fixed_weights":
            cfg_kwargs["dynamic_loss_scaling"] = False
        elif case == "same_vocab_teacher":
            same_vocab_teacher = True
            same_vocab = fx["same_vocab_teacher_logits"]

            def contiguous_window(_entries, cp_group, device):
                del device
                size = dist.get_world_size(cp_group) if cp_group is not None else 1
                rank_in_cp = dist.get_rank(cp_group) if cp_group is not None else 0
                width = SEQ // size
                return same_vocab[:, rank_in_cp * width : (rank_in_cp + 1) * width], 0

            loss_utils.rebuild_teacher_full_logits_from_ipc = contiguous_window
        elif case == "no_owned_chunks":
            # Every chunk's final student predictor row is held by CP rank 0
            # (rows 0, 1, 6, 7 at CP=2); CP rank 1 still holds prefix rows.
            fx["pair_valid"][:, [2, 3]] = False

        if transport == "sparse":
            # Reference: owner-local at TP1/CP1, then cross-checked against the
            # replicated path by test_sparse_owner_local_matches_replicated_sparse.
            ref_loss, ref_grad, ref_metrics, _ = _owner_local_step(
                fx,
                transport="sparse",
                tp_rank=0,
                cp_rank=0,
                tp_size=1,
                cp_size=1,
                tp_group=None,
                cp_group=None,
                same_vocab_teacher=same_vocab_teacher,
                **cfg_kwargs,
            )
        else:
            ref_loss, ref_grad, ref_metrics = _replicated_reference(
                fx, same_vocab_teacher=same_vocab_teacher, **cfg_kwargs
            )

        original_all_gather = dist.all_gather

        def all_gather_without_cp(tensor_list, tensor, group=None, async_op=False):
            if not same_vocab_teacher and cp_group is not None and group is cp_group:
                raise AssertionError("owner-local KD gathered across CP")
            return original_all_gather(
                tensor_list, tensor, group=group, async_op=async_op
            )

        dist.all_gather = all_gather_without_cp
        try:
            loss, local_grad, metrics, _ = _owner_local_step(
                fx,
                transport=transport,
                tp_rank=tp_rank,
                cp_rank=cp_rank,
                tp_size=tp_size,
                cp_size=cp_size,
                tp_group=tp_group,
                cp_group=cp_group,
                same_vocab_teacher=same_vocab_teacher,
                **cfg_kwargs,
            )
        finally:
            dist.all_gather = original_all_gather

        # The fixture must exercise both chunk kinds on every case.
        assert ref_metrics["num_common_chunks_t0"] > 0
        assert ref_metrics["num_mismatch_chunks_t0"] > 0
        cp_loss = loss.clone()
        if cp_group is not None:
            dist.all_reduce(cp_loss, group=cp_group)
        torch.testing.assert_close(cp_loss, ref_loss, rtol=2e-5, atol=2e-6)
        for key in ("loss", "kl_loss", "ce_loss", "kl_loss_scale", "accuracy"):
            assert metrics[key] == pytest.approx(
                ref_metrics[key], rel=2e-5, abs=2e-6
            ), key
        for key in ("num_common_chunks_t0", "num_mismatch_chunks_t0"):
            assert metrics[key] == ref_metrics[key], key
        full_grad = _reassemble_gradient(
            local_grad, world_size=world_size, tp_size=tp_size, cp_size=cp_size
        )
        torch.testing.assert_close(full_grad, ref_grad, rtol=5e-5, atol=5e-6)
    finally:
        dist.destroy_process_group()


_TOPOLOGIES = [(1, 1), (2, 1), (1, 2), (2, 2)]


def _spawn(tmp_path: Path, fn, world_size: int, *args) -> None:
    mp.spawn(
        fn,
        args=(
            world_size,
            os.fspath(tmp_path / "gloo_init"),
            os.fspath(tmp_path),
            *args,
        ),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend unavailable")
@pytest.mark.parametrize("transport", ["dense", "sparse"])
@pytest.mark.parametrize("case", ["prefix_support", "realized_only"])
@pytest.mark.parametrize("tp_size,cp_size", _TOPOLOGIES, ids=lambda v: str(v))
def test_owner_local_loss_and_gradient_are_topology_invariant(
    tmp_path, tp_size, cp_size, transport, case
):
    """CP-summed loss and reassembled gradient equal the reference, no 1/CP."""
    _spawn(
        tmp_path,
        _run_topology,
        tp_size * cp_size,
        tp_size,
        cp_size,
        transport,
        case,
    )


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend unavailable")
@pytest.mark.parametrize("transport", ["dense", "sparse"])
@pytest.mark.parametrize(
    "case", ["noise_filter", "fixed_weights", "same_vocab_teacher", "no_owned_chunks"]
)
def test_owner_local_tp2_cp2_variants(tmp_path, transport, case):
    _spawn(tmp_path, _run_topology, 4, 2, 2, transport, case)


def test_sparse_owner_local_matches_replicated_sparse(tmp_path):
    """At TP1/CP1, owner-local sparse equals the replicated sparse path.

    The replicated payload holds one forced label per row, so chunk 0 (whose
    teacher row would need two labels) is left out of this comparison.
    """
    fx = _fixture(os.fspath(tmp_path))
    fx["pair_valid"][:, 0] = False
    records = _sparse_records(fx, k=SPARSE_K, noise_filter_topk=0, teacher_cp=2)
    owner_loss, owner_grad, owner_metrics, _ = _owner_local_step(
        fx,
        transport="sparse",
        tp_rank=0,
        cp_rank=0,
        tp_size=1,
        cp_size=1,
        tp_group=None,
        cp_group=None,
        records=records,
    )

    # Materialize the replicated [B, T, K] payload: each row's natural top-k
    # with its single forced label kept, exactly as the reader serves it.
    force = _build_teacher_force_include_token_id_slots(
        _batch(fx), teacher_idx=0, kl_chunk_shift=True
    )
    assert int((force[..., 1] >= 0).sum()) == 0
    reader = SparseTeacherRowReader(records, device=torch.device("cpu"))
    natural_logits, natural_ids = torch.topk(fx["teacher_logits"], SPARSE_K, dim=-1)
    natural_ids, order = natural_ids.sort(dim=-1)
    topk_logits = natural_logits.gather(-1, order)
    topk_ids = natural_ids.to(torch.int32)
    labeled = (force[..., 0] >= 0).nonzero().tolist()
    rows = reader.gather_rows(
        torch.tensor([b for b, _ in labeled]),
        torch.tensor([t for _, t in labeled]),
        torch.stack([force[b, t, 0] for b, t in labeled]),
    )
    for row, (b, t) in enumerate(labeled):
        topk_logits[b, t] = rows[0][row]
        topk_ids[b, t] = rows[1][row]
    log_z = torch.logsumexp(fx["teacher_logits"] / TEMPERATURE, dim=-1)

    loss_fn = CSDLossFn(_loss_config(fx, owner_local=False, k=SPARSE_K))
    batch = _batch(fx)
    logits = fx["student_logits"].clone().requires_grad_(True)
    max_pairs = int(fx["pair_valid"].shape[1])
    align = LocalizedAlignment(
        sample_mask=fx["sample_mask"],
        pair_valid=fx["pair_valid"],
        student_input_ids=fx["student_ids"],
        student_token_mask=batch["token_mask"],
        student_kd_token_mask=batch["kd_token_mask"],
        teacher_input_ids=fx["teacher_ids"],
        student_spans=_chunk_ids_to_spans(fx["student_chunk_id"], max_pairs),
        teacher_spans=_chunk_ids_to_spans(fx["teacher_chunk_id"], max_pairs),
        num_chunks=torch.full((2,), max_pairs),
    )
    valid_seqs, valid_toks, valid_chunks = _normalizers(fx)
    loss, metrics = loss_fn(
        batch,
        valid_seqs,
        valid_toks,
        logits,
        logits,
        {},
        {0: align},
        teacher_sparse_logits_by_idx={0: (topk_logits, topk_ids, log_z, None)},
        global_valid_chunks_by_idx={0: valid_chunks},
        global_valid_kd_toks=valid_toks,
    )
    loss.backward()

    torch.testing.assert_close(owner_loss, loss.detach(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(owner_grad, logits.grad, rtol=5e-5, atol=5e-6)
    assert owner_metrics["num_mismatch_chunks_t0"] == metrics["num_mismatch_chunks_t0"]
    assert owner_metrics["num_common_chunks_t0"] == metrics["num_common_chunks_t0"]


_TEACHER_LABEL_CASES = [
    None,
    # Chunk c3 (1-to-1): the realized teacher 7 is common but is not the
    # realized student's pair.
    (5, 7),
    # Chunk c1 (1-to-2): the teacher prefix (7,) has no forward-table pairs.
    (1, 7),
]


def _relabeled_fixture(tmp_path, teacher_label) -> dict:
    """The owner-local fixture with one teacher label replaced, if given."""
    fx = _fixture(os.fspath(tmp_path))
    if teacher_label is not None:
        position, token = teacher_label
        fx["teacher_ids"] = fx["teacher_ids"].clone()
        fx["teacher_ids"][:, position] = token
    return fx


_TP1_CP1_STEP = dict(
    tp_rank=0,
    cp_rank=0,
    tp_size=1,
    cp_size=1,
    tp_group=None,
    cp_group=None,
    dynamic_loss_scaling=False,
)


@pytest.mark.parametrize("teacher_label", _TEACHER_LABEL_CASES)
def test_owner_local_dense_matches_sparse_full_support(tmp_path, teacher_label):
    """Owner-local dense K=0 optimizes the same loss as sparse K=V."""
    fx = _relabeled_fixture(tmp_path, teacher_label)
    dense_loss, dense_grad, dense_metrics, _ = _owner_local_step(
        fx, transport="dense", **_TP1_CP1_STEP
    )
    sparse_loss, sparse_grad, sparse_metrics, _ = _owner_local_step(
        fx, transport="sparse", sparse_k=V, **_TP1_CP1_STEP
    )

    torch.testing.assert_close(dense_loss, sparse_loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(dense_grad, sparse_grad, rtol=5e-5, atol=5e-6)
    for key in ("num_common_chunks_t0", "num_mismatch_chunks_t0"):
        assert dense_metrics[key] == sparse_metrics[key]


@pytest.mark.parametrize("keep_realized", [False, True])
@pytest.mark.parametrize("teacher_label", _TEACHER_LABEL_CASES)
def test_owner_local_dense_matches_replicated_dense(
    tmp_path, teacher_label, keep_realized
):
    """Owner-local dense K=0 equals the replicated loss for either keep setting."""
    fx = _relabeled_fixture(tmp_path, teacher_label)
    owner_loss, owner_grad, owner_metrics, _ = _owner_local_step(
        fx, transport="dense", keep_realized=keep_realized, **_TP1_CP1_STEP
    )
    loss, grad, metrics = _replicated_reference(
        fx, keep_realized=keep_realized, dynamic_loss_scaling=False
    )

    torch.testing.assert_close(owner_loss, loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(owner_grad, grad, rtol=5e-5, atol=5e-6)
    for key in ("num_common_chunks_t0", "num_mismatch_chunks_t0"):
        assert owner_metrics[key] == metrics[key]


def test_owner_local_ce_matches_replicated_ce(tmp_path):
    fx = _fixture(os.fspath(tmp_path))
    fx["sample_mask"] = torch.tensor([1, 0])
    batch = _batch(fx)
    _, valid_toks, _ = _normalizers(fx)
    owner = CSDLossFn(_loss_config(fx, owner_local=True, k=0))
    replicated = CSDLossFn(_loss_config(fx, owner_local=False, k=0))
    logits = fx["student_logits"].clone().requires_grad_(True)
    reference_logits = fx["student_logits"].clone().requires_grad_(True)

    owner_ce = owner._compute_owner_local_ce(
        logits, batch, valid_toks, tp_group=None, cp_group=None
    )
    reference_ce = replicated._compute_ce(reference_logits, batch, valid_toks)
    owner_ce.backward()
    reference_ce.backward()

    torch.testing.assert_close(owner_ce, reference_ce, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-6, atol=1e-7)
    assert torch.equal(logits.grad[1], torch.zeros_like(logits.grad[1]))


def _run_error_agreement(
    rank: int, world_size: int, init_file: str, tmpdir: str
) -> None:
    _init_gloo(rank, world_size, init_file)
    try:
        _, cp_rank, _, cp_group = _groups(rank, 1, world_size)
        fx = _fixture(os.path.join(tmpdir, f"rank{rank}"))
        records = _sparse_records(fx, k=SPARSE_K, noise_filter_topk=0, teacher_cp=2)
        if cp_rank == 1:
            records[0]["teacher_shards"][0]["transport"] = "unknown"
        try:
            _owner_local_step(
                fx,
                transport="sparse",
                tp_rank=0,
                cp_rank=cp_rank,
                tp_size=1,
                cp_size=world_size,
                tp_group=None,
                cp_group=cp_group,
                records=records,
            )
        except (RuntimeError, ValueError) as exc:
            Path(tmpdir, f"error_{rank}.txt").write_text(f"{type(exc).__name__}: {exc}")
        else:
            Path(tmpdir, f"error_{rank}.txt").write_text("no error")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend unavailable")
def test_owner_local_setup_failure_raises_on_every_rank(tmp_path):
    """A setup error on one CP rank makes every rank raise instead of hanging."""
    _spawn(tmp_path, _run_error_agreement, 2)
    errors = [(tmp_path / f"error_{rank}.txt").read_text() for rank in range(2)]
    assert errors[0].startswith("RuntimeError") and "another" in errors[0]
    assert errors[1].startswith("ValueError") and "transport" in errors[1]


def test_owner_local_config_violations_cover_the_restrictions():
    base = {"student_vocab_size": 8, "teacher_topk_ipc_k": 128}
    assert owner_local_config_violations(CSDLossConfig(**base)) == []
    cases = {
        "kd_loss_mode": ("averaged_logits", "kd_loss_mode"),
        "sum_weights_metric": ("ce", "sum_weights_metric"),
        "csd_pure_alm": (True, "csd_pure_alm"),
        "csd_loss_fn": ("bce", "csd_loss_fn='bce'"),
        "csd_last_pos_loss_fn": ("bce", "csd_last_pos_loss_fn='bce'"),
        "teacher_topk_ipc_keep_realized": (False, "keep_realized=false"),
    }
    for field, (value, message) in cases.items():
        violations = owner_local_config_violations(
            CSDLossConfig(**{**base, field: value})
        )
        assert len(violations) == 1 and message in violations[0], field
    # BCE on the binary realized-only partitions, keep_realized implied by
    # realized-only mismatch, and K=1 are all supported.
    assert (
        owner_local_config_violations(
            CSDLossConfig(
                **base,
                csd_last_pos_loss_fn="bce",
                csd_realized_only_mismatch=True,
                teacher_topk_ipc_keep_realized=False,
            )
        )
        == []
    )
    assert (
        owner_local_config_violations(
            CSDLossConfig(
                student_vocab_size=8,
                teacher_topk_ipc_k=1,
                teacher_topk_ipc_keep_realized=False,
            )
        )
        == []
    )
    with pytest.raises(ValueError, match="does not support csd_pure_alm=true"):
        CSDLossFn(CSDLossConfig(**base, csd_pure_alm=True, owner_local=True))
    CSDLossFn(CSDLossConfig(**base, csd_pure_alm=True))


def test_owner_local_requires_the_step_global_chunk_count(tmp_path):
    fx = _fixture(os.fspath(tmp_path))
    loss_fn = CSDLossFn(_loss_config(fx, owner_local=True, k=0))
    with pytest.raises(ValueError, match="step-global valid-chunk count"):
        loss_fn._owner_local_sum_kd(
            fx["student_logits"],
            None,
            {},
            {},
            {},
            torch.tensor(1.0),
            None,
            tp_group=None,
            cp_group=None,
        )
