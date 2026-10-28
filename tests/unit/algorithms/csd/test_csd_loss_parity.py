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
"""CPU regression coverage for CSD values, gradients, masking, and transport."""

import os
import tempfile

import pytest
import torch


def _write_subtok_tables(tmpdir, v_s, v_t):
    """Tiny forward/reverse pseudo-target tables.

    Student/teacher ids 0..9 are 1-to-1 common (length-1 chain, mapped to the
    same id). To exercise the mismatch (prefix-support) path with
    ``keep_realized=False`` (which needs >= 2 support pairs per chunk):

      * Forward (student->teacher): students 10 and 11 both have a length-2
        chain with teacher prefix ``(8,)`` and distinct finals -> the shared
        key ``(2, (8,))`` maps to two ``(s, t)`` pairs, so a 1-to-2 chunk whose
        teacher prefix is ``8`` finds 2 support pairs.
      * Reverse (teacher->student): teachers 10 and 11 both have a length-2
        chain with student prefix ``(9,)`` and distinct finals -> the shared
        key ``(2, (9,))`` maps to two pairs, so a 2-to-1 chunk whose student
        prefix is ``9`` finds 2 support pairs.
    """
    max_chain = 3
    subtoks = torch.full((v_s, max_chain), -1, dtype=torch.long)
    lengths = torch.zeros((v_s,), dtype=torch.long)
    for s in range(10):
        subtoks[s, 0] = s
        lengths[s] = 1
    # Shared forward prefix (8,) with two distinct finals.
    subtoks[10, 0], subtoks[10, 1], lengths[10] = 8, 3, 2
    subtoks[11, 0], subtoks[11, 1], lengths[11] = 8, 4, 2
    # Filler length-2 chains (distinct prefixes; harmless).
    for s in range(12, v_s):
        subtoks[s, 0], subtoks[s, 1], lengths[s] = s % 10, (s + 3) % 10, 2
    fwd_path = os.path.join(tmpdir, "fwd_subtoks.pt")
    torch.save({"subtoks": subtoks, "lengths": lengths}, fwd_path)

    subtoks_r = torch.full((v_t, max_chain), -1, dtype=torch.long)
    lengths_r = torch.zeros((v_t,), dtype=torch.long)
    for t in range(10):
        subtoks_r[t, 0] = t
        lengths_r[t] = 1
    # Shared reverse prefix (9,) with two distinct finals.
    subtoks_r[10, 0], subtoks_r[10, 1], lengths_r[10] = 9, 3, 2
    subtoks_r[11, 0], subtoks_r[11, 1], lengths_r[11] = 9, 4, 2
    for t in range(12, v_t):
        subtoks_r[t, 0], subtoks_r[t, 1], lengths_r[t] = t % 10, (t + 3) % 10, 2
    rev_path = os.path.join(tmpdir, "rev_subtoks.pt")
    torch.save({"subtoks": subtoks_r, "lengths": lengths_r}, rev_path)
    return fwd_path, rev_path


def _csd_knobs(fwd_path, rev_path, v_t):
    """CSD settings shared by dense and sparse transport checks."""
    return {
        "temperature": 1.0,
        "teacher_vocab_size": v_t,
        "common_indices_from_subtoks": True,
        "student_to_teacher_table_path": fwd_path,
        "teacher_to_student_table_path": rev_path,
        "kl_chunk_shift": True,
        "csd_loss_fn": "jsd",
        "csd_last_pos_loss_fn": "kl",
        "csd_jsd_beta": 0.5,
        "csd_mismatch_loss_beta": 2.0,
        "csd_noise_filter_topk": 0,
        "reverse_kl": False,
    }


def _build_case(name, tmpdir, seed=0):
    """Return a serializable fixture dict for the named case.

    Cases:
      * ``common_only``: three 1-to-1 common chunks per sample (exercises the
        common-vocab JSD partition KL, no mismatch/prefix-index path).
      * ``with_mismatch``: adds a 1-to-2 and a 2-to-1 chunk (exercises the
        prefix support index + batched mismatch partition KL).
      * ``mismatch_only``: a 1-to-2 chunk at the sequence start (never
        shifted) followed by a shifted 2-to-1 chunk, with no common chunk.
    """
    torch.manual_seed(seed)
    v_s, v_t = 16, 16
    fwd_path, rev_path = _write_subtok_tables(tmpdir, v_s, v_t)
    B = 2

    if name == "common_only":
        S = T = 4
        # Every position carries a common id (0..9).
        student_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)
        teacher_ids = student_ids.clone()
        chunks = [(0, 1, 0, 1), (1, 2, 1, 2), (2, 3, 2, 3)]  # (ss, se, ts, te)
        per_sample_chunks = [chunks, chunks]
    elif name == "with_mismatch":
        S = T = 6
        # chunk 0: 1-to-1 common (id in 0..9). chunk 1: 1-to-2 (M=1,N=2) whose
        # teacher prefix is 8 -> forward key (2,(8,)) has 2 support pairs.
        # chunk 2: 2-to-1 (M=2,N=1) whose student prefix is 9 -> reverse key
        # (2,(9,)) has 2 support pairs. Positions 2 and 5 are gaps.
        student_ids = torch.tensor(
            [[0, 5, 0, 9, 5, 0], [1, 6, 0, 9, 6, 0]], dtype=torch.long
        )
        teacher_ids = torch.tensor(
            [[0, 8, 3, 0, 5, 0], [1, 8, 4, 0, 6, 0]], dtype=torch.long
        )
        chunks = [(0, 1, 0, 1), (1, 2, 1, 3), (3, 5, 4, 5)]
        per_sample_chunks = [chunks, chunks]
    elif name == "mismatch_only":
        S = T = 3
        student_ids = torch.tensor([[5, 9, 5], [6, 9, 6]], dtype=torch.long)
        teacher_ids = torch.tensor([[8, 3, 5], [8, 4, 6]], dtype=torch.long)
        chunks = [(0, 1, 0, 2), (1, 3, 2, 3)]
        per_sample_chunks = [chunks, chunks]
    else:
        raise ValueError(name)

    max_pairs = max(len(c) for c in per_sample_chunks)
    s_spans = torch.zeros(B, max_pairs, 2, dtype=torch.long)
    t_spans = torch.zeros(B, max_pairs, 2, dtype=torch.long)
    pair_valid = torch.zeros(B, max_pairs, dtype=torch.bool)
    num_chunks = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        cs = per_sample_chunks[b]
        num_chunks[b] = len(cs)
        for k, (ss, se, ts, te) in enumerate(cs):
            s_spans[b, k, 0], s_spans[b, k, 1] = ss, se
            t_spans[b, k, 0], t_spans[b, k, 1] = ts, te
            pair_valid[b, k] = True

    student_logits = torch.randn(B, S, v_s, dtype=torch.float32)
    teacher_logits = torch.randn(B, T, v_t, dtype=torch.float32)
    # global_valid_chunks = total valid chunks across the (single-rank) batch.
    gvc = float(sum(len(c) for c in per_sample_chunks))

    return {
        "v_s": v_s,
        "v_t": v_t,
        "student_logits": student_logits,
        "teacher_logits": teacher_logits,
        "student_ids": student_ids,
        "teacher_ids": teacher_ids,
        "s_spans": s_spans,
        "t_spans": t_spans,
        "pair_valid": pair_valid,
        "num_chunks": num_chunks,
        "global_valid_chunks": gvc,
        "knobs": _csd_knobs(fwd_path, rev_path, v_t),
    }


# ----------------------------------------------------------------------------- #
# In-process CSD loss runner.
# ----------------------------------------------------------------------------- #
def _run_csd(
    fx,
    *,
    teacher_sparse_payload=None,
    sample_mask=None,
    global_valid_chunks=None,
    cfg_overrides=None,
):
    from nemo_rl.algorithms.loss.csd import (
        CSDLossFn,
    )
    from nemo_rl.algorithms.x_token.loss_utils import LocalizedAlignment

    v_s, v_t = fx["v_s"], fx["v_t"]
    knobs = fx["knobs"]
    cfg = {
        "temperature": knobs["temperature"],
        "vocab_topk": 8,  # same-vocab path only; unused here (cross-tok teacher)
        "reverse_kl": knobs["reverse_kl"],
        "kl_loss_weight": 1.0,
        "ce_loss_scale": 1.0,
        "dynamic_loss_scaling": False,
        "kd_loss_mode": "sum",
        "normalize_teacher_by_vocab": False,
        "alpha": 1.0,
        "sum_weights_metric": None,
        "student_vocab_size": v_s,
        "teacher_vocab_sizes": [v_t],
        "projection_matrix_paths": [None],
        "teacher_weights": [1.0],
        "common_indices_from_subtoks": knobs["common_indices_from_subtoks"],
        "student_to_teacher_table_paths": [knobs["student_to_teacher_table_path"]],
        "teacher_to_student_table_paths": [knobs["teacher_to_student_table_path"]],
        "kl_chunk_shift": knobs["kl_chunk_shift"],
        "csd_loss_fn": knobs["csd_loss_fn"],
        "csd_last_pos_loss_fn": knobs["csd_last_pos_loss_fn"],
        "csd_jsd_beta": knobs["csd_jsd_beta"],
        "csd_mismatch_loss_beta": knobs["csd_mismatch_loss_beta"],
        "csd_noise_filter_topk": knobs["csd_noise_filter_topk"],
        "teacher_topk_ipc_k": v_t if teacher_sparse_payload is not None else 0,
        "teacher_topk_ipc_support_mode": "row_topk",
        # Full support needs no forced insertion. The probability references
        # below score prefix-support alternatives only, without the realized pair.
        "teacher_topk_ipc_keep_realized": False,
    }
    cfg.update(cfg_overrides or {})
    loss_fn = CSDLossFn(cfg)
    student_logits = fx["student_logits"].clone().requires_grad_(True)
    if sample_mask is None:
        sample_mask = torch.ones(fx["student_ids"].shape[0], dtype=torch.bool)
    if global_valid_chunks is None:
        global_valid_chunks = fx["global_valid_chunks"]
    align = LocalizedAlignment(
        sample_mask=sample_mask,
        pair_valid=fx["pair_valid"],
        student_input_ids=fx["student_ids"],
        teacher_input_ids=fx["teacher_ids"],
        student_spans=fx["s_spans"],
        teacher_spans=fx["t_spans"],
        num_chunks=fx["num_chunks"],
    )
    loss, metrics = loss_fn._compute_csd_loss(
        0,
        student_logits,
        None if teacher_sparse_payload is not None else fx["teacher_logits"].clone(),
        align,
        teacher_sparse_payload=teacher_sparse_payload,
        teacher_vocab_size=v_t,
        global_valid_chunks=torch.as_tensor(global_valid_chunks, dtype=torch.float32),
    )
    loss.backward()
    return loss.detach(), metrics, student_logits.grad.detach()


def _full_vocab_payload(fx):
    """Sparse payload at K = V: every teacher id with its exact logZ."""
    teacher_logits = fx["teacher_logits"].clone()
    token_ids = torch.arange(fx["v_t"], dtype=torch.int32).view(1, 1, -1)
    token_ids = token_ids.expand(*teacher_logits.shape).clone()
    log_z = torch.logsumexp(teacher_logits / fx["knobs"]["temperature"], dim=-1)
    return (teacher_logits, token_ids, log_z, None)


def _realized_pair_not_common(fx):
    """1-to-1 chunks whose realized pair is not a common pair.

    In sample 0, chunk 1 realizes (5, 7), two different common pairs, and
    chunk 2 realizes (12, 3), a non-common student with a common teacher.
    """
    fx = dict(fx)
    fx["student_ids"] = torch.tensor([[0, 5, 12, 3], [4, 5, 6, 7]])
    fx["teacher_ids"] = torch.tensor([[0, 7, 3, 3], [4, 5, 6, 7]])
    return fx


def _empty_teacher_prefix(fx):
    """The 1-to-2 chunk's teacher prefix (7,) has no forward-table pairs."""
    fx = dict(fx)
    fx["teacher_ids"] = fx["teacher_ids"].clone()
    fx["teacher_ids"][:, 1] = 7
    return fx


def _chain_longer_than_table(fx):
    """A 1-to-3 chunk longer than the forward table's longest chain (2)."""
    fx = dict(fx)
    generator = torch.Generator().manual_seed(5)
    fx["student_ids"] = torch.tensor([[5, 9, 5], [6, 9, 6]])
    fx["teacher_ids"] = torch.tensor([[8, 3, 1, 5], [8, 4, 2, 6]])
    fx["teacher_logits"] = torch.randn(2, 4, fx["v_t"], generator=generator)
    fx["s_spans"] = torch.tensor([[[0, 1], [1, 3]]] * 2)
    fx["t_spans"] = torch.tensor([[[0, 3], [3, 4]]] * 2)
    return fx


_FULL_SUPPORT_CASES = {
    "with_mismatch": ("with_mismatch", None),
    "mismatch_only": ("mismatch_only", None),
    "realized_pair_not_common": ("common_only", _realized_pair_not_common),
    "empty_teacher_prefix": ("with_mismatch", _empty_teacher_prefix),
    "chain_longer_than_table": ("mismatch_only", _chain_longer_than_table),
}


@pytest.mark.parametrize("keep_realized", [False, True])
@pytest.mark.parametrize("case", sorted(_FULL_SUPPORT_CASES))
def test_csd_sparse_full_support_matches_dense(case, keep_realized):
    """Dense K=0 and sparse K=V optimize the same loss for either keep setting."""
    base, mutate = _FULL_SUPPORT_CASES[case]
    overrides = {"teacher_topk_ipc_keep_realized": keep_realized}
    with tempfile.TemporaryDirectory() as tmp:
        fx = _build_case(base, tmp)
        if mutate is not None:
            fx = mutate(fx)
        dense_loss, dense_metrics, dense_grad = _run_csd(fx, cfg_overrides=overrides)
        sparse_loss, sparse_metrics, sparse_grad = _run_csd(
            fx,
            teacher_sparse_payload=_full_vocab_payload(fx),
            cfg_overrides=overrides,
        )

    torch.testing.assert_close(sparse_loss, dense_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sparse_grad, dense_grad, rtol=1e-5, atol=1e-6)
    assert sparse_metrics["num_common_chunks"] == dense_metrics["num_common_chunks"]
    assert sparse_metrics["num_mismatch_chunks"] == dense_metrics["num_mismatch_chunks"]


def test_csd_dense_keep_realized_scores_chunks_without_table_support():
    """Dense K=0 scores a chunk without table alternatives only when kept."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = _empty_teacher_prefix(_build_case("with_mismatch", tmp))
        _, skip_metrics, _ = _run_csd(
            fx, cfg_overrides={"teacher_topk_ipc_keep_realized": False}
        )
        _, keep_metrics, _ = _run_csd(
            fx, cfg_overrides={"teacher_topk_ipc_keep_realized": True}
        )

    assert skip_metrics["num_mismatch_chunks"] == 2
    assert keep_metrics["num_mismatch_chunks"] == 4


def test_csd_dense_teacher_stays_cp_local_and_slices_padded_vocab(monkeypatch):
    """Dense CSD never sequence-gathers teacher logits and excludes padded cols."""
    from nemo_rl.algorithms.loss import csd as loss_functions_module

    with tempfile.TemporaryDirectory() as tmp:
        fx = _build_case("with_mismatch", tmp)
        baseline_loss, baseline_metrics, baseline_grad = _run_csd(fx)

        padded_fx = dict(fx)
        padded_fx["teacher_logits"] = torch.cat(
            [
                fx["teacher_logits"],
                torch.full(
                    (*fx["teacher_logits"].shape[:-1], 3),
                    1.0e4,
                    dtype=fx["teacher_logits"].dtype,
                ),
            ],
            dim=-1,
        )
        original_gather = loss_functions_module.allgather_cp_contiguous_tensor

        def reject_dense_teacher_gather(tensor, group, seq_dim=1):
            if (
                tensor.ndim == 3
                and not tensor.requires_grad
                and tensor.shape[-1] >= fx["v_t"]
            ):
                raise AssertionError(
                    "dense teacher logits entered a full-sequence CP gather"
                )
            return original_gather(tensor, group, seq_dim)

        monkeypatch.setattr(
            loss_functions_module,
            "allgather_cp_contiguous_tensor",
            reject_dense_teacher_gather,
        )
        padded_loss, padded_metrics, padded_grad = _run_csd(padded_fx)

    torch.testing.assert_close(padded_loss, baseline_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(padded_grad, baseline_grad, rtol=1e-5, atol=1e-6)
    for key in (
        "kl_common_per_chunk",
        "kl_partition_last_per_chunk",
        "kl_mismatch_scaled_per_chunk",
        "top1_acc_per_chunk",
        "num_common_chunks",
        "num_mismatch_chunks",
    ):
        assert padded_metrics[key] == pytest.approx(baseline_metrics[key])


def test_dense_teacher_lookup_broadcasts_repeated_out_of_order_queries():
    """The CP1 lookup seam preserves arbitrary advanced-index semantics."""
    from nemo_rl.algorithms.loss.csd import (
        CSDLossFn,
    )

    teacher = torch.tensor(
        [
            [[-4.0, 0.0, 2.0], [3.0, -2.0, 1.0]],
            [[7.0, 5.0, -1.0], [0.0, -6.0, 8.0]],
        ]
    )
    batch = torch.tensor([[1], [0], [1]])
    positions = torch.tensor([[1], [0], [1]])
    token_ids = torch.tensor([[2, 0, 2], [1, 2, 1], [0, 1, 0]])
    actual = CSDLossFn._lookup_cp_sharded_teacher_logits(
        teacher,
        batch,
        positions,
        token_ids,
        cp_group=None,
    )
    b_full, p_full, t_full = torch.broadcast_tensors(batch, positions, token_ids)
    expected = teacher[b_full, p_full, t_full]
    torch.testing.assert_close(actual, expected)


def test_student_lookup_tp1_cp1_preserves_repeated_query_gradients():
    """The owner-lookup fast path matches advanced indexing and its gradient."""
    from nemo_rl.algorithms.loss.csd import (
        CSDLossFn,
    )

    student = torch.randn(2, 3, 5, requires_grad=True)
    reference = student.detach().clone().requires_grad_(True)
    batch = torch.tensor([[1], [0], [1]])
    positions = torch.tensor([[2], [0], [2]])
    token_ids = torch.tensor([[4, 0, 4], [1, 3, 1], [2, 2, 2]])
    weights = torch.arange(1, 10, dtype=student.dtype).view(3, 3)

    actual = CSDLossFn._lookup_tp_cp_sharded_student_logits(
        student,
        batch,
        positions,
        token_ids,
        tp_group=None,
        cp_group=None,
    )
    b_full, p_full, t_full = torch.broadcast_tensors(batch, positions, token_ids)
    expected = reference[b_full, p_full, t_full]
    torch.testing.assert_close(actual, expected)

    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(student.grad, reference.grad)


def test_student_lookup_tp2_cp2_preserves_outer_product_gradients(monkeypatch):
    """Compact broadcast queries preserve owner masking and local gradients."""
    from nemo_rl.algorithms.loss import csd as loss_functions_module

    tp_group = object()
    cp_group = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        torch.distributed,
        "get_rank",
        lambda group: 1 if group is tp_group else 0,
    )
    collective_calls = []

    def tp_reduce(values, group):
        collective_calls.append(("tp", group))
        return values

    def cp_reduce(values, group):
        collective_calls.append(("cp", group))
        return values

    monkeypatch.setattr(
        loss_functions_module,
        "group_all_reduce_sum_with_grad",
        tp_reduce,
    )
    monkeypatch.setattr(
        loss_functions_module,
        "group_all_reduce_sum_with_grad_backward_sum",
        cp_reduce,
    )

    student = torch.randn(2, 3, 5, requires_grad=True)
    reference = student.detach().clone().requires_grad_(True)
    batch = torch.tensor([[1], [0], [1]])
    positions = torch.tensor([[0], [4], [2]])
    token_ids = torch.tensor([[0, 6, 9, 4]])
    weights = torch.arange(1, 13, dtype=student.dtype).view(3, 4)

    actual = loss_functions_module.CSDLossFn._lookup_tp_cp_sharded_student_logits(
        student,
        batch,
        positions,
        token_ids,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    expected = reference[
        batch,
        positions.remainder(reference.shape[1]),
        token_ids.remainder(reference.shape[2]),
    ]
    expected = torch.where(positions // reference.shape[1] == 0, expected, 0.0)
    expected = torch.where(token_ids // reference.shape[2] == 1, expected, 0.0)

    torch.testing.assert_close(actual, expected)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(student.grad, reference.grad)
    assert collective_calls == [("tp", tp_group), ("cp", cp_group)]


def test_student_lookup_dense_common_query_saves_compact_indices(monkeypatch):
    """The qualification-sized owner lookup retains no dense index storage."""
    from nemo_rl.algorithms.loss import csd as loss_functions_module

    tp_group = object()
    cp_group = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    collective_calls = []

    def tp_reduce(values, group):
        collective_calls.append(("tp", group))
        return values

    def cp_reduce(values, group):
        collective_calls.append(("cp", group))
        return values

    monkeypatch.setattr(
        loss_functions_module,
        "group_all_reduce_sum_with_grad",
        tp_reduce,
    )
    monkeypatch.setattr(
        loss_functions_module,
        "group_all_reduce_sum_with_grad_backward_sum",
        cp_reduce,
    )

    row_count = 64
    common_width = 109_567
    student = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
    student.requires_grad_(True)
    batch = torch.zeros((row_count, 1), dtype=torch.long)
    # Rank CP0/TP0 owns none of these queries. It must still execute both
    # collectives, and the autograd graph must retain only the compact axes.
    positions = torch.full((row_count, 1), 2, dtype=torch.long)
    token_ids = (torch.arange(common_width) % 4 + 4).view(1, common_width)
    saved_storage = []

    def save_tensor(tensor):
        if tensor.dtype in (torch.long, torch.bool):
            saved_storage.append((tensor.dtype, tensor.untyped_storage().nbytes()))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save_tensor, lambda tensor: tensor):
        actual = loss_functions_module.CSDLossFn._lookup_tp_cp_sharded_student_logits(
            student,
            batch,
            positions,
            token_ids,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        actual.sum().backward()

    assert actual.shape == (row_count, common_width)
    assert torch.count_nonzero(actual) == 0
    assert torch.count_nonzero(student.grad) == 0
    assert collective_calls == [("tp", tp_group), ("cp", cp_group)]
    long_storage = sorted(size for dtype, size in saved_storage if dtype == torch.long)
    bool_storage = sorted(size for dtype, size in saved_storage if dtype == torch.bool)
    assert long_storage == [row_count * 8, row_count * 8, common_width * 8]
    assert bool_storage == [row_count, common_width]


@pytest.mark.parametrize(
    ("batch", "position", "token", "match"),
    [
        (torch.tensor([-1]), torch.tensor([0]), torch.tensor([0]), "batch"),
        (torch.tensor([0]), torch.tensor([3]), torch.tensor([0]), "sequence"),
        (torch.tensor([0]), torch.tensor([0]), torch.tensor([5]), "vocabulary"),
    ],
)
def test_student_lookup_rejects_queries_before_collectives(
    batch: torch.Tensor,
    position: torch.Tensor,
    token: torch.Tensor,
    match: str,
):
    """Malformed globally replicated owner queries fail before communication."""
    from nemo_rl.algorithms.loss.csd import (
        CSDLossFn,
    )

    with pytest.raises(IndexError, match=match):
        CSDLossFn._lookup_tp_cp_sharded_student_logits(
            torch.zeros(1, 3, 5),
            batch,
            position,
            token,
            tp_group=None,
            cp_group=None,
        )


def test_student_lookup_rejects_nonbroadcastable_queries_before_collectives(
    monkeypatch,
):
    """Incompatible compact query axes fail before either collective."""
    from nemo_rl.algorithms.loss import csd as loss_functions_module

    collective_calls = []

    def unexpected_collective(values, group):
        collective_calls.append(group)
        return values

    monkeypatch.setattr(
        loss_functions_module,
        "group_all_reduce_sum_with_grad",
        unexpected_collective,
    )
    monkeypatch.setattr(
        loss_functions_module,
        "group_all_reduce_sum_with_grad_backward_sum",
        unexpected_collective,
    )

    with pytest.raises(RuntimeError):
        loss_functions_module.CSDLossFn._lookup_tp_cp_sharded_student_logits(
            torch.zeros(1, 3, 5),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(3, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            tp_group=object(),
            cp_group=object(),
        )

    assert collective_calls == []


def test_csd_student_logits_never_enter_rank3_gathers(monkeypatch):
    """The CSD student path gathers only scalar log-normalizers across CP."""
    from nemo_rl.algorithms.loss import csd as loss_functions_module

    with tempfile.TemporaryDirectory() as tmp:
        fx = _build_case("with_mismatch", tmp)
        baseline_loss, baseline_metrics, baseline_grad = _run_csd(fx)
        original_gather = loss_functions_module.allgather_cp_contiguous_tensor

        def reject_rank3_gather(tensor, group, seq_dim=1):
            if tensor.ndim == 3:
                raise AssertionError("CSD attempted a rank-3 CP gather")
            return original_gather(tensor, group, seq_dim)

        monkeypatch.setattr(
            loss_functions_module,
            "allgather_cp_contiguous_tensor",
            reject_rank3_gather,
        )
        loss, metrics, grad = _run_csd(fx)

    torch.testing.assert_close(loss, baseline_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad, baseline_grad, rtol=1e-5, atol=1e-6)
    for key in (
        "kl_common_per_chunk",
        "kl_partition_last_per_chunk",
        "top1_acc_per_chunk",
        "num_common_chunks",
        "num_mismatch_chunks",
    ):
        assert metrics[key] == pytest.approx(baseline_metrics[key])


def _single_sample_case(fx, sample_idx):
    """Slice the batched fixture while retaining its on-disk CSD tables/config."""
    single = dict(fx)
    for key in (
        "student_logits",
        "teacher_logits",
        "student_ids",
        "teacher_ids",
        "s_spans",
        "t_spans",
        "pair_valid",
        "num_chunks",
    ):
        single[key] = fx[key][sample_idx : sample_idx + 1].clone()
    single["global_valid_chunks"] = float(
        single["pair_valid"][0, : single["num_chunks"][0]].sum().item()
    )
    return single


@pytest.mark.parametrize("case_name", ["common_only", "with_mismatch"])
def test_csd_sample_mask_matches_single_sample_and_zeroes_masked_grad(case_name):
    """Masked rows affect neither CSD cohorts nor forward/backward numerators."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = _build_case(case_name, tmp)
        single = _single_sample_case(fx, 0)
        single_loss, single_metrics, single_grad = _run_csd(single)

        sample_mask = torch.tensor([True, False])
        masked_loss, masked_metrics, masked_grad = _run_csd(
            fx,
            sample_mask=sample_mask,
            global_valid_chunks=single["global_valid_chunks"],
        )

        # Corrupt every logit in the masked row. It must not change selected
        # cohorts, scalar numerators, or the valid row's gradient.
        corrupted = dict(fx)
        corrupted["student_logits"] = fx["student_logits"].clone()
        corrupted["teacher_logits"] = fx["teacher_logits"].clone()
        corrupted["student_logits"][1] = torch.linspace(
            -50.0,
            50.0,
            corrupted["student_logits"][1].numel(),
        ).reshape_as(corrupted["student_logits"][1])
        corrupted["teacher_logits"][1] = torch.linspace(
            75.0,
            -75.0,
            corrupted["teacher_logits"][1].numel(),
        ).reshape_as(corrupted["teacher_logits"][1])
        corrupt_loss, corrupt_metrics, corrupt_grad = _run_csd(
            corrupted,
            sample_mask=sample_mask,
            global_valid_chunks=single["global_valid_chunks"],
        )

    torch.testing.assert_close(masked_loss, single_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(masked_grad[0], single_grad[0], rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(masked_grad[1]).item() == 0
    torch.testing.assert_close(corrupt_loss, masked_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(corrupt_grad, masked_grad, rtol=1e-5, atol=1e-6)

    for key in (
        "kl_common_per_chunk",
        "kl_partition_last_per_chunk",
        "top1_acc_per_chunk",
        "num_common_chunks",
        "num_mismatch_chunks",
        "num_noise_filtered_common_chunks",
        "num_noise_filtered_mismatch_chunks",
        "num_valid_samples",
    ):
        assert masked_metrics[key] == pytest.approx(single_metrics[key])
        assert corrupt_metrics[key] == pytest.approx(masked_metrics[key])


@pytest.mark.parametrize("case_name", ["common_only", "with_mismatch"])
def test_csd_all_samples_masked_is_zero_safe(case_name):
    """An empty CSD cohort returns connected zero with an exactly zero gradient."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = _build_case(case_name, tmp)
        loss, metrics, grad = _run_csd(
            fx,
            sample_mask=torch.zeros(2, dtype=torch.bool),
            global_valid_chunks=0.0,
        )

    assert torch.isfinite(loss)
    assert torch.equal(loss, torch.zeros_like(loss))
    assert torch.count_nonzero(grad).item() == 0
    for key in (
        "kl_common_per_chunk",
        "kl_partition_last_per_chunk",
        "top1_acc_per_chunk",
        "num_common_chunks",
        "num_mismatch_chunks",
        "num_noise_filtered_common_chunks",
        "num_noise_filtered_mismatch_chunks",
        "num_valid_samples",
    ):
        assert metrics[key] == pytest.approx(0.0)


def test_csd_common_partition_matches_probability_reference(tmp_path):
    """The common-token objective and gradient match an independent JSD equation."""
    fx = _build_case("common_only", str(tmp_path))
    loss, metrics, grad = _run_csd(fx)
    student = fx["student_logits"].clone().requires_grad_(True)
    # Shifted predictor positions for the three fixture chunks are 0, 0, 1.
    positions = torch.tensor([0, 0, 1])
    student_prob = student.softmax(-1)[:, positions, :]
    teacher_prob = fx["teacher_logits"].softmax(-1)[:, positions, :]
    student_partition = torch.cat(
        [student_prob[..., :10], student_prob[..., 10:].sum(-1, keepdim=True)],
        dim=-1,
    )
    teacher_partition = torch.cat(
        [teacher_prob[..., :10], teacher_prob[..., 10:].sum(-1, keepdim=True)],
        dim=-1,
    )
    mixture = (student_partition + teacher_partition) / 2
    reference = (
        (
            0.5 * student_partition * (student_partition.log() - mixture.log())
            + 0.5 * teacher_partition * (teacher_partition.log() - mixture.log())
        )
        .sum(-1)
        .mean()
    )
    reference.backward()
    torch.testing.assert_close(loss, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad, student.grad, rtol=1e-5, atol=1e-6)
    assert metrics["num_common_chunks"] == 6
    assert metrics["num_mismatch_chunks"] == 0


def test_csd_mismatch_partition_aggregation(tmp_path):
    """Forward/reverse mismatch cohorts keep their beta weight and global counts."""
    fx = _build_case("with_mismatch", str(tmp_path))
    loss, metrics, grad = _run_csd(fx)
    assert metrics["num_common_chunks"] == 2
    assert metrics["num_mismatch_chunks"] == 4
    weighted_mismatch = 2 * metrics["kl_partition_last_per_chunk"]
    assert metrics["kl_mismatch_scaled_per_chunk"] == pytest.approx(weighted_mismatch)
    expected = (2 * metrics["kl_common_per_chunk"] + 4 * weighted_mismatch) / fx[
        "global_valid_chunks"
    ]
    assert loss.item() == pytest.approx(expected)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0


_REALIZED_ONLY = {"csd_realized_only_mismatch": True}


def _realized_chain_reference(fx, student_logits):
    """Independent realized-chain-vs-REST KL for every mismatch chunk."""
    knobs = fx["knobs"]
    temperature = knobs["temperature"]
    student_logp = torch.log_softmax(student_logits / temperature, dim=-1)
    teacher_logp = torch.log_softmax(fx["teacher_logits"] / temperature, dim=-1)

    def positions(start, end, shift):
        if shift:
            return list(range(start - 1, end - 1)), list(range(start, end))
        return list(range(start, end)), list(range(start, end))

    def chain(logp, ids, b, preds, labels):
        return sum(logp[b, p, ids[b, lab]] for p, lab in zip(preds, labels))

    def with_rest(chain_logp):
        rest = (1.0 - chain_logp.exp()).clamp_min(1e-12).log()
        return torch.stack([chain_logp, rest])

    total = torch.zeros(())
    for b in range(fx["student_ids"].shape[0]):
        for k in range(int(fx["num_chunks"][b])):
            ss, se = fx["s_spans"][b, k].tolist()
            ts, te = fx["t_spans"][b, k].tolist()
            shift = knobs["kl_chunk_shift"] and ss > 0 and ts > 0
            s_pred, s_lab = positions(ss, se, shift)
            t_pred, t_lab = positions(ts, te, shift)
            student = with_rest(
                chain(student_logp, fx["student_ids"], b, s_pred, s_lab)
            )
            teacher = with_rest(
                chain(teacher_logp, fx["teacher_ids"], b, t_pred, t_lab)
            )
            total = total + torch.nn.functional.kl_div(
                student, teacher, reduction="sum", log_target=True
            )
    beta = knobs["csd_mismatch_loss_beta"]
    return beta * total * temperature**2 / fx["global_valid_chunks"]


def _realized_rows_payload(fx, k):
    """Sparse teacher rows holding each position's realized labels plus fillers."""
    teacher_logits = fx["teacher_logits"]
    batch, seq_len, vocab = teacher_logits.shape
    required = [[set() for _ in range(seq_len)] for _ in range(batch)]
    for b in range(batch):
        for c in range(int(fx["num_chunks"][b])):
            ss = int(fx["s_spans"][b, c, 0])
            ts, te = fx["t_spans"][b, c].tolist()
            shift = fx["knobs"]["kl_chunk_shift"] and ss > 0 and ts > 0
            for label in range(ts, te):
                pred = label - 1 if shift else label
                required[b][pred].add(int(fx["teacher_ids"][b, label]))
    ids = torch.empty(batch, seq_len, k, dtype=torch.long)
    for b in range(batch):
        for pos in range(seq_len):
            row = set(required[b][pos])
            filler = (t for t in range(vocab) if t not in row)
            while len(row) < k:
                row.add(next(filler))
            ids[b, pos] = torch.tensor(sorted(row))
    temperature = fx["knobs"]["temperature"]
    return (
        teacher_logits.gather(-1, ids),
        ids.to(torch.int32),
        torch.logsumexp(teacher_logits / temperature, dim=-1),
        None,
    )


def test_csd_realized_only_mismatch_matches_chain_reference(tmp_path):
    """Realized-only mismatch loss and gradient match an independent chain KL."""
    fx = _build_case("mismatch_only", str(tmp_path))
    fx["knobs"]["temperature"] = 1.7
    loss, metrics, grad = _run_csd(fx, cfg_overrides=_REALIZED_ONLY)

    student = fx["student_logits"].clone().requires_grad_(True)
    reference = _realized_chain_reference(fx, student)
    reference.backward()
    torch.testing.assert_close(loss, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad, student.grad, rtol=1e-5, atol=1e-6)
    assert metrics["num_common_chunks"] == 0
    assert metrics["num_mismatch_chunks"] == 4


def test_csd_realized_only_mismatch_sparse_rows_match_dense(tmp_path):
    """Sparse rows holding only the realized labels reproduce the dense loss."""
    fx = _build_case("mismatch_only", str(tmp_path))
    fx["knobs"]["temperature"] = 1.7
    dense_loss, _, dense_grad = _run_csd(fx, cfg_overrides=_REALIZED_ONLY)
    sparse_loss, sparse_metrics, sparse_grad = _run_csd(
        fx,
        teacher_sparse_payload=_realized_rows_payload(fx, k=3),
        cfg_overrides={**_REALIZED_ONLY, "teacher_topk_ipc_k": 3},
    )
    torch.testing.assert_close(sparse_loss, dense_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sparse_grad, dense_grad, rtol=1e-5, atol=1e-6)
    assert sparse_metrics["num_mismatch_chunks"] == 4


def test_csd_realized_only_mismatch_leaves_common_chunks_unchanged(tmp_path):
    """With the realized pair kept, the option changes mismatch scoring only."""
    fx = _build_case("with_mismatch", str(tmp_path))
    realized_overrides = {**_REALIZED_ONLY, "teacher_topk_ipc_keep_realized": True}
    prefix_overrides = {**realized_overrides, "csd_realized_only_mismatch": False}
    _, prefix_metrics, _ = _run_csd(fx, cfg_overrides=prefix_overrides)
    _, realized_metrics, _ = _run_csd(fx, cfg_overrides=realized_overrides)
    assert realized_metrics["num_common_chunks"] == prefix_metrics["num_common_chunks"]
    assert realized_metrics["kl_common_per_chunk"] == pytest.approx(
        prefix_metrics["kl_common_per_chunk"]
    )
    assert (
        realized_metrics["kl_partition_last_per_chunk"]
        != prefix_metrics["kl_partition_last_per_chunk"]
    )


@pytest.mark.parametrize(
    "removed_key",
    [
        "csd_position_0_kl",
        "csd_mismatch_pos0_alpha",
        "csd_mismatch_pos0_weight",
        "csd_mismatch_loss_scale",
    ],
)
def test_csd_rejects_removed_position_0_keys(removed_key):
    """Configs written for the removed position-0 term fail instead of being ignored."""
    from nemo_rl.algorithms.loss.csd import CSDLossConfig

    with pytest.raises(ValueError, match=f"loss_fn.{removed_key} was removed"):
        CSDLossConfig.model_validate({removed_key: 0.0})
