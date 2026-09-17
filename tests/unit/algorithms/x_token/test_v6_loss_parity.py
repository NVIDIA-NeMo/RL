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
"""Cross-repo parity for the v6 prefix-bidir partition KL loss.

Compares this branch's ``CrossTokenizerDistillationLossFn._compute_prefix_bidir_partition_kl_v3``
against the upstream (mingyu) implementation on an identical CPU fixture. The
math is a byte-exact port; only the I/O contract differs (this branch reads
per-teacher args + a ``LocalizedAlignment``, upstream reads a single-teacher
``cfg`` + a padded ``data`` dict). So the scalar loss, the metrics, and the
student-logit gradient must match to fp32 precision.

The upstream loss is run in a **subprocess** whose ``sys.path`` is prepended
with the upstream checkout — both repos are named ``nemo_rl`` and cannot be
co-imported. The subprocess asserts ``nemo_rl.__file__`` resolves under the
upstream root before running.

Run in-container via the direct runner (plain pytest trips the session-autouse
``init_ray`` fixture); this module exposes ``__main__`` for that.
"""

import os
import subprocess
import sys
import tempfile

import pytest
import torch

# The external reference checkout is intentionally opt-in so this test remains
# portable. Its nemo_rl package is loaded in a subprocess to avoid co-importing
# two repositories with the same package name.
UPSTREAM_ROOT = os.environ.get("XTOKEN_UPSTREAM_ROOT", "")


# ----------------------------------------------------------------------------- #
# Fixture builders (pure torch — no nemo_rl import, so the upstream subprocess
# reconstructs the same fixture from the serialized file).
# ----------------------------------------------------------------------------- #
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


def _v6_knobs(fwd_path, rev_path, v_t):
    """The v6 preset knobs shared by both sides (global math + per-teacher paths)."""
    return {
        "temperature": 1.0,
        "teacher_vocab_size": v_t,
        "common_indices_from_subtoks": True,
        "pseudo_target_path": fwd_path,
        "reverse_pseudo_target_path": rev_path,
        "kl_chunk_shift": True,
        "prefix_bidir_v3_position_0_kl": True,
        "prefix_bidir_v3_loss_fn": "jsd",
        "prefix_bidir_v3_last_pos_loss_fn": "kl",
        "prefix_bidir_v3_jsd_beta": 0.5,
        "prefix_bidir_v3_mismatch_pos0_alpha": 0.2,
        "prefix_bidir_v3_mismatch_loss_beta": 2.0,
        "prefix_bidir_v3_noise_filter_topk": 0,
        "reverse_kl": False,
    }


def _build_case(name, tmpdir, seed=0):
    """Return a serializable fixture dict for the named case.

    Cases:
      * ``common_only``: three 1-to-1 common chunks per sample (exercises the
        common-vocab JSD partition KL, no mismatch/prefix-index path).
      * ``with_mismatch``: adds a 1-to-2 and a 2-to-1 chunk (exercises the
        prefix support index + batched mismatch + position-0 KL).
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
        "knobs": _v6_knobs(fwd_path, rev_path, v_t),
    }


# ----------------------------------------------------------------------------- #
# This-branch runner (in-process).
# ----------------------------------------------------------------------------- #
def _run_this_branch(
    fx,
    *,
    teacher_sparse_payload=None,
    sample_mask=None,
    global_valid_chunks=None,
):
    from nemo_rl.algorithms.loss.loss_functions import (
        CrossTokenizerDistillationLossFn,
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
        "projection_matrix_paths": ["dummy_proj.pt"],
        "teacher_weights": [1.0],
        "common_indices_from_subtoks": knobs["common_indices_from_subtoks"],
        "pseudo_target_paths": [knobs["pseudo_target_path"]],
        "reverse_pseudo_target_paths": [knobs["reverse_pseudo_target_path"]],
        "kl_chunk_shift": knobs["kl_chunk_shift"],
        "prefix_bidir_v3_position_0_kl": knobs["prefix_bidir_v3_position_0_kl"],
        "prefix_bidir_v3_loss_fn": knobs["prefix_bidir_v3_loss_fn"],
        "prefix_bidir_v3_last_pos_loss_fn": knobs["prefix_bidir_v3_last_pos_loss_fn"],
        "prefix_bidir_v3_jsd_beta": knobs["prefix_bidir_v3_jsd_beta"],
        "prefix_bidir_v3_mismatch_pos0_alpha": knobs[
            "prefix_bidir_v3_mismatch_pos0_alpha"
        ],
        "prefix_bidir_v3_mismatch_loss_beta": knobs[
            "prefix_bidir_v3_mismatch_loss_beta"
        ],
        "prefix_bidir_v3_noise_filter_topk": knobs["prefix_bidir_v3_noise_filter_topk"],
        "teacher_topk_ipc_k": v_t if teacher_sparse_payload is not None else 0,
        "teacher_topk_ipc_support_mode": "row_topk",
        # Full support needs no forced insertion. Keeping this false also makes
        # dense and sparse mismatch eligibility identical for this regression.
        "teacher_topk_ipc_keep_realized": False,
    }
    loss_fn = CrossTokenizerDistillationLossFn(cfg)
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
    loss, metrics = loss_fn._compute_prefix_bidir_partition_kl_v3(
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


# ----------------------------------------------------------------------------- #
# Upstream (mingyu) subprocess runner — written to disk, run with the upstream
# nemo_rl on sys.path. Reconstructs the same fixture from the serialized file.
# ----------------------------------------------------------------------------- #
_UPSTREAM_RUNNER = """
import os, sys
UPSTREAM_ROOT = sys.argv[1]
FIXTURE = sys.argv[2]
OUT = sys.argv[3]
sys.path.insert(0, UPSTREAM_ROOT)
os.environ.setdefault("NRL_IGNORE_VERSION_MISMATCH", "1")
import torch
import nemo_rl
assert nemo_rl.__file__.startswith(UPSTREAM_ROOT), (
    f"upstream nemo_rl shadowed: {nemo_rl.__file__}"
)
from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn

fx = torch.load(FIXTURE, weights_only=False)
knobs = fx["knobs"]
v_t = fx["v_t"]
teacher_logits = fx["teacher_logits"].clone()

# Monkeypatch the IPC rebuild to inject the fixture teacher logits (no CUDA).
CrossTokenizerDistillationLossFn._rebuild_teacher_full_logits = staticmethod(
    lambda data: teacher_logits.clone()
)

cfg = {
    "gold_loss": False,
    "xtoken_loss": False,
    "temperature": knobs["temperature"],
    "teacher_vocab_size": v_t,
    "common_indices_from_subtoks": knobs["common_indices_from_subtoks"],
    "projection_matrix_path": "",
    "pseudo_target_path": knobs["pseudo_target_path"],
    "reverse_pseudo_target_path": knobs["reverse_pseudo_target_path"],
    "kl_chunk_shift": knobs["kl_chunk_shift"],
    "prefix_bidir_v3_position_0_kl": knobs["prefix_bidir_v3_position_0_kl"],
    "prefix_bidir_v3_loss_fn": knobs["prefix_bidir_v3_loss_fn"],
    "prefix_bidir_v3_last_pos_loss_fn": knobs["prefix_bidir_v3_last_pos_loss_fn"],
    "prefix_bidir_v3_jsd_beta": knobs["prefix_bidir_v3_jsd_beta"],
    "prefix_bidir_v3_mismatch_pos0_alpha": knobs["prefix_bidir_v3_mismatch_pos0_alpha"],
    "prefix_bidir_v3_mismatch_loss_beta": knobs["prefix_bidir_v3_mismatch_loss_beta"],
    "prefix_bidir_v3_noise_filter_topk": knobs["prefix_bidir_v3_noise_filter_topk"],
    "reverse_kl": knobs["reverse_kl"],
}
loss_fn = CrossTokenizerDistillationLossFn(cfg)

student_logits = fx["student_logits"].clone().requires_grad_(True)
data = {
    "input_ids": fx["student_ids"],
    "teacher_input_ids": fx["teacher_ids"],
    "teacher_full_logits_ipc": [None] * fx["student_ids"].shape[0],
    "alignment_student_spans": fx["s_spans"],
    "alignment_teacher_spans": fx["t_spans"],
    "alignment_pair_valid": fx["pair_valid"],
    "alignment_num_chunks": fx["num_chunks"],
}
loss, metrics = loss_fn._compute_prefix_bidir_partition_kl_v3(
    student_logits, data, global_valid_chunks=torch.tensor(fx["global_valid_chunks"])
)
loss.backward()
torch.save(
    {"loss": loss.detach(), "metrics": metrics, "grad": student_logits.grad.detach()},
    OUT,
)
print("[upstream] loss=", float(loss))
"""


def _run_upstream(fx, tmpdir):
    fixture_path = os.path.join(tmpdir, "fixture.pt")
    torch.save(fx, fixture_path)
    runner_path = os.path.join(tmpdir, "upstream_runner.py")
    with open(runner_path, "w") as f:
        f.write(_UPSTREAM_RUNNER)
    out_path = os.path.join(tmpdir, "upstream_out.pt")
    proc = subprocess.run(
        [sys.executable, runner_path, UPSTREAM_ROOT, fixture_path, out_path],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"upstream runner failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    out = torch.load(out_path, weights_only=False)
    return out["loss"], out["metrics"], out["grad"]


# ----------------------------------------------------------------------------- #
# Tests
# ----------------------------------------------------------------------------- #
def _assert_parity(case_name):
    if not UPSTREAM_ROOT:
        pytest.skip("set XTOKEN_UPSTREAM_ROOT to run cross-repository parity")
    if not os.path.isdir(UPSTREAM_ROOT):
        pytest.skip(f"upstream checkout not found at {UPSTREAM_ROOT}")
    with tempfile.TemporaryDirectory() as tmp:
        fx = _build_case(case_name, tmp)
        this_loss, this_metrics, this_grad = _run_this_branch(fx)
        up_loss, up_metrics, up_grad = _run_upstream(fx, tmp)

    torch.testing.assert_close(this_loss, up_loss, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(this_grad, up_grad, rtol=1e-4, atol=1e-4)
    for key in (
        "kl_common_per_chunk",
        "kl_partition_first_per_chunk",
        "kl_partition_last_per_chunk",
        "kl_mismatch_combined_per_chunk",
        "kl_mismatch_scaled_per_chunk",
        "top1_acc_per_chunk",
    ):
        assert abs(float(this_metrics[key]) - float(up_metrics[key])) < 1e-4, (
            f"{case_name}: metric {key} differs: "
            f"{this_metrics[key]} vs {up_metrics[key]}"
        )
    assert this_metrics["num_common_chunks"] == up_metrics["num_common_chunks"]
    assert this_metrics["num_mismatch_chunks"] == up_metrics["num_mismatch_chunks"]
    return this_loss, this_metrics


def test_v6_parity_common_only():
    _, metrics = _assert_parity("common_only")
    assert metrics["num_common_chunks"] > 0


def test_v6_parity_with_mismatch():
    _, metrics = _assert_parity("with_mismatch")
    # Guard: the fixture must actually exercise the batched-mismatch +
    # prefix-support path (else this case degenerates to a common-only check).
    assert metrics["num_mismatch_chunks"] > 0, (
        "with_mismatch fixture produced 0 kept mismatch chunks — the "
        "_partition_kl_mismatch_batched path is not being exercised."
    )
    assert metrics["num_common_chunks"] > 0


def test_v6_sparse_full_support_matches_dense():
    """The production sparse consumer matches dense when support is complete."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = _build_case("with_mismatch", tmp)
        dense_loss, dense_metrics, dense_grad = _run_this_branch(fx)

        teacher_logits = fx["teacher_logits"].clone()
        token_ids = torch.arange(fx["v_t"], dtype=torch.int32).view(1, 1, -1)
        token_ids = token_ids.expand(*teacher_logits.shape).clone()
        sparse_payload = (
            teacher_logits,
            token_ids,
            torch.logsumexp(teacher_logits / fx["knobs"]["temperature"], dim=-1),
            None,
        )
        sparse_loss, sparse_metrics, sparse_grad = _run_this_branch(
            fx,
            teacher_sparse_payload=sparse_payload,
        )

    torch.testing.assert_close(sparse_loss, dense_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sparse_grad, dense_grad, rtol=1e-5, atol=1e-6)
    assert sparse_metrics["num_common_chunks"] == dense_metrics["num_common_chunks"]
    assert sparse_metrics["num_mismatch_chunks"] == dense_metrics["num_mismatch_chunks"]


def test_v6_dense_teacher_stays_cp_local_and_slices_padded_vocab(monkeypatch):
    """Dense v6 never sequence-gathers teacher logits and excludes padded cols."""
    from nemo_rl.algorithms.loss import loss_functions as loss_functions_module

    with tempfile.TemporaryDirectory() as tmp:
        fx = _build_case("with_mismatch", tmp)
        baseline_loss, baseline_metrics, baseline_grad = _run_this_branch(fx)

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
        padded_loss, padded_metrics, padded_grad = _run_this_branch(padded_fx)

    torch.testing.assert_close(padded_loss, baseline_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(padded_grad, baseline_grad, rtol=1e-5, atol=1e-6)
    for key in (
        "kl_common_per_chunk",
        "kl_partition_first_per_chunk",
        "kl_partition_last_per_chunk",
        "kl_mismatch_combined_per_chunk",
        "kl_mismatch_scaled_per_chunk",
        "top1_acc_per_chunk",
        "num_common_chunks",
        "num_mismatch_chunks",
    ):
        assert padded_metrics[key] == pytest.approx(baseline_metrics[key])


def test_dense_teacher_lookup_broadcasts_repeated_out_of_order_queries():
    """The CP1 lookup seam preserves arbitrary advanced-index semantics."""
    from nemo_rl.algorithms.loss.loss_functions import (
        CrossTokenizerDistillationLossFn,
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
    actual = CrossTokenizerDistillationLossFn._lookup_cp_sharded_teacher_logits(
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
    from nemo_rl.algorithms.loss.loss_functions import (
        CrossTokenizerDistillationLossFn,
    )

    student = torch.randn(2, 3, 5, requires_grad=True)
    reference = student.detach().clone().requires_grad_(True)
    batch = torch.tensor([[1], [0], [1]])
    positions = torch.tensor([[2], [0], [2]])
    token_ids = torch.tensor([[4, 0, 4], [1, 3, 1], [2, 2, 2]])
    weights = torch.arange(1, 10, dtype=student.dtype).view(3, 3)

    actual = CrossTokenizerDistillationLossFn._lookup_tp_cp_sharded_student_logits(
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
    from nemo_rl.algorithms.loss import loss_functions as loss_functions_module

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

    actual = loss_functions_module.CrossTokenizerDistillationLossFn._lookup_tp_cp_sharded_student_logits(
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
    from nemo_rl.algorithms.loss import loss_functions as loss_functions_module

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
        actual = loss_functions_module.CrossTokenizerDistillationLossFn._lookup_tp_cp_sharded_student_logits(
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
    from nemo_rl.algorithms.loss.loss_functions import (
        CrossTokenizerDistillationLossFn,
    )

    with pytest.raises(IndexError, match=match):
        CrossTokenizerDistillationLossFn._lookup_tp_cp_sharded_student_logits(
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
    from nemo_rl.algorithms.loss import loss_functions as loss_functions_module

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
        loss_functions_module.CrossTokenizerDistillationLossFn._lookup_tp_cp_sharded_student_logits(
            torch.zeros(1, 3, 5),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(3, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            tp_group=object(),
            cp_group=object(),
        )

    assert collective_calls == []


def test_v6_student_logits_never_enter_rank3_gathers(monkeypatch):
    """The v6 student path gathers only scalar log-normalizers across CP."""
    from nemo_rl.algorithms.loss import loss_functions as loss_functions_module

    with tempfile.TemporaryDirectory() as tmp:
        fx = _build_case("with_mismatch", tmp)
        baseline_loss, baseline_metrics, baseline_grad = _run_this_branch(fx)
        original_gather = loss_functions_module.allgather_cp_contiguous_tensor

        def reject_rank3_gather(tensor, group, seq_dim=1):
            if tensor.ndim == 3:
                raise AssertionError("v6 attempted a rank-3 CP gather")
            return original_gather(tensor, group, seq_dim)

        monkeypatch.setattr(
            loss_functions_module,
            "allgather_cp_contiguous_tensor",
            reject_rank3_gather,
        )
        loss, metrics, grad = _run_this_branch(fx)

    torch.testing.assert_close(loss, baseline_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad, baseline_grad, rtol=1e-5, atol=1e-6)
    for key in (
        "kl_common_per_chunk",
        "kl_partition_first_per_chunk",
        "kl_partition_last_per_chunk",
        "top1_acc_per_chunk",
        "num_common_chunks",
        "num_mismatch_chunks",
    ):
        assert metrics[key] == pytest.approx(baseline_metrics[key])


def _single_sample_case(fx, sample_idx):
    """Slice the batched fixture while retaining its on-disk v6 tables/config."""
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
def test_v6_sample_mask_matches_single_sample_and_zeroes_masked_grad(case_name):
    """Masked rows affect neither v6 cohorts nor forward/backward numerators."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = _build_case(case_name, tmp)
        single = _single_sample_case(fx, 0)
        single_loss, single_metrics, single_grad = _run_this_branch(single)

        sample_mask = torch.tensor([True, False])
        masked_loss, masked_metrics, masked_grad = _run_this_branch(
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
        corrupt_loss, corrupt_metrics, corrupt_grad = _run_this_branch(
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
        "kl_partition_first_per_chunk",
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
def test_v6_all_samples_masked_is_zero_safe(case_name):
    """An empty v6 cohort returns connected zero with an exactly zero gradient."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = _build_case(case_name, tmp)
        loss, metrics, grad = _run_this_branch(
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
        "kl_partition_first_per_chunk",
        "top1_acc_per_chunk",
        "num_common_chunks",
        "num_mismatch_chunks",
        "num_noise_filtered_common_chunks",
        "num_noise_filtered_mismatch_chunks",
        "num_valid_samples",
    ):
        assert metrics[key] == pytest.approx(0.0)


if __name__ == "__main__":
    # Direct runner (in-container): ensure this RL checkout's nemo_rl wins over
    # the container's baked copy / venv site-package.
    _RL_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    sys.path.insert(0, _RL_ROOT)
    import nemo_rl as _nrl  # noqa: E402

    assert _nrl.__file__.startswith(_RL_ROOT), (
        f"nemo_rl resolved outside the RL checkout: {_nrl.__file__}"
    )
    print(f"[parity] nemo_rl: {_nrl.__file__}")
    print(f"[parity] upstream: {UPSTREAM_ROOT}")
    for _case in ("common_only", "with_mismatch"):
        loss, metrics = _assert_parity(_case)
        print(
            f"[parity:{_case}] PASS  loss={float(loss):.6f}  "
            f"common={metrics['num_common_chunks']}  "
            f"mismatch={metrics['num_mismatch_chunks']}"
        )
    print("V6 CROSS-REPO PARITY PASSED")
