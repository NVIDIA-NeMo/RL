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

# Upstream (mingyu) checkout — sibling of RL/. Its nemo_rl package is the
# parity reference. Overridable for a differently-located checkout.
UPSTREAM_ROOT = os.environ.get(
    "XTOKEN_UPSTREAM_ROOT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/xtoken_nemorl",
)


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
        "kl_chunk_shift": False,
        "prefix_bidir_v3_position_0_kl": True,
        "prefix_bidir_v3_loss_fn": "jsd",
        "prefix_bidir_v3_last_pos_loss_fn": "jsd",
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
def _run_this_branch(fx):
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
        "prefix_bidir_v3_mismatch_loss_beta": knobs["prefix_bidir_v3_mismatch_loss_beta"],
        "prefix_bidir_v3_noise_filter_topk": knobs["prefix_bidir_v3_noise_filter_topk"],
    }
    loss_fn = CrossTokenizerDistillationLossFn(cfg)
    student_logits = fx["student_logits"].clone().requires_grad_(True)
    align = LocalizedAlignment(
        sample_mask=torch.ones(fx["student_ids"].shape[0], dtype=torch.bool),
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
        fx["teacher_logits"].clone(),
        align,
        teacher_vocab_size=v_t,
        global_valid_chunks=torch.tensor(fx["global_valid_chunks"]),
    )
    loss.backward()
    return loss.detach(), metrics, student_logits.grad.detach()


# ----------------------------------------------------------------------------- #
# Upstream (mingyu) subprocess runner — written to disk, run with the upstream
# nemo_rl on sys.path. Reconstructs the same fixture from the serialized file.
# ----------------------------------------------------------------------------- #
_UPSTREAM_RUNNER = '''
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
'''


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
    if not os.path.isdir(UPSTREAM_ROOT):
        pytest.skip(f"upstream checkout not found at {UPSTREAM_ROOT}")
    with tempfile.TemporaryDirectory() as tmp:
        fx = _build_case(case_name, tmp)
        this_loss, this_metrics, this_grad = _run_this_branch(fx)
        up_loss, up_metrics, up_grad = _run_upstream(fx, tmp)

    torch.testing.assert_close(this_loss, up_loss, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(this_grad, up_grad, rtol=1e-4, atol=1e-4)
    for key in ("kl_common_per_chunk", "kl_partition_last_per_chunk", "top1_acc_per_chunk"):
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


if __name__ == "__main__":
    # Direct runner (in-container): ensure this RL checkout's nemo_rl wins over
    # the container's baked copy / venv site-package.
    _RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
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
