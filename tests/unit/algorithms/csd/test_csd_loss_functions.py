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

import pytest
import torch

from nemo_rl.algorithms.loss import prepare_loss_input
from nemo_rl.algorithms.loss.csd import CSDLossFn
from nemo_rl.algorithms.x_token.loss_utils import (
    LocalizedAlignment,
    same_tokenizer_topk_kl,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_CT_V_STUDENT = 6
_CT_V_TEACHER = 5
_CT_TOPK = 2
_CT_TEMPERATURE = 2.0


def test_csd_rebuild_single_teacher_logits_without_copy(monkeypatch):
    """MBS=1 must not duplicate the full teacher-logit IPC tensor."""
    import sys
    from types import SimpleNamespace

    teacher_logits = torch.randn(4, 7, dtype=torch.float32)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setitem(
        sys.modules,
        "nemo_rl.models.policy.utils",
        SimpleNamespace(
            rebuild_cuda_tensor_from_ipc=lambda _handle, _device: teacher_logits
        ),
    )

    rebuilt = CSDLossFn._rebuild_teacher_full_logits(
        {"teacher_full_logits_ipc": [{"logits_ipc": object()}]}
    )

    assert rebuilt.shape == (1, 4, 7)
    assert rebuilt.dtype == torch.float32
    assert rebuilt.data_ptr() == teacher_logits.data_ptr()


def _write_ct_projection(tmp_path):
    """Dense top-k projection with two strict exact maps (s0->t0, s5->t4).

    Strict exact-map rule: slot-0 weight 1.0 and a ``-1`` sentinel in slot 1.
    The four middle rows are fuzzy (0.7/0.3, both slots real) so they fall in
    the uncommon partition. Result: common student {0, 5} / teacher {0, 4};
    uncommon teacher {1, 2, 3}.
    """
    v_s, v_t, k = _CT_V_STUDENT, _CT_V_TEACHER, _CT_TOPK
    indices = torch.full((v_s, k), -1, dtype=torch.long)
    likelihoods = torch.zeros((v_s, k), dtype=torch.float32)
    indices[0, 0], likelihoods[0, 0] = 0, 1.0
    indices[v_s - 1, 0], likelihoods[v_s - 1, 0] = v_t - 1, 1.0
    for s, (t1, t2) in {1: (1, 2), 2: (2, 3), 3: (3, 1), 4: (1, 3)}.items():
        indices[s, 0], indices[s, 1] = t1, t2
        likelihoods[s, 0], likelihoods[s, 1] = 0.7, 0.3
    path = tmp_path / "ct_projection.pt"
    torch.save({"indices": indices, "likelihoods": likelihoods}, path)
    return str(path)


def _ct_loss_cfg(projection_path):
    # Per-teacher metadata as ``setup`` injects it: parallel lists, one entry
    # per teacher (here a single teacher).
    return {
        "temperature": _CT_TEMPERATURE,
        "vocab_topk": 4,
        "reverse_kl": False,
        "kl_loss_weight": 1.0,
        "ce_loss_scale": 1.0,
        "dynamic_loss_scaling": False,
        "kd_loss_mode": "sum",
        "alpha": 1.0,
        "normalize_teacher_by_vocab": False,
        "student_vocab_size": _CT_V_STUDENT,
        "projection_matrix_paths": [projection_path],
        "teacher_vocab_sizes": [_CT_V_TEACHER],
        "teacher_weights": [1.0],
    }


def test_csd_mismatched_per_teacher_lists_raises(tmp_path):
    """Unequal-length per-teacher lists fail loudly at construction."""
    cfg = _ct_loss_cfg(_write_ct_projection(tmp_path))
    # Two weights but a single entry in every other per-teacher list.
    cfg["teacher_weights"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="per-teacher lists must be equal length"):
        CSDLossFn(cfg)


def test_normalize_teacher_by_vocab_rejected_outside_sum_mode(tmp_path):
    """normalize_teacher_by_vocab is a no-op outside sum mode, so reject it there."""
    cfg = _ct_loss_cfg(_write_ct_projection(tmp_path))
    cfg["kd_loss_mode"] = "averaged_logits"
    cfg["normalize_teacher_by_vocab"] = True
    with pytest.raises(ValueError, match="normalize_teacher_by_vocab"):
        CSDLossFn(cfg)


def test_csd_ce_uniform_logits_equals_log_vocab(tmp_path):
    """_compute_ce on uniform logits equals log(V_student) per valid token."""
    loss_fn = CSDLossFn(_ct_loss_cfg(_write_ct_projection(tmp_path)))
    b, t_s = 1, 5
    logits = torch.zeros(b, t_s, _CT_V_STUDENT)  # uniform -> CE == log(V)
    data = BatchedDataDict(
        {
            "input_ids": torch.randint(0, _CT_V_STUDENT, (b, t_s)),
            "token_mask": torch.ones(b, t_s),
            "sample_mask": torch.ones(b),
        }
    )
    gvt = (data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)).sum()
    ce = loss_fn._compute_ce(logits, data, gvt)
    assert torch.allclose(ce, torch.log(torch.tensor(float(_CT_V_STUDENT))), atol=1e-5)


def test_csd_ce_megatron_uses_chunked_raw_shards(tmp_path, monkeypatch):
    """Plain Megatron CE routes raw TP/CP shards without rank-3 gathers."""
    loss_fn = CSDLossFn(_ct_loss_cfg(_write_ct_projection(tmp_path)))
    local_logits = torch.randn(1, 2, 3)
    tp_group = object()
    cp_group = object()
    calls = {}

    def reject_rank3_gather(*_args, **_kwargs):
        raise AssertionError("CE attempted a rank-3 CP gather")

    def fake_get_rank(group):
        assert group is tp_group
        return 1

    def fake_get_world_size(group):
        assert group in (tp_group, cp_group)
        return 2

    def fake_logprobs(**kwargs):
        calls.update(kwargs)
        return torch.full((1, 3), -torch.log(torch.tensor(float(_CT_V_STUDENT))))

    monkeypatch.setattr(
        "nemo_rl.algorithms.loss.csd.allgather_cp_contiguous_tensor",
        reject_rank3_gather,
    )
    monkeypatch.setattr(
        "nemo_rl.algorithms.loss.csd.get_next_token_logprobs_from_logits",
        fake_logprobs,
    )
    monkeypatch.setattr(torch.distributed, "get_rank", fake_get_rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", fake_get_world_size)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    data = BatchedDataDict(
        {
            "input_ids": torch.randint(0, _CT_V_STUDENT, (1, 4)),
            "token_mask": torch.ones(1, 4),
            "sample_mask": torch.ones(1),
        }
    )
    global_valid_toks = data["token_mask"][:, 1:].sum()
    ce = loss_fn._compute_ce(
        local_logits,
        data,
        global_valid_toks,
        student_logits_contig=local_logits,
        tp_group=tp_group,
        cp_group=cp_group,
    )

    assert calls["next_token_logits"] is local_logits
    assert calls["input_ids"] is data["input_ids"]
    assert calls["vocab_parallel_rank"] == 1
    assert calls["vocab_parallel_group"] is tp_group
    assert calls["context_parallel_group"] is cp_group
    assert calls["chunk_size"] == 256
    assert torch.allclose(ce, torch.log(torch.tensor(float(_CT_V_STUDENT))), atol=1e-5)


def test_csd_ce_respects_sample_mask(tmp_path):
    """A masked sample must not contribute to _compute_ce: the B=2 result with
    sample 1 masked equals the CE over sample 0 alone."""
    loss_fn = CSDLossFn(_ct_loss_cfg(_write_ct_projection(tmp_path)))
    b, t_s = 2, 5
    torch.manual_seed(0)
    logits = torch.randn(b, t_s, _CT_V_STUDENT)
    input_ids = torch.randint(0, _CT_V_STUDENT, (b, t_s))
    token_mask = torch.ones(b, t_s)

    data_masked = BatchedDataDict(
        {
            "input_ids": input_ids,
            "token_mask": token_mask,
            "sample_mask": torch.tensor([1.0, 0.0]),
        }
    )
    gvt_masked = (token_mask[:, 1:] * data_masked["sample_mask"].unsqueeze(-1)).sum()
    ce_masked = loss_fn._compute_ce(logits, data_masked, gvt_masked)

    data_single = BatchedDataDict(
        {
            "input_ids": input_ids[:1],
            "token_mask": token_mask[:1],
            "sample_mask": torch.tensor([1.0]),
        }
    )
    gvt_single = (token_mask[:1, 1:] * data_single["sample_mask"].unsqueeze(-1)).sum()
    ce_single = loss_fn._compute_ce(logits[:1], data_single, gvt_single)

    assert torch.allclose(ce_masked, ce_single, atol=1e-6)


def test_csd_prepare_loss_input_partitions_canonical_ce(tmp_path, monkeypatch):
    """Automodel CP assigns one disjoint canonical CE window to each rank."""
    loss_fn = CSDLossFn(_ct_loss_cfg(_write_ct_projection(tmp_path)))
    full_logprobs = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=True)
    token_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    cp_group = object()
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[0, 1, 2, 3]]),
            "token_mask": token_mask,
            "sample_mask": torch.ones(1),
        }
    )

    monkeypatch.setattr(
        "nemo_rl.algorithms.loss.loss_input.prepare_csd_cross_tokenizer_loss_input",
        lambda *args, **kwargs: (torch.empty(0), {}, {}, {}, {}, None, cp_group, None),
    )
    monkeypatch.setattr(
        "nemo_rl.algorithms.loss.loss_input.get_cp_sharded_next_token_logprobs",
        lambda *args, **kwargs: full_logprobs,
    )
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group: 1)

    loss_input, prepared_data = prepare_loss_input(
        torch.empty(0),
        data,
        loss_fn,
        context_parallel_group=cp_group,
        cp_sharder=object(),
    )

    torch.testing.assert_close(
        loss_input["student_next_token_logprobs"], torch.tensor([[-3.0, 0.0]])
    )
    torch.testing.assert_close(
        loss_input["student_next_token_mask"], torch.tensor([[1.0, 0.0]])
    )
    assert prepared_data is data


def test_csd_prepare_loss_input_rejects_nondivisible_cp_window(tmp_path, monkeypatch):
    """Automodel CP rejects canonical CE windows that cannot partition evenly."""
    loss_fn = CSDLossFn(_ct_loss_cfg(_write_ct_projection(tmp_path)))
    next_token_logprobs = torch.tensor(
        [[-1.0, -2.0, -3.0, -4.0, -5.0]], requires_grad=True
    )
    cp_group = object()
    data = BatchedDataDict(
        {
            "input_ids": torch.zeros((1, 6), dtype=torch.long),
            "token_mask": torch.ones((1, 6)),
            "sample_mask": torch.ones(1),
        }
    )

    monkeypatch.setattr(
        "nemo_rl.algorithms.loss.loss_input.prepare_csd_cross_tokenizer_loss_input",
        lambda *args, **kwargs: (torch.empty(0), {}, {}, {}, {}, None, cp_group, None),
    )
    monkeypatch.setattr(
        "nemo_rl.algorithms.loss.loss_input.get_cp_sharded_next_token_logprobs",
        lambda *args, **kwargs: next_token_logprobs,
    )
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 4)

    with pytest.raises(ValueError, match=r"sequence_length=6, cp_size=4"):
        prepare_loss_input(
            torch.empty(0),
            data,
            loss_fn,
            context_parallel_group=cp_group,
            cp_sharder=object(),
        )


def test_csd_precomputed_ce_reduces_partitioned_cp_window(tmp_path, monkeypatch):
    """Precomputed CE sums CP-window values while preserving local gradients."""
    loss_fn = CSDLossFn(_ct_loss_cfg(_write_ct_projection(tmp_path)))
    data = BatchedDataDict({"sample_mask": torch.ones(1)})
    next_token_logprobs = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    next_token_mask = torch.ones_like(next_token_logprobs)
    cp_group = object()
    reduce_calls = []

    def reduce_sum(value, group):
        reduce_calls.append((value.detach().clone(), group))
        return value

    monkeypatch.setattr(
        "nemo_rl.algorithms.loss.csd.group_all_reduce_sum_with_grad",
        reduce_sum,
    )

    ce = loss_fn._compute_ce(
        torch.empty(0),
        data,
        torch.tensor(2.0),
        student_next_token_logprobs=next_token_logprobs,
        student_next_token_mask=next_token_mask,
        cp_group=cp_group,
    )

    torch.testing.assert_close(ce, torch.tensor(1.5))
    ce.backward()
    torch.testing.assert_close(
        next_token_logprobs.grad,
        torch.tensor([[-0.5, -0.5]]),
    )
    assert len(reduce_calls) == 1
    torch.testing.assert_close(reduce_calls[0][0], torch.tensor(1.5))
    assert reduce_calls[0][1] is cp_group


def test_csd_partitioned_cp_ce_matches_cp1_value_and_gradient(tmp_path):
    """Summed CP2 CE windows are numerically identical to the CP1 full CE."""
    loss_fn = CSDLossFn(_ct_loss_cfg(_write_ct_projection(tmp_path)))
    data = BatchedDataDict(
        {
            "token_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
            "sample_mask": torch.tensor([1.0, 0.0]),
        }
    )
    global_valid_toks = torch.tensor(2.0)
    full_mask = data["token_mask"].roll(shifts=-1, dims=1)
    full_mask[:, -1] = 0

    cp1_logprobs = torch.tensor(
        [[-1.0, -2.0, -3.0], [-10.0, -20.0, -30.0]], requires_grad=True
    )
    cp1_loss = loss_fn._compute_ce(
        torch.empty(0),
        data,
        global_valid_toks,
        student_next_token_logprobs=cp1_logprobs,
        student_next_token_mask=full_mask[:, :-1],
        cp_group=None,
    )
    cp1_loss.backward()

    cp2_logprobs = cp1_logprobs.detach().clone().requires_grad_(True)
    cp2_padded = torch.cat([cp2_logprobs, torch.zeros_like(cp2_logprobs[:, :1])], dim=1)
    cp2_local_seq_len = cp2_padded.shape[1] // 2
    cp2_losses = []
    for cp_rank in range(2):
        seq_start = cp_rank * cp2_local_seq_len
        cp2_losses.append(
            loss_fn._compute_ce(
                torch.empty(0),
                data,
                global_valid_toks,
                student_next_token_logprobs=cp2_padded.narrow(
                    1, seq_start, cp2_local_seq_len
                ),
                student_next_token_mask=full_mask.narrow(
                    1, seq_start, cp2_local_seq_len
                ),
                cp_group=None,
            )
        )
    cp2_loss = sum(cp2_losses)
    cp2_loss.backward()

    torch.testing.assert_close(cp2_loss, cp1_loss)
    torch.testing.assert_close(cp2_logprobs.grad, cp1_logprobs.grad)


def test_csd_same_vocab_topk_ignores_masked_and_padding_rows(tmp_path):
    """Invalid teacher rows below the valid column maxima cannot change same-vocab loss or gradient."""
    cfg = _ct_loss_cfg(str(tmp_path / "unused_projection.pt"))
    cfg["projection_matrix_paths"] = [None]
    cfg["teacher_vocab_sizes"] = [_CT_V_STUDENT]
    cfg["vocab_topk"] = 2
    cfg["temperature"] = 1.0
    cfg["ce_loss_scale"] = 0.0
    loss_fn = CSDLossFn(cfg)

    torch.manual_seed(7)
    student_base = torch.randn(2, 4, _CT_V_STUDENT)
    input_ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    token_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    sample_mask = torch.tensor([True, False])
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": torch.tensor([3, 4]),
            "token_mask": token_mask,
            "sample_mask": sample_mask,
        }
    )
    align = LocalizedAlignment(
        sample_mask=sample_mask,
        student_input_ids=input_ids,
        student_token_mask=token_mask,
    )

    teacher_clean = torch.full((2, 4, _CT_V_STUDENT), -10.0)
    teacher_clean[0, 0, 0:2] = torch.tensor([8.0, 7.0])
    teacher_clean[0, 1, 0:2] = torch.tensor([7.0, 8.0])
    teacher_corrupt = teacher_clean.clone()
    # The top-K columns come from the per-vocab max over every teacher row, so
    # the corruption stays below the valid rows' column maxima (8.0) and the
    # selected columns {0, 1} are unchanged. Padding in the valid sample and
    # every predictor in the masked sample still get a different distribution
    # over those columns and over the rest of the vocab.
    teacher_corrupt[0, 2:, 0:4] = torch.tensor([7.5, -5.0, 6.0, 5.0])
    teacher_corrupt[1, :, 0:6] = torch.tensor([-5.0, 7.5, -8.0, -9.0, 6.0, 5.0])

    def run(teacher_logits):
        student_logits = student_base.clone().requires_grad_(True)
        loss, metrics = loss_fn(
            data=data,
            global_valid_seqs=sample_mask.sum(),
            global_valid_toks=torch.tensor(2.0),
            logits=student_logits,
            student_logits_contig=student_logits,
            teacher_full_logits_by_idx={0: teacher_logits},
            aligns_by_idx={0: align},
        )
        loss.backward()
        return loss.detach(), metrics, student_logits.grad.detach()

    clean_loss, clean_metrics, clean_grad = run(teacher_clean)
    corrupt_loss, corrupt_metrics, corrupt_grad = run(teacher_corrupt)

    torch.testing.assert_close(corrupt_loss, clean_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(corrupt_grad, clean_grad, rtol=1e-6, atol=1e-6)
    assert clean_metrics["num_valid_samples"] == 1
    assert corrupt_metrics["num_valid_samples"] == 1
    assert torch.count_nonzero(clean_grad[1]).item() == 0
    assert torch.count_nonzero(clean_grad[0, 2:]).item() == 0


def test_same_vocab_chat_kd_uses_semantic_mask_and_its_own_denominator(tmp_path):
    """Template scaffold stays in CE but cannot enter direct same-vocab KD."""
    cfg = _ct_loss_cfg(str(tmp_path / "unused_projection.pt"))
    cfg["projection_matrix_paths"] = [None]
    cfg["teacher_vocab_sizes"] = [_CT_V_STUDENT]
    cfg["vocab_topk"] = 3
    cfg["temperature"] = 1.0
    cfg["ce_loss_scale"] = 0.0
    loss_fn = CSDLossFn(cfg)

    torch.manual_seed(19)
    student = torch.randn(1, 5, _CT_V_STUDENT, requires_grad=True)
    teacher = torch.randn(1, 5, _CT_V_STUDENT)
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    ce_mask = torch.ones(1, 5, dtype=torch.long)
    # Only labels at positions 3 (content) and 4 (explicit EOT) are KD targets,
    # hence predictors 2 and 3 receive KD gradients.
    kd_mask = torch.tensor([[0, 0, 0, 1, 1]])
    sample_mask = torch.ones(1)
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": torch.tensor([5]),
            "token_mask": ce_mask,
            "kd_token_mask": kd_mask,
            "sample_mask": sample_mask,
        }
    )
    align = LocalizedAlignment(
        sample_mask=sample_mask,
        student_input_ids=input_ids,
        student_token_mask=ce_mask,
        student_kd_token_mask=kd_mask,
    )
    expected_kd = same_tokenizer_topk_kl(
        student,
        teacher,
        kd_mask=kd_mask,
        sample_mask=sample_mask,
        num_kd_tokens=torch.tensor(2.0),
        vocab_size=_CT_V_STUDENT,
        vocab_topk=3,
        temperature=1.0,
        reverse_kl=loss_fn.reverse_kl,
        tp_group=None,
        cp_group=None,
    )
    loss, metrics = loss_fn(
        data=data,
        global_valid_seqs=torch.tensor(1.0),
        # CE has four next-token targets, while semantic KD has two.
        global_valid_toks=torch.tensor(4.0),
        global_valid_kd_toks=torch.tensor(2.0),
        logits=student,
        student_logits_contig=student,
        teacher_full_logits_by_idx={0: teacher},
        aligns_by_idx={0: align},
    )
    torch.testing.assert_close(loss, expected_kd)
    assert metrics["kl_loss"] == pytest.approx(expected_kd.item())
    loss.backward()
    assert torch.count_nonzero(student.grad[0, :2]).item() == 0
    assert torch.count_nonzero(student.grad[0, 2:4]).item() > 0
    assert torch.count_nonzero(student.grad[0, 4]).item() == 0


def _run_kd_ce_combination(tmp_path, cfg_overrides, *, kd_mask=None):
    """Run the full loss on a fixed same-vocab batch; return loss, metrics, grad.

    The KD/CE combination does not depend on how the KD term was computed, so a
    same-vocab teacher keeps the batch small.
    """
    cfg = _ct_loss_cfg(str(tmp_path / "unused_projection.pt"))
    cfg["projection_matrix_paths"] = [None]
    cfg["teacher_vocab_sizes"] = [_CT_V_STUDENT]
    cfg["vocab_topk"] = 3
    cfg["temperature"] = 1.0
    cfg.update(cfg_overrides)
    loss_fn = CSDLossFn(cfg)

    torch.manual_seed(23)
    student = torch.randn(1, 5, _CT_V_STUDENT, requires_grad=True)
    teacher = torch.randn(1, 5, _CT_V_STUDENT)
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    token_mask = torch.ones(1, 5, dtype=torch.long)
    if kd_mask is None:
        kd_mask = token_mask
    sample_mask = torch.ones(1)
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": torch.tensor([5]),
            "token_mask": token_mask,
            "kd_token_mask": kd_mask,
            "sample_mask": sample_mask,
        }
    )
    align = LocalizedAlignment(
        sample_mask=sample_mask,
        student_input_ids=input_ids,
        student_token_mask=token_mask,
        student_kd_token_mask=kd_mask,
    )
    loss, metrics = loss_fn(
        data=data,
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(4.0),
        global_valid_kd_toks=torch.tensor(4.0),
        logits=student,
        student_logits_contig=student,
        teacher_full_logits_by_idx={0: teacher},
        aligns_by_idx={0: align},
    )
    loss.backward()
    return loss.detach(), metrics, student.grad.detach()


_KD_ONLY = {"dynamic_loss_scaling": False, "kl_loss_weight": 1.0, "ce_loss_scale": 0.0}
_CE_ONLY = {"dynamic_loss_scaling": False, "kl_loss_weight": 0.0, "ce_loss_scale": 1.0}


def test_csd_fixed_weighting_scales_kd_and_ce(tmp_path):
    """Fixed weighting is kl_loss_weight * KD + ce_loss_scale * CE in value and gradient."""
    _, _, kd_grad = _run_kd_ce_combination(tmp_path, _KD_ONLY)
    _, _, ce_grad = _run_kd_ce_combination(tmp_path, _CE_ONLY)
    loss, metrics, grad = _run_kd_ce_combination(
        tmp_path,
        {"dynamic_loss_scaling": False, "kl_loss_weight": 5.0, "ce_loss_scale": 0.3},
    )

    kd, ce = metrics["kl_loss"], metrics["ce_loss"]
    assert kd > 0 and ce > 0
    assert loss.item() == pytest.approx(5.0 * kd + 0.3 * ce)
    assert metrics["kl_loss_scale"] == 1.0
    torch.testing.assert_close(grad, 5.0 * kd_grad + 0.3 * ce_grad)


def test_csd_dynamic_scaling_matches_kd_to_ce_and_ignores_weights(tmp_path):
    """Dynamic scaling is sg(CE / KD) * KD + CE; the fixed weights have no effect."""
    _, _, kd_grad = _run_kd_ce_combination(tmp_path, _KD_ONLY)
    _, _, ce_grad = _run_kd_ce_combination(tmp_path, _CE_ONLY)
    loss, metrics, grad = _run_kd_ce_combination(
        tmp_path,
        {"dynamic_loss_scaling": True, "kl_loss_weight": 1.0, "ce_loss_scale": 0.1},
    )
    reweighted_loss, _, reweighted_grad = _run_kd_ce_combination(
        tmp_path,
        {"dynamic_loss_scaling": True, "kl_loss_weight": 5.0, "ce_loss_scale": 1.0},
    )

    kd, ce = metrics["kl_loss"], metrics["ce_loss"]
    assert kd > 0 and ce > 0
    assert metrics["kl_loss_scale"] == pytest.approx(ce / kd)
    # The rescaled KD term equals CE in value, so the loss reads as 2 * CE.
    assert loss.item() == pytest.approx(2 * ce)
    torch.testing.assert_close(grad, (ce / kd) * kd_grad + ce_grad)
    torch.testing.assert_close(reweighted_loss, loss)
    torch.testing.assert_close(reweighted_grad, grad)


def test_csd_dynamic_scaling_keeps_unit_scale_when_kd_is_zero(tmp_path):
    """A microbatch without KD positions keeps scale 1, so the loss is CE alone."""
    loss, metrics, grad = _run_kd_ce_combination(
        tmp_path,
        {"dynamic_loss_scaling": True},
        kd_mask=torch.zeros(1, 5, dtype=torch.long),
    )
    _, _, ce_grad = _run_kd_ce_combination(tmp_path, _CE_ONLY)

    assert metrics["kl_loss"] == 0.0
    assert metrics["kl_loss_scale"] == 1.0
    assert loss.item() == pytest.approx(metrics["ce_loss"])
    torch.testing.assert_close(grad, ce_grad)


def test_csd_table_paths_identify_cross_tokenizer_without_projection(tmp_path):
    """CSD table artifacts select the cross-tokenizer path without dummy matrices."""
    cfg = _ct_loss_cfg(None)
    cfg["student_to_teacher_table_paths"] = [str(tmp_path / "forward.pt")]
    cfg["teacher_to_student_table_paths"] = [str(tmp_path / "reverse.pt")]
    cfg["common_indices_from_subtoks"] = True
    loss_fn = CSDLossFn(cfg)
    assert not loss_fn._teacher_is_same_vocab(0)
    assert loss_fn.projection_matrix_paths == [None]


def test_csd_runtime_table_lists_must_match_teachers(tmp_path):
    cfg = _ct_loss_cfg(None)
    cfg["student_to_teacher_table_paths"] = [None, None]
    with pytest.raises(ValueError, match="one entry per teacher"):
        CSDLossFn(cfg)
