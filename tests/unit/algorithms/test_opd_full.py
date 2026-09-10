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
"""Full-vocabulary MOPD (``on_policy_distillation.full``) config and loss.

Everything here is CPU-only, and process-group free except for
``prepare_opd_full_loss_input``, whose TP collectives are neutralized the way
``tests/unit/distributed/test_model_utils.py`` does. The divergence kernels
themselves are covered there, and ``_opd_full_call`` only masks and normalizes a
divergence tensor that ``prepare_loss_input`` has already produced.
"""

from __future__ import annotations

import pytest
import torch

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import LossInputType, MetricNormalizer
from nemo_rl.algorithms.loss.utils import (
    prepare_opd_full_loss_input,
    reconstruct_opd_full_teacher_logits,
)
from nemo_rl.algorithms.loss.wrapper import _SEQ_METRIC_MAX, _SEQ_METRIC_MIN
from nemo_rl.algorithms.opd import (
    OnPolicyDistillationFullConfig,
    get_opd_full_config,
    opd_full_teacher_index_field,
)
from nemo_rl.data_plane.column_io import TOKEN_ALIGNED_FIELDS
from nemo_rl.data_plane.schema import (
    OPD_FULL_FIELDS,
    OPD_FULL_HIDDEN_STATES_FIELD,
    OPD_FULL_LOGITS_FIELD,
    OPD_FULL_TEACHER_INDEX_FIELD,
    SC_ROLLOUT_SCHEMA_FIELDS,
    fields_with_optional_opd_full,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _full(**overrides) -> OnPolicyDistillationFullConfig:
    return OnPolicyDistillationFullConfig(enabled=True, **overrides)


def _loss_fn(*, opd_full=None, **cfg_overrides) -> ClippedPGLossFn:
    """Build an opd_full loss with the knobs its validator insists on.

    ``reference_policy_kl_penalty`` defaults to 0.01, which under opd_full is
    legal but stacks a second KL and then demands differentiable current-policy
    logprobs. Default it off so each test opts in explicitly.
    """
    cfg_overrides.setdefault("disable_ppo_ratio", True)
    cfg_overrides.setdefault("reference_policy_kl_penalty", 0.0)
    return ClippedPGLossFn(
        ClippedPGLossConfig(**cfg_overrides),
        opd_full=_full() if opd_full is None else opd_full,
    )


# ── Config schema ──────────────────────────────────────────────────────────


def test_opd_full_config_rejects_a_degenerate_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        OnPolicyDistillationFullConfig(chunk_size=0)
    # None means "the whole sequence in one chunk" and must stay legal.
    assert OnPolicyDistillationFullConfig(chunk_size=None).chunk_size is None


def test_get_opd_full_config_requires_both_switches():
    """``full.enabled`` alone must not arm the teacher payload column.

    The payload column is only written when OPD itself is on, so a run with
    ``full.enabled`` but no OPD would ask the data plane for a column nobody
    wrote.
    """
    assert get_opd_full_config({}) is None
    assert get_opd_full_config({"on_policy_distillation": {"enabled": True}}) is None
    assert (
        get_opd_full_config(
            {"on_policy_distillation": {"enabled": True, "full": {"enabled": False}}}
        )
        is None
    )
    assert (
        get_opd_full_config(
            {"on_policy_distillation": {"enabled": False, "full": {"enabled": True}}}
        )
        is None
    )

    resolved = get_opd_full_config(
        {"on_policy_distillation": {"enabled": True, "full": {"enabled": True}}}
    )
    assert resolved is not None
    assert resolved.teacher_payload == "hidden_states"


# ── Data-plane column registration ─────────────────────────────────────────


def test_fields_with_optional_opd_full_is_a_no_op_when_disabled():
    base = ["input_ids", "advantages"]
    assert fields_with_optional_opd_full(base, field=None) == base

    once = fields_with_optional_opd_full(base, field=OPD_FULL_LOGITS_FIELD)
    assert once == [*base, OPD_FULL_LOGITS_FIELD]
    # Several call sites wrap overlapping field lists; wrapping twice must not
    # register the column twice.
    assert fields_with_optional_opd_full(once, field=OPD_FULL_LOGITS_FIELD) == once
    # The input list is never mutated in place.
    assert base == ["input_ids", "advantages"]


def test_teacher_index_field_is_only_needed_by_the_hidden_state_path():
    """The logits payload ships an already-projected distribution.

    Requesting the routing column there would make every consumer fetch a
    column no teacher ever wrote.
    """
    assert (
        opd_full_teacher_index_field(_full(teacher_payload="hidden_states"))
        == OPD_FULL_TEACHER_INDEX_FIELD
    )
    assert opd_full_teacher_index_field(_full(teacher_payload="logits")) is None


def test_fields_with_optional_opd_full_registers_the_teacher_index_column():
    """The routing column rides the same helper as the payload column."""
    base = ["input_ids", "advantages"]
    both = fields_with_optional_opd_full(
        base,
        field=OPD_FULL_HIDDEN_STATES_FIELD,
        teacher_index_field=OPD_FULL_TEACHER_INDEX_FIELD,
    )
    assert both == [*base, OPD_FULL_HIDDEN_STATES_FIELD, OPD_FULL_TEACHER_INDEX_FIELD]
    # Several call sites wrap overlapping field lists; neither column may be
    # registered twice.
    assert (
        fields_with_optional_opd_full(
            both,
            field=OPD_FULL_HIDDEN_STATES_FIELD,
            teacher_index_field=OPD_FULL_TEACHER_INDEX_FIELD,
        )
        == both
    )
    # The logits path passes None and must be left exactly as before.
    assert fields_with_optional_opd_full(
        base, field=OPD_FULL_LOGITS_FIELD, teacher_index_field=None
    ) == [*base, OPD_FULL_LOGITS_FIELD]
    # The input list is never mutated in place.
    assert base == ["input_ids", "advantages"]


def test_teacher_index_column_is_registered_per_sample_not_token_aligned():
    """One int per row, not per token.

    Declaring it token-aligned would make ``pack_jagged_fields`` trim it to the
    row's token length, so the routing tag would desynchronize from the rows it
    routes after any repack.
    """
    assert OPD_FULL_TEACHER_INDEX_FIELD in SC_ROLLOUT_SCHEMA_FIELDS
    assert OPD_FULL_TEACHER_INDEX_FIELD not in TOKEN_ALIGNED_FIELDS


def test_opd_full_columns_are_registered_token_aligned():
    """A payload column that is not token-aligned would be stored rectangular.

    ``pack_jagged_fields`` only trims a column to its per-row length when the
    column is declared token-aligned; otherwise the payload silently
    desynchronizes from ``input_ids`` after any repack.
    """
    assert set(OPD_FULL_FIELDS) <= TOKEN_ALIGNED_FIELDS
    # Pre-registered so TransferQueue does not race on lazy field creation.
    assert set(OPD_FULL_FIELDS) <= set(SC_ROLLOUT_SCHEMA_FIELDS)


# ── Loss construction ──────────────────────────────────────────────────────


def test_disabled_opd_full_leaves_the_logprob_path_intact():
    """``full.enabled=false`` must not flip input_type nor run the validator.

    The disabled block still reaches the constructor because it is resolved
    from config; if it armed the validator, an ordinary GRPO run that happened
    to carry the block would be rejected for its policy-gradient knobs.
    """
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(use_cispo=True),
        opd_full=OnPolicyDistillationFullConfig(enabled=False),
    )
    assert loss_fn.opd_full is None
    assert loss_fn.input_type is LossInputType.LOGPROB


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"disable_ppo_ratio": False}, "disable_ppo_ratio"),
        ({"ratio_clip_c": 3.0}, "dual PPO clipping"),
        ({"use_cispo": True}, "CISPO"),
        ({"force_on_policy_ratio": True}, "force_on_policy_ratio"),
        ({"use_on_policy_kl_approximation": True}, "use_on_policy_kl_approximation"),
        (
            {"sequence_level_importance_ratios": True},
            "sequence_level_importance_ratios",
        ),
        ({"use_importance_sampling_correction": True}, "exact expectation"),
        (
            {"truncated_importance_sampling_type": "tis"},
            "truncated_importance_sampling_type",
        ),
        ({"positive_example_nll_weight": 0.1}, "positive_example_nll_weight"),
        ({"use_kl_in_reward": True}, "use_kl_in_reward"),
    ],
)
def test_opd_full_rejects_knobs_with_no_code_path(overrides, match):
    """Every dead knob fails at construction rather than being ignored.

    ``use_on_policy_kl_approximation`` is the one the shipped recipe inherits
    as ``true`` from its grandparent, so without this rejection it would have
    been silently dropped on a real run.
    """
    with pytest.raises(ValueError, match=match):
        _loss_fn(**overrides)


def test_opd_full_rejects_fused_linear_logprobs():
    """The fused forward returns logprobs, but the reverse KL needs raw logits."""
    with pytest.raises(ValueError, match="use_fused_linear_logprobs"):
        ClippedPGLossFn(
            ClippedPGLossConfig(
                disable_ppo_ratio=True, reference_policy_kl_penalty=0.0
            ),
            use_fused_linear_logprobs=True,
            opd_full=_full(),
        )


def test_opd_full_warns_but_allows_a_second_reference_kl():
    """Stacking a reference KL on the teacher KL is legal but worth saying."""
    with pytest.warns(UserWarning, match="second KL"):
        loss_fn = _loss_fn(reference_policy_kl_penalty=0.01)
    assert loss_fn.reference_policy_kl_penalty == 0.01


def test_opd_full_declares_a_normalizer_for_every_metric_it_emits():
    """Split-API trainers rescale by these; a missing key silently mis-scales."""
    plain = _loss_fn()
    assert set(plain.metric_normalizations) == {
        "loss",
        "kl_penalty",
        "num_valid_samples",
        "opd_full_reverse_kl",
        "opd_full_reverse_kl_min",
        "opd_full_reverse_kl_max",
    }

    decomposed = _loss_fn(opd_full=_full(validate_decomposition=True))
    assert set(decomposed.metric_normalizations) == set(plain.metric_normalizations) | {
        "opd_full_entropy",
        "opd_full_cross_entropy",
        "opd_full_decomposition_error",
    }
    # Extrema and the residual are already reduced, so rescaling them by a token
    # or sequence count would be meaningless.
    for key in (
        "opd_full_reverse_kl_min",
        "opd_full_reverse_kl_max",
        "opd_full_decomposition_error",
    ):
        assert decomposed.metric_normalizations[key] is MetricNormalizer.NONE


def test_opd_full_extrema_are_reduced_by_both_accumulation_layers():
    """Two independent layers combine microbatch metrics, and both must agree.

    ``SequencePackingLossWrapper`` folds per-sequence metrics and the
    SingleController utils fold per-microbatch ones. A name present in one set
    but not the other gets *summed* there, reporting a "min" larger than any
    individual value.
    """
    # Deferred: single_controller_utils pulls the whole SingleController and
    # data-plane stack, which nothing else in this loss-level module needs.
    from nemo_rl.algorithms.single_controller_utils.utils import (
        _MB_METRIC_MAX,
        _MB_METRIC_MIN,
    )

    assert "opd_full_reverse_kl_min" in _SEQ_METRIC_MIN & _MB_METRIC_MIN
    assert {"opd_full_reverse_kl_max", "opd_full_decomposition_error"} <= (
        _SEQ_METRIC_MAX & _MB_METRIC_MAX
    )


# ── _opd_full_call reduction ───────────────────────────────────────────────


def _microbatch() -> BatchedDataDict:
    """Two sequences of 3 predicted tokens; the second one is a token short."""
    return BatchedDataDict(
        {
            "token_mask": torch.tensor(
                [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]],
            ),
            "sample_mask": torch.tensor([1.0, 1.0]),
        }
    )


_DIVERGENCE = torch.tensor([[1.0, 3.0, 2.0], [5.0, 7.0, 0.0]])
_GLOBAL_VALID_SEQS = torch.tensor(2.0)
_GLOBAL_VALID_TOKS = torch.tensor(5.0)


def test_opd_full_token_level_loss_normalizes_by_the_global_token_count():
    loss, metrics = _loss_fn(token_level_loss=True)(
        data=_microbatch(),
        global_valid_seqs=_GLOBAL_VALID_SEQS,
        global_valid_toks=_GLOBAL_VALID_TOKS,
        opd_full_divergence=_DIVERGENCE,
    )

    assert loss.item() == pytest.approx((1 + 3 + 2 + 5 + 7) / 5)
    assert metrics["num_valid_samples"] == pytest.approx(2.0)


def test_opd_full_sequence_level_loss_averages_within_each_sequence_first():
    """A flat masked_mean here would scale the loss by the mean sequence length.

    SEQUENCE_LEVEL divides by a *sequence* count, so summing tokens first and
    dividing once would mix the two units; each sequence must be averaged over
    its own valid tokens before the outer mean.
    """
    loss, _ = _loss_fn(token_level_loss=False)(
        data=_microbatch(),
        global_valid_seqs=_GLOBAL_VALID_SEQS,
        global_valid_toks=_GLOBAL_VALID_TOKS,
        opd_full_divergence=_DIVERGENCE,
    )

    # seq0 averages 1, 3, 2 -> 2.0; seq1 has two valid tokens -> 6.0.
    assert loss.item() == pytest.approx((2.0 + 6.0) / 2)


def test_opd_full_diagnostics_stay_token_normalized_under_sequence_level_loss():
    """``opd_full_reverse_kl`` declares MetricNormalizer.TOKENS either way."""
    _, metrics = _loss_fn(token_level_loss=False)(
        data=_microbatch(),
        global_valid_seqs=_GLOBAL_VALID_SEQS,
        global_valid_toks=_GLOBAL_VALID_TOKS,
        opd_full_divergence=_DIVERGENCE,
    )

    assert metrics["opd_full_reverse_kl"] == pytest.approx((1 + 3 + 2 + 5 + 7) / 5)
    # The masked-out trailing position must not drag the minimum to 0.
    assert metrics["opd_full_reverse_kl_min"] == pytest.approx(1.0)
    assert metrics["opd_full_reverse_kl_max"] == pytest.approx(7.0)


def test_opd_full_reports_no_kl_penalty_when_the_coefficient_is_zero():
    """Regression: ``kl_penalty`` divides by the coefficient to undo it.

    ``kl`` is a 0-dim tensor whose ``numel()`` is always 1, so guarding on
    ``kl.numel()`` instead of the coefficient divides by zero on the very first
    microbatch of a default-configured run.
    """
    _, metrics = _loss_fn(reference_policy_kl_penalty=0.0)(
        data=_microbatch(),
        global_valid_seqs=_GLOBAL_VALID_SEQS,
        global_valid_toks=_GLOBAL_VALID_TOKS,
        opd_full_divergence=_DIVERGENCE,
    )

    assert metrics["kl_penalty"] == 0


def test_opd_full_requires_a_differentiable_logprob_for_the_reference_kl():
    """``prev_logprobs`` is stale and detached, so it cannot carry the penalty."""
    with pytest.warns(UserWarning, match="second KL"):
        loss_fn = _loss_fn(reference_policy_kl_penalty=0.01)

    with pytest.raises(ValueError, match="next_token_logprobs"):
        loss_fn(
            data=_microbatch(),
            global_valid_seqs=_GLOBAL_VALID_SEQS,
            global_valid_toks=_GLOBAL_VALID_TOKS,
            opd_full_divergence=_DIVERGENCE,
        )


def test_opd_full_reference_kl_gradient_carries_the_score_function_term():
    """The sampled penalty needs a score-function term the divergence does not.

    Its weight is 1 in the forward pass, so only the gradient and the metric
    together make a dropped weight visible.
    """
    penalty = 0.01
    next_token_logprobs = torch.tensor(
        [[-0.5, -1.0, -2.0], [-0.25, -1.5, -3.0]], requires_grad=True
    )
    data = _microbatch()
    data["reference_policy_logprobs"] = torch.tensor(
        [[0.0, -0.75, -1.25, -2.5], [0.0, -0.5, -2.0, -1.0]]
    )

    with pytest.warns(UserWarning, match="second KL"):
        loss_fn = _loss_fn(reference_policy_kl_penalty=penalty, token_level_loss=True)

    loss, metrics = loss_fn(
        next_token_logprobs=next_token_logprobs,
        data=data,
        global_valid_seqs=_GLOBAL_VALID_SEQS,
        global_valid_toks=_GLOBAL_VALID_TOKS,
        opd_full_divergence=_DIVERGENCE,
    )
    loss.backward()

    mask = data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)
    logr = data["reference_policy_logprobs"][:, 1:] - next_token_logprobs.detach()
    k3 = torch.exp(logr) - 1.0 - logr
    # d(k3)/dx = -(exp(r) - 1); the weight adds k3 on top.
    expected_grad = penalty * mask * (k3 - (torch.exp(logr) - 1.0)) / _GLOBAL_VALID_TOKS

    # The score-function term is ~6e-5 here, which the default atol of 1e-5
    # would nearly swallow.
    torch.testing.assert_close(
        next_token_logprobs.grad, expected_grad, rtol=1e-5, atol=1e-9
    )
    # Forward is unweighted: the metric stays the plain k3.
    assert metrics["kl_penalty"] == pytest.approx(
        float((k3 * mask).sum() / _GLOBAL_VALID_TOKS), rel=1e-6
    )


def test_opd_full_requires_the_divergence_tensor():
    """Reaching the loss without it means prepare_loss_input silently no-oped."""
    with pytest.raises(ValueError, match="opd_full_divergence"):
        _loss_fn()(
            data=_microbatch(),
            global_valid_seqs=_GLOBAL_VALID_SEQS,
            global_valid_toks=_GLOBAL_VALID_TOKS,
        )


def test_opd_full_reports_the_decomposition_residual():
    """entropy + cross entropy - reverse KL, reported when asked for.

    This residual is an algebraic identity, not a correctness signal: all three
    kernels read the same logits, so a corrupted teacher cancels out of it. It
    pins the kernels' own arithmetic and nothing upstream of them.
    """
    entropy = torch.tensor([[-2.0, -2.0, -2.0], [-2.0, -2.0, 0.0]])
    cross_entropy = _DIVERGENCE - entropy

    _, metrics = _loss_fn(opd_full=_full(validate_decomposition=True))(
        data=_microbatch(),
        global_valid_seqs=_GLOBAL_VALID_SEQS,
        global_valid_toks=_GLOBAL_VALID_TOKS,
        opd_full_divergence=_DIVERGENCE,
        opd_full_entropy=entropy,
        opd_full_cross_entropy=cross_entropy,
    )

    assert metrics["opd_full_decomposition_error"] == pytest.approx(0.0, abs=1e-6)
    # Entropy is reported sign-flipped: the kernel returns sum_v p log p.
    assert metrics["opd_full_entropy"] == pytest.approx(2.0 * 5 / 5)
    assert metrics["opd_full_cross_entropy"] == pytest.approx(
        (3.0 + 5.0 + 4.0 + 7.0 + 9.0) / 5
    )


def test_opd_full_decomposition_residual_is_masked():
    """A residual at a masked position must not fail an otherwise clean step."""
    entropy = torch.tensor([[-2.0, -2.0, -2.0], [-2.0, -2.0, 0.0]])
    cross_entropy = _DIVERGENCE - entropy
    # Position [1, 2] is masked out by token_mask.
    cross_entropy[1, 2] += 100.0

    _, metrics = _loss_fn(opd_full=_full(validate_decomposition=True))(
        data=_microbatch(),
        global_valid_seqs=_GLOBAL_VALID_SEQS,
        global_valid_toks=_GLOBAL_VALID_TOKS,
        opd_full_divergence=_DIVERGENCE,
        opd_full_entropy=entropy,
        opd_full_cross_entropy=cross_entropy,
    )

    assert metrics["opd_full_decomposition_error"] == pytest.approx(0.0, abs=1e-6)


def test_opd_full_loss_is_differentiable_through_the_divergence():
    """The reverse KL *is* the objective; nothing else carries the gradient."""
    divergence = _DIVERGENCE.clone().requires_grad_(True)

    loss, _ = _loss_fn()(
        data=_microbatch(),
        global_valid_seqs=_GLOBAL_VALID_SEQS,
        global_valid_toks=_GLOBAL_VALID_TOKS,
        opd_full_divergence=divergence,
    )
    loss.backward()

    assert divergence.grad is not None
    # Masked positions get no gradient; valid ones share 1 / global_valid_toks.
    assert divergence.grad[1, 2].item() == pytest.approx(0.0)
    assert divergence.grad[0, 0].item() == pytest.approx(1 / 5, rel=1e-5)


# ── Teacher payload reconstruction ─────────────────────────────────────────
# With no context-parallel group this is pure torch. It owns three silent
# failure modes -- a transposed projection, the wrong vocabulary window, and
# padding the wrong dimension -- each of which yields a merely plausible teacher.


def test_reconstruct_projects_hidden_states_through_the_teacher_lm_head():
    """Deliberately non-square: a transposed matmul would still run on a square."""
    payload = torch.randn(2, 6, 3)  # [B, S, H_teacher]
    lm_head = torch.randn(5, 3)  # [V_local, H_teacher]

    teacher_logits = reconstruct_opd_full_teacher_logits(
        payload,
        teacher_payload="hidden_states",
        student_logits=torch.zeros(2, 6, 5),
        vocab_parallel_rank=0,
        context_parallel_group=None,
        teacher_output_layer_weight_by_index={0: lm_head},
    )

    torch.testing.assert_close(teacher_logits, payload @ lm_head.t())


def test_reconstruct_routes_each_row_through_its_own_teachers_lm_head():
    """Multi-teacher: the row's ``teacher_index`` picks the LM head, not the dict.

    Projecting the whole microbatch with one head is the silent failure this
    guards: the shapes stay right and only the distribution is another
    teacher's, so the alternating index below is what makes it visible.
    """
    payload = torch.randn(3, 4, 3)  # [B, S, H_teacher]
    heads = {0: torch.randn(5, 3), 1: torch.randn(5, 3)}
    teacher_index = torch.tensor([1, 0, 1])

    teacher_logits = reconstruct_opd_full_teacher_logits(
        payload,
        teacher_payload="hidden_states",
        student_logits=torch.zeros(3, 4, 5),
        vocab_parallel_rank=0,
        context_parallel_group=None,
        teacher_output_layer_weight_by_index=heads,
        teacher_index=teacher_index,
    )

    expected = torch.stack(
        [payload[row] @ heads[int(teacher_index[row])].t() for row in range(3)]
    )
    torch.testing.assert_close(teacher_logits, expected)
    # Whole-batch projection through either head alone would be wrong.
    assert not torch.allclose(teacher_logits, payload @ heads[0].t())
    assert not torch.allclose(teacher_logits, payload @ heads[1].t())


@pytest.mark.parametrize("teacher_index", [None, torch.tensor([0, 0])])
def test_reconstruct_uses_the_only_loaded_head_for_a_single_teacher_run(teacher_index):
    """One loaded shard needs no routing, with or without the column present."""
    payload = torch.randn(2, 3, 3)
    head = torch.randn(5, 3)

    teacher_logits = reconstruct_opd_full_teacher_logits(
        payload,
        teacher_payload="hidden_states",
        student_logits=torch.zeros(2, 3, 5),
        vocab_parallel_rank=0,
        context_parallel_group=None,
        teacher_output_layer_weight_by_index={0: head},
        teacher_index=teacher_index,
    )

    torch.testing.assert_close(teacher_logits, payload @ head.t())


def test_reconstruct_rejects_a_row_tagged_with_an_unloaded_teacher():
    """A tag with no shard means the routing and the load disagree."""
    with pytest.raises(ValueError, match="no teacher LM"):
        reconstruct_opd_full_teacher_logits(
            torch.randn(2, 3, 3),
            teacher_payload="hidden_states",
            student_logits=torch.zeros(2, 3, 5),
            vocab_parallel_rank=0,
            context_parallel_group=None,
            teacher_output_layer_weight_by_index={
                0: torch.randn(5, 3),
                1: torch.randn(5, 3),
            },
            teacher_index=torch.tensor([0, 7]),
        )


def test_reconstruct_rejects_a_teacher_whose_hidden_size_disagrees():
    """Teachers may differ in hidden size; the mismatch must name the culprit."""
    with pytest.raises(ValueError, match="for teacher_index=1"):
        reconstruct_opd_full_teacher_logits(
            torch.randn(2, 3, 3),
            teacher_payload="hidden_states",
            student_logits=torch.zeros(2, 3, 5),
            vocab_parallel_rank=0,
            context_parallel_group=None,
            teacher_output_layer_weight_by_index={
                0: torch.randn(5, 3),
                1: torch.randn(5, 4),
            },
            teacher_index=torch.tensor([0, 1]),
        )


@pytest.mark.parametrize("vocab_parallel_rank", [0, 1, 2])
def test_reconstruct_slices_this_ranks_vocabulary_window(vocab_parallel_rank):
    """A window off by one shard distills against another rank's vocabulary."""
    payload = torch.randn(1, 3, 12)
    shard = 4

    teacher_logits = reconstruct_opd_full_teacher_logits(
        payload,
        teacher_payload="logits",
        student_logits=torch.zeros(1, 3, shard),
        vocab_parallel_rank=vocab_parallel_rank,
        context_parallel_group=None,
    )

    start = vocab_parallel_rank * shard
    torch.testing.assert_close(teacher_logits, payload[..., start : start + shard])


def test_reconstruct_right_pads_a_short_payload_on_the_sequence_dim():
    """Padding the vocabulary dim instead would put mass on real tokens."""
    payload = torch.randn(1, 3, 4)

    teacher_logits = reconstruct_opd_full_teacher_logits(
        payload,
        teacher_payload="logits",
        student_logits=torch.zeros(1, 5, 4),
        vocab_parallel_rank=0,
        context_parallel_group=None,
    )

    assert teacher_logits.shape == (1, 5, 4)
    torch.testing.assert_close(teacher_logits[:, :3], payload)
    torch.testing.assert_close(teacher_logits[:, 3:], torch.zeros(1, 2, 4))


def test_reconstruct_takes_the_cp_window_before_projecting(monkeypatch):
    """Projecting first costs cp_size times the matmul and a full-sequence tensor.

    The result is identical either way, so only the shape handed to the matmul
    distinguishes them -- and that allocation is ``[B, S, V_local]``, which the
    divergence kernel's chunking cannot bound.
    """
    from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank

    cp_size, cp_rank = 4, 0
    payload = torch.randn(1, 32, 3)
    lm_head = torch.randn(5, 3)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: cp_size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: cp_rank)

    projected_shapes = []
    real_matmul = torch.matmul

    def spy(a, b, *args, **kwargs):
        projected_shapes.append(tuple(a.shape))
        return real_matmul(a, b, *args, **kwargs)

    monkeypatch.setattr(torch, "matmul", spy)

    teacher_logits = reconstruct_opd_full_teacher_logits(
        payload,
        teacher_payload="hidden_states",
        student_logits=torch.zeros(1, 8, 5),
        vocab_parallel_rank=0,
        context_parallel_group=object(),  # opaque: world size and rank are stubbed
        teacher_output_layer_weight_by_index={0: lm_head},
    )

    window = _get_tokens_on_this_cp_rank(payload, cp_rank, cp_size, seq_dim=1)
    assert projected_shapes == [tuple(window.shape)]
    torch.testing.assert_close(teacher_logits, window @ lm_head.t())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "teacher_payload": "hidden_states",
                "teacher_output_layer_weight_by_index": None,
            },
            "requires a loaded teacher",
        ),
        (
            {
                "teacher_payload": "hidden_states",
                "teacher_output_layer_weight_by_index": {0: torch.randn(4, 7)},
            },
            "do not match the loaded teacher LM head",
        ),
        (
            {
                "teacher_payload": "hidden_states",
                "teacher_output_layer_weight_by_index": {0: torch.randn(9, 3)},
            },
            "must match the student vocabulary shard",
        ),
    ],
)
def test_reconstruct_rejects_an_unusable_teacher_lm_head(kwargs, match):
    with pytest.raises(ValueError, match=match):
        reconstruct_opd_full_teacher_logits(
            torch.randn(1, 3, 3),
            student_logits=torch.zeros(1, 3, 4),
            vocab_parallel_rank=0,
            context_parallel_group=None,
            **kwargs,
        )


def test_reconstruct_rejects_a_payload_narrower_than_this_ranks_window():
    """Silently short-slicing here would hand the top rank a partial vocabulary."""
    with pytest.raises(ValueError, match="narrower than this rank's vocabulary"):
        reconstruct_opd_full_teacher_logits(
            torch.randn(1, 3, 6),
            teacher_payload="logits",
            student_logits=torch.zeros(1, 3, 4),
            vocab_parallel_rank=1,
            context_parallel_group=None,
        )


def test_reconstruct_rejects_a_payload_longer_than_the_forward_window():
    """A longer payload means the teacher and student disagree on the batch."""
    with pytest.raises(ValueError, match="longer than the student forward window"):
        reconstruct_opd_full_teacher_logits(
            torch.randn(1, 9, 4),
            teacher_payload="logits",
            student_logits=torch.zeros(1, 3, 4),
            vocab_parallel_rank=0,
            context_parallel_group=None,
        )


# ── Filtered samples ───────────────────────────────────────────────────────


def _microbatch_with_a_filtered_sequence() -> BatchedDataDict:
    """Three sequences, the third dropped by ``sample_mask``.

    ``overlong_filtering`` zeroes ``sample_mask`` rather than ``token_mask``, so
    a reduction that forgot the outer mask would still see this row's tokens.
    """
    return BatchedDataDict(
        {
            "token_mask": torch.tensor(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ),
            "sample_mask": torch.tensor([1.0, 1.0, 0.0]),
        }
    )


def test_opd_full_excludes_sequences_masked_out_by_sample_mask():
    """Regression: ``mask = token_mask * sample_mask`` -- both factors matter.

    Dropping the ``sample_mask`` factor leaves every other assertion in this
    module passing while filtered samples keep contributing gradient. The third
    row's divergence is set far above the others so its leakage is unmissable.
    """
    divergence = torch.tensor([[1.0, 3.0, 2.0], [5.0, 7.0, 0.0], [99.0, 99.0, 99.0]])

    loss, metrics = _loss_fn(token_level_loss=True)(
        data=_microbatch_with_a_filtered_sequence(),
        global_valid_seqs=_GLOBAL_VALID_SEQS,
        global_valid_toks=_GLOBAL_VALID_TOKS,
        opd_full_divergence=divergence,
    )

    # Identical to the unfiltered fixture: the third row contributes nothing.
    assert loss.item() == pytest.approx((1 + 3 + 2 + 5 + 7) / 5)
    assert metrics["opd_full_reverse_kl"] == pytest.approx((1 + 3 + 2 + 5 + 7) / 5)
    assert metrics["opd_full_reverse_kl_max"] == pytest.approx(7.0)
    assert metrics["num_valid_samples"] == pytest.approx(2.0)


def test_opd_full_filtered_sequences_receive_no_gradient():
    """A filtered row must not steer the update at all."""
    divergence = torch.tensor(
        [[1.0, 3.0, 2.0], [5.0, 7.0, 0.0], [99.0, 99.0, 99.0]], requires_grad=True
    )

    loss, _ = _loss_fn(token_level_loss=True)(
        data=_microbatch_with_a_filtered_sequence(),
        global_valid_seqs=_GLOBAL_VALID_SEQS,
        global_valid_toks=_GLOBAL_VALID_TOKS,
        opd_full_divergence=divergence,
    )
    loss.backward()

    assert divergence.grad is not None
    torch.testing.assert_close(divergence.grad[2], torch.zeros(3))
    assert divergence.grad[0, 0].item() == pytest.approx(1 / 5, rel=1e-5)


def test_opd_full_normalizes_by_the_global_counts_not_the_microbatch():
    """The other fixtures use global counts equal to the microbatch's own.

    A microbatch is one slice of the step, so dividing by its own counts would
    over-weight small microbatches.
    """
    global_valid_seqs = torch.tensor(8.0)
    global_valid_toks = torch.tensor(20.0)

    loss, metrics = _loss_fn(token_level_loss=True)(
        data=_microbatch(),
        global_valid_seqs=global_valid_seqs,
        global_valid_toks=global_valid_toks,
        opd_full_divergence=_DIVERGENCE,
    )

    assert loss.item() == pytest.approx((1 + 3 + 2 + 5 + 7) / 20)
    assert metrics["opd_full_reverse_kl"] == pytest.approx((1 + 3 + 2 + 5 + 7) / 20)

    loss, _ = _loss_fn(token_level_loss=False)(
        data=_microbatch(),
        global_valid_seqs=global_valid_seqs,
        global_valid_toks=global_valid_toks,
        opd_full_divergence=_DIVERGENCE,
    )

    # seq0 averages to 2.0, seq1 to 6.0; the outer mean divides by the global count.
    assert loss.item() == pytest.approx((2.0 + 6.0) / 8)


# ── prepare_opd_full_loss_input ────────────────────────────────────────────
# TP=1 limit with the kernels' collectives neutralized, as in test_model_utils.py.


@pytest.fixture
def _single_rank_collectives(monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, *a, **kw: None)
    monkeypatch.setattr(
        torch.distributed.nn.functional,
        "all_reduce",
        lambda tensor, *a, **kw: tensor,
    )
    # from_parallel_logits_to_logprobs reads the CP rank off the default group
    # even at cp_size == 1; there is no process group here.
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)


def test_prepare_opd_full_loss_input_projects_the_payload_and_drops_the_last_position(
    _single_rank_collectives,
):
    """The [B, S-1] divergence pairs position t with token t+1, like LOGPROB.

    Shifting the window by one or forgetting the slice both leave a tensor of a
    plausible shape; only the closed form at every position catches it.
    """
    torch.manual_seed(11)
    batch_size, seq_len, hidden, vocab = 2, 5, 3, 6
    student_logits = torch.randn(batch_size, seq_len, vocab, requires_grad=True)
    lm_head = torch.randn(vocab, hidden)
    payload = torch.randn(batch_size, seq_len, hidden)
    data = BatchedDataDict(
        {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            OPD_FULL_HIDDEN_STATES_FIELD: payload,
        }
    )

    loss_input = prepare_opd_full_loss_input(
        student_logits,
        data,
        _loss_fn(),
        vocab_parallel_rank=0,
        vocab_parallel_group=object(),  # opaque: every collective is neutralized
        context_parallel_group=None,
        sampling_params=None,
        chunk_size=None,
        teacher_output_layer_weight_by_index={0: lm_head},
    )

    student_log_probs = torch.log_softmax(student_logits.detach(), dim=-1)
    teacher_log_probs = torch.log_softmax(payload @ lm_head.t(), dim=-1)
    expected = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(
        -1
    )
    divergence = loss_input["opd_full_divergence"]

    assert divergence.shape == (batch_size, seq_len - 1)
    torch.testing.assert_close(divergence, expected[:, :-1], rtol=1e-5, atol=1e-6)
    # The divergence is the objective, so it must carry the student's gradient.
    assert divergence.requires_grad
    # Decomposition off and no reference KL: nothing else is computed.
    assert loss_input["opd_full_entropy"] is None
    assert loss_input["opd_full_cross_entropy"] is None
    assert "next_token_logprobs" not in loss_input


def test_prepare_opd_full_loss_input_routes_rows_by_the_teacher_index_column(
    _single_rank_collectives,
):
    """The routing column is read off the microbatch, not passed in by hand.

    This is the only place ``opd_full_teacher_index_field`` -> ``data[...]`` ->
    per-row projection is exercised end to end; a column that is configured but
    never read degrades silently to "everyone gets teacher 0".
    """
    torch.manual_seed(13)
    batch_size, seq_len, hidden, vocab = 2, 4, 3, 6
    student_logits = torch.randn(batch_size, seq_len, vocab, requires_grad=True)
    heads = {0: torch.randn(vocab, hidden), 1: torch.randn(vocab, hidden)}
    payload = torch.randn(batch_size, seq_len, hidden)
    teacher_index = torch.tensor([1, 0])
    data = BatchedDataDict(
        {
            "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            OPD_FULL_HIDDEN_STATES_FIELD: payload,
            OPD_FULL_TEACHER_INDEX_FIELD: teacher_index,
        }
    )

    loss_input = prepare_opd_full_loss_input(
        student_logits,
        data,
        _loss_fn(),
        vocab_parallel_rank=0,
        vocab_parallel_group=object(),  # opaque: every collective is neutralized
        context_parallel_group=None,
        sampling_params=None,
        chunk_size=None,
        teacher_output_layer_weight_by_index=heads,
    )

    teacher_logits = torch.stack(
        [payload[row] @ heads[int(teacher_index[row])].t() for row in range(batch_size)]
    )
    student_log_probs = torch.log_softmax(student_logits.detach(), dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    expected = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(
        -1
    )
    divergence = loss_input["opd_full_divergence"]

    assert divergence.shape == (batch_size, seq_len - 1)
    torch.testing.assert_close(divergence, expected[:, :-1], rtol=1e-5, atol=1e-6)
    assert divergence.requires_grad
