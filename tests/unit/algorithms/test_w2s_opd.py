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

"""W2S-OPD (arXiv:2607.26246): numerics, configuration, and data-plane contracts.

CPU-only and process-group free: the collectives the distributed kernels issue
are neutralized for the TP=1 limit. ``tests/functional/w2s_*_distributed.py``
cover TP2 x CP2.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from nemo_rl.algorithms import opd
from nemo_rl.algorithms.async_utils.staleness_sampler import ReadyFirstSamplerConfig
from nemo_rl.algorithms.loss import (
    ClippedPGLossConfig,
    ClippedPGLossFn,
    prepare_loss_input,
)
from nemo_rl.algorithms.loss.interfaces import LossInputType, MetricNormalizer
from nemo_rl.algorithms.loss.wrapper import _SEQ_METRIC_MAX, _SEQ_METRIC_MIN
from nemo_rl.algorithms.opd import (
    OnPolicyDistillationFullConfig,
    W2SOPDConfig,
    get_opd_distribution_config,
    opd_full_teacher_index_field,
    w2s_anchor_model,
)
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.algorithms.single_controller_utils.utils import (
    _MB_METRIC_MAX,
    _MB_METRIC_MIN,
)
from nemo_rl.algorithms.w2s_opd import (
    validate_w2s_scoring,
    w2s_compact_proxy,
    w2s_packed_proxy_support,
    w2s_proxy_agreement_from_support,
    w2s_proxy_support,
    w2s_reverse_kl_from_support,
    w2s_reverse_kl_on_support,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import (
    W2S_ARM_FIELDS,
    W2S_ARM_ROW_SUFFIX,
    W2S_PROXY_FIELDS,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.teacher_worker_group import (
    create_teacher_configs_from_opd_config,
)
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

_REPO_ROOT = Path(__file__).resolve().parents[3]
# Same student, teacher resources, and loss block as the full-vocabulary MOPD
# recipe; ``full`` is swapped for ``w2s`` and the self-distilling teacher
# doubles as positive and negative, so the proxy equals the student's start.
_BASE_RECIPE = (
    _REPO_ROOT / "examples/configs/recipes/llm/"
    "mopd-qwen3-1.7b-3n8g-megatron-pack-single-controller-fullvocab.yaml"
)
_W2S_OVERRIDES = (
    "on_policy_distillation.full=null",
    "on_policy_distillation.reference_model_by_agent_name.default_teacher="
    "${policy.model_name}",
    "on_policy_distillation.w2s.enabled=true",
)


def _w2s_master_config(monkeypatch, *overrides: str) -> MasterConfig:
    monkeypatch.setenv("HF_HOME", "/tmp/nemo-rl-test-hf")
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(
        OmegaConf.merge(
            load_config(_BASE_RECIPE),
            OmegaConf.from_dotlist([*_W2S_OVERRIDES, *overrides]),
        ),
        resolve=True,
    )
    assert isinstance(resolved, dict)
    return MasterConfig.model_validate(resolved)


def _w2s_loss_fn(**overrides) -> ClippedPGLossFn:
    return ClippedPGLossFn(
        ClippedPGLossConfig(disable_ppo_ratio=True, reference_policy_kl_penalty=0),
        opd_full=W2SOPDConfig(enabled=True, **overrides),
    )


@pytest.fixture
def _single_rank_collectives(monkeypatch):
    """Neutralize the vocabulary-parallel collectives for the TP=1 limit."""
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        torch.distributed.nn.functional,
        "all_reduce",
        lambda tensor, *args, **kwargs: tensor,
    )


def reference(student, proxy, k, vocab):
    ids = proxy[..., :vocab].topk(k, dim=-1).indices
    p = student.gather(-1, ids).softmax(-1)
    q = proxy.gather(-1, ids).softmax(-1)
    return (p * (p.log() - q.log())).sum(-1)


def _support_loss(student, proxy, k, vocab):
    """The production student loss on a dense proxy's exact top-K support."""
    support, target = w2s_proxy_support(
        proxy, top_k=k, vocab_size=vocab, vocab_parallel_group=None
    )
    return w2s_reverse_kl_on_support(
        student, support, target, vocab_parallel_group=None
    )


def agreement_reference(student, support, vocab):
    """Argmax match and ``|top-K(student) ∩ support| / K`` on one rank."""
    k = support.shape[-1]
    student_top = student[..., :vocab].topk(k, dim=-1).indices
    top1 = (student_top[..., 0] == support[..., 0]).float()
    overlap = torch.tensor(
        [
            len(set(mine.tolist()) & set(theirs.tolist())) / k
            for mine, theirs in zip(
                student_top.reshape(-1, k), support.reshape(-1, k), strict=True
            )
        ]
    ).reshape(top1.shape)
    return top1, overlap


# ── Numerics ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("k", [2, 5, 9])
def test_dense_oracle_value_gradient_padding_and_detach(k):
    torch.manual_seed(19)
    student = torch.randn(2, 4, 12, requires_grad=True)
    proxy = torch.randn(2, 4, 12, requires_grad=True)
    proxy.data[..., 9:] = 1000  # padded vocab must never enter the support
    expected = reference(student, proxy.detach(), k, 9)
    (expected_grad,) = torch.autograd.grad(expected.sum(), student)
    actual = _support_loss(student, proxy, k, 9)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(student.grad, expected_grad, atol=2e-6, rtol=2e-6)
    assert proxy.grad is None
    assert student.grad[..., 9:].count_nonzero() == 0


@pytest.mark.parametrize(
    "alpha,equal_pair", [(0.0, False), (0.75, True), (0.75, False)]
)
@pytest.mark.parametrize("chunk", [1, 3, 8])
@pytest.mark.parametrize("window", [7, 9])
def test_proxy_matches_logit_arithmetic(alpha, equal_pair, chunk, window):
    torch.manual_seed(27)
    heads = tuple(torch.randn(13, h, requires_grad=True) for h in [3, 5])
    hidden = tuple(torch.randn(2, 7, h, requires_grad=True) for h in [3, 5])
    base, positive = [h @ w.t() for h, w in zip(hidden, heads)]
    # The negative arm is the negative teacher's own logits; an equal pair cancels.
    negative = (
        positive.detach().clone() if equal_pair else torch.randn(2, 7, 13)
    ).requires_grad_()
    proxy = base + alpha * (positive - negative)
    # A student window past the arms is padding that masks always drop.
    student = torch.randn(2, window, 13)
    student[:, :7] = base.detach()
    student.requires_grad_()
    expected = reference(student[:, :7], proxy.detach(), 5, 11)
    (expected_grad,) = torch.autograd.grad(expected.sum(), student)
    support, target = w2s_compact_proxy(
        hidden,
        heads,
        negative,
        alpha=alpha,
        top_k=5,
        vocab_size=11,
        chunk_size=chunk,
        vocab_parallel_group=None,
    )
    assert support.shape == target.shape == (2, 7, 5)
    assert support.max() < 11
    got = w2s_reverse_kl_from_support(
        student,
        support.to(torch.int32),
        target,
        chunk_size=chunk,
        vocab_parallel_group=None,
        context_parallel_group=None,
    )
    assert got.shape == (2, window)
    got[:, :7].sum().backward()
    torch.testing.assert_close(got[:, :7], expected, atol=4e-6, rtol=4e-6)
    torch.testing.assert_close(student.grad, expected_grad, atol=4e-6, rtol=4e-6)
    assert all(t.grad is None for t in (*heads, *hidden, negative))
    # Padded window positions stay finite (uniform target on a repeated id).
    assert torch.isfinite(got).all()
    assert got[:, 7:].count_nonzero() == 0
    if alpha == 0 or equal_pair:
        assert got.abs().max() < 1e-6
        assert student.grad.abs().max() < 1e-6
    else:
        assert student.grad.abs().max() > 0.01


def test_proxy_support_is_selected_after_contrast():
    # Token 2 is outside every component's top-2 but enters the proxy top-2.
    base = torch.tensor([[[10.0, 9.0, 0.0, -1.0]]])
    positive = torch.tensor([[[10.0, 9.0, 0.0, -1.0]]])
    negative = torch.tensor([[[10.0, 9.0, -100.0, -1.0]]])
    proxy = base + positive - negative
    assert 2 in proxy.topk(2).indices
    student = base.clone().requires_grad_()
    loss = _support_loss(student, proxy, 2, 4)
    loss.sum().backward()
    assert student.grad[0, 0, 2] < 0
    torch.testing.assert_close(loss, reference(student, proxy, 2, 4))


def test_nonfinite_scoring_rejects_predicted_tokens_but_not_padding(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    scores = torch.tensor([[float("nan"), -1.0, float("nan")]])
    validate_w2s_scoring(scores, torch.tensor([2]))
    with pytest.raises(FloatingPointError, match="local_count=1"):
        validate_w2s_scoring(scores, torch.tensor([3]))


def test_nonfinite_scoring_propagates_failure_from_another_rank(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda flag, **kwargs: flag.fill_(1)
    )
    with pytest.raises(FloatingPointError, match="local_count=0"):
        validate_w2s_scoring(torch.zeros(1, 2), torch.tensor([2]))


def _random_arms(seed, batch=2, length=7, widths=(3, 5), vocab=13):
    """Anchor/positive hidden states, their LM heads, and the negative logits."""
    torch.manual_seed(seed)
    heads = tuple(torch.randn(vocab, width) for width in widths)
    arms = tuple(torch.randn(batch, length, width) for width in widths)
    return arms, heads, torch.randn(batch, length, vocab)


def test_compact_support_rejects_mismatched_columns():
    student = torch.randn(1, 4, 9)
    with pytest.raises(ValueError, match="both be"):
        w2s_reverse_kl_from_support(
            student,
            torch.zeros(1, 4, 3, dtype=torch.long),
            torch.zeros(1, 4, 2),
            chunk_size=2,
            vocab_parallel_group=None,
            context_parallel_group=None,
        )
    with pytest.raises(ValueError, match="exceeds"):
        w2s_reverse_kl_from_support(
            student,
            torch.zeros(1, 5, 3, dtype=torch.long),
            torch.zeros(1, 5, 3),
            chunk_size=2,
            vocab_parallel_group=None,
            context_parallel_group=None,
        )


@pytest.mark.parametrize("payload_dtype", [torch.float32, torch.bfloat16])
def test_packed_proxy_rows_match_per_row_dense_support(payload_dtype):
    torch.manual_seed(41)
    heads = tuple(torch.randn(13, width) for width in (3, 5))
    lengths = [5, 3]
    spans = [(0, 8), (8, 12)]
    rows = [torch.randn(sum(lengths), width) for width in (3, 5)]
    negative = torch.randn(1, 12, 13)
    got = w2s_packed_proxy_support(
        tuple(rows),
        negative,
        heads,
        lengths=lengths,
        spans=spans,
        alpha=0.75,
        top_k=4,
        vocab_size=11,
        chunk_size=3,
        payload_dtype=payload_dtype,
        vocab_parallel_group=None,
        context_parallel_group=None,
    )
    offset = 0
    for (support, target), length, (start, _) in zip(got, lengths, spans):
        arms = (
            rows[0][None, offset : offset + length].to(payload_dtype),
            rows[1][None, offset : offset + length].to(payload_dtype),
        )
        offset += length
        expected = w2s_compact_proxy(
            arms,
            heads,
            negative[:, start : start + length],
            alpha=0.75,
            top_k=4,
            vocab_size=11,
            chunk_size=7,
            vocab_parallel_group=None,
        )
        torch.testing.assert_close(support, expected[0][0], atol=0, rtol=0)
        torch.testing.assert_close(target, expected[1][0], atol=0, rtol=0)


def test_packed_proxy_rejects_nonfinite_and_misaligned_rows():
    heads = tuple(torch.randn(9, width) for width in (2, 3))
    rows = [torch.randn(4, 2), torch.randn(4, 3)]
    kwargs = dict(
        alpha=0.5,
        top_k=3,
        vocab_size=9,
        chunk_size=2,
        payload_dtype=torch.float32,
        vocab_parallel_group=None,
        context_parallel_group=None,
    )
    with pytest.raises(ValueError, match="do not match"):
        w2s_packed_proxy_support(
            (rows[0][:3], rows[1]),
            torch.randn(1, 4, 9),
            heads,
            lengths=[4],
            spans=[(0, 4)],
            **kwargs,
        )
    with pytest.raises(ValueError, match="spans"):
        w2s_packed_proxy_support(
            tuple(rows),
            torch.randn(1, 4, 9),
            heads,
            lengths=[5],
            spans=[(0, 4)],
            **kwargs,
        )
    with pytest.raises(ValueError, match="vocabulary shard"):
        w2s_packed_proxy_support(
            tuple(rows),
            torch.randn(1, 4, 8),
            heads,
            lengths=[4],
            spans=[(0, 4)],
            **kwargs,
        )
    rows[1][2, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        w2s_packed_proxy_support(
            tuple(rows),
            torch.randn(1, 4, 9),
            heads,
            lengths=[4],
            spans=[(0, 4)],
            **kwargs,
        )


@pytest.mark.parametrize("chunk", [1, 2, 8])
def test_proxy_agreement_matches_a_dense_reference(chunk):
    torch.manual_seed(41)
    proxy = torch.randn(2, 5, 12)
    support = proxy[..., :9].topk(3, dim=-1).indices
    # One window position past the 5-token payload, as in a CP-padded window.
    student = torch.randn(2, 6, 12)
    student[0, 1] = proxy[0, 1]  # same ranking: full agreement
    student[1, 2, :9] = -proxy[1, 2, :9]  # reversed ranking: no agreement
    student[..., 9:] = 1e4  # padded vocab must never enter the student's top-K
    top1, overlap = w2s_proxy_agreement_from_support(
        student,
        support.to(torch.int32),
        vocab_size=9,
        chunk_size=chunk,
        vocab_parallel_group=None,
        context_parallel_group=None,
    )
    expected_top1, expected_overlap = agreement_reference(student[:, :5], support, 9)
    assert top1.shape == overlap.shape == (2, 6)
    torch.testing.assert_close(top1[:, :5], expected_top1)
    torch.testing.assert_close(overlap[:, :5], expected_overlap)
    assert top1[0, 1] == 1 and overlap[0, 1] == 1
    assert top1[1, 2] == 0 and overlap[1, 2] == 0
    # The padded window position stays finite; masks drop it.
    assert torch.isfinite(top1).all() and torch.isfinite(overlap).all()


def test_proxy_agreement_rejects_mismatched_support():
    student = torch.randn(2, 4, 9)
    kwargs = dict(
        vocab_size=9,
        chunk_size=2,
        vocab_parallel_group=None,
        context_parallel_group=None,
    )
    with pytest.raises(ValueError, match=r"\[B, S, K\]"):
        w2s_proxy_agreement_from_support(
            student, torch.zeros(1, 4, 3, dtype=torch.long), **kwargs
        )
    with pytest.raises(ValueError, match="K <= vocabulary"):
        w2s_proxy_agreement_from_support(
            student, torch.zeros(2, 4, 10, dtype=torch.long), **kwargs
        )


# ── Config schema and dispatch ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -1},
        {"alpha": float("nan")},
        {"alpha": float("inf")},
        {"top_k": 1},
        {"chunk_size": 0},
        {"chunk_size": None},
        {"teacher_payload": "logits"},
        {"teacher_lm_head_lifecycle": "offload"},
        {"teacher_lm_head_lifecycle": "evict"},
        {"validate_decomposition": True},
        # Keys an earlier W2S-OPD config carried; the anchor is now always
        # policy.model_name and the proxy support the only transport.
        {"anchor_model": "/student"},
        {"transport": "proxy_support"},
    ],
)
def test_schema_rejects_unsupported_settings(kwargs):
    with pytest.raises(ValueError):
        W2SOPDConfig(enabled=True, **kwargs)


def test_method_dispatch_requires_opd_and_excludes_full():
    assert get_opd_distribution_config({}) is None
    w2s = {"enabled": True}
    resolved = get_opd_distribution_config(
        {"on_policy_distillation": {"enabled": True, "w2s": w2s}}
    )
    assert isinstance(resolved, W2SOPDConfig)
    with pytest.raises(ValueError, match="on_policy_distillation.enabled"):
        get_opd_distribution_config(
            {"on_policy_distillation": {"enabled": False, "w2s": w2s}}
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        get_opd_distribution_config(
            {
                "on_policy_distillation": {
                    "enabled": True,
                    "full": {"enabled": True},
                    "w2s": w2s,
                }
            }
        )
    # A disabled w2s block leaves full-vocabulary OPD alone.
    full = get_opd_distribution_config(
        {
            "on_policy_distillation": {
                "enabled": True,
                "full": {"enabled": True},
                "w2s": {**w2s, "enabled": False},
            }
        }
    )
    assert type(full) is OnPolicyDistillationFullConfig


def test_w2s_rows_carry_no_teacher_index():
    """Every row needs all three arms, so no per-row teacher routing applies."""
    assert opd_full_teacher_index_field(W2SOPDConfig()) is None


@pytest.mark.parametrize("shared", [False, True])
def test_teacher_configs_add_the_frozen_negative_and_anchor(shared):
    configs = create_teacher_configs_from_opd_config(
        {
            "teacher_model_by_agent_name": {"swe": "/post"},
            "reference_model_by_agent_name": {"swe": "/post" if shared else "/pre"},
            "w2s": {"enabled": True},
            "non_colocated_teachers": {
                "default_teacher_cfg": {"num_nodes": 2},
                "teacher_overrides": {
                    "swe": {"num_nodes": 4},
                    "anchor:": {"num_nodes": 1},
                },
            },
        },
        anchor_model="/student",
    )
    by_alias = {config.alias: config for config in configs}
    if shared:
        # Deduplicated: the positive group also serves the negative arm.
        assert list(by_alias) == ["swe", "anchor:"]
    else:
        assert list(by_alias) == ["swe", "ref:swe", "anchor:"]
        assert by_alias["ref:swe"].model_name == "/pre"
        # A ref:<alias> teacher falls back to the <alias> override.
        assert by_alias["ref:swe"].num_nodes == 4
    assert by_alias["anchor:"].model_name == "/student"
    assert by_alias["anchor:"].num_nodes == 1
    # The anchor is given exactly when W2S-OPD is on.
    w2s_on = {"teacher_model_by_agent_name": {"swe": "/post"}, "w2s": {"enabled": True}}
    with pytest.raises(ValueError, match="exactly when"):
        create_teacher_configs_from_opd_config(w2s_on)
    with pytest.raises(ValueError, match="exactly when"):
        create_teacher_configs_from_opd_config(
            {**w2s_on, "w2s": None}, anchor_model="/student"
        )


# ── SingleController validation ────────────────────────────────────────────


def test_w2s_config_validates_and_builds_the_w2s_loss(monkeypatch):
    cfg = _w2s_master_config(monkeypatch)
    validate_single_controller_config(cfg)

    w2s_cfg = get_opd_distribution_config(cfg)
    assert isinstance(w2s_cfg, W2SOPDConfig)
    assert w2s_anchor_model(cfg) == cfg.policy["model_name"]
    loss_fn = ClippedPGLossFn(cfg.loss_fn, opd_full=w2s_cfg)
    assert loss_fn.input_type is LossInputType.OPD_FULL
    assert (
        loss_fn.metric_normalizations["w2s_opd_reverse_kl"] is MetricNormalizer.TOKENS
    )
    assert not any(
        name.startswith("opd_full_") for name in loss_fn.metric_normalizations
    )
    assert "w2s_opd_entropy" not in loss_fn.metric_normalizations
    assert "w2s_opd_proxy_top1_agreement" not in loss_fn.metric_normalizations


def test_log_student_entropy_registers_the_entropy_metric(monkeypatch):
    cfg = _w2s_master_config(
        monkeypatch, "on_policy_distillation.w2s.log_student_entropy=true"
    )
    validate_single_controller_config(cfg)
    loss_fn = ClippedPGLossFn(cfg.loss_fn, opd_full=get_opd_distribution_config(cfg))
    assert loss_fn.metric_normalizations["w2s_opd_entropy"] is MetricNormalizer.TOKENS


def test_log_proxy_agreement_registers_the_agreement_metrics(monkeypatch):
    cfg = _w2s_master_config(
        monkeypatch, "on_policy_distillation.w2s.log_proxy_agreement=true"
    )
    validate_single_controller_config(cfg)
    loss_fn = ClippedPGLossFn(cfg.loss_fn, opd_full=get_opd_distribution_config(cfg))
    for name in ("w2s_opd_proxy_top1_agreement", "w2s_opd_proxy_topk_overlap"):
        assert loss_fn.metric_normalizations[name] is MetricNormalizer.TOKENS


@pytest.mark.parametrize(
    "overrides,match",
    [
        pytest.param(
            [
                "on_policy_distillation.teacher_model_by_agent_name.other=/other",
                "on_policy_distillation.reference_model_by_agent_name.other=/other-pre",
            ],
            "exactly one positive",
            id="two_pairs",
        ),
        pytest.param(
            ["on_policy_distillation.deduplicate_shared_teacher_checkpoints=false"],
            "deduplicate_shared_teacher_checkpoints",
            id="no_dedup",
        ),
        pytest.param(["policy.generation.top_p=0.9"], "untempered", id="filtered"),
        pytest.param(
            ["policy.generation.temperature=0.8"], "untempered", id="temperature"
        ),
        pytest.param(
            ["loss_fn.reference_policy_kl_penalty=0.01"],
            "reference_policy_kl_penalty=0",
            id="reference_kl",
        ),
        pytest.param(
            ["loss_fn.use_importance_sampling_correction=true"],
            "use_importance_sampling_correction",
            id="importance_sampling",
        ),
        pytest.param(
            ["policy.megatron_cfg.use_fused_linear_logprobs=true"],
            "use_fused_linear_logprobs",
            id="fused",
        ),
        pytest.param(
            ["policy.sequence_packing.fuse_loss=true"], "fuse_loss", id="fuse_loss"
        ),
        pytest.param(
            [
                "on_policy_distillation.non_colocated_teachers.default_teacher_cfg"
                ".pipeline_model_parallel_size=2"
            ],
            "pipeline_model_parallel_size=1",
            id="negative_teacher_pp",
        ),
    ],
)
def test_invalid_w2s_config_fails_before_gpu_setup(monkeypatch, overrides, match):
    cfg = _w2s_master_config(monkeypatch, *overrides)
    with pytest.raises(ValueError, match=match):
        validate_single_controller_config(cfg)


def test_missing_negative_teacher_is_rejected(monkeypatch):
    cfg = _w2s_master_config(monkeypatch)
    cfg.on_policy_distillation.reference_model_by_agent_name = {}
    with pytest.raises(ValueError, match="reference_model_by_agent_name"):
        validate_single_controller_config(cfg)


def test_w2s_and_full_vocabulary_opd_are_mutually_exclusive(monkeypatch):
    cfg = _w2s_master_config(monkeypatch)
    cfg.on_policy_distillation.full = OnPolicyDistillationFullConfig(enabled=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_single_controller_config(cfg)


def test_reserved_aliases_are_rejected(monkeypatch):
    cfg = _w2s_master_config(monkeypatch)
    model = cfg.policy["model_name"]
    cfg.on_policy_distillation.teacher_model_by_agent_name = {"anchor:": model}
    cfg.on_policy_distillation.reference_model_by_agent_name = {"anchor:": model}
    cfg.on_policy_distillation.default_teacher_alias = None
    with pytest.raises(ValueError, match="reserves"):
        validate_single_controller_config(cfg)


def test_reference_teachers_require_w2s(monkeypatch):
    cfg = _w2s_master_config(monkeypatch)
    cfg.on_policy_distillation.w2s = None
    with pytest.raises(ValueError, match="only used by on_policy_distillation.w2s"):
        validate_single_controller_config(cfg)


def test_ready_first_accepts_w2s_only_with_the_logprob_error_filter(monkeypatch):
    # ready_first without importance weights, which the W2S loss rejects; the
    # logprob-error filter stands in for them.
    cfg = _w2s_master_config(monkeypatch)
    cfg.async_rl.sampler = ReadyFirstSamplerConfig()
    assert not cfg.loss_fn.use_importance_sampling_correction
    assert cfg.grpo.seq_logprob_error_threshold is not None
    validate_single_controller_config(cfg)

    cfg.grpo.seq_logprob_error_threshold = None
    with pytest.raises(ValueError, match="seq_logprob_error_threshold"):
        validate_single_controller_config(cfg)


# ── Loss ───────────────────────────────────────────────────────────────────


def test_w2s_extrema_are_reduced_by_both_accumulation_layers():
    assert "w2s_opd_reverse_kl_min" in _SEQ_METRIC_MIN & _MB_METRIC_MIN
    assert "w2s_opd_reverse_kl_max" in _SEQ_METRIC_MAX & _MB_METRIC_MAX


def test_dense_loss_masks_tools_padding_and_invalid_rows_ignores_rewards():
    loss_fn = _w2s_loss_fn(vocab_size=9)
    divergence = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    data = BatchedDataDict(
        {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "token_mask": torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]]),
            "sample_mask": torch.tensor([1.0, 0.0]),
            "advantages": torch.full((2, 4), float("nan")),
        }
    )
    loss, metrics = loss_fn(
        data=data,
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(2.0),
        opd_full_divergence=divergence,
    )
    loss.backward()
    assert loss.item() == 2.0
    torch.testing.assert_close(
        divergence.grad, torch.tensor([[0.5, 0.0, 0.5], [0.0, 0.0, 0.0]])
    )
    assert metrics["w2s_opd_reverse_kl"] == 2.0


def test_dense_loss_reports_sampler_drift_without_weighting_the_loss():
    loss_fn = _w2s_loss_fn(vocab_size=9)
    divergence = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    token_mask = torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]])
    # log p_train - log q_gen; the 9.0 sits on a masked (tool) token.
    log_ratio = torch.tensor([[0.0, 0.2, 9.0, -0.4], [0.0, 0.1, 0.1, 0.1]])
    data = BatchedDataDict(
        {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "token_mask": token_mask,
            "sample_mask": torch.tensor([1.0, 1.0]),
            "generation_logprobs": torch.full((2, 4), -1.0),
            "prev_logprobs": log_ratio - 1.0,
        }
    )
    loss, metrics = loss_fn(
        data=data,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=torch.tensor(5.0),
        opd_full_divergence=divergence,
    )
    # Diagnostics only: the loss is the plain masked mean (1 + 3 + 4 + 5 + 6) / 5.
    assert loss.item() == pytest.approx(19.0 / 5.0)
    expected_ratio = torch.tensor([0.2, -0.4, 0.1, 0.1, 0.1]).exp().sum().item() / 5.0
    assert metrics["w2s_opd_sampling_importance_ratio"] == pytest.approx(expected_ratio)
    # Per-sequence mean log-ratio: -0.1 and 0.1; sums: -0.2 and 0.3.
    assert metrics["w2s_opd_seq_mean_log_ratio_abs"] == pytest.approx(0.1)
    assert metrics["w2s_opd_seq_sum_log_ratio_abs"] == pytest.approx(0.25)
    for name in (
        "w2s_opd_sampling_importance_ratio",
        "w2s_opd_seq_mean_log_ratio_abs",
        "w2s_opd_seq_sum_log_ratio_abs",
    ):
        assert name in loss_fn.metric_normalizations


def test_proxy_loss_aligns_next_tokens_and_backpropagates_only_valid_positions(
    _single_rank_collectives,
):
    torch.manual_seed(54)
    student = torch.randn(2, 6, 9, requires_grad=True)
    arms = tuple(torch.randn(2, 6, w) for w in (2, 3))
    heads = tuple(torch.randn(9, w) for w in (2, 3))
    negative = torch.randn(2, 6, 9)
    support, target = w2s_compact_proxy(
        arms,
        heads,
        negative,
        alpha=0.75,
        top_k=4,
        vocab_size=9,
        chunk_size=128,
        vocab_parallel_group=None,
    )
    base = {
        "input_ids": torch.zeros(2, 6, dtype=torch.long),
        "token_mask": torch.tensor(
            [[0.0, 1.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        ),
        "sample_mask": torch.tensor([1.0, 0.0]),
    }
    data = BatchedDataDict(
        {
            **base,
            **dict(
                zip(W2S_PROXY_FIELDS, (support.to(torch.int32), target), strict=True)
            ),
        }
    )
    loss_fn = _w2s_loss_fn(vocab_size=9, top_k=4)
    kwargs = dict(
        vocab_parallel_rank=0,
        vocab_parallel_group=object(),
        context_parallel_group=None,
        sampling_params=None,
        chunk_size=None,
    )
    prepared, _ = prepare_loss_input(student, data, loss_fn, **kwargs)
    projected = [h @ w.t() for h, w in zip(arms, heads)]
    proxy = projected[0] + 0.75 * (projected[1] - negative)
    expected = reference(student, proxy, 4, 9)[:, :-1]
    torch.testing.assert_close(
        prepared["opd_full_divergence"], expected, atol=4e-6, rtol=4e-6
    )
    loss, _ = loss_fn(
        data=data,
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(3.0),
        **prepared,
    )
    expected_loss = expected[0, [0, 2, 3]].mean()
    gradient = torch.autograd.grad(expected_loss, student, retain_graph=True)[0]
    loss.backward()
    torch.testing.assert_close(loss, expected_loss, atol=4e-6, rtol=4e-6)
    torch.testing.assert_close(student.grad, gradient, atol=4e-6, rtol=4e-6)
    assert student.grad[1].count_nonzero() == 0
    assert student.grad[:, -1].count_nonzero() == 0
    with pytest.raises(ValueError, match="proxy columns"):
        prepare_loss_input(student, BatchedDataDict(base), loss_fn, **kwargs)


def test_prepared_loss_reports_exact_student_entropy_only_when_asked(
    _single_rank_collectives,
):
    arms, heads, negative = _random_arms(7, length=6, vocab=9)
    student = torch.randn(2, 6, 9, requires_grad=True)
    base = {
        "input_ids": torch.zeros(2, 6, dtype=torch.long),
        "token_mask": torch.ones(2, 6),
        "sample_mask": torch.ones(2),
    }
    support, target = w2s_compact_proxy(
        arms,
        heads,
        negative,
        alpha=0.75,
        top_k=4,
        vocab_size=9,
        chunk_size=128,
        vocab_parallel_group=None,
    )
    columns = dict(
        zip(W2S_PROXY_FIELDS, (support.to(torch.int32), target), strict=True)
    )

    def prepare(log_student_entropy):
        loss_fn = _w2s_loss_fn(
            vocab_size=9,
            top_k=4,
            chunk_size=4,
            log_student_entropy=log_student_entropy,
        )
        return loss_fn, prepare_loss_input(
            student,
            BatchedDataDict({**base, **columns}),
            loss_fn,
            vocab_parallel_rank=0,
            vocab_parallel_group=object(),
            context_parallel_group=None,
            sampling_params=None,
            chunk_size=None,
        )[0]

    plain_fn, plain = prepare(False)
    assert "opd_full_entropy" not in plain
    assert "w2s_opd_entropy" not in plain_fn.metric_normalizations

    loss_fn, prepared = prepare(True)
    # Same divergence with or without the diagnostic.
    torch.testing.assert_close(
        prepared["opd_full_divergence"], plain["opd_full_divergence"]
    )
    log_p = student.detach().log_softmax(-1)
    expected = (log_p.exp() * log_p).sum(-1)[:, :5]
    torch.testing.assert_close(prepared["opd_full_entropy"], expected)
    assert not prepared["opd_full_entropy"].requires_grad

    data = BatchedDataDict(
        {**base, "token_mask": torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 1.0]] * 2)}
    )
    _, metrics = loss_fn(
        data=data,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=torch.tensor(8.0),
        **prepared,
    )
    # token_mask[:, 1:] keeps positions 0, 1, 3, 4 of the next-token axis.
    kept = torch.tensor([0, 1, 3, 4])
    assert metrics["w2s_opd_entropy"] == pytest.approx(
        (-expected[:, kept]).sum().item() / 8.0, rel=1e-5
    )
    assert loss_fn.metric_normalizations["w2s_opd_entropy"] is MetricNormalizer.TOKENS


def test_prepared_loss_reports_proxy_agreement_only_when_asked(
    _single_rank_collectives,
):
    arms, heads, negative = _random_arms(8, length=6, vocab=9)
    anchor, positive = [arm @ head.t() for arm, head in zip(arms, heads, strict=True)]
    proxy = anchor + 0.75 * (positive - negative)
    support, target = w2s_compact_proxy(
        arms,
        heads,
        negative,
        alpha=0.75,
        top_k=4,
        vocab_size=9,
        chunk_size=128,
        vocab_parallel_group=None,
    )
    student = torch.randn(2, 6, 9)
    student[0, :3] = proxy[0, :3]  # the student ranks like the proxy here
    student.requires_grad_()
    base = {
        "input_ids": torch.zeros(2, 6, dtype=torch.long),
        "token_mask": torch.ones(2, 6),
        "sample_mask": torch.ones(2),
        **dict(zip(W2S_PROXY_FIELDS, (support.to(torch.int32), target), strict=True)),
    }
    names = ("w2s_proxy_top1_agreement", "w2s_proxy_topk_overlap")
    metric_names = ("w2s_opd_proxy_top1_agreement", "w2s_opd_proxy_topk_overlap")

    def prepare(log_proxy_agreement):
        loss_fn = _w2s_loss_fn(
            vocab_size=9,
            top_k=4,
            chunk_size=4,
            log_proxy_agreement=log_proxy_agreement,
        )
        return loss_fn, prepare_loss_input(
            student,
            BatchedDataDict(base),
            loss_fn,
            vocab_parallel_rank=0,
            vocab_parallel_group=object(),
            context_parallel_group=None,
            sampling_params=None,
            chunk_size=None,
        )[0]

    plain_fn, plain = prepare(False)
    assert not set(names) & set(plain)
    assert not set(metric_names) & set(plain_fn.metric_normalizations)

    loss_fn, prepared = prepare(True)
    # Same divergence with or without the diagnostic.
    torch.testing.assert_close(
        prepared["opd_full_divergence"], plain["opd_full_divergence"]
    )
    expected = agreement_reference(student.detach(), support, 9)
    for name, values in zip(names, expected, strict=True):
        torch.testing.assert_close(prepared[name], values[:, :5])
        assert not prepared[name].requires_grad
    assert prepared["w2s_proxy_top1_agreement"][0, :3].eq(1).all()
    assert prepared["w2s_proxy_topk_overlap"][0, :3].eq(1).all()

    data = BatchedDataDict(
        {**base, "token_mask": torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 1.0]] * 2)}
    )
    loss, metrics = loss_fn(
        data=data,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=torch.tensor(8.0),
        **prepared,
    )
    # token_mask[:, 1:] keeps positions 0, 1, 3, 4 of the next-token axis.
    kept = torch.tensor([0, 1, 3, 4])
    for metric_name, values in zip(metric_names, expected, strict=True):
        assert metrics[metric_name] == pytest.approx(values[:, kept].sum().item() / 8.0)
        assert loss_fn.metric_normalizations[metric_name] is MetricNormalizer.TOKENS
    # Diagnostics only: the loss does not change.
    plain_loss, plain_metrics = plain_fn(
        data=data,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=torch.tensor(8.0),
        **plain,
    )
    torch.testing.assert_close(loss, plain_loss)
    assert not set(metric_names) & set(plain_metrics)


# ── Teacher coordinator ────────────────────────────────────────────────────


def _extra_env_info(agent_name):
    """The only part of a rollout's env info the coordinator reads."""
    return {"agent_ref": {"name": agent_name}}


class _FakeDataPlane:
    """Wire rows, tags, and clears keyed by sample id, enough for coordinator tests."""

    def __init__(self, rows=None, tags=None):
        self.rows = dict(rows or {})
        self.tags = dict(tags or {})
        self.cleared = []

    def get_samples(self, *, sample_ids, partition_id, select_fields):
        return TensorDict(
            {
                field: torch.stack(
                    [self.rows[sample_id][field] for sample_id in sample_ids]
                )
                for field in select_fields
            },
            batch_size=[len(sample_ids)],
        )

    def put_samples(self, *, sample_ids, partition_id, fields, tags=None):
        for index, sample_id in enumerate(sample_ids):
            row = self.rows.setdefault(sample_id, {})
            for key in fields.keys():
                row[key] = fields[key][index]
            if tags is not None:
                self.tags[sample_id] = tags[index]

    def clear_samples(self, *, sample_ids, partition_id):
        self.cleared.append(list(sample_ids))
        for sample_id in sample_ids:
            self.rows.pop(sample_id, None)
            self.tags.pop(sample_id, None)


@pytest.mark.parametrize("fail_on", [None, "anchor", "proxy"])
def test_proxy_transport_scores_arms_on_temporary_rows(fail_on):
    canonical_tags = [{"row": "a"}, {"row": "b"}]
    plane = _FakeDataPlane(
        rows={
            "a": {"input_ids": torch.arange(4), "input_lengths": torch.tensor(4)},
            "b": {"input_ids": torch.arange(4, 8), "input_lengths": torch.tensor(3)},
        },
    )
    calls = []

    class Teacher:
        sharding_annotations = SimpleNamespace(get_axis_size=lambda name: 1)

        def __init__(self, name):
            self.name = name

        def get_logprobs_from_meta(self, meta, *, payload_field):
            calls.append((self.name, list(meta.sample_ids), payload_field))
            if fail_on == "anchor":
                raise RuntimeError("anchor failed")
            for sample_id, canonical_id in zip(meta.sample_ids, ["a", "b"]):
                assert sample_id in plane.rows, "arm row must exist before scoring"
                # The arm row is a wire clone of the canonical row, tags included.
                for field in ("input_ids", "input_lengths"):
                    assert torch.equal(
                        plane.rows[sample_id][field], plane.rows[canonical_id][field]
                    )
                assert plane.tags[sample_id] == plane.tags[canonical_id]
                assert plane.tags[sample_id] is not plane.tags[canonical_id]
                plane.rows[sample_id][payload_field] = self.name

        def get_w2s_proxy_from_meta(self, meta, *, arm_sample_ids, arm_fields):
            calls.append((self.name, list(meta.sample_ids), dict(arm_sample_ids)))
            assert tuple(arm_fields) == W2S_ARM_FIELDS
            for sample_id in meta.sample_ids:
                arm_row = plane.rows[arm_sample_ids[sample_id]]
                assert [arm_row[field] for field in arm_fields] == ["base", "pos"]
            if fail_on == "proxy":
                raise RuntimeError("proxy failed")

    coord = opd.TQTeacherLogprobCoordinator(
        dp_client=plane,
        teacher_worker_groups={name: Teacher(name) for name in ("base", "pos", "neg")},
        alias_to_group_alias={"anchor:": "base", "swe": "pos", "ref:swe": "neg"},
        on_policy_distillation_cfg={
            "enabled": True,
            "teacher_model_by_agent_name": {"swe": "/post"},
            "w2s": {"enabled": True},
        },
    )
    plane.tags.update(zip(["a", "b"], canonical_tags, strict=True))
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["a", "b"],
        fields=["input_ids", "input_lengths"],
        sequence_lengths=[4, 3],
        tags=canonical_tags,
    )
    run = coord.enrich(meta, _extra_env_info("swe"))
    arm_ids = [f"a{W2S_ARM_ROW_SUFFIX}", f"b{W2S_ARM_ROW_SUFFIX}"]
    if fail_on is not None:
        with pytest.raises(RuntimeError, match=f"{fail_on} failed"):
            asyncio.run(run)
    else:
        enriched = asyncio.run(run)
        assert set(W2S_PROXY_FIELDS).issubset(enriched.fields)
        assert not set(W2S_ARM_FIELDS) & set(enriched.fields)
        assert calls == [
            ("base", arm_ids, W2S_ARM_FIELDS[0]),
            ("pos", arm_ids, W2S_ARM_FIELDS[1]),
            ("neg", ["a", "b"], dict(zip(["a", "b"], arm_ids, strict=True))),
        ]
    # Temporary arm rows never outlive the enrichment, success or failure, and
    # the canonical rows are left as they were.
    assert plane.cleared == [arm_ids]
    assert set(plane.rows) == {"a", "b"}
    assert all(
        set(row) == {"input_ids", "input_lengths"} for row in plane.rows.values()
    )
