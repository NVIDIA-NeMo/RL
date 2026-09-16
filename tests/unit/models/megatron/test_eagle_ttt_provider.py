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

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_provider():
    loss_package_name = "nemo_rl.algorithms.loss"
    loss_package = ModuleType(loss_package_name)
    loss_package.__path__ = [str(Path(__file__).parents[4] / "nemo_rl/algorithms/loss")]
    sys.modules[loss_package_name] = loss_package
    package_name = "nemo_rl.models.megatron.draft"
    package = ModuleType(package_name)
    package.__path__ = [
        str(Path(__file__).parents[4] / "nemo_rl/models/megatron/draft")
    ]
    sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.eagle_provider")


def _load_loss():
    _load_provider()
    return importlib.import_module("nemo_rl.algorithms.loss.draft")


class _FakeStructuredPassRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.begin_calls: list[dict[str, object]] = []
        self.reset_calls = 0

    def begin(self, **kwargs) -> None:
        self.begin_calls.append(kwargs)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        hidden_states = kwargs["hidden_states"]
        embeddings = kwargs["embeddings"]
        output = hidden_states + embeddings
        return output, output * 0.5

    def reset(self) -> None:
        self.reset_calls += 1


def _provider(module, *, budget: int = 1 << 30):
    return module.EagleTTTProvider(
        max_passes=8,
        activation_budget_bytes=budget,
        layer_count=3,
        kv_heads=2,
        head_dim=2,
        rope_dim=2,
    )


def _project(hidden_states: torch.Tensor) -> torch.Tensor:
    weight = torch.arange(
        hidden_states.shape[-1] * 7,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    ).reshape(hidden_states.shape[-1], 7)
    return hidden_states @ weight


def _soft_ce_pass_loss(
    *,
    teacher: torch.Tensor,
    valid: torch.Tensor,
    token_chunk_size: int,
):
    loss_module = _load_loss()

    def pass_loss(*, hidden_states: torch.Tensor, plan):
        logits = _project(hidden_states).transpose(0, 1)
        return loss_module.streaming_vocab_parallel_soft_ce(
            student_logits=logits,
            teacher_logits=teacher[:, plan.teacher_slice],
            mask=valid[:, plan.teacher_slice],
            token_chunk_size=token_chunk_size,
            tp_group=None,
        )

    return pass_loss


@pytest.mark.parametrize("pass_count", [1, 2, 4, 8])
def test_provider_runs_structured_passes_without_dense_masks(pass_count: int) -> None:
    module = _load_provider()
    runner = _FakeStructuredPassRunner()
    provider = _provider(module)
    sequence, batch, hidden = 12, 2, 4
    target = torch.randn(sequence, batch, hidden, requires_grad=True)
    embeds = torch.randn_like(target, requires_grad=True)
    teacher = torch.randn(batch, sequence, 7)
    valid = torch.ones(batch, sequence, dtype=torch.bool)
    output = provider.forward_loss_stats(
        pass_runner=runner,
        pass_loss=_soft_ce_pass_loss(
            teacher=teacher,
            valid=valid,
            token_chunk_size=4,
        ),
        target_trunk_states=target,
        input_embeds=embeds,
        pass_count=pass_count,
        pass_weights=torch.ones(pass_count),
    )

    assert output.stats.numerators.shape == (pass_count,)
    assert output.stats.counts.shape == (pass_count,)
    assert output.stats.weights.shape == (pass_count,)
    assert [plan.pass_index for plan in output.plans] == list(range(pass_count))
    assert len(runner.calls) == pass_count
    assert len(runner.begin_calls) == 1
    begin_call = runner.begin_calls[0]
    assert begin_call["layout"].valid_tokens.shape == (batch, sequence)
    assert begin_call["storage_plan"].pass_count == pass_count
    assert runner.reset_calls == 1
    assert not hasattr(output, "pass_logits")
    for index, call in enumerate(runner.calls):
        plan = call["plan"]
        assert plan.pass_index == index
        rope_positions = call["rope_positions"]
        torch.testing.assert_close(rope_positions, torch.arange(sequence))
        assert rope_positions is runner.calls[0]["rope_positions"]
        assert "attention_mask" not in call


def test_one_pass_matches_direct_loss_and_gradient() -> None:
    module = _load_provider()
    sequence, batch, hidden = 6, 2, 4
    target = torch.randn(sequence, batch, hidden, requires_grad=True)
    embeds = torch.randn_like(target, requires_grad=True)
    teacher = torch.randn(batch, sequence, 7)
    valid = torch.ones(batch, sequence, dtype=torch.bool)
    provider = _provider(module)
    output = provider.forward_loss_stats(
        pass_runner=_FakeStructuredPassRunner(),
        pass_loss=_soft_ce_pass_loss(
            teacher=teacher,
            valid=valid,
            token_chunk_size=4,
        ),
        target_trunk_states=target,
        input_embeds=embeds,
        pass_count=1,
        pass_weights=torch.tensor([1.0]),
    )

    direct_logits = _project(target + embeds).transpose(0, 1)
    direct_stats = _load_loss().streaming_vocab_parallel_soft_ce(
        student_logits=direct_logits[:, :-1],
        teacher_logits=teacher[:, 1:],
        mask=valid[:, 1:],
        token_chunk_size=4,
        tp_group=None,
    )
    torch.testing.assert_close(output.stats.numerators, direct_stats.numerators)
    torch.testing.assert_close(output.stats.counts, direct_stats.counts)
    actual_gradients = torch.autograd.grad(
        output.stats.numerators.sum(), (target, embeds)
    )
    expected_gradients = torch.autograd.grad(
        direct_stats.numerators.sum(), (target, embeds)
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


def test_cross_pass_state_is_explicitly_stop_gradient() -> None:
    module = _load_provider()
    sequence, batch, hidden, pass_count = 10, 1, 4, 4
    target = torch.randn(sequence, batch, hidden, requires_grad=True)
    embeds = torch.randn_like(target, requires_grad=True)
    teacher = torch.randn(batch, sequence, 7)
    valid = torch.ones(batch, sequence, dtype=torch.bool)
    runner = _FakeStructuredPassRunner()
    provider = _provider(module)
    output = provider.forward_loss_stats(
        pass_runner=runner,
        pass_loss=_soft_ce_pass_loss(
            teacher=teacher,
            valid=valid,
            token_chunk_size=8,
        ),
        target_trunk_states=target,
        input_embeds=embeds,
        pass_count=pass_count,
        pass_weights=torch.ones(pass_count),
    )

    assert runner.calls[0]["hidden_states"] is target
    assert all(not call["hidden_states"].requires_grad for call in runner.calls[1:])
    assert not output.final_branch_state.requires_grad
    target_gradient = torch.autograd.grad(
        output.stats.numerators.sum(), target, retain_graph=True
    )[0]
    first_pass = _soft_ce_pass_loss(
        teacher=teacher,
        valid=valid,
        token_chunk_size=8,
    )(
        hidden_states=(target + embeds)[:-1],
        plan=output.plans[0],
    )
    expected_gradient = torch.autograd.grad(first_pass.numerators.sum(), target)[0]
    torch.testing.assert_close(target_gradient, expected_gradient)


def test_full_and_split_additive_stats_match_with_unequal_weights() -> None:
    module = _load_provider()
    sequence, batch, hidden, pass_count = 9, 4, 4, 4
    target = torch.randn(sequence, batch, hidden)
    embeds = torch.randn_like(target)
    teacher = torch.randn(batch, sequence, 7)
    valid = torch.ones(batch, sequence, dtype=torch.bool)
    weights = torch.tensor([1.0, 0.6, 0.3, 0.1])
    provider = _provider(module)

    def run(start: int, end: int):
        return provider.forward_loss_stats(
            pass_runner=_FakeStructuredPassRunner(),
            pass_loss=_soft_ce_pass_loss(
                teacher=teacher[start:end],
                valid=valid[start:end],
                token_chunk_size=5,
            ),
            target_trunk_states=target[:, start:end],
            input_embeds=embeds[:, start:end],
            pass_count=pass_count,
            pass_weights=weights,
        ).stats

    full = run(0, batch)
    left = run(0, 1)
    right = run(1, batch)
    torch.testing.assert_close(full.numerators, left.numerators + right.numerators)
    torch.testing.assert_close(full.counts, left.counts + right.counts)
    torch.testing.assert_close(full.weights, weights)


@pytest.mark.parametrize("sequence", [32_768, 262_144])
def test_long_context_provider_retains_no_square_state_or_vocab_logits(
    sequence: int,
) -> None:
    module = _load_provider()
    provider = _provider(module)
    target = torch.zeros(sequence, 1, 1)
    seen_shapes: list[torch.Size] = []

    def pass_loss(*, hidden_states: torch.Tensor, plan):
        seen_shapes.append(hidden_states.shape)
        return _load_loss().DraftLossStats(
            numerators=hidden_states[:1].reshape(1).float() + plan.pass_index,
            counts=torch.ones(1),
            weights=torch.ones(1),
        )

    output = provider.forward_loss_stats(
        pass_runner=_FakeStructuredPassRunner(),
        pass_loss=pass_loss,
        target_trunk_states=target,
        input_embeds=torch.ones_like(target),
        pass_count=4,
        pass_weights=torch.ones(4),
    )
    assert output.stats.numerators.shape == (4,)
    assert output.final_branch_state.numel() == sequence
    assert seen_shapes == [
        torch.Size((sequence - pass_index - 1, 1, 1)) for pass_index in range(4)
    ]
    assert not any(
        len(shape) >= 2 and shape[-2:] == (sequence, sequence) for shape in seen_shapes
    )


@pytest.mark.mcore
@pytest.mark.parametrize("pass_count", [1, 2, 4, 8])
def test_pinned_modelopt_and_megatron_public_contracts(pass_count: int) -> None:
    module = _load_provider()
    megatron_eagle = pytest.importorskip(
        "modelopt.torch.speculative.plugins.megatron_eagle"
    )
    contexts = pytest.importorskip("megatron.core.inference.contexts")
    assert list(inspect.signature(megatron_eagle.EagleModule.forward).parameters) == [
        "self",
        "embeddings",
        "hidden_states",
        "attention_mask",
        "rotary_pos_emb",
        "inference_params",
        "packed_seq_params",
        "inference_context",
        "extra_block_kwargs",
    ]
    assert list(
        inspect.signature(megatron_eagle.set_multi_step_attention_mask).parameters
    ) == ["attn_mask", "step"]
    assert "detach()" in inspect.getsource(
        megatron_eagle.EagleModule._eagle3_layer_forward_hook
    )
    assert list(inspect.signature(contexts.StaticInferenceContext).parameters) == [
        "max_batch_size",
        "max_sequence_length",
        "use_flashinfer_fused_rope",
    ]
    context = contexts.StaticInferenceContext(
        max_batch_size=1,
        max_sequence_length=13 * 4,
    )
    assert context.sequence_len_offset == 0
    sequence = 12
    base_mask = torch.ones(sequence, sequence, dtype=torch.bool).triu(diagonal=1)
    base_mask = base_mask[None, None]
    shifted_mask = base_mask.clone()
    shifted_mask[:, :, :-1, :-1] = shifted_mask[:, :, 1:, 1:]
    shifted_mask[:, :, -1, :] = True
    shifted_mask[:, :, :, -1] = True
    pass_index = pass_count - 1
    modelopt_mask = megatron_eagle.set_multi_step_attention_mask(
        shifted_mask,
        pass_index,
    )
    plan = module.EagleTTTAttentionPlan(
        pass_index=pass_index,
        pass_count=pass_count,
        max_passes=8,
        sequence_length=sequence,
    )
    torch.testing.assert_close(
        ~modelopt_mask[0, 0, plan.student_slice],
        plan.dense_visibility_mask()[plan.student_slice],
    )


@pytest.mark.parametrize("pass_index", [0, 1, 2, 3, 7])
def test_provider_pass_loss_receives_exact_student_teacher_alignment(
    pass_index: int,
) -> None:
    module = _load_provider()
    sequence, batch, hidden = 12, 1, 4
    seen: list[tuple[torch.Tensor, slice]] = []

    def pass_loss(*, hidden_states: torch.Tensor, plan):
        seen.append((hidden_states.detach().clone(), plan.teacher_slice))
        return _load_loss().DraftLossStats(
            numerators=hidden_states.sum().reshape(1),
            counts=torch.tensor([hidden_states.shape[0]], dtype=torch.float32),
            weights=torch.ones(1),
        )

    target = torch.arange(sequence, dtype=torch.float32)[:, None, None].repeat(
        1, batch, hidden
    )

    class AlignmentRunner(_FakeStructuredPassRunner):
        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            aligned = torch.arange(
                sequence,
                dtype=target.dtype,
                device=target.device,
            )[:, None, None].repeat(1, batch, hidden)
            return aligned, aligned

    runner = AlignmentRunner()
    _provider(module).forward_loss_stats(
        pass_runner=runner,
        pass_loss=pass_loss,
        target_trunk_states=target,
        input_embeds=torch.zeros_like(target),
        pass_count=pass_index + 1,
        pass_weights=torch.ones(pass_index + 1),
    )

    aligned_student, teacher_slice = seen[pass_index]
    assert teacher_slice == slice(pass_index + 1, None)
    torch.testing.assert_close(
        aligned_student[:, 0, 0],
        torch.arange(pass_index, sequence - 1, dtype=torch.float32),
    )


def test_runner_session_resets_when_a_pass_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_provider()
    attention_reset_calls = 0

    def reset_attention_state() -> None:
        nonlocal attention_reset_calls
        attention_reset_calls += 1

    monkeypatch.setattr(
        module, "reset_eagle_ttt_attention_state", reset_attention_state
    )

    class FailingRunner(_FakeStructuredPassRunner):
        def __call__(self, **kwargs):
            super().__call__(**kwargs)
            raise RuntimeError("pass failed")

    runner = FailingRunner()
    target = torch.zeros(4, 1, 4)
    with pytest.raises(RuntimeError, match="pass failed"):
        _provider(module).forward_loss_stats(
            pass_runner=runner,
            pass_loss=lambda **_: None,
            target_trunk_states=target,
            input_embeds=target,
            pass_count=1,
            pass_weights=torch.ones(1),
        )
    assert runner.reset_calls == 1
    assert attention_reset_calls == 1


def test_resource_ledger_deduplicates_saved_storage_views() -> None:
    module = _load_provider()
    ledger = module.EagleTTTResourceLedger(limit_bytes=1 << 20)
    source = torch.randn(16, requires_grad=True)
    ledger.exclude((source,))

    with ledger.saved_tensors():
        intermediate = source.sin()
        loss = (intermediate * intermediate.view(4, 4).reshape(16)).sum()

    assert ledger.owned_bytes == 0
    assert ledger.autograd_bytes == intermediate.untyped_storage().nbytes()
    assert ledger.total_bytes == ledger.autograd_bytes
    loss.backward()


def test_provider_detects_actual_saved_state_overrun_and_resets() -> None:
    module = _load_provider()

    class OversizedSaveRunner(_FakeStructuredPassRunner):
        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            hidden_states = kwargs["hidden_states"]
            oversized = hidden_states.sum() * torch.ones(2048)
            output = hidden_states + oversized.square().sum() * 0
            return output, output

    runner = OversizedSaveRunner()
    target = torch.randn(4, 1, 4, requires_grad=True)
    provider = _provider(module, budget=1024)

    with pytest.raises(module.EagleTTTResourceLimitError, match="autograd"):
        provider.forward_loss_stats(
            pass_runner=runner,
            pass_loss=lambda **_: None,
            target_trunk_states=target,
            input_embeds=torch.zeros_like(target),
            pass_count=1,
            pass_weights=torch.ones(1),
        )

    assert len(runner.begin_calls) == 1
    ledger = runner.begin_calls[0]["resource_ledger"]
    assert runner.reset_calls == 1
    assert ledger.total_bytes == 0


def test_provider_resets_ledger_after_runner_exception_and_can_run_again() -> None:
    module = _load_provider()

    class FailOnceRunner(_FakeStructuredPassRunner):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        def __call__(self, **kwargs):
            if self.fail:
                self.fail = False
                raise RuntimeError("injected runner failure")
            return super().__call__(**kwargs)

    runner = FailOnceRunner()
    target = torch.zeros(4, 1, 4)
    provider = _provider(module)
    arguments = {
        "pass_runner": runner,
        "pass_loss": lambda **kwargs: _load_loss().DraftLossStats(
            numerators=kwargs["hidden_states"].sum().reshape(1),
            counts=torch.ones(1),
            weights=torch.ones(1),
        ),
        "target_trunk_states": target,
        "input_embeds": target,
        "pass_count": 1,
        "pass_weights": torch.ones(1),
    }

    with pytest.raises(RuntimeError, match="injected runner failure"):
        provider.forward_loss_stats(**arguments)

    first_ledger = runner.begin_calls[0]["resource_ledger"]
    assert first_ledger.total_bytes == 0
    output = provider.forward_loss_stats(**arguments)
    assert output.stats.numerators.shape == (1,)
    assert runner.reset_calls == 2
    assert len(runner.begin_calls) == 2


def test_projected_pass_loss_uses_indexed_teacher_rows_without_retained_logits() -> (
    None
):
    module = _load_provider()
    loss_module = _load_loss()
    sequence, batch, hidden, vocab = 7, 2, 4, 11
    teacher = torch.randn(batch, sequence, vocab)
    valid = torch.ones(batch, sequence, dtype=torch.bool)
    output_weight = torch.randn(vocab, hidden)
    captured: list[dict[str, object]] = []

    def projected_loss(**kwargs):
        captured.append(kwargs)
        student_hidden = kwargs["student_hidden"]
        indices = kwargs["teacher_row_indices"]
        selected_teacher = teacher.reshape(-1, vocab).index_select(
            0, indices.reshape(-1)
        )
        return loss_module.streaming_vocab_parallel_soft_ce(
            student_logits=student_hidden @ output_weight.T,
            teacher_logits=selected_teacher.reshape(batch, -1, vocab),
            mask=kwargs["mask"],
            token_chunk_size=kwargs["token_chunk_size"],
            tp_group=None,
        )

    adapter = module.ProjectedEaglePassLoss(
        projected_loss=projected_loss,
        output_weight=output_weight,
        teacher_logits=teacher,
        valid_mask=valid,
        token_chunk_size=3,
        tp_group=None,
    )
    hidden_states = torch.randn(sequence - 3, batch, hidden, requires_grad=True)
    plan = module.EagleTTTAttentionPlan(
        pass_index=2,
        pass_count=4,
        max_passes=8,
        sequence_length=sequence,
    )
    stats = adapter(hidden_states=hidden_states, plan=plan)

    assert stats.numerators.shape == (1,)
    assert len(captured) == 1
    call = captured[0]
    assert call["student_hidden"].shape == (batch, sequence - 3, hidden)
    expected_rows = torch.tensor([[3, 4, 5, 6], [10, 11, 12, 13]])
    torch.testing.assert_close(call["teacher_row_indices"], expected_rows)
    assert not hasattr(adapter, "student_logits")


@pytest.mark.parametrize("pass_count", [1, 2, 4, 8])
def test_each_projected_pass_matches_explicit_modelopt_loss_slice(
    pass_count: int,
) -> None:
    module = _load_provider()
    loss_module = _load_loss()
    sequence, batch, hidden, vocab = 12, 2, 4, 7
    pass_index = pass_count - 1
    full_hidden = torch.randn(
        sequence,
        batch,
        hidden,
        dtype=torch.float64,
        requires_grad=True,
    )
    output_weight = torch.randn(vocab, hidden, dtype=torch.float64)
    teacher = torch.randn(batch, sequence, vocab, dtype=torch.float64)
    valid = torch.ones(batch, sequence, dtype=torch.bool)

    def projected_loss(**kwargs):
        selected_teacher = teacher.reshape(-1, vocab).index_select(
            0,
            kwargs["teacher_row_indices"].reshape(-1),
        )
        return loss_module.streaming_vocab_parallel_soft_ce(
            student_logits=kwargs["student_hidden"] @ output_weight.T,
            teacher_logits=selected_teacher.reshape(batch, -1, vocab),
            mask=kwargs["mask"],
            token_chunk_size=kwargs["token_chunk_size"],
            tp_group=None,
        )

    adapter = module.ProjectedEaglePassLoss(
        projected_loss=projected_loss,
        output_weight=output_weight,
        teacher_logits=teacher,
        valid_mask=valid,
        token_chunk_size=3,
        tp_group=None,
    )
    plan = module.EagleTTTAttentionPlan(
        pass_index=pass_index,
        pass_count=pass_count,
        max_passes=8,
        sequence_length=sequence,
    )
    actual = adapter(
        hidden_states=full_hidden[plan.student_slice],
        plan=plan,
    )
    direct = loss_module.streaming_vocab_parallel_soft_ce(
        student_logits=(full_hidden[pass_index:-1].transpose(0, 1) @ output_weight.T),
        teacher_logits=teacher[:, pass_index + 1 :],
        mask=valid[:, pass_index + 1 :],
        token_chunk_size=3,
        tp_group=None,
    )
    torch.testing.assert_close(actual.numerators, direct.numerators)
    torch.testing.assert_close(actual.counts, direct.counts)
    actual_gradient = torch.autograd.grad(
        actual.numerators.sum(),
        full_hidden,
        retain_graph=True,
    )[0]
    direct_gradient = torch.autograd.grad(direct.numerators.sum(), full_hidden)[0]
    torch.testing.assert_close(actual_gradient, direct_gradient)
