from pathlib import Path

import pytest

OBJECTIVE_MODULE = (
    Path(__file__).parents[3] / "nemo_rl/algorithms/loss/dspark.py"
)
if not OBJECTIVE_MODULE.is_file():
    pytest.fail(f"missing DSpark objective API: {OBJECTIVE_MODULE}", pytrace=False)

import torch
import torch.nn.functional as F

from nemo_rl.algorithms.loss.dspark import (
    DSparkLossBins,
    dspark_tiled_objective,
)


def _inputs() -> dict[str, object]:
    generator = torch.Generator().manual_seed(314159)
    target_logits = torch.randn(
        2, 3, 7, generator=generator, dtype=torch.float64, requires_grad=True
    )
    base_logits = torch.randn(
        2, 3, 7, generator=generator, dtype=torch.float64, requires_grad=True
    )
    markov_bias = torch.randn(
        2, 3, 7, generator=generator, dtype=torch.float64, requires_grad=True
    )
    confidence_logits = torch.randn(
        2, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    return {
        "target_logits": target_logits,
        "base_logits": base_logits,
        "markov_bias": markov_bias,
        "confidence_logits": confidence_logits,
        "hard_labels": torch.tensor([[1, 4, 2], [5, 0, 6]]),
        "valid_mask": torch.tensor([[True, True, False], [True, False, True]]),
        "slot_bins": torch.tensor([[0, 1, 2], [0, 1, 2]]),
        "token_chunk_size": 2,
        "tp_group": None,
    }


def _dense_oracle(inputs: dict[str, object]) -> tuple[DSparkLossBins, ...]:
    target_logits = inputs["target_logits"]
    base_logits = inputs["base_logits"]
    markov_bias = inputs["markov_bias"]
    confidence_logits = inputs["confidence_logits"]
    hard_labels = inputs["hard_labels"]
    valid_mask = inputs["valid_mask"]
    slot_bins = inputs["slot_bins"]
    assert isinstance(target_logits, torch.Tensor)
    assert isinstance(base_logits, torch.Tensor)
    assert isinstance(markov_bias, torch.Tensor)
    assert isinstance(confidence_logits, torch.Tensor)
    assert isinstance(hard_labels, torch.Tensor)
    assert isinstance(valid_mask, torch.Tensor)
    assert isinstance(slot_bins, torch.Tensor)

    corrected_logits = base_logits + markov_bias
    target_probs = torch.softmax(target_logits.detach(), dim=-1)
    draft_probs = torch.softmax(corrected_logits, dim=-1)
    ce_rows = -torch.log_softmax(corrected_logits, dim=-1).gather(
        -1, hard_labels.unsqueeze(-1)
    ).squeeze(-1)
    tv_rows = 0.5 * (target_probs - draft_probs).abs().sum(dim=-1)
    verifier_correct = corrected_logits.detach().argmax(dim=-1).eq(hard_labels).float()
    confidence_rows = F.binary_cross_entropy_with_logits(
        confidence_logits,
        verifier_correct.to(dtype=confidence_logits.dtype),
        reduction="none",
    )

    num_bins = corrected_logits.shape[1]
    counts = torch.zeros(num_bins, dtype=torch.float32)
    counts.scatter_add_(0, slot_bins.reshape(-1), valid_mask.reshape(-1).float())
    components = []
    for rows in (ce_rows, tv_rows, confidence_rows):
        numerators = torch.zeros(num_bins, dtype=torch.float64)
        numerators.scatter_add_(
            0,
            slot_bins.reshape(-1),
            rows.reshape(-1) * valid_mask.reshape(-1),
        )
        components.append(DSparkLossBins(numerators=numerators, counts=counts))
    return tuple(components)


@pytest.mark.parametrize(
    "loss_weights",
    [
        pytest.param((1.0, 0.0, 0.0), id="ce_only"),
        pytest.param((0.0, 1.0, 0.0), id="tv_only"),
        pytest.param((0.0, 0.0, 1.0), id="confidence_only"),
        pytest.param((1.75, 0.375, 2.25), id="mixed_non_unit"),
    ],
)
def test_dspark_objective_matches_dense_component_oracles(
    loss_weights: tuple[float, float, float],
) -> None:
    """Wrong CE, TV factor, verifier target, bins, or weights breaks literals."""
    inputs = _inputs()
    expected_ce, expected_tv, expected_confidence = _dense_oracle(inputs)

    actual = dspark_tiled_objective(**inputs, loss_weights=loss_weights)

    for actual_component, expected_component in zip(
        (actual.ce, actual.tv, actual.confidence),
        (expected_ce, expected_tv, expected_confidence),
        strict=True,
    ):
        torch.testing.assert_close(
            actual_component.numerators, expected_component.numerators
        )
        torch.testing.assert_close(actual_component.counts, expected_component.counts)
    expected_combined = sum(
        weight * component.numerators
        for weight, component in zip(
            loss_weights,
            (expected_ce, expected_tv, expected_confidence),
            strict=True,
        )
    )
    torch.testing.assert_close(actual.combined.numerators, expected_combined)
    torch.testing.assert_close(actual.combined.counts, expected_ce.counts)


def test_dspark_objective_stops_teacher_but_trains_all_draft_inputs() -> None:
    """Target/live-teacher gradients must not leak through CE or TV."""
    inputs = _inputs()
    actual = dspark_tiled_objective(
        **inputs,
        loss_weights=(1.5, 0.75, 2.0),
    )

    actual.combined.normalized().backward()

    target_logits = inputs["target_logits"]
    assert isinstance(target_logits, torch.Tensor)
    assert target_logits.grad is None
    for name in ("base_logits", "markov_bias", "confidence_logits"):
        tensor = inputs[name]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_invalid_nonfinite_rows_are_excluded_before_probability_math() -> None:
    """Masked NaN/Inf values cannot poison numerators or gradients."""
    inputs = _inputs()
    valid_mask = inputs["valid_mask"]
    assert isinstance(valid_mask, torch.Tensor)
    for name, replacements in {
        "target_logits": (torch.nan, torch.inf),
        "base_logits": (torch.inf, torch.nan),
        "markov_bias": (-torch.inf, torch.nan),
    }.items():
        tensor = inputs[name]
        assert isinstance(tensor, torch.Tensor)
        data = tensor.detach().clone()
        data[~valid_mask] = torch.tensor(replacements, dtype=data.dtype).repeat(
            data.shape[-1] // 2 + 1
        )[: data.shape[-1]]
        inputs[name] = data.requires_grad_()
    confidence = inputs["confidence_logits"]
    assert isinstance(confidence, torch.Tensor)
    confidence_data = confidence.detach().clone()
    confidence_data[~valid_mask] = torch.tensor([torch.nan, torch.inf])
    inputs["confidence_logits"] = confidence_data.requires_grad_()

    actual = dspark_tiled_objective(
        **inputs,
        loss_weights=(1.0, 1.0, 1.0),
    )
    actual.combined.normalized().backward()

    for component in (actual.ce, actual.tv, actual.confidence, actual.combined):
        assert torch.isfinite(component.numerators).all()
    for name in ("base_logits", "markov_bias", "confidence_logits"):
        tensor = inputs[name]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.equal(tensor.grad[~valid_mask], torch.zeros_like(tensor.grad[~valid_mask]))


@pytest.mark.parametrize("num_blocks", [0, 2])
def test_empty_or_all_invalid_slots_return_zero_raw_bins(num_blocks: int) -> None:
    """Zero-count microbatches remain additive and normalize to finite zero."""
    num_slots, vocab_size = 3, 5
    shape = (num_blocks, num_slots, vocab_size)
    logits_shape = (num_blocks, num_slots)
    actual = dspark_tiled_objective(
        target_logits=torch.full(shape, torch.nan, requires_grad=True),
        base_logits=torch.full(shape, torch.inf, requires_grad=True),
        markov_bias=torch.full(shape, -torch.inf, requires_grad=True),
        confidence_logits=torch.full(logits_shape, torch.nan, requires_grad=True),
        hard_labels=torch.zeros(logits_shape, dtype=torch.long),
        valid_mask=torch.zeros(logits_shape, dtype=torch.bool),
        slot_bins=torch.arange(num_slots).expand(num_blocks, num_slots),
        loss_weights=(2.0, 3.0, 4.0),
        token_chunk_size=2,
        tp_group=None,
    )

    for component in (actual.ce, actual.tv, actual.confidence, actual.combined):
        torch.testing.assert_close(component.numerators, torch.zeros(num_slots))
        torch.testing.assert_close(component.counts, torch.zeros(num_slots))
        torch.testing.assert_close(component.normalized(), torch.tensor(0.0))


def test_raw_stats_add_across_dp_splits_with_one_zero_rank() -> None:
    """DP combination must sum raw bins, including a rank with no valid rows."""
    inputs = _inputs()
    full = dspark_tiled_objective(
        **inputs,
        loss_weights=(1.25, 0.5, 2.0),
    )
    rank_stats = []
    for block_index in range(2):
        rank_inputs: dict[str, object] = {}
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                rank_inputs[name] = value[block_index : block_index + 1]
            else:
                rank_inputs[name] = value
        if block_index == 1:
            rank_inputs["valid_mask"] = torch.zeros_like(rank_inputs["valid_mask"])
        rank_stats.append(
            dspark_tiled_objective(
                **rank_inputs,
                loss_weights=(1.25, 0.5, 2.0),
            )
        )
    first_rank_inputs = dict(inputs)
    for name, value in tuple(first_rank_inputs.items()):
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            first_rank_inputs[name] = value[:1]
    expected_nonzero_rank = dspark_tiled_objective(
        **first_rank_inputs,
        loss_weights=(1.25, 0.5, 2.0),
    )

    combined = rank_stats[0] + rank_stats[1]
    for actual_component, expected_component in zip(
        (combined.ce, combined.tv, combined.confidence, combined.combined),
        (
            expected_nonzero_rank.ce,
            expected_nonzero_rank.tv,
            expected_nonzero_rank.confidence,
            expected_nonzero_rank.combined,
        ),
        strict=True,
    ):
        torch.testing.assert_close(
            actual_component.numerators, expected_component.numerators
        )
        torch.testing.assert_close(actual_component.counts, expected_component.counts)
    assert not torch.equal(full.combined.counts, combined.combined.counts)


def test_probability_core_uses_one_normalizer_pass_and_saves_no_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CE and TV share each tile and backward recomputes ephemeral probabilities."""
    import nemo_rl.algorithms.loss.dspark as dspark

    inputs = _inputs()
    tile_sizes = []
    original = dspark._tile_log_normalizers

    def record_tiles(*args: object, **kwargs: object) -> torch.Tensor:
        student = args[0]
        assert isinstance(student, torch.Tensor)
        tile_sizes.append(student.shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(dspark, "_tile_log_normalizers", record_tiles)
    saved_shapes = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: (saved_shapes.append(tuple(tensor.shape)) or tensor),
        lambda tensor: tensor,
    ):
        actual = dspark_tiled_objective(
            **inputs,
            loss_weights=(1.0, 1.0, 1.0),
        )
        assert tile_sizes == [2, 2, 2]
        actual.combined.normalized().backward()

    vocab_shape = (2, 3, 7)
    assert saved_shapes.count(vocab_shape) <= 2


def test_confidence_can_be_disabled_without_changing_ce_or_tv() -> None:
    """A runtime without confidence still exposes explicit zero confidence bins."""
    inputs = _inputs()
    enabled = dspark_tiled_objective(
        **inputs,
        loss_weights=(1.25, 0.5, 0.0),
    )
    inputs["confidence_logits"] = None
    disabled = dspark_tiled_objective(
        **inputs,
        loss_weights=(1.25, 0.5, 0.0),
    )

    for name in ("ce", "tv", "combined"):
        actual = getattr(disabled, name)
        expected = getattr(enabled, name)
        torch.testing.assert_close(actual.numerators, expected.numerators)
        torch.testing.assert_close(actual.counts, expected.counts)
    torch.testing.assert_close(disabled.confidence.numerators, torch.zeros(3))
    torch.testing.assert_close(disabled.confidence.counts, torch.zeros(3))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("valid_mask", torch.ones(2, 3), TypeError),
        ("hard_labels", torch.zeros(2, 3, dtype=torch.int32), TypeError),
        ("slot_bins", torch.zeros(2, 3, dtype=torch.int32), TypeError),
        ("loss_weights", (1.0, -1.0, 1.0), ValueError),
    ],
)
def test_objective_rejects_ambiguous_metadata(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    inputs = _inputs()
    loss_weights = (1.0, 1.0, 1.0)
    if field == "loss_weights":
        loss_weights = value
    else:
        inputs[field] = value
    with pytest.raises(error):
        dspark_tiled_objective(**inputs, loss_weights=loss_weights)
