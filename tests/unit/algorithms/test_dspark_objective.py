import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F


def _load_objective_without_poisoning_packages() -> ModuleType:
    repository_root = Path(__file__).parents[3]
    module_names = (
        "nemo_rl",
        "nemo_rl.algorithms",
        "nemo_rl.algorithms.loss",
        "nemo_rl.algorithms.loss.draft",
        "nemo_rl.algorithms.loss.dspark",
    )
    previous_modules = {name: sys.modules.get(name) for name in module_names}
    try:
        for package_name in module_names[:3]:
            package = ModuleType(package_name)
            package.__path__ = []  # type: ignore[attr-defined]
            sys.modules[package_name] = package
        for module_name, relative_path in (
            ("nemo_rl.algorithms.loss.draft", "nemo_rl/algorithms/loss/draft.py"),
            ("nemo_rl.algorithms.loss.dspark", "nemo_rl/algorithms/loss/dspark.py"),
        ):
            spec = importlib.util.spec_from_file_location(
                module_name, repository_root / relative_path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load {relative_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        loaded = sys.modules["nemo_rl.algorithms.loss.dspark"]
        assert isinstance(loaded, ModuleType)
        return loaded
    finally:
        for module_name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous


_OBJECTIVE = _load_objective_without_poisoning_packages()
DSparkLossBins = _OBJECTIVE.DSparkLossBins
dspark_tiled_objective = _OBJECTIVE.dspark_tiled_objective


def _inputs() -> dict[str, object]:
    generator = torch.Generator().manual_seed(314159)
    target_logits = torch.randn(
        2, 3, 7, generator=generator, dtype=torch.float64, requires_grad=True
    )
    draft_hidden = torch.randn(
        2, 3, 5, generator=generator, dtype=torch.float64, requires_grad=True
    )
    target_output_weight = torch.randn(
        7, 5, generator=generator, dtype=torch.float64, requires_grad=True
    )
    markov_w1 = torch.randn(
        7, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    markov_w2 = torch.randn(
        7, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    confidence_logits = torch.randn(
        2, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    return {
        "target_logits": target_logits,
        "draft_hidden": draft_hidden,
        "target_output_weight": target_output_weight,
        "markov_w1": markov_w1,
        "markov_w2": markov_w2,
        "previous_token_ids": torch.tensor([[2, 2, 4], [2, 6, 2]]),
        "confidence_logits": confidence_logits,
        "hard_labels": torch.tensor([[1, 4, 2], [5, 0, 6]]),
        "valid_mask": torch.tensor([[True, True, False], [True, False, True]]),
        "slot_bins": torch.tensor([[0, 1, 2], [0, 1, 2]]),
        "token_chunk_size": 2,
        "vocab_start_index": 0,
        "tp_group": None,
    }


def _dense_oracle(inputs: dict[str, object]) -> tuple[DSparkLossBins, ...]:
    target_logits = inputs["target_logits"]
    draft_hidden = inputs["draft_hidden"]
    target_output_weight = inputs["target_output_weight"]
    markov_w1 = inputs["markov_w1"]
    markov_w2 = inputs["markov_w2"]
    previous_token_ids = inputs["previous_token_ids"]
    confidence_logits = inputs["confidence_logits"]
    hard_labels = inputs["hard_labels"]
    valid_mask = inputs["valid_mask"]
    slot_bins = inputs["slot_bins"]
    assert isinstance(target_logits, torch.Tensor)
    assert isinstance(draft_hidden, torch.Tensor)
    assert isinstance(target_output_weight, torch.Tensor)
    assert isinstance(markov_w1, torch.Tensor)
    assert isinstance(markov_w2, torch.Tensor)
    assert isinstance(previous_token_ids, torch.Tensor)
    assert isinstance(confidence_logits, torch.Tensor)
    assert isinstance(hard_labels, torch.Tensor)
    assert isinstance(valid_mask, torch.Tensor)
    assert isinstance(slot_bins, torch.Tensor)

    corrected_logits = (
        draft_hidden @ target_output_weight.detach().T
        + F.linear(F.embedding(previous_token_ids, markov_w1), markov_w2)
    ).float()
    target_probs = torch.softmax(target_logits.detach().float(), dim=-1)
    draft_probs = torch.softmax(corrected_logits, dim=-1)
    ce_rows = (
        -torch.log_softmax(corrected_logits, dim=-1)
        .gather(-1, hard_labels.unsqueeze(-1))
        .squeeze(-1)
    )
    tv_rows = 0.5 * (target_probs - draft_probs).abs().sum(dim=-1)
    verifier_correct = corrected_logits.detach().argmax(dim=-1).eq(hard_labels).float()
    confidence_rows = F.binary_cross_entropy_with_logits(
        confidence_logits.float(),
        verifier_correct,
        reduction="none",
    )

    num_bins = corrected_logits.shape[1]
    counts = torch.zeros(num_bins, dtype=torch.float32)
    counts.scatter_add_(0, slot_bins.reshape(-1), valid_mask.reshape(-1).float())
    components = []
    for rows in (ce_rows, tv_rows, confidence_rows):
        numerators = torch.zeros(num_bins, dtype=torch.float32)
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

    actual.combined.normalized(normalization_counts=actual.combined.counts).backward()

    target_logits = inputs["target_logits"]
    assert isinstance(target_logits, torch.Tensor)
    assert target_logits.grad is None
    target_output_weight = inputs["target_output_weight"]
    assert isinstance(target_output_weight, torch.Tensor)
    assert target_output_weight.grad is None
    for name in ("draft_hidden", "markov_w1", "markov_w2", "confidence_logits"):
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
        "draft_hidden": (torch.inf, torch.nan),
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
    confidence_data[~valid_mask] = torch.tensor(
        [torch.nan, torch.inf], dtype=confidence_data.dtype
    )
    inputs["confidence_logits"] = confidence_data.requires_grad_()

    actual = dspark_tiled_objective(
        **inputs,
        loss_weights=(1.0, 1.0, 1.0),
    )
    actual.combined.normalized(normalization_counts=actual.combined.counts).backward()

    for component in (actual.ce, actual.tv, actual.confidence, actual.combined):
        assert torch.isfinite(component.numerators).all()
    for name in ("draft_hidden", "confidence_logits"):
        tensor = inputs[name]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.equal(
            tensor.grad[~valid_mask], torch.zeros_like(tensor.grad[~valid_mask])
        )


@pytest.mark.parametrize("num_blocks", [0, 2])
def test_empty_or_all_invalid_slots_return_zero_raw_bins(num_blocks: int) -> None:
    """Zero-count microbatches remain additive and normalize to finite zero."""
    num_slots, vocab_size = 3, 5
    shape = (num_blocks, num_slots, vocab_size)
    logits_shape = (num_blocks, num_slots)
    actual = dspark_tiled_objective(
        target_logits=torch.full(shape, torch.nan, requires_grad=True),
        draft_hidden=torch.full(
            (num_blocks, num_slots, 4), torch.inf, requires_grad=True
        ),
        target_output_weight=torch.randn(vocab_size, 4, requires_grad=True),
        markov_w1=torch.randn(vocab_size, 2, requires_grad=True),
        markov_w2=torch.randn(vocab_size, 2, requires_grad=True),
        previous_token_ids=torch.full(logits_shape, -1, dtype=torch.long),
        confidence_logits=torch.full(logits_shape, torch.nan, requires_grad=True),
        hard_labels=torch.zeros(logits_shape, dtype=torch.long),
        valid_mask=torch.zeros(logits_shape, dtype=torch.bool),
        slot_bins=torch.arange(num_slots).expand(num_blocks, num_slots),
        loss_weights=(2.0, 3.0, 4.0),
        token_chunk_size=2,
        vocab_start_index=0,
        tp_group=None,
    )

    for component in (actual.ce, actual.tv, actual.confidence, actual.combined):
        torch.testing.assert_close(component.numerators, torch.zeros(num_slots))
        torch.testing.assert_close(component.counts, torch.zeros(num_slots))
        torch.testing.assert_close(
            component.normalized(normalization_counts=component.counts),
            torch.tensor(0.0),
        )


def test_raw_stats_add_across_dp_splits_with_one_zero_rank() -> None:
    """DP combination must sum raw bins, including a rank with no valid rows."""
    inputs = _inputs()
    full = dspark_tiled_objective(
        **inputs,
        loss_weights=(1.25, 0.5, 2.0),
    )
    rank_stats = []
    per_block_inputs = {
        "target_logits",
        "draft_hidden",
        "previous_token_ids",
        "confidence_logits",
        "hard_labels",
        "valid_mask",
        "slot_bins",
    }
    for block_index in range(2):
        rank_inputs: dict[str, object] = {}
        for name, value in inputs.items():
            if name in per_block_inputs and isinstance(value, torch.Tensor):
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
        if name in per_block_inputs and isinstance(value, torch.Tensor):
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
    dspark = _OBJECTIVE
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
        lambda tensor: saved_shapes.append(tuple(tensor.shape)) or tensor,
        lambda tensor: tensor,
    ):
        actual = dspark_tiled_objective(
            **inputs,
            loss_weights=(1.0, 1.0, 1.0),
        )
        assert tile_sizes == [2, 2, 2]
        actual.combined.normalized(
            normalization_counts=actual.combined.counts
        ).backward()

    vocab_shape = (2, 3, 7)
    assert saved_shapes.count(vocab_shape) <= 1


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


def test_normalized_requires_externally_reduced_counts() -> None:
    """A DP-local count must never silently define the optimizer gradient scale."""
    local_numerator = torch.tensor([6.0], requires_grad=True)
    stats = DSparkLossBins(
        numerators=local_numerator,
        counts=torch.tensor([2.0]),
    )

    with pytest.raises(TypeError, match="normalization_counts"):
        stats.normalized()  # type: ignore[call-arg]

    stats.normalized(normalization_counts=torch.tensor([3.0])).backward()
    torch.testing.assert_close(local_numerator.grad, torch.tensor([1.0 / 3.0]))


@pytest.mark.parametrize("bad_label", [-1, 7])
def test_valid_hard_labels_must_be_inside_global_vocabulary(bad_label: int) -> None:
    """Valid labels below zero or at vocab_size cannot be clamped to a shard."""
    inputs = _inputs()
    hard_labels = inputs["hard_labels"]
    assert isinstance(hard_labels, torch.Tensor)
    hard_labels[0, 0] = bad_label

    with pytest.raises(ValueError, match="hard_labels"):
        dspark_tiled_objective(**inputs, loss_weights=(1.0, 1.0, 1.0))


def test_invalid_slots_may_carry_out_of_range_sentinel_labels_and_tokens() -> None:
    """Only valid slots participate in label and previous-token validation."""
    inputs = _inputs()
    valid_mask = inputs["valid_mask"]
    hard_labels = inputs["hard_labels"]
    previous_token_ids = inputs["previous_token_ids"]
    assert isinstance(valid_mask, torch.Tensor)
    assert isinstance(hard_labels, torch.Tensor)
    assert isinstance(previous_token_ids, torch.Tensor)
    hard_labels[~valid_mask] = 999
    previous_token_ids[~valid_mask] = -999

    stats = dspark_tiled_objective(**inputs, loss_weights=(1.0, 1.0, 1.0))
    assert torch.isfinite(stats.combined.numerators).all()


def test_dspark_imports_coexist_with_normal_package_modules() -> None:
    """Collecting DSpark tests must not replace package modules in sys.modules."""
    assert sys.modules.get("nemo_rl.algorithms.loss.dspark") is not _OBJECTIVE
    existing_nemo = sys.modules.get("nemo_rl")
    assert existing_nemo is None or getattr(existing_nemo, "__file__", None) is not None


@pytest.mark.parametrize("source_position", [32_767, 262_143])
def test_qwen_vocab_retained_state_is_independent_of_source_position(
    source_position: int,
) -> None:
    """Only selected slots and one projected tile may be retained for long context."""
    generator = torch.Generator().manual_seed(source_position)
    local_vocab_size = 151_936 // 8
    inputs = {
        "target_logits": torch.randn(2, 3, local_vocab_size, generator=generator),
        "draft_hidden": torch.randn(2, 3, 16, generator=generator, requires_grad=True),
        "target_output_weight": torch.randn(
            local_vocab_size, 16, generator=generator, requires_grad=True
        ),
        "markov_w1": torch.randn(
            local_vocab_size, 2, generator=generator, requires_grad=True
        ),
        "markov_w2": torch.randn(
            local_vocab_size, 2, generator=generator, requires_grad=True
        ),
        "previous_token_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "confidence_logits": None,
        "hard_labels": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "valid_mask": torch.ones(2, 3, dtype=torch.bool),
        "slot_bins": torch.tensor([[0, 1, 2], [0, 1, 2]]),
        "loss_weights": (1.0, 1.0, 0.0),
        "token_chunk_size": 2,
        "vocab_start_index": 0,
        "tp_group": None,
    }
    saved_shapes: list[tuple[int, ...]] = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: saved_shapes.append(tuple(tensor.shape)) or tensor,
        lambda tensor: tensor,
    ):
        stats = dspark_tiled_objective(**inputs)
        stats.combined.normalized(normalization_counts=stats.combined.counts).backward()

    full_vocab_shape = (2, 3, local_vocab_size)
    assert saved_shapes.count(full_vocab_shape) <= 1
    assert (2, local_vocab_size) not in saved_shapes
    assert not any(
        len(shape) >= 2 and shape[0] in (32_768, 262_144) for shape in saved_shapes
    )
