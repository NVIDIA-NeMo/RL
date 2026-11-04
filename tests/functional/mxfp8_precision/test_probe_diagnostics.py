"""CPU regression tests for the diagnostic checkers, including injected failures."""

import asyncio
import json
import sys

import pytest
import torch
import analyze_rollout_boundaries as boundaries

from analyze_rollout_boundaries import STAGES, attribution, read_phase
from precision_probe_common import (
    canonical_request_records,
    collect_indexed_outputs,
    generate_single_sample_rows,
    order_logit_captures,
    ordered_single_sequence_plan,
    forward_input_fingerprint,
    mxfp8_storage_snapshot,
    semantic_model_state,
    compare_tensor_directory,
    compare_gradient_prefixes,
    first_boundary_difference,
    save_json,
    tensor_error,
    tensor_hash,
)


def test_prefix_comparison_locates_a_difference_even_when_final_gradient_matches():
    expected = [
        {"next_row": row, "gradients": {"norm.weight": torch.tensor([value])}}
        for row, value in [(0, 0.0), (1, 1.0), (None, 3.0)]
    ]
    actual = [
        {
            "next_row": item["next_row"],
            "gradients": {k: v.clone() for k, v in item["gradients"].items()},
        }
        for item in expected
    ]
    assert compare_gradient_prefixes(actual, expected, atol=0, rtol=0)["pass"]
    actual[1]["gradients"]["norm.weight"].add_(0.5)
    result = compare_gradient_prefixes(actual, expected, atol=0, rtol=0)
    assert result["snapshot"] == 1 and result["before_row"] == 1
    actual[0]["next_row"] = 3
    assert (
        compare_gradient_prefixes(actual, expected, atol=0, rtol=0)["reason"]
        == "different_accumulation_order"
    )


def test_matching_packed_order_across_chunk_boundaries_preserves_lengths():
    ids = [3, 2, 1, 0]
    order, bounds, lengths = ordered_single_sequence_plan(
        ids, [[0, 1], [1, 2], [2, 3], [3, 4]], [64, 128, 256, 32]
    )
    assert [ids[i] for i in order] == [0, 1, 2, 3]
    assert lengths == [32, 256, 128, 64]
    assert bounds == [[0, 1], [1, 2], [2, 3], [3, 4]]
    chunk_ids = []
    for chunk in [[1, 0], [3, 2]]:
        permutation, _, _ = ordered_single_sequence_plan(
            chunk, [[0, 1], [1, 2]], [128, 128]
        )
        chunk_ids.extend(chunk[i] for i in permutation)
    assert chunk_ids == [0, 1, 2, 3]
    with pytest.raises(AssertionError):
        ordered_single_sequence_plan([0, 1], [[0, 2]], [256])


def test_bf16_rounding_changes_sum_when_the_observed_orders_differ():
    def accumulate(dtype, order):
        values = torch.tensor([256.0, 1.0, -256.0, 1.0], dtype=dtype)
        result = torch.zeros((), dtype=dtype)
        for i in order:
            result.add_(values[i])
        return result

    combined, chunked = [3, 2, 1, 0], [1, 0, 3, 2]
    assert accumulate(torch.bfloat16, combined) != accumulate(torch.bfloat16, chunked)
    assert accumulate(torch.float32, combined) == accumulate(torch.float32, chunked)


def test_serial_logit_window_excludes_later_requests_and_requires_coverage():
    stages = {
        "generated_outputs": {"-1": [{"lengths": [2]}, {"lengths": [1]}]},
        "sampler_logits": {
            "0": [{"shape": [1, 32], "sha256": str(i)} for i in range(5)]
        },
    }
    window = boundaries.restrict_sequential_logit_window(stages, 2)
    assert window["pass"] and len(stages["sampler_logits"]["0"]) == 3
    assert not boundaries.restrict_sequential_logit_window(stages, 3)["pass"]
    stages["sampler_logits"]["0"].pop()
    assert not boundaries.restrict_sequential_logit_window(stages, 2)["pass"]


def test_missing_forward_context_cannot_pass_strict_serial_control(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys, "argv", ["analyzer", str(tmp_path), "--require-forward-context"]
    )
    monkeypatch.setattr(
        boundaries,
        "read_phase",
        lambda root, phase: (
            {stage: "same" for stage in STAGES},
            {"sampler_logits": 1},
        ),
    )
    with pytest.raises(SystemExit) as error:
        boundaries.main()
    assert error.value.code == 1
    result = json.loads((tmp_path / "boundary-result.json").read_text())
    assert result["conclusion"] == "incomplete_forward_context"
    assert (
        json.loads(capsys.readouterr().out)["conclusion"]
        == "incomplete_forward_context"
    )


def test_sampler_forward_context_handles_positional_and_keyword_inputs():
    def forward(input_ids, positions):
        pass

    ids, positions = torch.tensor([3, 7]), torch.tensor([0, 1])
    a = forward_input_fingerprint(forward, (ids, positions), {})
    b = forward_input_fingerprint(
        forward, (), {"input_ids": ids, "positions": positions}
    )
    assert a == b and a["input_ids"]["values"] == [3, 7]
    assert forward_input_fingerprint(forward, (), {}) == {}
    c = forward_input_fingerprint(forward, (ids, positions + 1), {})
    assert a != c

    def head_forward(hidden_states):
        pass

    hidden = forward_input_fingerprint(
        head_forward, (ids,), {}, fields=("hidden_states",), include_values=False
    )
    assert hidden["hidden_states"]["sha256"] == tensor_hash(ids)
    assert "values" not in hidden["hidden_states"]


def test_teacher_logits_are_aligned_by_prompt_after_packing():
    a, b = torch.tensor([1.0]), torch.tensor([2.0])
    result = order_logit_captures([b, a], [1, 0])
    assert torch.equal(result[0], a) and torch.equal(result[1], b)
    with pytest.raises(AssertionError):
        order_logit_captures([a, b], [0, 0])
    with pytest.raises(AssertionError):
        order_logit_captures([a, b], [0])


class SampleBatch:
    def __init__(self, values):
        self.values = values
        self.size = len(values)

    def slice(self, start, end):
        return SampleBatch(self.values[start:end])


def test_vllm_requests_are_single_samples_and_local_indices_preserve_prompt_order():
    inputs = SampleBatch([17, 23, 31, 43])
    seen = []

    async def generate(sample, *, greedy):
        assert sample.size == 1 and greedy
        seen.extend(sample.values)
        # The real API returns local index zero for each independent request.
        yield 0, SampleBatch([sample.values[0] * 10])

    rows = asyncio.run(generate_single_sample_rows(generate, inputs, greedy=True))
    assert seen == inputs.values
    assert [row.values for row in rows] == [[170], [230], [310], [430]]


@pytest.mark.parametrize(
    "invalid", ["missing", "duplicate", "wrong_index", "multiple_rows"]
)
def test_single_sample_generation_rejects_invalid_results(invalid):
    async def generate(sample, *, greedy):
        if invalid == "missing":
            return
        if invalid == "wrong_index":
            yield 1, sample
        elif invalid == "multiple_rows":
            yield 0, SampleBatch([1, 2])
        else:
            yield 0, sample
            yield 0, sample

    with pytest.raises(AssertionError):
        asyncio.run(
            generate_single_sample_rows(generate, SampleBatch([17]), greedy=True)
        )


def test_request_order_does_not_hide_token_or_duplicate_count_differences():
    a = {"request_key": "a", "output_ids": [1, 2]}
    b = {"request_key": "b", "output_ids": [3, 4]}
    assert canonical_request_records([a, b]) == canonical_request_records([b, a])
    assert canonical_request_records([a, a, b]) != canonical_request_records([a, b])
    assert canonical_request_records([a, b]) != canonical_request_records(
        [a, {**b, "output_ids": [3, 5]}]
    )


def test_mxfp8_padding_is_separate_from_logical_scales():
    metadata = {
        "with_gemm_swizzled_scales": False,
        "rowwise_data": torch.zeros(64, 96, dtype=torch.uint8),
        "rowwise_scale_inv": torch.zeros(128, 8, dtype=torch.uint8),
        "columnwise_scale_inv": torch.zeros(4, 128, dtype=torch.uint8),
    }
    baseline = mxfp8_storage_snapshot((64, 96), metadata)
    metadata["rowwise_scale_inv"][64:, :] = 17
    metadata["rowwise_scale_inv"][:64, 3:] = 19
    metadata["columnwise_scale_inv"][2:, :] = 23
    metadata["columnwise_scale_inv"][:2, 96:] = 29
    padded = mxfp8_storage_snapshot((64, 96), metadata)
    assert semantic_model_state(baseline) == semantic_model_state(padded)
    assert baseline["raw_storage"] != padded["raw_storage"]
    metadata["columnwise_scale_inv"][1, 95] = 31
    changed = mxfp8_storage_snapshot((64, 96), metadata)
    assert semantic_model_state(changed) != semantic_model_state(baseline)


def test_mxfp8_partial_last_scale_block_is_not_padding():
    metadata = {
        "with_gemm_swizzled_scales": False,
        "columnwise_scale_inv": torch.zeros(4, 128, dtype=torch.uint8),
    }
    before = mxfp8_storage_snapshot((65, 96), metadata)
    metadata["columnwise_scale_inv"][2, 95] = 1
    assert semantic_model_state(before) != semantic_model_state(
        mxfp8_storage_snapshot((65, 96), metadata)
    )


def test_old_transposed_column_crop_reads_padding_and_misses_used_scales():
    a = torch.zeros(4, 128, dtype=torch.uint8)
    b = a.clone()
    b[2, 0] = 17  # Padding for a [64,96] weight, but read by the old crop.
    assert tensor_hash(a[:96, :2]) != tensor_hash(b[:96, :2])
    assert tensor_hash(a[:2, :96]) == tensor_hash(b[:2, :96])
    b.copy_(a)
    b[1, 95] = 23  # Used scale, but outside the old crop's two columns.
    assert tensor_hash(a[:96, :2]) == tensor_hash(b[:96, :2])
    assert tensor_hash(a[:2, :96]) != tensor_hash(b[:2, :96])


@pytest.mark.parametrize("swizzled", [True, None])
def test_unknown_or_swizzled_scale_layout_fails_closed(swizzled):
    with pytest.raises(AssertionError, match="layout"):
        mxfp8_storage_snapshot(
            (64, 96),
            {
                "with_gemm_swizzled_scales": swizzled,
                "rowwise_scale_inv": torch.zeros(128, 8, dtype=torch.uint8),
            },
        )


def test_async_rollouts_are_compared_in_prompt_order():
    async def outputs():
        yield 1, "second"
        yield 0, "first"

    assert asyncio.run(collect_indexed_outputs(outputs(), 2)) == ["first", "second"]


@pytest.mark.parametrize("indices", [[0], [0, 0], [0, 2]])
def test_async_rollout_coverage_fails_closed(indices):
    async def outputs():
        for index in indices:
            yield index, "output"

    with pytest.raises(AssertionError):
        asyncio.run(collect_indexed_outputs(outputs(), 2))


def test_tensor_hash_checks_shape_dtype_and_every_value():
    x = torch.arange(16, dtype=torch.float32)
    assert tensor_hash(x) != tensor_hash(x.view(4, 4))
    assert tensor_hash(x) != tensor_hash(x.double())
    y = x.clone()
    y[-1] += 1
    assert tensor_hash(x) != tensor_hash(y)
    assert tensor_hash(x) == tensor_hash(x.clone())
    assert tensor_hash(torch.tensor(1.0)) == tensor_hash(torch.tensor(1.0))


def test_gradient_overwrite_is_detected_even_with_same_norm():
    reference = torch.tensor([1.0, -1.0, 2.0, -2.0])
    corrupted = reference.flip(0)
    assert reference.norm() == corrupted.norm()
    result = tensor_error(corrupted, reference, atol=0, rtol=0)
    assert not result["pass"] and result["different"] == 4


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_cannot_pass(bad):
    assert not tensor_error(torch.tensor([bad]), torch.zeros(1), atol=1, rtol=1)["pass"]


def test_full_gradient_comparison_detects_stale_tensor_and_missing_parameter(tmp_path):
    a, b = tmp_path / "actual", tmp_path / "expected"
    for path in [a, b]:
        path.mkdir()
        save_json(path / "manifest.json", {"expert.weight": "w.pt"})
        torch.save(torch.ones(32), path / "w.pt")
    assert compare_tensor_directory(a, b, atol=0, rtol=0)["pass"]
    v = torch.ones(32)
    v[17] = 0
    torch.save(v, a / "w.pt")
    assert not compare_tensor_directory(a, b, atol=0, rtol=0)["pass"]
    save_json(a / "manifest.json", {})
    assert compare_tensor_directory(a, b, atol=0, rtol=0)["reason"] == "parameter_set"


def test_missing_boundary_never_counts_as_equivalent():
    result = first_boundary_difference({}, {}, STAGES)
    assert result == {"stage": "trainer_before_refit", "reason": "missing"}


def test_first_difference_orders_wire_before_logits():
    a = {k: 1 for k in STAGES}
    b = dict(a)
    b["wire_receive"] = 2
    b["sampler_logits"] = 2
    assert first_boundary_difference(a, b, STAGES)["stage"] == "wire_receive"


def test_clean_clean_difference_blocks_poison_attribution():
    result = attribution(
        {"stage": "sampler_logits", "reason": "different"},
        {"stage": "generated_outputs", "reason": "different"},
    )
    assert result["conclusion"] == "baseline_not_repeatable_or_incomplete"


def test_poison_only_difference_is_candidate_not_proven_cause():
    result = attribution(None, {"stage": "sampler_after_refit", "reason": "different"})
    assert result["conclusion"] == "poison_specific_candidate"
    assert "repeat" in result["causal_claim"]


def test_boundary_reader_ignores_pid_and_records_rng_separately(tmp_path):
    for phase, pid, rng in [("clean_a", 123, 1), ("clean_b", 456, 2)]:
        folder = (
            tmp_path / "boundaries" / phase / f"sampler_after_refit-rank-0-pid-{pid}"
        )
        save_json(
            folder / "00000.json", {"model": {"parameters": {"w": "same"}}, "rng": rng}
        )
    a, ac = read_phase(tmp_path, "clean_a")
    b, bc = read_phase(tmp_path, "clean_b")
    assert a == b and ac == bc
