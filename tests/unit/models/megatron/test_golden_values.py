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
"""CPU tests for the golden-value batch, fingerprints and comparison."""

import copy

import pytest
import torch

from nemo_rl.models.megatron import golden_values
from nemo_rl.models.megatron.golden_values import (
    compare_goldens,
    fingerprint_named_tensors,
    read_golden_file,
    synthetic_token_batch,
    tensor_fingerprint,
    write_golden_file,
)


def test_synthetic_batch_is_seeded_and_skips_excluded_ids():
    kwargs = dict(num_rows=4, seq_length=64, token_low=10, token_high=20)
    first = synthetic_token_batch(seed=7, excluded_token_ids=[11, 12], **kwargs)
    torch.testing.assert_close(
        first, synthetic_token_batch(seed=7, excluded_token_ids=[11, 12], **kwargs)
    )
    assert not torch.equal(
        first, synthetic_token_batch(seed=8, excluded_token_ids=[11, 12], **kwargs)
    )
    assert first.min() >= 10 and first.max() < 20
    assert not torch.isin(first, torch.tensor([11, 12])).any()
    with pytest.raises(ValueError, match="excluded"):
        synthetic_token_batch(seed=7, excluded_token_ids=[10], **kwargs)


def test_fingerprint_is_linear_and_layout_independent(monkeypatch):
    # Small chunks, so the chunked accumulation itself is exercised.
    monkeypatch.setattr(golden_values, "_CHUNK_NUMEL", 7)
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(6, 5, generator=generator)
    update = torch.randn(6, 5, generator=generator)

    before, after = tensor_fingerprint(weight), tensor_fingerprint(weight + update)
    delta = tensor_fingerprint(update)
    assert after["sum"] - before["sum"] == pytest.approx(delta["sum"])
    for a, b, d in zip(after["proj"], before["proj"], delta["proj"]):
        assert a - b == pytest.approx(d)
    assert before["sumsq"] == pytest.approx(weight.double().square().sum().item())
    # Flat-index based: a reshaped (or re-sharded and gathered) tensor matches.
    assert tensor_fingerprint(weight.reshape(-1))["proj"] == pytest.approx(
        before["proj"]
    )


def test_fingerprint_projection_sees_a_sign_flip():
    """Adam's first step is about lr * sign(grad); a flipped sign must show."""
    update = torch.full((256,), 1e-3)
    flipped = update.clone()
    flipped[::2] *= -1
    assert tensor_fingerprint(update)["sumsq"] == pytest.approx(
        tensor_fingerprint(flipped)["sumsq"]
    )
    assert tensor_fingerprint(update)["proj"] != pytest.approx(
        tensor_fingerprint(flipped)["proj"], rel=0.1
    )


def _golden(updates: dict[str, torch.Tensor], loss=2.0, grad_norm=1.5):
    weights = {"a.weight": torch.ones(4, 4), "b.weight": torch.zeros(8)}
    return {
        "format_version": golden_values.GOLDEN_FORMAT_VERSION,
        "steps": [{"loss": loss, "grad_norm": grad_norm}],
        "weights": [
            fingerprint_named_tensors(weights.items()),
            fingerprint_named_tensors(
                (name, weight + updates[name]) for name, weight in weights.items()
            ),
        ],
    }


def test_compare_passes_within_tolerance_and_flags_mismatches():
    updates = {"a.weight": torch.full((4, 4), 0.1), "b.weight": torch.full((8,), -0.2)}
    golden = _golden(updates)

    close = _golden(
        {name: update * 1.001 for name, update in updates.items()},
        loss=2.001,
        grad_norm=1.51,
    )
    mismatches, worst = compare_goldens(golden, close)
    assert mismatches == []
    assert 0 < worst["weight_update"] < 0.01

    # A rank-local update: half of b.weight moves the other way.
    wrong = copy.deepcopy(updates)
    wrong["b.weight"][:4] *= -1
    mismatches, _ = compare_goldens(golden, _golden(wrong, grad_norm=1.0))
    flagged = {(m.metric, m.name) for m in mismatches}
    assert {
        ("grad_norm", None),
        ("weight_update", None),
        ("weight_update_per_tensor", "b.weight"),
    } <= flagged
    assert ("weight_update_per_tensor", "a.weight") not in flagged
    assert all(str(m) for m in mismatches)


def test_compare_flags_different_initial_weights():
    updates = {"a.weight": torch.full((4, 4), 0.1), "b.weight": torch.full((8,), -0.2)}
    golden = _golden(updates)
    # A check run that loaded another model but applied the same update.
    stale = copy.deepcopy(golden)
    stale["weights"] = [
        fingerprint_named_tensors(weights.items())
        for weights in (
            {"a.weight": torch.ones(4, 4) * 1.01, "b.weight": torch.zeros(8)},
            {
                "a.weight": torch.ones(4, 4) * 1.01 + updates["a.weight"],
                "b.weight": updates["b.weight"],
            },
        )
    ]
    mismatches, worst = compare_goldens(golden, stale)
    assert [(m.metric, m.step, m.name) for m in mismatches] == [
        ("initial_weights", 0, "a.weight")
    ]
    assert "before the first step" in str(mismatches[0])
    assert compare_goldens(golden, golden)[1]["initial_weights"] == 0


def test_compare_rejects_different_weight_names():
    golden = _golden({"a.weight": torch.zeros(4, 4), "b.weight": torch.zeros(8)})
    other = copy.deepcopy(golden)
    for snapshot in other["weights"]:
        snapshot["c.weight"] = snapshot.pop("b.weight")
    with pytest.raises(ValueError, match="names differ"):
        compare_goldens(golden, other)


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_golden_file_round_trip(tmp_path, suffix):
    golden = _golden({"a.weight": torch.zeros(4, 4), "b.weight": torch.zeros(8)})
    path = tmp_path / f"golden{suffix}"
    write_golden_file(path, golden)
    assert read_golden_file(path) == golden
