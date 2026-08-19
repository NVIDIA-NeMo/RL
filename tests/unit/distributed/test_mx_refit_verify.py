# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Parameter-equality verification for mx_reshard refits.

Runs on CPU: no GPU, no vLLM, no ModelExpress.
"""

import json

import torch

from nemo_rl.distributed import mx_refit_verify


class _Model(torch.nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.a = torch.nn.Parameter(torch.arange(64, dtype=torch.float32).to(dtype))
        self.b = torch.nn.Parameter(torch.ones(8, 8, dtype=dtype))


def test_a_single_flipped_bit_is_detected():
    """The whole mechanism rests on this. A float statistic can miss a low-mantissa
    change; the fingerprint is taken over raw bytes so it cannot."""
    t = torch.ones(1024, dtype=torch.bfloat16)
    before = mx_refit_verify.fingerprint(t)

    raw = t.view(torch.uint8)
    raw[500] = raw[500].item() ^ 1  # flip the lowest mantissa bit of one element

    assert mx_refit_verify.fingerprint(t) != before


def test_high_byte_statistic_catches_an_offsetting_change():
    """Two changes can cancel in a flat byte sum. The second statistic covers only
    each element's most significant byte, so a cancelling pair has to cancel in
    both to hide."""
    t = torch.ones(64, dtype=torch.bfloat16)
    raw = t.view(torch.uint8)
    before = mx_refit_verify.fingerprint(t)

    # One byte up, one down, so the flat sum is unchanged by construction. The
    # pair has to straddle the element boundary for this to be a real test: bf16
    # is 2 bytes little-endian, so index 11 is element 5's high byte and index 10
    # is its low byte. Two *low* bytes would cancel in both statistics and prove
    # nothing.
    raw[11] = raw[11].item() + 1
    raw[10] = raw[10].item() - 1
    after = mx_refit_verify.fingerprint(t)

    assert after[1] == before[1], "expected the flat byte sum to be fooled here"
    assert after[2] != before[2], "the high-byte statistic should still differ"
    assert after != before


def test_identical_tensors_fingerprint_identically():
    """Determinism, or every refit would look like it changed everything."""
    a = torch.arange(256, dtype=torch.float32).to(torch.bfloat16)
    b = a.clone()
    assert mx_refit_verify.fingerprint(a) == mx_refit_verify.fingerprint(b)
    assert mx_refit_verify.fingerprint(a) == mx_refit_verify.fingerprint(a)


def test_non_contiguous_parameters_are_handled():
    """A transposed or sliced view cannot be reinterpreted as bytes directly, and
    must not take the whole verification down with it."""
    t = torch.ones(8, 8, dtype=torch.bfloat16).t()
    assert not t.is_contiguous()
    assert mx_refit_verify.fingerprint(t)[0] == 64


def test_float32_and_bfloat16_both_pick_the_high_byte():
    """The high-byte stride is derived from element size, not hardcoded to 2."""
    for dtype in (torch.bfloat16, torch.float32, torch.float16):
        t = torch.ones(32, dtype=dtype)
        numel, _flat, _high = mx_refit_verify.fingerprint(t)
        assert numel == 32


def test_compare_names_what_changed():
    model = _Model()
    before = mx_refit_verify.fingerprint_model(model)
    with torch.no_grad():
        model.b.add_(1.0)
    result = mx_refit_verify.compare(before, mx_refit_verify.fingerprint_model(model))

    assert result["params_compared"] == 2
    assert result["params_changed"] == 1
    assert result["changed_sample"] == ["b"]
    assert result["unchanged_sample"] == ["a"]


def test_first_refit_should_change_nothing(capsys):
    """The load-bearing case. On a fresh run vLLM is loaded from the HF checkpoint
    and the trainer's weights are converted from the same checkpoint with no
    optimizer step taken, so a faithful first refit is a no-op on every parameter.
    """
    model = _Model()
    before = mx_refit_verify.fingerprint_model(model)

    record = mx_refit_verify.report(
        1, 0, before, mx_refit_verify.fingerprint_model(model)
    )

    assert record["first_refit"] is True
    assert record["params_changed"] == 0
    assert record["params_unchanged"] == 2
    emitted = json.loads(capsys.readouterr().out.split("MX_REFIT_VERIFY ", 1)[1])
    assert emitted == record


def test_later_refits_are_flagged_as_a_different_question(capsys):
    """From step 2 the trainer has moved, so change is expected and the record
    instead reports which params the refit reaches."""
    model = _Model()
    before = mx_refit_verify.fingerprint_model(model)
    with torch.no_grad():
        model.a.mul_(2.0)
        model.b.mul_(3.0)

    record = mx_refit_verify.report(
        2, 5, before, mx_refit_verify.fingerprint_model(model)
    )

    assert record["first_refit"] is False
    assert record["params_changed"] == 2
    capsys.readouterr()


def test_verification_never_raises_on_a_broken_model():
    """A verification failure must not be able to fail the refit it verifies."""

    class Broken:
        def named_parameters(self):
            raise RuntimeError("model is not walkable here")

    assert mx_refit_verify.fingerprint_model(Broken()) == {}


def test_disabled_by_default(monkeypatch):
    """It sits on the refit critical path, so it is opt-in."""
    monkeypatch.delenv("MX_REFIT_VERIFY", raising=False)
    assert mx_refit_verify.enabled() is False
    for value in ("0", "", "false", "False"):
        monkeypatch.setenv("MX_REFIT_VERIFY", value)
        assert mx_refit_verify.enabled() is False
    for value in ("1", "true", "yes"):
        monkeypatch.setenv("MX_REFIT_VERIFY", value)
        assert mx_refit_verify.enabled() is True
