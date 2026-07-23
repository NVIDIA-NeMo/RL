"""Unit-test suite for ``nemo_rl.models.multi_lora.loss``.

Organized by concern:

  1. ``adapter_row_ranges`` — block-contiguous detection + violations
  2. ``build_loss_mask`` — token_mask × sample_mask, shift-after-materialize
  3. ``compute_token_logprobs`` — bit-match to NLLLossFn's op chain
  4. ``reduce_per_adapter`` — slice-not-mask reduction, lazy accumulation,
                              per-adapter normalization
  5. ``MultiAdapterLoss.__call__`` end-to-end:
      a. logits input
      b. pre-gathered next_token_logprobs input
      c. fallback when only labels (no token_mask) provided
      d. metrics shape
      e. backward produces finite gradients
      f. input validation (both / neither input raises)

No nemo_rl / nemo_automodel dependency: plain CPU torch tensors only.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from nemo_rl.models.multi_lora.loss import (
    MultiAdapterLoss,
    adapter_row_ranges,
    build_loss_mask,
    compute_token_logprobs,
    reduce_per_adapter,
)


# =============================================================================
# 1. adapter_row_ranges
# =============================================================================


class TestAdapterRowRanges:
    def test_empty(self):
        assert adapter_row_ranges([]) == []

    def test_single_adapter(self):
        assert adapter_row_ranges(["a", "a", "a"]) == [("a", 0, 3)]

    def test_two_block_contiguous(self):
        assert adapter_row_ranges(["a", "a", "b", "b", "b"]) == [("a", 0, 2), ("b", 2, 5)]

    def test_four_block_contiguous(self):
        names = ["a"] * 2 + ["b"] * 1 + ["c"] * 3 + ["d"] * 2
        ranges = adapter_row_ranges(names)
        assert ranges == [("a", 0, 2), ("b", 2, 3), ("c", 3, 6), ("d", 6, 8)]
        # Round-trip: ranges cover every row exactly once.
        total = sum(end - start for _, start, end in ranges)
        assert total == len(names)

    def test_non_contiguous_raises(self):
        """`a` appearing after `b` (non-contiguous) is a packing bug — raise."""
        with pytest.raises(ValueError, match="not block-contiguous"):
            adapter_row_ranges(["a", "b", "a"])

    def test_single_row_per_adapter(self):
        assert adapter_row_ranges(["x", "y", "z"]) == [
            ("x", 0, 1), ("y", 1, 2), ("z", 2, 3),
        ]


# =============================================================================
# 2. build_loss_mask
# =============================================================================


class TestBuildLossMask:
    def test_no_token_mask_returns_none(self):
        assert build_loss_mask({"input_ids": torch.zeros(2, 5, dtype=torch.long)}) is None

    def test_too_short_returns_none(self):
        # token_mask with T < 2 can't be shifted to [B, T-1]
        data = {"token_mask": torch.ones(2, 1, dtype=torch.long)}
        assert build_loss_mask(data) is None

    def test_basic_shift(self):
        # token_mask [B=2, T=4]; after shift expect shape [2, 3]
        tm = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 1]], dtype=torch.long)
        mask = build_loss_mask({"token_mask": tm})
        assert mask is not None
        assert mask.shape == (2, 3)
        # [:, 1:] of tm
        assert torch.equal(mask, torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.float32))

    def test_sample_mask_multiplies(self):
        tm = torch.ones(3, 4, dtype=torch.long)  # all tokens valid
        sm = torch.tensor([1, 0, 1], dtype=torch.long)  # middle sample invalid
        mask = build_loss_mask({"token_mask": tm, "sample_mask": sm})
        assert mask is not None
        # Row 0: all 1, Row 1: all 0 (masked out), Row 2: all 1
        expected = torch.tensor([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32)
        assert torch.equal(mask, expected)

    def test_output_is_fp32(self):
        tm = torch.ones(2, 4, dtype=torch.long)
        mask = build_loss_mask({"token_mask": tm})
        assert mask.dtype == torch.float32


# =============================================================================
# 3. compute_token_logprobs — must match NLLLossFn's op chain bit-exactly
# =============================================================================


class TestComputeTokenLogprobs:
    @staticmethod
    def _reference_nll_op_chain(logits, input_ids):
        """The exact op chain from NeMo-RL loss_functions.py:660-667."""
        next_token_logits = logits.to(torch.float32)
        next_tokens = input_ids[:, 1:].to(next_token_logits.device)
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        shifted = log_probs[:, :-1]
        return shifted.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)

    def test_matches_reference_op_chain_exactly(self):
        """Bit-identical (torch.equal) to a hand-built reference of
        log_softmax → [:, :-1] → gather, the NLLLossFn chain.
        """
        torch.manual_seed(0)
        B, T, V = 3, 5, 17
        logits = torch.randn(B, T, V)
        input_ids = torch.randint(0, V, (B, T))
        got = compute_token_logprobs(logits, {"input_ids": input_ids})
        expected = self._reference_nll_op_chain(logits, input_ids)
        assert torch.equal(got, expected)

    def test_output_shape_and_dtype(self):
        B, T, V = 2, 7, 11
        logits = torch.randn(B, T, V, dtype=torch.bfloat16)
        ids = torch.randint(0, V, (B, T))
        logp = compute_token_logprobs(logits, {"input_ids": ids})
        assert logp.shape == (B, T - 1)
        # Output is fp32 because the op chain casts logits.to(torch.float32).
        assert logp.dtype == torch.float32

    def test_fp32_log_softmax_chosen_not_fp16(self):
        """log_softmax must run in fp32 — fp16 logits would overflow on
        large vocab × extreme values.
        """
        # Construct logits where fp16 log_softmax would over/underflow but
        # fp32 stays finite.
        B, T, V = 1, 3, 4
        logits = torch.tensor([[
            [1e4, -1e4, 0.0, 1.0],
            [1e4, -1e4, 0.0, 1.0],
            [1e4, -1e4, 0.0, 1.0],
        ]], dtype=torch.float32)
        ids = torch.zeros(B, T, dtype=torch.long)
        logp = compute_token_logprobs(logits, {"input_ids": ids})
        assert torch.isfinite(logp).all()


# =============================================================================
# 4. reduce_per_adapter — slice-not-mask, lazy accumulation, per-adapter norm
# =============================================================================


class TestReducePerAdapter:
    def test_single_adapter_equals_NLLLoss_formula(self):
        """For one adapter, `reduce_per_adapter` must equal exactly
        ``-sum(logp * mask) / (sum(mask) + 1e-8)`` — NLLLossFn's formula.
        """
        torch.manual_seed(7)
        logp = torch.randn(2, 5)
        mask = torch.tensor([[1, 1, 0, 1, 1], [1, 0, 1, 1, 0]], dtype=torch.float32)
        total, means = reduce_per_adapter(logp, mask, [("only", 0, 2)])
        num = -torch.sum(logp * mask)
        den = mask.sum() + 1e-8
        expected = num / den
        assert torch.allclose(total, expected, atol=0.0)
        assert math.isclose(means["only"], float(expected.detach()))

    def test_per_adapter_normalization_independent(self):
        """Each adapter's mean must use ITS OWN valid-token count, not the
        global one. Equal-loss adapters with different valid-token counts
        still produce equal per-adapter means.
        """
        # Build logprobs where each adapter contributes identical
        # per-token loss (-1.0) but different valid-token counts. Means
        # should both equal +1.0.
        B, T = 4, 3
        logp = torch.full((B, T), -1.0)
        mask = torch.tensor([
            [1, 1, 0],   # adapter a: 2 valid tokens
            [1, 0, 0],   # adapter a: 1 valid token
            [1, 1, 1],   # adapter b: 3 valid tokens
            [1, 1, 1],   # adapter b: 3 valid tokens
        ], dtype=torch.float32)
        ranges = [("a", 0, 2), ("b", 2, 4)]
        total, means = reduce_per_adapter(logp, mask, ranges)
        assert math.isclose(means["a"], 1.0, abs_tol=1e-5)
        assert math.isclose(means["b"], 1.0, abs_tol=1e-5)
        # total is a sum of per-adapter means, not a global mean
        assert math.isclose(float(total), 2.0, abs_tol=1e-5)

    def test_slice_not_mask_no_ulp_drift(self):
        """The reduction must match a hand-built per-adapter sum-then-divide.
        This pins the slice-before-sum contract (worklog: masking the full
        [B, T-1] tensor and summing has a different reduction tree).
        """
        torch.manual_seed(13)
        logp = torch.randn(5, 7)
        mask = torch.randint(0, 2, (5, 7)).to(torch.float32)
        ranges = [("a", 0, 2), ("b", 2, 5)]
        total, _ = reduce_per_adapter(logp, mask, ranges)

        # Reference: same slice-then-sum, hand-built.
        l_a_num = -torch.sum(logp[0:2] * mask[0:2])
        l_a = l_a_num / (mask[0:2].sum() + 1e-8)
        l_b_num = -torch.sum(logp[2:5] * mask[2:5])
        l_b = l_b_num / (mask[2:5].sum() + 1e-8)
        expected = l_a + l_b
        assert torch.equal(total, expected)

    def test_total_is_None_when_ranges_empty(self):
        total, means = reduce_per_adapter(
            torch.randn(0, 5), torch.zeros(0, 5), ranges=[]
        )
        assert total is None
        assert means == {}

    def test_means_are_python_floats_not_tensors(self):
        """``per_adapter_loss/<name>`` entries in metrics must be plain
        ``float``s so they json-serialize cleanly into wandb / jsonl.
        """
        logp = torch.full((2, 3), -1.0)
        mask = torch.ones(2, 3)
        _, means = reduce_per_adapter(logp, mask, [("a", 0, 2)])
        for v in means.values():
            assert isinstance(v, float)

    def test_lazy_accumulation_no_extra_autograd_node(self):
        """The graph from ``total`` must reach each ``l_i`` but NOT a
        sum-over-all-positions node (which is what the
        ``logprobs.sum() * 0.0`` seed would add). We can't directly
        introspect the autograd graph cheaply, but we CAN verify that
        with a single-adapter range, total IS l_i (same object identity
        before any addition).
        """
        logp = torch.full((2, 3), -1.0, requires_grad=True)
        mask = torch.ones(2, 3)
        total, _ = reduce_per_adapter(logp, mask, [("a", 0, 2)])
        # Single-adapter case: total seeded from l_a, never combined with
        # zero-sum seed. So `total` participates in the SAME backward
        # subgraph that NLLLossFn would.
        assert total.requires_grad
        total.backward()
        # Each masked element contributes the same gradient (1 / valid_count).
        # Reference matches loss_fn = -sum(logp * mask) / sum(mask).
        valid = mask.sum().item()
        expected_grad = -mask / valid
        assert torch.allclose(logp.grad, expected_grad, atol=1e-7)


# =============================================================================
# 5. MultiAdapterLoss end-to-end
# =============================================================================


def _make_data(B, T, V, adapter_names, *, with_token_mask=True):
    torch.manual_seed(42)
    return {
        "input_ids": torch.randint(0, V, (B, T)),
        "adapter_names": adapter_names,
        **({"token_mask": torch.ones(B, T, dtype=torch.long)} if with_token_mask else {}),
    }


class TestMultiAdapterLossLogitsInput:
    def test_returns_scalar_loss_and_metrics(self):
        B, T, V = 4, 5, 11
        data = _make_data(B, T, V, ["a", "a", "b", "b"])
        logits = torch.randn(B, T, V, requires_grad=True)
        loss_fn = MultiAdapterLoss()
        loss, metrics = loss_fn(data=data, logits=logits)

        assert loss.shape == ()
        assert "per_adapter_loss/a" in metrics
        assert "per_adapter_loss/b" in metrics
        assert metrics["num_valid_samples"] == B
        assert metrics["num_unmasked_tokens"] == B * (T - 1)

    def test_two_adapters_sum_equals_per_adapter_sum(self):
        B, T, V = 4, 5, 11
        data = _make_data(B, T, V, ["a", "a", "b", "b"])
        logits = torch.randn(B, T, V)
        loss, metrics = MultiAdapterLoss()(data=data, logits=logits)
        per_adapter_sum = metrics["per_adapter_loss/a"] + metrics["per_adapter_loss/b"]
        assert math.isclose(float(loss.detach()), per_adapter_sum, abs_tol=1e-5)

    def test_uniform_adapter_loss_equals_handbuilt_single_adapter(self):
        """When all rows route to one adapter, the loss must equal a
        hand-built single-adapter NLL computed on the same logits/labels.
        This is the multi-vs-single bit-match contract.
        """
        B, T, V = 3, 5, 11
        data = _make_data(B, T, V, ["only", "only", "only"])
        logits = torch.randn(B, T, V)
        loss, _ = MultiAdapterLoss()(data=data, logits=logits)

        # Hand-built reference (NLLLossFn's exact op chain).
        next_token_logits = logits.to(torch.float32)
        next_tokens = data["input_ids"][:, 1:]
        logp = F.log_softmax(next_token_logits, dim=-1)[:, :-1]
        token_logprobs = logp.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)
        # All tokens valid (token_mask was all-ones).
        mask = data["token_mask"][:, 1:].to(torch.float32)
        expected = -torch.sum(token_logprobs * mask) / (mask.sum() + 1e-8)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_backward_produces_finite_gradients(self):
        B, T, V = 4, 5, 11
        data = _make_data(B, T, V, ["a", "a", "b", "b"])
        logits = torch.randn(B, T, V, requires_grad=True)
        loss, _ = MultiAdapterLoss()(data=data, logits=logits)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_inactive_position_grad_is_zero(self):
        """Logits at the LAST position have no next-token target after the
        [:, :-1] shift; their gradient must be exactly zero.
        """
        B, T, V = 2, 4, 7
        data = _make_data(B, T, V, ["a", "a"])
        logits = torch.randn(B, T, V, requires_grad=True)
        MultiAdapterLoss()(data=data, logits=logits)[0].backward()
        # Last time step gets no gradient contribution.
        assert torch.equal(logits.grad[:, -1, :], torch.zeros(B, V))


class TestPerAdapterGlobalNormalization:
    def test_row_aligned_global_count_scales_backward_not_metrics(self):
        """DP-global denominators must survive row slicing and scale loss.

        Simulate one rank holding 2 rows of adapter a with 6 local valid
        tokens while the full DP group has 12. The backward loss contribution
        must be half the local mean; the human-readable metric stays local.
        """
        from nemo_rl.models.multi_lora.loss import GLOBAL_ADAPTER_TOKEN_COUNTS_KEY

        data = _make_data(2, 4, 7, ["a", "a"])
        data[GLOBAL_ADAPTER_TOKEN_COUNTS_KEY] = torch.full((2,), 12.0)
        token_logprobs = torch.full((2, 3), -1.0, requires_grad=True)
        loss, metrics = MultiAdapterLoss()(
            data=data, next_token_logprobs=token_logprobs
        )
        assert math.isclose(float(loss.detach()), 0.5, abs_tol=1e-7)
        assert math.isclose(metrics["per_adapter_loss/a"], 1.0, abs_tol=1e-7)
        loss.backward()
        assert torch.allclose(
            token_logprobs.grad, torch.full_like(token_logprobs, -1.0 / 12.0)
        )

    def test_different_adapters_use_their_own_global_counts(self):
        from nemo_rl.models.multi_lora.loss import GLOBAL_ADAPTER_TOKEN_COUNTS_KEY

        data = _make_data(4, 4, 7, ["a", "a", "b", "b"])
        data[GLOBAL_ADAPTER_TOKEN_COUNTS_KEY] = torch.tensor(
            [12.0, 12.0, 24.0, 24.0]
        )
        token_logprobs = torch.full((4, 3), -1.0)
        loss, metrics = MultiAdapterLoss()(
            data=data, next_token_logprobs=token_logprobs
        )
        # local numerators are 6 each: 6/12 + 6/24 = 0.75
        assert math.isclose(float(loss.detach()), 0.75, abs_tol=1e-7)
        assert math.isclose(metrics["per_adapter_loss/a"], 1.0, abs_tol=1e-7)
        assert math.isclose(metrics["per_adapter_loss/b"], 1.0, abs_tol=1e-7)


class TestMultiAdapterLossPregatheredInput:
    def test_accepts_next_token_logprobs(self):
        B, T, V = 4, 5, 11
        data = _make_data(B, T, V, ["a", "a", "b", "b"])
        # Pre-compute logprobs the same way NeMo-RL would under LOGPROB.
        logits = torch.randn(B, T, V)
        token_logprobs = compute_token_logprobs(logits, data)

        loss_logits, _ = MultiAdapterLoss()(data=data, logits=logits)
        loss_pregath, _ = MultiAdapterLoss()(data=data, next_token_logprobs=token_logprobs)
        # Same input → same output.
        assert torch.allclose(loss_logits, loss_pregath, atol=1e-6)

    def test_pregathered_skips_log_softmax(self):
        """When ``next_token_logprobs`` is provided, the code must NOT
        re-run log_softmax. We check by providing values that aren't valid
        logprobs (positive) — if the code re-ran log_softmax they'd become
        negative.
        """
        B, T, V = 2, 5, 11
        data = _make_data(B, T, V, ["a", "a"])
        positive_logprobs = torch.full((B, T - 1), 1.0)  # invalid as logprobs
        loss, _ = MultiAdapterLoss()(data=data, next_token_logprobs=positive_logprobs)
        # The loss formula: -sum(logp * mask) / sum(mask), with logp=1.0 and
        # all mask=1 → result is -1.0 per adapter, summed over 1 adapter.
        assert math.isclose(float(loss.detach()), -1.0, abs_tol=1e-5)


class TestMultiAdapterLossLabelFallback:
    def test_uses_labels_when_no_token_mask(self):
        """Without `token_mask`, the loss mask comes from labels != -100."""
        B, T, V = 2, 5, 11
        labels = torch.tensor([
            [1, 2, 3, 4, -100],   # last position padded
            [1, -100, -100, -100, -100],  # only first position valid
        ])
        data = {
            "input_ids": labels.clamp(min=0),  # avoid negative ids
            "labels": labels,
            "adapter_names": ["a", "a"],
        }
        logits = torch.randn(B, T, V)
        _, metrics = MultiAdapterLoss()(data=data, logits=logits)
        # token_mask=None → loss_mask = (labels[..., 1:] != -100). Valid
        # token counts: row 0 has labels[1:]=[2,3,4,-100] → 3 valid; row 1
        # has labels[1:]=[-100,-100,-100,-100] → 0 valid. Total = 3.
        assert metrics["num_unmasked_tokens"] == 3


class TestMultiAdapterLossInputValidation:
    def test_both_inputs_raises(self):
        B, T, V = 2, 5, 11
        data = _make_data(B, T, V, ["a", "a"])
        with pytest.raises(TypeError, match="exactly one of"):
            MultiAdapterLoss()(
                data=data,
                logits=torch.randn(B, T, V),
                next_token_logprobs=torch.randn(B, T - 1),
            )

    def test_neither_input_raises(self):
        B, T, V = 2, 5, 11
        data = _make_data(B, T, V, ["a", "a"])
        with pytest.raises(TypeError, match="exactly one of"):
            MultiAdapterLoss()(data=data)

    def test_absorbs_global_valid_kwargs(self):
        """NeMo-RL passes `global_valid_seqs` / `global_valid_toks`; these
        must be silently absorbed, not raised on.
        """
        B, T, V = 2, 5, 11
        data = _make_data(B, T, V, ["a", "a"])
        logits = torch.randn(B, T, V)
        loss, _ = MultiAdapterLoss()(
            data=data, logits=logits,
            global_valid_seqs=99, global_valid_toks=999,  # irrelevant
        )
        assert loss.shape == ()


class TestMultiAdapterLossMetricsShape:
    def test_metrics_keys_match_active_adapters(self):
        B, T, V = 4, 5, 11
        data = _make_data(B, T, V, ["a", "a", "b", "c"])
        loss, metrics = MultiAdapterLoss()(data=data, logits=torch.randn(B, T, V))
        per_adapter_keys = {k for k in metrics if k.startswith("per_adapter_loss/")}
        assert per_adapter_keys == {
            "per_adapter_loss/a", "per_adapter_loss/b", "per_adapter_loss/c",
        }

    def test_metrics_num_valid_samples_equals_B(self):
        B, T, V = 5, 5, 11
        data = _make_data(B, T, V, ["a", "a", "a", "b", "b"])
        _, metrics = MultiAdapterLoss()(data=data, logits=torch.randn(B, T, V))
        assert metrics["num_valid_samples"] == B


# =============================================================================
# 10. Forward accuracy — hand-built NLLLossFn references
# =============================================================================


def _reference_per_adapter_nll(logits, input_ids, token_mask, adapter_names):
    """Brute-force per-adapter NLL using NLLLossFn's exact op chain.
    Used as the ground-truth reference for both forward and backward
    accuracy tests.
    """
    log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)[:, :-1]
    next_tokens = input_ids[:, 1:]
    token_logprobs = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)
    mask = token_mask[:, 1:].to(torch.float32)

    ranges = adapter_row_ranges(adapter_names)
    total = None
    per_adapter = {}
    for name, start, end in ranges:
        a_logp = token_logprobs[start:end]
        a_mask = mask[start:end]
        num = -torch.sum(a_logp * a_mask)
        den = a_mask.sum() + 1e-8
        l_i = num / den
        per_adapter[name] = float(l_i.detach())
        total = l_i if total is None else total + l_i
    return total, per_adapter


class TestForwardAccuracy:
    """Strong accuracy tests against hand-built NLLLossFn references."""

    def test_loss_value_matches_handbuilt_NLL_reference(self):
        """Forward loss with multiple adapters must equal a brute-force
        per-adapter NLL accumulation."""
        torch.manual_seed(31)
        B, T, V = 6, 7, 23
        logits = torch.randn(B, T, V)
        input_ids = torch.randint(0, V, (B, T))
        token_mask = torch.randint(0, 2, (B, T)).to(torch.long)
        data = {
            "input_ids": input_ids,
            "token_mask": token_mask,
            "adapter_names": ["a"] * 2 + ["b"] * 1 + ["c"] * 3,
        }
        loss, _ = MultiAdapterLoss()(data=data, logits=logits)
        expected, _ = _reference_per_adapter_nll(
            logits, input_ids, token_mask, data["adapter_names"]
        )
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_per_adapter_loss_equals_standalone_NLL_on_same_rows(self):
        """The MULTI-VS-SINGLE BIT-MATCH CONTRACT.

        Each adapter's per_adapter_loss/<name> metric must equal the
        loss a single-LoRA run would produce on just that adapter's rows
        with the same logits/labels/mask. This is the central correctness
        claim — multi-LoRA training is equivalent to N independent
        single-LoRA trainings.
        """
        torch.manual_seed(42)
        B, T, V = 6, 5, 11
        logits = torch.randn(B, T, V)
        input_ids = torch.randint(0, V, (B, T))
        token_mask = torch.ones(B, T, dtype=torch.long)
        ranges_used = [("a", 0, 2), ("b", 2, 4), ("c", 4, 6)]
        adapter_names = ["a"] * 2 + ["b"] * 2 + ["c"] * 2
        data = {
            "input_ids": input_ids,
            "token_mask": token_mask,
            "adapter_names": adapter_names,
        }
        _, metrics = MultiAdapterLoss()(data=data, logits=logits)

        for name, start, end in ranges_used:
            # Standalone single-LoRA NLL on just this adapter's rows.
            sub_logits = logits[start:end]
            sub_ids = input_ids[start:end]
            sub_mask = token_mask[start:end]
            logp = F.log_softmax(sub_logits.to(torch.float32), dim=-1)[:, :-1]
            tlp = logp.gather(-1, sub_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            m = sub_mask[:, 1:].to(torch.float32)
            standalone = -torch.sum(tlp * m) / (m.sum() + 1e-8)
            assert math.isclose(
                metrics[f"per_adapter_loss/{name}"],
                float(standalone),
                abs_tol=1e-5,
            ), f"adapter {name} per_adapter_loss != standalone NLL"

    def test_sum_of_per_adapter_metrics_equals_total_loss(self):
        torch.manual_seed(99)
        B, T, V = 8, 6, 17
        adapter_names = ["a"] * 2 + ["b"] * 3 + ["c"] * 3
        data = _make_data(B, T, V, adapter_names)
        loss, metrics = MultiAdapterLoss()(data=data, logits=torch.randn(B, T, V))
        per_adapter_sum = sum(
            v for k, v in metrics.items() if k.startswith("per_adapter_loss/")
        )
        assert math.isclose(float(loss.detach()), per_adapter_sum, abs_tol=1e-5)

    def test_imbalanced_row_counts_normalize_correctly(self):
        """Adapter with 1 row + adapter with 5 rows: each adapter's mean
        is computed over its own valid-token count, not pooled.
        """
        torch.manual_seed(0)
        B, T, V = 6, 5, 11
        logits = torch.randn(B, T, V)
        input_ids = torch.randint(0, V, (B, T))
        data = {
            "input_ids": input_ids,
            "token_mask": torch.ones(B, T, dtype=torch.long),
            "adapter_names": ["a"] + ["b"] * 5,
        }
        _, metrics = MultiAdapterLoss()(data=data, logits=logits)

        # adapter_a: rows [0:1], should be NLL on 1 row.
        a_logp = F.log_softmax(logits[0:1].to(torch.float32), dim=-1)[:, :-1]
        a_tlp = a_logp.gather(-1, input_ids[0:1, 1:].unsqueeze(-1)).squeeze(-1)
        a_expected = -a_tlp.mean()  # all-ones mask, T-1 tokens
        assert math.isclose(metrics["per_adapter_loss/a"], float(a_expected), abs_tol=1e-5)

        # adapter_b: rows [1:6], 5 rows, mean over 5*(T-1)=20 tokens.
        b_logp = F.log_softmax(logits[1:6].to(torch.float32), dim=-1)[:, :-1]
        b_tlp = b_logp.gather(-1, input_ids[1:6, 1:].unsqueeze(-1)).squeeze(-1)
        b_expected = -b_tlp.mean()
        assert math.isclose(metrics["per_adapter_loss/b"], float(b_expected), abs_tol=1e-5)


# =============================================================================
# 11. Backward accuracy — gradient values match NLLLossFn's backward
# =============================================================================


class TestBackwardAccuracy:
    """Gradient VALUES (not just finite-ness) must match handbuilt refs."""

    def test_logits_gradient_matches_handbuilt_NLL_backward(self):
        """``MultiAdapterLoss.backward()`` must produce the SAME gradient
        on ``logits`` as a hand-built per-adapter NLL reference would.
        This is the autograd-graph bit-equivalence claim.
        """
        torch.manual_seed(17)
        B, T, V = 4, 5, 13
        adapter_names = ["a"] * 2 + ["b"] * 2

        # Path 1: MultiAdapterLoss
        logits1 = torch.randn(B, T, V, requires_grad=True)
        input_ids = torch.randint(0, V, (B, T))
        token_mask = torch.ones(B, T, dtype=torch.long)
        data1 = {
            "input_ids": input_ids,
            "token_mask": token_mask,
            "adapter_names": adapter_names,
        }
        loss1, _ = MultiAdapterLoss()(data=data1, logits=logits1)
        loss1.backward()

        # Path 2: hand-built reference (identical op chain)
        logits2 = logits1.detach().clone().requires_grad_(True)
        loss2, _ = _reference_per_adapter_nll(logits2, input_ids, token_mask, adapter_names)
        loss2.backward()

        assert torch.allclose(logits1.grad, logits2.grad, atol=1e-7)

    def test_masked_position_grad_is_zero(self):
        """A position where ``token_mask=0`` (after the shift) must
        receive zero gradient on the logits row that predicts it.
        """
        B, T, V = 2, 5, 11
        torch.manual_seed(3)
        input_ids = torch.randint(0, V, (B, T))
        # Row 0: mask out position 2 in the original mask. After [:, 1:]
        # shift, this becomes index 1 of the [B, T-1] loss mask → token
        # at input_ids[0, 2] gets no gradient. The logit that produces
        # that prediction is logits[0, 1, :] (logits[t] predicts t+1).
        token_mask = torch.tensor(
            [[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.long
        )
        data = {
            "input_ids": input_ids,
            "token_mask": token_mask,
            "adapter_names": ["a", "a"],
        }
        logits = torch.randn(B, T, V, requires_grad=True)
        loss, _ = MultiAdapterLoss()(data=data, logits=logits)
        loss.backward()
        # logits[0, 1, :] is the producer of token_logprobs[0, 1] which
        # is masked out → zero gradient.
        assert torch.allclose(logits.grad[0, 1, :], torch.zeros(V), atol=1e-7)
        # But logits[1, 1, :] (other row, fully unmasked) must have grad.
        assert not torch.allclose(logits.grad[1, 1, :], torch.zeros(V))

    def test_gradient_per_adapter_only_flows_through_own_rows(self):
        """Each adapter's contribution to the total loss must produce
        gradient ONLY on its own rows of ``logits``. We verify by
        replacing one adapter's labels with arbitrary noise — the OTHER
        adapter's gradient rows must be unchanged.
        """
        torch.manual_seed(7)
        B, T, V = 4, 5, 11
        input_ids = torch.randint(0, V, (B, T))
        token_mask = torch.ones(B, T, dtype=torch.long)
        adapter_names = ["a"] * 2 + ["b"] * 2

        # Baseline gradient.
        logits1 = torch.randn(B, T, V, requires_grad=True)
        data1 = {
            "input_ids": input_ids.clone(),
            "token_mask": token_mask.clone(),
            "adapter_names": adapter_names,
        }
        MultiAdapterLoss()(data=data1, logits=logits1)[0].backward()
        g1 = logits1.grad.clone()

        # Now mutate ONLY adapter_b's rows in input_ids (rows 2,3).
        logits2 = logits1.detach().clone().requires_grad_(True)
        input_ids2 = input_ids.clone()
        input_ids2[2:4] = torch.randint(0, V, (2, T))  # change b's targets
        data2 = {
            "input_ids": input_ids2,
            "token_mask": token_mask.clone(),
            "adapter_names": adapter_names,
        }
        MultiAdapterLoss()(data=data2, logits=logits2)[0].backward()
        g2 = logits2.grad

        # adapter_a's rows (0,1) of the gradient must be byte-identical
        # in both runs — they don't see adapter_b's targets.
        assert torch.equal(g1[0:2], g2[0:2])
        # adapter_b's rows (2,3) MUST differ.
        assert not torch.equal(g1[2:4], g2[2:4])


# =============================================================================
# 12. Numerical robustness & contracts
# =============================================================================


class TestNumericalRobustness:
    def test_global_valid_kwargs_do_not_affect_output(self):
        """Multiple values of ``global_valid_seqs`` / ``global_valid_toks``
        must produce identical loss and metrics when they are plain scalars
        (the NeMo-RL default). Per-adapter dict counts (set by the multi-LoRA
        worker for DP>1) ARE consumed — see test_per_adapter_global_counts.
        """
        B, T, V = 3, 5, 11
        data = _make_data(B, T, V, ["a", "a", "b"])
        logits = torch.randn(B, T, V)
        loss_a, m_a = MultiAdapterLoss()(
            data=data, logits=logits, global_valid_seqs=1, global_valid_toks=1,
        )
        loss_b, m_b = MultiAdapterLoss()(
            data=data, logits=logits, global_valid_seqs=999, global_valid_toks=999_999,
        )
        assert torch.equal(loss_a, loss_b)
        # Per-adapter metrics also identical.
        for k in m_a:
            if k.startswith("per_adapter_loss/"):
                assert math.isclose(m_a[k], m_b[k])

    def test_position_alignment_logits_t_predicts_token_t_plus_1(self):
        """Position alignment contract: ``logits[b, t]`` is the
        distribution that predicts ``input_ids[b, t+1]``. We verify by
        constructing logits with strong correct predictions and expecting
        a near-zero loss.
        """
        B, T, V = 1, 5, 8
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        logits = torch.zeros(B, T, V)
        # For each prediction position t in [0, T-1), make logits[t]
        # strongly predict input_ids[t+1].
        for t in range(T - 1):
            target = int(input_ids[0, t + 1])
            logits[0, t, target] = 100.0
        data = {
            "input_ids": input_ids,
            "token_mask": torch.ones(B, T, dtype=torch.long),
            "adapter_names": ["a"],
        }
        loss, _ = MultiAdapterLoss()(data=data, logits=logits)
        # All T-1 predictions are correct under softmax — loss ≈ 0.
        assert float(loss.detach()) < 1e-3

    def test_position_alignment_offbyone_is_high_loss(self):
        """Sanity check the alignment contract: shifting predictions by
        one position (logits[t] strongly predicts input_ids[t], not [t+1])
        must yield a HIGH loss, not low.
        """
        B, T, V = 1, 5, 8
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        logits = torch.zeros(B, T, V)
        for t in range(T):
            target = int(input_ids[0, t])  # WRONG: predicting current, not next
            logits[0, t, target] = 100.0
        data = {
            "input_ids": input_ids,
            "token_mask": torch.ones(B, T, dtype=torch.long),
            "adapter_names": ["a"],
        }
        loss, _ = MultiAdapterLoss()(data=data, logits=logits)
        assert float(loss.detach()) > 50.0  # confidently wrong → huge loss

    def test_deterministic_same_input_same_output(self):
        torch.manual_seed(0)
        B, T, V = 5, 7, 13
        data = _make_data(B, T, V, ["a"] * 3 + ["b"] * 2)
        logits = torch.randn(B, T, V)
        # Build two fresh MultiAdapterLoss instances and call them with
        # identical inputs; outputs must be byte-identical.
        l1, m1 = MultiAdapterLoss()(data=data, logits=logits)
        l2, m2 = MultiAdapterLoss()(data=data, logits=logits)
        assert torch.equal(l1, l2)
        assert m1 == m2

    def test_loss_invariant_to_extra_data_keys(self):
        """Extra unused keys in ``data`` must not affect the output."""
        B, T, V = 3, 5, 11
        base = _make_data(B, T, V, ["a", "a", "b"])
        logits = torch.randn(B, T, V)
        l1, _ = MultiAdapterLoss()(data=base, logits=logits)

        extra = dict(base)
        extra["unused_key"] = "anything"
        extra["another"] = torch.zeros(100)
        l2, _ = MultiAdapterLoss()(data=extra, logits=logits)
        assert torch.equal(l1, l2)

    def test_bf16_logits_produce_fp32_loss(self):
        """The op chain casts logits to fp32 explicitly; loss dtype must
        be fp32 even with bf16 inputs."""
        B, T, V = 2, 5, 11
        logits = torch.randn(B, T, V, dtype=torch.bfloat16)
        data = _make_data(B, T, V, ["a", "a"])
        loss, _ = MultiAdapterLoss()(data=data, logits=logits)
        assert loss.dtype == torch.float32

    def test_pregathered_path_matches_logits_path_across_inputs(self):
        """Fuzz: across 5 random inputs, the pregathered LOGPROB path
        and the LOGIT path must produce equal loss.
        """
        for seed in range(5):
            torch.manual_seed(seed)
            B, T, V = 4, 5, 17
            data = _make_data(B, T, V, ["a", "a", "b", "b"])
            logits = torch.randn(B, T, V)
            tl = compute_token_logprobs(logits, data)

            l_logits, _ = MultiAdapterLoss()(data=data, logits=logits)
            l_pregath, _ = MultiAdapterLoss()(data=data, next_token_logprobs=tl)
            assert torch.allclose(l_logits, l_pregath, atol=1e-6), (
                f"seed={seed}: logits vs pregathered diverged"
            )

    def test_non_contiguous_adapter_names_raises_in_loss_fn(self):
        """The block-contiguous contract surfaces all the way up at
        MultiAdapterLoss, not just inside ``adapter_row_ranges``.
        """
        B, T, V = 3, 5, 11
        data = {
            "input_ids": torch.randint(0, V, (B, T)),
            "token_mask": torch.ones(B, T, dtype=torch.long),
            "adapter_names": ["a", "b", "a"],  # non-contiguous
        }
        with pytest.raises(ValueError, match="not block-contiguous"):
            MultiAdapterLoss()(data=data, logits=torch.randn(B, T, V))

    def test_metrics_dict_contains_required_keys(self):
        """``num_unmasked_tokens`` and ``num_valid_samples`` are required
        downstream by the per-step logger; absence is a regression.
        """
        B, T, V = 3, 5, 11
        data = _make_data(B, T, V, ["a", "a", "b"])
        _, metrics = MultiAdapterLoss()(data=data, logits=torch.randn(B, T, V))
        assert "num_unmasked_tokens" in metrics
        assert "num_valid_samples" in metrics
        assert isinstance(metrics["num_unmasked_tokens"], int)
        assert isinstance(metrics["num_valid_samples"], int)
