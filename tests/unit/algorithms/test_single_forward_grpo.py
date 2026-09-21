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

import math

import pytest
import torch

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.utils import rescale_loss_metrics
from nemo_rl.algorithms.loss.wrapper import SequencePackingLossWrapper
from nemo_rl.algorithms.utils import compute_seq_logprob_errors
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"force_on_policy_ratio": False}, "force_on_policy_ratio"),
        ({"token_level_loss": False}, "token_level_loss"),
        ({"positive_example_nll_weight": 0.1}, "NLL"),
    ],
)
def test_in_loss_filter_rejects_incompatible_loss_configuration(
    overrides: dict[str, bool | float], message: str
) -> None:
    """Passing a threshold cannot silently enable filtering for ordinary PPO."""
    cfg = ClippedPGLossConfig(force_on_policy_ratio=True)
    cfg = cfg.model_copy(update=overrides)
    with pytest.raises(ValueError, match=message):
        ClippedPGLossFn(cfg, seq_logprob_error_threshold=2.0)


def _batch() -> tuple[BatchedDataDict, torch.Tensor]:
    current = torch.full((4, 3), -2.0, dtype=torch.float64)
    # Row 1 fails absolute error (>2) but passes signed geometric-mean TIS:
    # exp(mean(log(4), -log(4))) = 1. Row 2 passes absolute error (1.3)
    # but fails TIS (upper bound 1.2). Row 3 was masked before training.
    delta = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [math.log(4), -math.log(4), 0.0],
            [math.log(1.3), 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    generation = torch.cat([torch.zeros(4, 1), current - delta], dim=1)
    data = BatchedDataDict(
        {
            "token_mask": torch.tensor(
                [[0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1]],
                dtype=torch.float64,
            ),
            "sample_mask": torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64),
            "generation_logprobs": generation,
            "reference_policy_logprobs": generation - 0.2,
            "advantages": torch.tensor(
                [
                    [0.0, 1.0, 2.0, -1.0],
                    [0.0, 2.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0],
                ],
                dtype=torch.float64,
            ),
        }
    )
    # Deliberately omit prev_logprobs: the single-forward path must not read it.
    return data, current


@pytest.mark.parametrize("chunks", [[4], [1, 1, 1, 1], [2, 2], [1, 3]])
@pytest.mark.parametrize("kl_penalty", [0.0, 0.1])
def test_accumulated_loss_and_gradients_match_upstream_mask(chunks, kl_penalty):
    """Post-backward global scaling matches pre-masking, even with empty chunks."""
    data, values = _batch()
    cfg = ClippedPGLossConfig(
        force_on_policy_ratio=True,
        token_level_loss=True,
        reference_policy_kl_penalty=kl_penalty,
        use_importance_sampling_correction=True,
        truncated_importance_sampling_type="seq-mask-tis",
        truncated_importance_sampling_ratio_min=0.8,
        truncated_importance_sampling_ratio=1.2,
    )
    expected_data = data.select_indices(list(range(4)))
    expected_data["sample_mask"] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    expected_lp = values.clone().requires_grad_()
    expected_loss, expected_metrics = ClippedPGLossFn(cfg)(
        next_token_logprobs=expected_lp,
        data=expected_data,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=torch.tensor(4.0),
    )
    expected_loss.backward()

    loss_fn = ClippedPGLossFn(cfg, seq_logprob_error_threshold=2.0)
    actual_lp = values.clone().requires_grad_()
    metrics = []
    start = 0
    for size in chunks:
        end = start + size
        loss, chunk_metrics = loss_fn(
            next_token_logprobs=actual_lp[start:end],
            data=data.slice(start, end),
            global_valid_seqs=torch.tensor(3.0),
            global_valid_toks=torch.tensor(6.0),
        )
        loss.backward()
        metrics.append(chunk_metrics)
        start = end

    kept_toks = sum(m["seq_logprob_error_valid_tokens"] for m in metrics)
    kept_seqs = sum(m["seq_logprob_error_valid_seqs"] for m in metrics)
    assert kept_toks == 4  # includes the TIS-only rejected row
    assert kept_seqs == 2
    assert sum(m["num_masked_seqs_by_logprob_error"] for m in metrics) == 1
    torch.testing.assert_close(actual_lp.grad * (6 / kept_toks), expected_lp.grad)
    rescaled = [
        rescale_loss_metrics(
            m,
            loss_fn.metric_normalizations,
            token_factor=6 / kept_toks,
            sequence_factor=3 / kept_seqs,
        )
        for m in metrics
    ]
    for key in (
        "loss",
        "is_oob_ratio",
        "probs_ratio",
        "kl_penalty",
        "num_valid_samples",
    ):
        assert sum(m[key] for m in rescaled) == pytest.approx(expected_metrics[key])
    assert sum(m["loss"] for m in rescaled) == pytest.approx(expected_loss.item())
    # Loss-local filtering must not change rollout/advantage bookkeeping.
    assert data["sample_mask"].tolist() == [1.0, 1.0, 1.0, 0.0]


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_rejected_nonfinite_logprobs_do_not_poison_loss(bad_value):
    data, values = _batch()
    values[1, :2] = bad_value
    lp = values.requires_grad_()
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(force_on_policy_ratio=True, reference_policy_kl_penalty=0),
        seq_logprob_error_threshold=2.0,
    )
    loss, metrics = loss_fn(
        next_token_logprobs=lp,
        data=data,
        global_valid_seqs=torch.tensor(3.0),
        global_valid_toks=torch.tensor(6.0),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(lp.grad).all()
    assert torch.count_nonzero(lp.grad[1]) == 0
    assert metrics["seq_logprob_error_valid_tokens"] == 4


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("poison", ["policy", "reference"])
def test_rejected_nonfinite_logprobs_stay_contained_under_reference_kl(
    bad_value: float, poison: str
) -> None:
    """The KL term must not resurrect a rejected sequence's nonfinite logprob."""
    data, values = _batch()
    if poison == "policy":
        values[1, :2] = bad_value
    else:
        data["reference_policy_logprobs"][1, 1:3] = bad_value
    lp = values.requires_grad_()
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            force_on_policy_ratio=True,
            token_level_loss=True,
            reference_policy_kl_penalty=0.1,
        ),
        seq_logprob_error_threshold=2.0,
    )
    loss, metrics = loss_fn(
        next_token_logprobs=lp,
        data=data,
        global_valid_seqs=torch.tensor(3.0),
        global_valid_toks=torch.tensor(6.0),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(lp.grad).all()
    assert torch.count_nonzero(lp.grad[1]) == 0
    assert math.isfinite(metrics["kl_penalty"])
    assert metrics["seq_logprob_error_valid_tokens"] == 4


def test_sequence_error_uses_only_loss_tokens_and_preserves_threshold_boundary():
    policy = torch.zeros(3, 3, requires_grad=True)
    generation = torch.tensor(
        [
            [math.log(2), math.log(2), float("nan")],
            [100.0, 100.0, 100.0],
            [0.0, 0.0, 0.0],
        ]
    )
    errors, valid = compute_seq_logprob_errors(
        policy_logprobs=policy,
        generation_logprobs=generation,
        token_mask=torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
        sample_mask=torch.tensor([1.0, 0.0, 1.0]),
    )
    assert errors.tolist() == [2.0, 0.0, 0.0]
    assert valid.tolist() == [True, False, False]
    assert not errors.requires_grad


def test_all_rejected_loss_reports_zero_survivors_and_zero_gradient():
    data, values = _batch()
    lp = values.requires_grad_()
    loss, metrics = ClippedPGLossFn(
        ClippedPGLossConfig(force_on_policy_ratio=True, reference_policy_kl_penalty=0),
        seq_logprob_error_threshold=0.5,
    )(
        next_token_logprobs=lp,
        data=data,
        global_valid_seqs=torch.tensor(3.0),
        global_valid_toks=torch.tensor(6.0),
    )
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(lp.grad) == 0
    assert metrics["seq_logprob_error_valid_tokens"] == 0
    assert metrics["num_masked_seqs_by_logprob_error"] == 3


def test_no_threshold_keeps_existing_loss_contract():
    data, values = _batch()
    _, metrics = ClippedPGLossFn(
        ClippedPGLossConfig(force_on_policy_ratio=True, reference_policy_kl_penalty=0)
    )(
        next_token_logprobs=values,
        data=data,
        global_valid_seqs=torch.tensor(3.0),
        global_valid_toks=torch.tensor(6.0),
    )
    assert metrics["num_valid_samples"] == 3
    assert "seq_logprob_error_valid_tokens" not in metrics


def test_packed_loss_preserves_sequence_decisions_and_survivor_counts():
    """Per-sequence packing must sum counts, never normalize each sequence alone."""
    data, values = _batch()
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(force_on_policy_ratio=True, reference_policy_kl_penalty=0),
        seq_logprob_error_threshold=2.0,
    )
    expected_lp = values.clone().requires_grad_()
    expected_loss, expected_metrics = loss_fn(
        next_token_logprobs=expected_lp,
        data=data,
        global_valid_seqs=torch.tensor(3.0),
        global_valid_toks=torch.tensor(6.0),
    )
    expected_loss.backward()
    packed_lp = values.clone().requires_grad_()
    lengths = (4, 3, 2, 4)
    stream = torch.cat(
        [
            torch.cat([packed_lp[i, : length - 1], torch.zeros(1)])
            for i, length in enumerate(lengths)
        ]
    ).unsqueeze(0)

    def prepare_logprobs(*, logits, data, **_kwargs):
        # Stand in only for the logits-to-logprobs projection. The real wrapper
        # must split the packed stream, slice masks, and aggregate the real loss.
        return {"next_token_logprobs": logits[:, :-1]}, data

    wrapper = SequencePackingLossWrapper(
        loss_fn=loss_fn,
        prepare_fn=prepare_logprobs,
        cu_seqlens_q=torch.tensor([0, 4, 7, 9, 13]),
    )
    loss, metrics = wrapper(
        next_token_logits=stream,
        data=data,
        global_valid_seqs=torch.tensor(3.0),
        global_valid_toks=torch.tensor(6.0),
    )
    loss.backward()
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(packed_lp.grad, expected_lp.grad)
    for key in (
        "seq_logprob_error_valid_tokens",
        "seq_logprob_error_valid_seqs",
        "num_masked_seqs_by_logprob_error",
    ):
        assert metrics[key] == expected_metrics[key]
