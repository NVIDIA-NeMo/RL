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

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from nemo_rl.algorithms.loss.loss_functions import (
    ClippedPGLossConfig,
    ClippedPGLossFn,
)


GOLDEN_PATH = Path(__file__).parent / "data" / "clipped_pg_loss_goldens.json"
GOLDENS = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
GOLDEN_CASES = {case["name"]: case for case in GOLDENS["cases"]}
EXPECTED_LEGACY_COMMIT = "737c4d2cd629d19e5c576b67ef2a6105b6b1bbb3"


def _tensor(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    integer: bool = False,
) -> torch.Tensor:
    return torch.tensor(
        value,
        device=device,
        dtype=torch.int64 if integer else dtype,
    )


def _run_case(
    case: dict[str, Any],
    *,
    metrics_level: str,
    enable_torch_compile: bool = False,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    inputs = copy.deepcopy(GOLDENS["base_inputs"])
    inputs.update(copy.deepcopy(case["input_overrides"]))

    next_token_logprobs = _tensor(
        inputs["next_token_logprobs"], device=device, dtype=dtype
    ).requires_grad_(True)
    data = {
        "input_ids": _tensor(
            inputs["input_ids"], device=device, dtype=dtype, integer=True
        ),
        "token_mask": _tensor(inputs["token_mask"], device=device, dtype=dtype),
        "sample_mask": _tensor(inputs["sample_mask"], device=device, dtype=dtype),
        "advantages": _tensor(inputs["advantages"], device=device, dtype=dtype),
        "prev_logprobs": _tensor(inputs["prev_logprobs"], device=device, dtype=dtype),
        "generation_logprobs": _tensor(
            inputs["generation_logprobs"], device=device, dtype=dtype
        ),
        "reference_policy_logprobs": _tensor(
            inputs["reference_policy_logprobs"], device=device, dtype=dtype
        ),
        "rewards": _tensor(inputs["rewards"], device=device, dtype=dtype),
    }
    gradient_inputs = [next_token_logprobs]
    gradient_names = ["next_token_logprobs"]
    if "curr_logprobs_unfiltered" in inputs:
        curr_logprobs_unfiltered = _tensor(
            inputs["curr_logprobs_unfiltered"], device=device, dtype=dtype
        ).requires_grad_(True)
        data["curr_logprobs_unfiltered"] = curr_logprobs_unfiltered
        gradient_inputs.append(curr_logprobs_unfiltered)
        gradient_names.append("curr_logprobs_unfiltered")

    config = ClippedPGLossConfig(
        **case["config"],
        metrics_level=metrics_level,
        enable_torch_compile=enable_torch_compile,
    )
    loss_fn = ClippedPGLossFn(config)
    global_valid_seqs = data["sample_mask"].sum()
    global_valid_toks = (
        data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)
    ).sum()
    loss, metrics = loss_fn(
        next_token_logprobs=next_token_logprobs,
        data=data,
        global_valid_seqs=global_valid_seqs,
        global_valid_toks=global_valid_toks,
    )
    gradients = torch.autograd.grad(loss, gradient_inputs)
    return (
        loss.detach(),
        dict(zip(gradient_names, gradients, strict=True)),
        metrics,
    )


def _assert_scalar_close(
    actual: float | torch.Tensor,
    expected: float,
    *,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    torch.testing.assert_close(
        torch.as_tensor(actual, device=device, dtype=dtype),
        torch.as_tensor(expected, device=device, dtype=dtype),
        rtol=rtol,
        atol=atol,
    )


def _expected_metric_under_compatibility_contract(
    *, case_name: str, name: str, legacy_value: float
) -> float:
    if case_name != "gspo_sequence_is" or name != "sampling_importance_ratio":
        return legacy_value

    # The compatibility golden records sequence-level weights shaped [batch, 1]
    # multiplied by a [batch] sample mask. That legacy broadcast produces
    # [batch, batch], doubling this B=2 fixture's result. This documented
    # compatibility exception instead reduces one weight per valid sample.
    inputs = GOLDENS["base_inputs"]
    prev_logprobs = torch.tensor(inputs["prev_logprobs"], dtype=torch.float64)[:, 1:]
    generation_logprobs = torch.tensor(
        inputs["generation_logprobs"], dtype=torch.float64
    )[:, 1:]
    token_mask = torch.tensor(inputs["token_mask"], dtype=torch.float64)[:, 1:]
    sample_mask = torch.tensor(inputs["sample_mask"], dtype=torch.float64)
    weights = torch.exp(
        ((prev_logprobs - generation_logprobs) * token_mask).sum(dim=-1)
    )
    corrected_value = (weights * sample_mask).sum() / (sample_mask.sum() + 1e-8)
    _assert_scalar_close(legacy_value, 2.0 * corrected_value)
    return corrected_value.item()


def test_clipped_pg_goldens_define_pinned_backward_compatibility_contract() -> None:
    assert GOLDENS["schema_version"] == 1
    assert GOLDENS["compatibility_contract"] == {
        "scope": (
            "Preserve loss, gradients, and metrics from the pinned legacy "
            "implementation except for the intentional corrections listed here."
        ),
        "intentional_corrections": [
            {
                "case": "gspo_sequence_is",
                "metric": "sampling_importance_ratio",
                "reason": (
                    "Reduce one sequence weight per valid sample instead of preserving "
                    "the legacy [B, 1] by [B] broadcast."
                ),
            }
        ],
    }
    assert GOLDENS["provenance"] == {
        "source_commit": EXPECTED_LEGACY_COMMIT,
        "source_file": "nemo_rl/algorithms/loss/loss_functions.py",
        "python_version": "3.12.8",
        "torch_version": "2.6.0",
        "device": "cpu",
        "dtype": "float64",
    }


@pytest.mark.parametrize("case_name", GOLDEN_CASES)
def test_clipped_pg_full_eager_preserves_backward_compatibility(
    case_name: str,
) -> None:
    """Preserve the pinned legacy numerics and documented compatibility exceptions."""
    case = GOLDEN_CASES[case_name]
    expected = case["expected"]
    loss, gradients, metrics = _run_case(case, metrics_level="full")

    _assert_scalar_close(loss, expected["loss"])
    assert gradients.keys() == expected["gradients"].keys()
    for name, gradient in gradients.items():
        torch.testing.assert_close(
            gradient,
            torch.tensor(expected["gradients"][name], dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )

    assert metrics.keys() == expected["metrics"].keys()
    for name, value in metrics.items():
        _assert_scalar_close(
            value,
            _expected_metric_under_compatibility_contract(
                case_name=case_name,
                name=name,
                legacy_value=expected["metrics"][name],
            ),
        )


@pytest.mark.parametrize("case_name", GOLDEN_CASES)
def test_clipped_pg_minimal_preserves_backward_compatible_loss_and_gradients(
    case_name: str,
) -> None:
    case = GOLDEN_CASES[case_name]
    expected = case["expected"]
    loss, gradients, metrics = _run_case(case, metrics_level="minimal")

    _assert_scalar_close(loss, expected["loss"])
    for name, gradient in gradients.items():
        torch.testing.assert_close(
            gradient,
            torch.tensor(expected["gradients"][name], dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )

    expected_metric_names = {
        "loss",
        "kl_penalty",
        "num_valid_samples",
        "positive_nll_loss",
    }
    if case["config"].get("use_importance_sampling_correction", False):
        expected_metric_names.add("sampling_importance_ratio")
    if case["config"].get("truncated_importance_sampling_type") is not None:
        expected_metric_names.add("is_oob_ratio")
    assert metrics.keys() == expected_metric_names
    for name, value in metrics.items():
        _assert_scalar_close(
            value,
            _expected_metric_under_compatibility_contract(
                case_name=case_name,
                name=name,
                legacy_value=expected["metrics"][name],
            ),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "case_name",
    ["ppo_token", "tis_token", "gspo_sequence_is"],
)
def test_clipped_pg_compiled_matches_eager(case_name: str) -> None:
    case = GOLDEN_CASES[case_name]
    device = torch.device("cuda")
    eager_loss, eager_gradients, eager_metrics = _run_case(
        case,
        metrics_level="minimal",
        device=device,
        dtype=torch.float32,
    )
    compiled_loss, compiled_gradients, compiled_metrics = _run_case(
        case,
        metrics_level="minimal",
        enable_torch_compile=True,
        device=device,
        dtype=torch.float32,
    )

    torch.testing.assert_close(compiled_loss, eager_loss, rtol=1e-5, atol=1e-6)
    assert compiled_gradients.keys() == eager_gradients.keys()
    for name, gradient in compiled_gradients.items():
        torch.testing.assert_close(
            gradient,
            eager_gradients[name],
            rtol=1e-5,
            atol=1e-6,
        )
    assert compiled_metrics.keys() == eager_metrics.keys()
    for name, value in compiled_metrics.items():
        _assert_scalar_close(
            value,
            eager_metrics[name],
            device=device,
            dtype=torch.float32,
            rtol=1e-5,
            atol=1e-6,
        )
