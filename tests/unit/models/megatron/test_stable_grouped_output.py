"""Memory-bounded expert forward writes preserve all real output rows."""

import pytest
import torch

from nemo_rl.models.megatron.stable_forward import write_groups

pytestmark = pytest.mark.mcore


@pytest.mark.parametrize("budget_rows", [1, 3, 19])
def test_grouped_output_matches_full_groups_with_strided_inputs(
    budget_rows: int,
) -> None:
    generator = torch.Generator().manual_seed(42)
    splits = [0, 7, 0, 5]
    features, hidden = 6, 9
    weights = [
        torch.randn(features, hidden, generator=generator).bfloat16() for _ in splits
    ]
    inputs = [
        torch.randn(rows, hidden * 2, generator=generator).bfloat16()[:, ::2]
        for rows in splits
    ]
    expected = torch.cat([x @ w.t() for w, x in zip(weights, inputs, strict=True)])
    destination = torch.full_like(expected, float("nan"))
    largest_temporary = 0
    kernel_calls = 0

    def checked_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        nonlocal largest_temporary, kernel_calls
        result = a @ b
        largest_temporary = max(
            largest_temporary, result.numel() * result.element_size()
        )
        kernel_calls += 1
        return result

    budget = budget_rows * features * destination.element_size()
    calls = write_groups(
        weights,
        inputs,
        destination,
        splits,
        kernel=checked_kernel,
        max_output_bytes=budget,
    )
    assert torch.equal(destination, expected)
    assert torch.isfinite(destination).all()
    assert largest_temporary <= budget
    assert calls == kernel_calls
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, 11])
def test_invalid_output_budget_rejected_before_any_kernel(budget: object) -> None:
    def forbidden_kernel(*args: object) -> torch.Tensor:
        raise AssertionError("Invalid budget must fail before running a kernel")

    with pytest.raises(ValueError, match="budget"):
        write_groups(
            [torch.zeros(6, 9, dtype=torch.bfloat16)],
            [torch.zeros(4, 9, dtype=torch.bfloat16)],
            torch.empty(4, 6, dtype=torch.bfloat16),
            [4],
            kernel=forbidden_kernel,
            max_output_bytes=budget,
        )


def test_all_empty_groups_preserve_native_empty_destination() -> None:
    destination = torch.empty(0, 6, dtype=torch.bfloat16)

    def forbidden_kernel(*args: object) -> torch.Tensor:
        raise AssertionError("Empty experts must not launch a kernel")

    assert (
        write_groups(
            [torch.zeros(6, 9, dtype=torch.bfloat16)] * 2,
            [torch.empty(0, 9, dtype=torch.bfloat16)] * 2,
            destination,
            [0, 0],
            kernel=forbidden_kernel,
        )
        == 0
    )
