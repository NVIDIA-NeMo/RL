from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DraftUpdateProbe:
    before: tuple[float, float]
    grad_l2: float


@dataclass(frozen=True)
class DraftUpdateResult:
    before: tuple[float, float]
    after: tuple[float, float]
    grad_l2: float
    checksum_delta: float


def format_draft_update_probe(result: DraftUpdateResult) -> str:
    return (
        f"draft_update_probe=complete grad_l2={result.grad_l2:.17g} "
        f"checksum_sum_before={result.before[0]:.17g} "
        f"checksum_sum_after={result.after[0]:.17g} "
        f"checksum_l2_before={result.before[1]:.17g} "
        f"checksum_l2_after={result.after[1]:.17g} "
        f"delta={result.checksum_delta:.17g}"
    )


@torch.no_grad()
def _parameter_checksum(module: nn.Module) -> tuple[float, float]:
    value_sum = 0.0
    l2_sum = 0.0
    for parameter in module.parameters():
        value_sum += float(parameter.detach().sum(dtype=torch.float64).item())
        norm = torch.linalg.vector_norm(parameter.detach())
        l2_sum += float(norm.double().square().item())
    return value_sum, l2_sum


@torch.no_grad()
def _gradient_l2(module: nn.Module) -> float:
    squared_norm = 0.0
    for parameter in module.parameters():
        gradient = getattr(parameter, "main_grad", None)
        if gradient is None:
            gradient = parameter.grad
        if gradient is None:
            continue
        norm = torch.linalg.vector_norm(gradient.detach())
        squared_norm += float(norm.double().square().item())
    return squared_norm**0.5


def start_draft_update_probe(module: nn.Module) -> DraftUpdateProbe:
    return DraftUpdateProbe(
        before=_parameter_checksum(module),
        grad_l2=_gradient_l2(module),
    )


def finalize_draft_update_probe(
    module: nn.Module, probe: DraftUpdateProbe
) -> DraftUpdateResult:
    after = _parameter_checksum(module)
    delta = abs(after[0] - probe.before[0]) + abs(after[1] - probe.before[1])
    return DraftUpdateResult(
        before=probe.before,
        after=after,
        grad_l2=probe.grad_l2,
        checksum_delta=delta,
    )


def require_draft_update(result: DraftUpdateResult) -> None:
    if result.grad_l2 <= 0:
        raise RuntimeError("draft update probe requires a nonzero gradient")
    if result.checksum_delta <= 0 or result.before == result.after:
        raise RuntimeError("draft update probe requires a parameter change")
