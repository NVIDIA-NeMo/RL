import math

import pytest
import torch
from logra.compression import make_projection

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@cuda
def test_make_projection_on_cuda_does_not_synchronize_host():
    torch.cuda.set_sync_debug_mode("error")
    try:
        projection = make_projection(
            256, 3584, 7, device="cuda", dtype=torch.float32, distribution="rademacher"
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert projection.device.type == "cuda"
    assert projection.shape == (256, 3584)


@cuda
@pytest.mark.parametrize("distribution", ["rademacher", "gaussian"])
def test_make_projection_on_cuda_is_deterministic_and_well_scaled(distribution):
    kwargs = dict(device="cuda", dtype=torch.float32, distribution=distribution)
    a = make_projection(256, 3584, 7, **kwargs)
    b = make_projection(256, 3584, 7, **kwargs)
    c = make_projection(256, 3584, 8, **kwargs)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    if distribution == "rademacher":
        assert torch.equal(a.abs(), torch.full_like(a, 1 / math.sqrt(256)))
    # Columns have unit norm in expectation, so A^T A has a unit diagonal and
    # ||A x|| approximates ||x|| (the Johnson-Lindenstrauss scaling).
    column_norms = (a * a).sum(dim=0)
    assert column_norms.mean().item() == pytest.approx(1.0, abs=0.05)
    assert abs(a.mean().item()) < 1e-3


def test_refresh_reuses_the_low_precision_projection_buffer():
    from logra.config import LoGRAConfig
    from logra.optimizer import LoGRAOptimizer
    from torch import nn

    net = nn.Sequential(nn.Linear(7, 5))
    opt = LoGRAOptimizer(
        net,
        torch.optim.AdamW(net.parameters(), lr=0.01),
        LoGRAConfig(rank=3, target_modules=["0"]),
    )
    layer = opt.layers[0]
    before = layer.projection_as(torch.bfloat16)
    opt.update_count += 1
    opt.refresh_projection()
    after = layer.projection_as(torch.bfloat16)
    assert after is before
    torch.testing.assert_close(after, layer.projection.to(torch.bfloat16))
