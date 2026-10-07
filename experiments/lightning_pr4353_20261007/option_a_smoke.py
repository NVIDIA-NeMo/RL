"""Exercise TE 2.18's grouped-tensor MXFP8 GEMM on one Blackwell GPU."""

import inspect
import os

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.module.grouped_linear import _GroupedLinear


def main() -> None:
    assert os.environ.get("NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM") == "1"
    assert "use_grouped_tensor" not in inspect.signature(te.GroupedLinear.__init__).parameters

    original = _GroupedLinear._forward_grouped_tensor
    calls = 0

    def capture(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    _GroupedLinear._forward_grouped_tensor = staticmethod(capture)
    try:
        layer = te.GroupedLinear(2, 128, 128, bias=False, params_dtype=torch.bfloat16)
        x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        splits = torch.tensor([128, 128], device="cuda", dtype=torch.int32)
        with te.fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
            output = layer(x, splits)
            output.float().square().mean().backward()
        assert calls > 0, "TE did not select the grouped-tensor path"
        assert torch.isfinite(output).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    finally:
        _GroupedLinear._forward_grouped_tensor = staticmethod(original)

    print(f"TE grouped-tensor MXFP8 forward/backward passed; calls={calls}")


if __name__ == "__main__":
    main()
