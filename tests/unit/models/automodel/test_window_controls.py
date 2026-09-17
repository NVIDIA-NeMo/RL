import random
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from nemo_rl.models.automodel.window_controls import collect_window_controls


@pytest.mark.parametrize("fail", [False, True])
def test_collection_restores_buffers_and_rng(fail: bool) -> None:
    model = torch.nn.BatchNorm1d(3)
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    torch_rng = torch.get_rng_state()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()

    class Control:
        phase: str | None = None

        def begin_logical_window_control_collection(self) -> None:
            self.phase = "collect"

        def finalize_logical_window_control_collection(self) -> None:
            self.phase = "apply"

        def end_logical_window_control(self) -> None:
            self.phase = None

    control = Control()

    def forward(data: list[None]) -> None:
        model(torch.randn(4, 3))
        random.random()
        np.random.rand()
        if fail:
            raise RuntimeError("prepass failed")

    with pytest.raises(RuntimeError, match="prepass failed") if fail else nullcontext():
        collect_window_controls(
            model=model,
            data=[None],
            loss_fn=control,
            forward=forward,
            num_valid_microbatches=1,
        )

    assert control.phase == (None if fail else "apply")
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert random.getstate() == python_rng
    np.testing.assert_equal(np.random.get_state(), numpy_rng)
