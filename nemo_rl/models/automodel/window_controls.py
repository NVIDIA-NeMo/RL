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

"""Collect batch-wide loss controls without changing model buffers or RNG state."""

import copy
import random
from typing import Any, Callable, Protocol

import numpy as np
import torch
from torch.distributed.tensor import DTensor


class WindowControlledLoss(Protocol):
    def begin_logical_window_control_collection(self) -> None: ...

    def finalize_logical_window_control_collection(self) -> None: ...

    def end_logical_window_control(self) -> None: ...


def collect_window_controls(
    *,
    model: torch.nn.Module,
    data: list[Any],
    loss_fn: WindowControlledLoss,
    forward: Callable[[list[Any]], Any],
    num_valid_microbatches: int | None,
) -> None:
    """Collect detached statistics, restoring registered buffers and random state."""
    if any(isinstance(p, DTensor) for p in model.parameters()):
        raise NotImplementedError(
            "Window controls do not yet support distributed parameters"
        )
    if num_valid_microbatches is not None and num_valid_microbatches != len(data):
        raise NotImplementedError(
            "Window controls do not yet support dummy microbatches"
        )
    # Complete registered state plus all supported RNGs. Optimizer/scheduler are
    # not called in this prepass and remain owned by the native worker.
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial_buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    train_flags = {name: mod.training for name, mod in model.named_modules()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    loss_fn.begin_logical_window_control_collection()
    try:
        with torch.no_grad():
            forward(copy.deepcopy(data))
        for name, param in model.named_parameters():
            if not torch.equal(param.detach(), initial[name]):
                raise RuntimeError(
                    "A model parameter changed in the control prepass: " + name
                )
        loss_fn.finalize_logical_window_control_collection()
    except BaseException:
        loss_fn.end_logical_window_control()
        raise
    finally:
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(initial_buffers[name])
        for name, mod in model.named_modules():
            mod.training = train_flags[name]
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
