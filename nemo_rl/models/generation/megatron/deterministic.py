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

"""Isolate inference initialization from the policy RNG streams."""

from contextlib import contextmanager
from typing import Iterator

import torch
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

from nemo_rl.utils.determinism import capture_rng_state, restore_rng_state


@contextmanager
def preserve_megatron_rng() -> Iterator[None]:
    """Prevent lazy inference setup from consuming the training RNG on restart."""
    state = capture_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    tracker_state = get_cuda_rng_tracker().get_states()
    try:
        yield
    finally:
        restore_rng_state(state)
        torch.cuda.set_rng_state(cuda_state)
        get_cuda_rng_tracker().set_states(tracker_state)
