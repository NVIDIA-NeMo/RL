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
"""Tests for the single-seed validation rollout mode (verl-omni val parity)."""

import torch

from nemo_rl.algorithms.flow_grpo import FlowGRPOValGenerationCfg
from nemo_rl.models.diffusion.flow_grpo_policy import derive_rollout_seed
from nemo_rl.models.diffusion.workers.flow_grpo_worker import (
    build_single_seed_latents_fn,
)


def test_derive_rollout_seed_offsets_per_worker_by_default():
    assert derive_rollout_seed(None, 5) is None
    assert derive_rollout_seed(42, 0) == 42
    assert derive_rollout_seed(42, 3) == 42 + 3 * 7919


def test_derive_rollout_seed_single_seed_ignores_worker_index():
    assert derive_rollout_seed(42, 3, single_seed=True) == 42
    assert derive_rollout_seed(None, 3, single_seed=True) is None


def test_single_seed_latents_repeat_the_first_draw():
    def prepare(batch_size: int, seed: int) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        return torch.randn((batch_size, 2, 3), generator=gen)

    fn = build_single_seed_latents_fn(prepare)
    out = fn(4, 42)

    assert out.shape == (4, 2, 3)
    # Every sample equals the first draw of a fresh generator seeded with 42
    # (verl-omni applies val_kwargs.seed to every request individually).
    assert torch.equal(out[0], prepare(1, 42)[0])
    for i in range(1, 4):
        assert torch.equal(out[i], out[0])


def test_val_generation_cfg_single_seed_defaults_off():
    cfg = FlowGRPOValGenerationCfg()
    assert cfg.single_seed is False
    assert cfg.model_dump()["single_seed"] is False
