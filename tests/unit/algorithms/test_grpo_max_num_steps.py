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
"""Tests for _clamp_max_num_steps_to_epochs."""

from nemo_rl.algorithms.grpo import GRPOConfig, _clamp_max_num_steps_to_epochs


def _config(max_num_steps: int, max_num_epochs: int) -> GRPOConfig:
    return GRPOConfig(max_num_steps=max_num_steps, max_num_epochs=max_num_epochs)


def test_clamps_to_one_epoch():
    cfg = _config(max_num_steps=1000000, max_num_epochs=1)
    _clamp_max_num_steps_to_epochs(cfg, 1425, use_multiple_dataloader=False)
    assert cfg.max_num_steps == 1425


def test_clamps_to_several_epochs():
    cfg = _config(max_num_steps=1000, max_num_epochs=3)
    _clamp_max_num_steps_to_epochs(cfg, 100, use_multiple_dataloader=False)
    assert cfg.max_num_steps == 300


def test_keeps_the_smaller_step_budget():
    cfg = _config(max_num_steps=50, max_num_epochs=1)
    _clamp_max_num_steps_to_epochs(cfg, 1425, use_multiple_dataloader=False)
    assert cfg.max_num_steps == 50


def test_multiple_dataloaders_are_left_alone():
    # MultipleDataloaderWrapper is an infinite iterator, so an epoch has no
    # length and the step budget is the only bound.
    cfg = _config(max_num_steps=1000000, max_num_epochs=1)
    _clamp_max_num_steps_to_epochs(cfg, 1425, use_multiple_dataloader=True)
    assert cfg.max_num_steps == 1000000


def test_non_positive_epochs_are_left_alone():
    # Clamping on 0 would silently budget zero training steps.
    cfg = _config(max_num_steps=1000000, max_num_epochs=0)
    _clamp_max_num_steps_to_epochs(cfg, 1425, use_multiple_dataloader=False)
    assert cfg.max_num_steps == 1000000
