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

from nemo_rl.algorithms.oapl import (
    _get_oapl_save_state,
    _initial_oapl_save_state,
)


def test_get_oapl_save_state_handles_legacy_checkpoint_and_filters_metrics():
    assert _get_oapl_save_state({}) == _initial_oapl_save_state()

    loaded_state = {
        "epoch": 1,
        "step": 3,
        "total_steps": 13,
        "consumed_samples": 32,
        "val:loss": 0.75,
    }

    save_state = _get_oapl_save_state(loaded_state)

    assert vars(save_state) == {
        "epoch": 1,
        "step": 3,
        "total_steps": 13,
        "consumed_samples": 32,
        "total_valid_tokens": 0,
    }
    assert "total_valid_tokens" not in loaded_state


def test_get_oapl_save_state_none_returns_initial():
    assert _get_oapl_save_state(None) == _initial_oapl_save_state()
