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

import pytest

from nemo_rl.models.policy.lm_policy import _megatron_gtp_weight_remat_size


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"megatron_cfg": None},
        {"megatron_cfg": {}},
        {"megatron_cfg": {"model_overrides": None}},
        {"megatron_cfg": {"model_overrides": {"gtp_weight_remat_size": None}}},
    ],
)
def test_gtp_config_defaults_to_one(config):
    assert _megatron_gtp_weight_remat_size(config) == 1


@pytest.mark.parametrize("size", [1, 2, 4])
def test_gtp_config_preserves_dense_size(size):
    assert (
        _megatron_gtp_weight_remat_size(
            {
                "megatron_cfg": {
                    "model_overrides": {
                        "gtp_weight_remat_size": size,
                        "expert_gtp_weight_remat_size": 1,
                    }
                }
            }
        )
        == size
    )


@pytest.mark.parametrize("size", [0, -1, True, 2.5, "2"])
def test_gtp_config_rejects_invalid_sizes(size):
    with pytest.raises(ValueError, match="positive integer"):
        _megatron_gtp_weight_remat_size(
            {"megatron_cfg": {"model_overrides": {"gtp_weight_remat_size": size}}}
        )
