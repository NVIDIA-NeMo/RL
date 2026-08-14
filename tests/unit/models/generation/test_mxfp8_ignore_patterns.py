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

from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm.quantization.ignore_patterns import (
    get_embedded_mtp_ignore_patterns,
)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            SimpleNamespace(
                num_hidden_layers=43,
                num_nextn_predict_layers=1,
            ),
            ["model.layers.43.*"],
        ),
        (
            SimpleNamespace(
                text_config=SimpleNamespace(
                    num_hidden_layers=78,
                    num_nextn_predict_layers=2,
                )
            ),
            ["model.layers.78.*", "model.layers.79.*"],
        ),
        (
            SimpleNamespace(
                text_config=SimpleNamespace(
                    num_hidden_layers=80,
                    mtp_num_hidden_layers=2,
                )
            ),
            ["model.layers.80.*", "model.layers.81.*"],
        ),
    ],
)
def test_get_embedded_mtp_ignore_patterns(config, expected):
    assert get_embedded_mtp_ignore_patterns(config) == expected


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(num_hidden_layers=61),
        SimpleNamespace(num_hidden_layers=61, num_nextn_predict_layers=0),
        SimpleNamespace(
            text_config=SimpleNamespace(
                num_hidden_layers=61,
                num_nextn_predict_layers=0,
            )
        ),
    ],
)
def test_get_embedded_mtp_ignore_patterns_without_mtp(config):
    assert get_embedded_mtp_ignore_patterns(config) == []


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(num_nextn_predict_layers=1),
        SimpleNamespace(num_hidden_layers=61, num_nextn_predict_layers=-1),
        SimpleNamespace(num_hidden_layers=61, num_nextn_predict_layers="1"),
    ],
)
def test_get_embedded_mtp_ignore_patterns_rejects_invalid_metadata(config):
    with pytest.raises(ValueError, match="MTP layer metadata"):
        get_embedded_mtp_ignore_patterns(config)
