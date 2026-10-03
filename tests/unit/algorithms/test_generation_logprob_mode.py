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

from nemo_rl.algorithms.grpo import _validate_generation_logprob_mode


@pytest.mark.parametrize(
    "temperature,top_k,top_p", [(0.7, None, 1.0), (1.0, 10, 1.0), (1.0, None, 0.9)]
)
@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
def test_processed_sampling_requires_processed_generation_logprobs(
    temperature, top_k, top_p, mode
):
    config = SimpleNamespace(
        policy={
            "generation": {
                "backend": "megatron",
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "mcore_generation_config": {"logprobs_mode": mode},
            }
        }
    )
    if mode == "raw_logprobs":
        with pytest.raises(ValueError, match="inconsistent importance ratios"):
            _validate_generation_logprob_mode(config.policy)
    else:
        _validate_generation_logprob_mode(config.policy)


@pytest.mark.parametrize("top_k", [None, 0, -1])
@pytest.mark.parametrize("top_p", [None, 1.0])
def test_raw_logprobs_allowed_for_identity_processing(top_k, top_p):
    config = SimpleNamespace(
        policy={
            "generation": {
                "backend": "megatron",
                "temperature": 1.0,
                "top_k": top_k,
                "top_p": top_p,
                "mcore_generation_config": {"logprobs_mode": "raw_logprobs"},
            }
        }
    )
    _validate_generation_logprob_mode(config.policy)


def test_single_controller_rejects_mismatch_before_worker_setup(monkeypatch):
    import importlib

    setup_module = importlib.import_module(
        "nemo_rl.algorithms.single_controller_utils.setup"
    )
    monkeypatch.setattr(
        setup_module, "validate_single_controller_config", lambda config: None
    )
    config = SimpleNamespace(
        policy={
            "generation": {
                "backend": "megatron",
                "temperature": 0.7,
                "top_k": None,
                "top_p": 1.0,
                "mcore_generation_config": {"logprobs_mode": "raw_logprobs"},
            }
        }
    )
    with pytest.raises(ValueError, match="inconsistent importance ratios"):
        setup_module.setup_single_controller(config, None)
