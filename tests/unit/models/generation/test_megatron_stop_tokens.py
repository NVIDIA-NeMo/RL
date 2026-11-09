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

"""Configured stop token ids reach the Megatron engine."""

from types import SimpleNamespace

import pytest
from megatron.core.inference.sampling_params import SamplingParams

from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin

pytestmark = pytest.mark.mcore


@pytest.mark.parametrize("stop_ids", [None, [], [7], [7, 9, 7]])
@pytest.mark.parametrize("greedy", [False, True])
def test_configured_stop_ids_reach_sampling_params(stop_ids, greedy):
    worker = SimpleNamespace(
        cfg={
            "generation": {
                "top_k": None,
                "top_p": 1.0,
                "temperature": 1.0,
                "max_new_tokens": 8,
                "stop_token_ids": stop_ids,
            }
        },
        megatron_tokenizer=SimpleNamespace(eod=11),
    )
    params = MegatronGenerationMixin._build_sampling_params(worker, greedy, ["END"])
    assert params.stop_token_ids == (
        sorted(set(stop_ids)) if stop_ids is not None else None
    )
    assert params.termination_id == 11
    assert params.stop_words == ["END"]
    assert params.detokenize_stop_sequence
    restored = SamplingParams.deserialize(params.serialize())
    assert restored.stop_token_ids == params.stop_token_ids
