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

from nemo_rl.models.generation.vllm.vllm_worker_async import _AsyncLLMHTTPClient

pytestmark = pytest.mark.vllm


def _client(*, flat_logprobs: bool) -> _AsyncLLMHTTPClient:
    engine_client = SimpleNamespace(
        model_config=None, renderer=None, input_processor=None, vllm_config=None
    )
    client = _AsyncLLMHTTPClient(
        engine_client, SimpleNamespace(), flat_logprobs=flat_logprobs
    )
    client._generate = lambda prompt, sampling_params, request_id, kwargs: (
        sampling_params
    )
    return client


@pytest.mark.parametrize(
    ("flat_logprobs", "logprobs", "prompt_logprobs", "expected"),
    [
        (True, 0, None, True),  # logprobs requested: store them flat
        (True, None, None, False),  # no logprobs requested: nothing to store
        (False, 0, None, False),  # off: vLLM's default per-token objects
        # Prompt logprobs requested: vLLM's -inf clamp cannot edit flat storage.
        (True, 0, 0, False),
    ],
)
def test_generate_sets_flat_logprobs(
    flat_logprobs, logprobs, prompt_logprobs, expected
):
    sampling_params = SimpleNamespace(
        logprobs=logprobs, prompt_logprobs=prompt_logprobs, flat_logprobs=False
    )

    _client(flat_logprobs=flat_logprobs).generate("prompt", sampling_params, "req-0")

    assert sampling_params.flat_logprobs is expected
