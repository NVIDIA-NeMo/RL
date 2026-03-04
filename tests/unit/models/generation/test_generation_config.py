# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from copy import deepcopy
from typing import cast

from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig


class _MockTokenizer:
    pad_token_id = 0
    eos_token_id = 2


def _make_vllm_config() -> VllmConfig:
    return cast(
        VllmConfig,
        {
            "backend": "vllm",
            "max_new_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": None,
            "stop_token_ids": None,
            "stop_strings": None,
            "vllm_cfg": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "expert_parallel_size": 1,
                "gpu_memory_utilization": 0.5,
                "max_model_len": 1024,
                "async_engine": False,
                "kv_cache_dtype": "auto",
            },
        },
    )


def test_configure_generation_config_disables_skip_tokenizer_init_for_stop_tokens():
    config = _make_vllm_config()

    configured = configure_generation_config(config, _MockTokenizer(), is_eval=False)

    assert configured["stop_token_ids"] == [_MockTokenizer.eos_token_id]
    assert configured["vllm_cfg"]["skip_tokenizer_init"] is False
    assert configured["vllm_cfg"]["load_format"] == "dummy"


def test_configure_generation_config_respects_explicit_skip_tokenizer_init():
    config = deepcopy(_make_vllm_config())
    config["vllm_cfg"]["skip_tokenizer_init"] = True

    configured = configure_generation_config(config, _MockTokenizer(), is_eval=False)

    assert configured["vllm_cfg"]["skip_tokenizer_init"] is True
