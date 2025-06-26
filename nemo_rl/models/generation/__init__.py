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
from typing import cast

from transformers import PreTrainedTokenizerBase

from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.generation.vllm import VllmConfig

TokenizerType = PreTrainedTokenizerBase


def configure_generation_config(
    config: GenerationConfig, tokenizer: TokenizerType, is_eval=False
) -> GenerationConfig:
    """Apply specific configurations to generation config."""
    # tokenizer setting
    config["pad_token_id"] = tokenizer.pad_token_id
    if config["stop_token_ids"] is None:
        config["stop_token_ids"] = [tokenizer.eos_token_id]

    # vllm setting
    if config["backend"] == "vllm":
        config = cast(VllmConfig, config)
        # set load_format
        config["vllm_cfg"]["load_format"] = "auto" if is_eval else "dummy"

        # set skip_tokenizer_init
        if is_eval or config["stop_strings"] is not None:
            config["vllm_cfg"]["skip_tokenizer_init"] = False
        else:
            config["vllm_cfg"]["skip_tokenizer_init"] = True

    return config
