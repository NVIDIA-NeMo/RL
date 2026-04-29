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

from typing import Any, NotRequired, TypedDict

from nemo_rl.models.generation.interfaces import GenerationConfig


class SpeculativeDecodingArgs(TypedDict, total=False):
    """Maps 1:1 to tensorrt_llm.llmapi speculative decoding config classes.

    ``method`` selects the config class; remaining keys become constructor kwargs.
    Supported methods (all work with backend="pytorch"):
      - ngram:        NGramDecodingConfig   (no draft model)
      - mtp:          MTPDecodingConfig     (model must have MTP heads)
      - eagle3:       Eagle3DecodingConfig  (requires Eagle3 draft checkpoint)
      - draft_target: DraftTargetDecodingConfig (separate draft model)
    """

    method: str
    max_draft_len: int
    speculative_model: str
    # NGram-specific
    max_matching_ngram_size: int
    is_public_pool: bool
    # MTP-specific
    num_nextn_predict_layers: int
    mtp_eagle_one_model: bool
    # Eagle3-specific
    eagle3_one_model: bool
    use_dynamic_tree: bool
    greedy_sampling: bool
    # Common tuning knobs
    max_concurrency: int
    draft_len_schedule: dict[int, int]
    acceptance_window: int
    acceptance_length_threshold: float


class TrtllmSpecificArgs(TypedDict):
    tensor_parallel_size: int
    gpu_memory_utilization: NotRequired[float]
    max_model_len: int
    precision: NotRequired[str]
    max_batch_size: NotRequired[int]
    max_num_tokens: NotRequired[int]
    speculative_decoding: NotRequired[SpeculativeDecodingArgs]
    expose_http_server: NotRequired[bool]


class TrtllmConfig(GenerationConfig):
    trtllm_cfg: TrtllmSpecificArgs
