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


class SglangSpecificArgs(TypedDict):
    """SGLang-specific configuration arguments.

    Most fields below map directly to SGLang's ServerArgs (see:
    https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py).
    """

    model_path: NotRequired[str]
    gpus_per_server: NotRequired[int]
    random_seed: NotRequired[int]
    skip_tokenizer_init: NotRequired[bool]
    disable_cuda_graph: NotRequired[bool]
    disable_radix_cache: NotRequired[bool]
    disable_cuda_graph_padding: NotRequired[bool]
    enable_nccl_nvls: NotRequired[bool]
    disable_outlines_disk_cache: NotRequired[bool]
    disable_custom_all_reduce: NotRequired[bool]
    disable_overlap_schedule: NotRequired[bool]
    enable_mixed_chunk: NotRequired[bool]
    enable_dp_attention: NotRequired[bool]
    enable_ep_moe: NotRequired[bool]
    enable_torch_compile: NotRequired[bool]
    torch_compile_max_bs: NotRequired[int]
    cuda_graph_max_bs: NotRequired[int | None]
    cuda_graph_bs: NotRequired[list[int] | None]
    torchao_config: NotRequired[str]
    enable_nan_detection: NotRequired[bool]
    enable_p2p_check: NotRequired[bool]
    triton_attention_reduce_in_fp32: NotRequired[bool]
    triton_attention_num_kv_splits: NotRequired[int]
    num_continuous_decode_steps: NotRequired[int]
    enable_memory_saver: NotRequired[bool]
    allow_auto_truncate: NotRequired[bool]
    attention_backend: NotRequired[str | None]
    enable_multimodal: NotRequired[bool]
    sampling_backend: NotRequired[str | None]
    context_length: NotRequired[int | None]
    mem_fraction_static: NotRequired[float | None]
    max_running_requests: NotRequired[int | None]
    chunked_prefill_size: NotRequired[int | None]
    max_prefill_tokens: NotRequired[int]
    schedule_policy: NotRequired[str]
    schedule_conservativeness: NotRequired[float]
    cpu_offload_gb: NotRequired[int]
    dtype: NotRequired[str]
    kv_cache_dtype: NotRequired[str]
    dp_size: NotRequired[int]  # only used for dp attention
    pp_size: NotRequired[int]  # pipeline parallel size
    ep_size: NotRequired[int]
    # lora
    enable_lora: NotRequired[bool | None]
    max_lora_rank: NotRequired[int | None]
    lora_target_modules: NotRequired[list[str] | None]
    lora_paths: NotRequired[list[str] | None]
    max_loaded_loras: NotRequired[int]
    max_loras_per_batch: NotRequired[int]
    lora_backend: NotRequired[str]
    # logging
    log_level: NotRequired[str]
    log_level_http: NotRequired[str | None]
    log_requests: NotRequired[bool]
    log_requests_level: NotRequired[int]
    show_time_cost: NotRequired[bool]
    enable_metrics: NotRequired[bool]  # Exports Prometheus-like metrics
    # The interval (in decoding iterations) to log throughput
    # and update prometheus metrics
    decode_log_interval: NotRequired[int]
    # Extra loader arguments
    enable_multithread_load: NotRequired[bool]
    enable_fast_load: NotRequired[bool]
    # Server warmup
    skip_server_warmup: NotRequired[bool]


class SGLangConfig(GenerationConfig):
    """Configuration for SGLang runtime."""

    sglang_cfg: SglangSpecificArgs
    sglang_kwargs: NotRequired[dict[str, Any]]
