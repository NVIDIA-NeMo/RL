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


class SGLangConfig():
    """Configuration for SGLang runtime. Refer to:
    https://github.com/sgl-project/sglang for detailed documentation.
    """

    model_path: str = ""
    random_seed: int = 1
    skip_tokenizer_init: bool = False
    disable_cuda_graph: bool = False
    disable_radix_cache: bool = True
    disable_cuda_graph_padding: bool = False
    enable_nccl_nvls: bool = False
    disable_outlines_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    disable_overlap_schedule: bool = False
    enable_mixed_chunk: bool = False
    enable_dp_attention: bool = False
    enable_ep_moe: bool = False
    enable_torch_compile: bool = False
    torch_compile_max_bs: int = 32
    cuda_graph_max_bs: int | None = None
    cuda_graph_bs: list[int] | None = None
    torchao_config: str = ""
    enable_nan_detection: bool = False
    enable_p2p_check: bool = False
    triton_attention_reduce_in_fp32: bool = False
    triton_attention_num_kv_splits: int = 8
    num_continuous_decode_steps: int = 1
    enable_memory_saver: bool = False
    allow_auto_truncate: bool = False
    attention_backend: str | None = "fa3"
    enable_multimodal: bool = False
    sampling_backend: str | None = None
    context_length: int | None = 32768
    mem_fraction_static: float | None = 0.9
    max_running_requests: int | None = None
    # NOTE: chunked_prefill_size is by default 8192 on GPUs with 80GB mem in SGLang,
    # but we disable it to avoid precision issues
    chunked_prefill_size: int | None = -1
    max_prefill_tokens: int = 32768
    schedule_policy: str = "lpm"
    schedule_conservativeness: float = 1.0
    cpu_offload_gb: int = 0
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"
    dp_size: int = 1  # only used for dp attention
    ep_size: int = 1
    # lora
    enable_lora: bool | None = None
    max_lora_rank: int | None = None
    lora_target_modules: list[str] | None = None
    lora_paths: list[str] | None = None
    max_loaded_loras: int = 1
    max_loras_per_batch: int = 1
    lora_backend: str = "triton"
    # logging
    log_level: str = "warning"
    log_level_http: str | None = "warning"
    log_requests: bool = False
    log_requests_level: int = 0
    show_time_cost: bool = False
    enable_metrics: bool = True  # Exports Prometheus-like metrics
    # The interval (in decoding iterations) to log throughput
    # and update prometheus metrics
    decode_log_interval: int = 1
    # Extra loader arguments
    # NOTE: These arguments will be parsed into a dict json-string
    # and passed as `model_loader_extra_config` to SGLang.
    enable_multithread_load: bool = False
    enable_fast_load: bool = False

    