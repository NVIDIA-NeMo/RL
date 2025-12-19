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

from typing import Any, Literal, NotRequired, TypedDict, Union

from nemo_rl.models.generation.interfaces import GenerationConfig


class LoRAConfigDisabled(TypedDict):
    enabled: Literal[False]


class LoRAConfig(TypedDict):
    enabled: Literal[True]
    target_modules: list[str]
    exclude_modules: list[str]
    match_all_linear: NotRequired[bool]
    dim: int
    alpha: int
    dropout: float
    dropout_position: Literal["pre", "post"]
    lora_A_init: str
    use_triton: NotRequired[bool]


class DTensorConfigDisabled(TypedDict):
    enabled: Literal[False]


class DTensorConfig(TypedDict):
    enabled: Literal[True]
    env_vars: NotRequired[dict[str, str] | None]
    _v2: NotRequired[bool]
    cpu_offload: bool
    sequence_parallel: bool
    activation_checkpointing: bool
    tensor_parallel_size: int
    context_parallel_size: int
    custom_parallel_plan: str | None
    clear_cache_every_n_steps: NotRequired[int | None]
    lora_cfg: NotRequired[LoRAConfig | LoRAConfigDisabled]


class SequencePackingConfigDisabled(TypedDict):
    enabled: Literal[False]


class SequencePackingConfig(TypedDict):
    enabled: Literal[True]
    train_mb_tokens: int
    # Not required because some algorithms like SFT don't calculate log probs
    logprob_mb_tokens: NotRequired[int]
    algorithm: str


class RewardModelConfig(TypedDict):
    enabled: bool
    reward_model_type: str


class MegatronOptimizerConfig(TypedDict):
    optimizer: str
    lr: float
    min_lr: float
    weight_decay: float
    bf16: bool
    fp16: bool
    params_dtype: str
    # adam
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    # sgd
    sgd_momentum: float
    # distributed optimizer
    use_distributed_optimizer: bool
    use_precision_aware_optimizer: bool
    clip_grad: float
    # knob to enable optimizer cpu offload
    optimizer_cpu_offload: bool
    # knob to set the fraction of parameters to keep on CPU
    # currently if optimizer_cpu_offload is true, this knob must be 1.0
    optimizer_offload_fraction: float


class MegatronSchedulerConfig(TypedDict):
    start_weight_decay: float
    end_weight_decay: float
    weight_decay_incr_style: str
    lr_decay_style: str
    lr_decay_iters: NotRequired[int | None]
    lr_warmup_iters: int
    lr_warmup_init: float


class MegatronDDPConfig(TypedDict):
    grad_reduce_in_fp32: bool
    overlap_grad_reduce: bool
    overlap_param_gather: bool
    use_custom_fsdp: bool
    data_parallel_sharding_strategy: str


# Type exists to be lax if not specified
class MegatronConfigDisabled(TypedDict):
    enabled: Literal[False]


class MegatronConfig(TypedDict):
    enabled: Literal[True]
    env_vars: NotRequired[dict[str, str] | None]
    # 1 is the minimum recommendation for RL since we almost always need to offload before beginning generation.
    # Setting to 0 is faster, but you are more likely to run out of GPU memory. In SFT/DPO, the default is 0.
    empty_unused_memory_level: int
    activation_checkpointing: bool
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    num_layers_in_first_pipeline_stage: int | None
    num_layers_in_last_pipeline_stage: int | None
    context_parallel_size: int
    pipeline_dtype: str
    sequence_parallel: bool
    freeze_moe_router: bool
    expert_tensor_parallel_size: int
    expert_model_parallel_size: int
    # If True, defer the casting of logits to float32 until the backward pass.
    # If you are using logprob_chunk_size, you must set this to True.
    defer_fp32_logits: NotRequired[bool]
    # gives ~20% training perf speedup with sequence packing
    apply_rope_fusion: bool
    # gives ~25% training perf speedup with sequence packing and apply_rope_fusion
    bias_activation_fusion: bool
    # Force overwrite of the initial checkpoint even if it exists (default: False)
    force_overwrite_initial_ckpt: NotRequired[bool]
    moe_per_layer_logging: bool
    optimizer: MegatronOptimizerConfig
    scheduler: MegatronSchedulerConfig
    distributed_data_parallel_config: MegatronDDPConfig


class TokenizerConfig(TypedDict):
    name: str
    chat_template: NotRequired[str]
    # Arguments to pass to tokenizer.apply_chat_template(...). This can be used to pass kwargs like enable_thinking=true
    chat_template_kwargs: NotRequired[dict[str, Any] | None]


class PytorchOptimizerConfig(TypedDict):
    name: str
    kwargs: dict[str, Any]


class SinglePytorchSchedulerConfig(TypedDict):
    name: str
    kwargs: dict[str, Any]


class SinglePytorchMilestonesConfig(TypedDict):
    milestones: list[int]  # Used in SequentialLR configuration


SchedulerMilestones = dict[str, list[int]]


class DynamicBatchingConfigDisabled(TypedDict):
    enabled: Literal[False]


class DynamicBatchingConfig(TypedDict):
    # dynamic_batching improves performance by ensuring logprob and training microbatches
    # have a sufficent number of tokens to maximize GPU utilization. Specifically, variable length
    # responses are sorted by sequence length and bucketed into microbatches with a total
    # amount of tokens is approximately close to 'train_mb_tokens' and 'logprob_mb_tokens' for the
    # training and logprob stages respectively.
    enabled: Literal[True]
    train_mb_tokens: int
    logprob_mb_tokens: NotRequired[int]  # Only used for some algorithms
    sequence_length_round: int


class PolicyConfig(TypedDict):
    model_name: str
    tokenizer: TokenizerConfig
    train_global_batch_size: int
    train_micro_batch_size: int
    logprob_batch_size: NotRequired[int]
    # If set, log probability computation is chunked along the sequence dimension to avoid GPU OOM (especially during backward pass).
    # Within each chunk loop, logits casting (from float16/bfloat16 to float32) is done to prevent holding the entire float32 logits tensor in memory.
    # If None, chunking is disabled and the full sequence is processed at once.
    logprob_chunk_size: NotRequired[int | None]
    generation: NotRequired[GenerationConfig]
    generation_batch_size: NotRequired[
        int
    ]  # used in static batched (framework) generation
    precision: str
    reward_model_cfg: NotRequired[RewardModelConfig]
    dtensor_cfg: DTensorConfig | DTensorConfigDisabled
    megatron_cfg: NotRequired[MegatronConfig | MegatronConfigDisabled]
    hf_config_overrides: NotRequired[dict[str, Any]]
    dynamic_batching: DynamicBatchingConfig | DynamicBatchingConfigDisabled
    sequence_packing: NotRequired[SequencePackingConfig | SequencePackingConfigDisabled]
    make_sequence_length_divisible_by: int
    max_total_sequence_length: int
    # This sets the clipping norm for the DTensorPolicyWorkers (Megatron's is called clip_grad)
    max_grad_norm: NotRequired[float | int | None]
    refit_buffer_size_gb: NotRequired[float]
    optimizer: NotRequired[PytorchOptimizerConfig | None]
    scheduler: NotRequired[
        list[SinglePytorchSchedulerConfig | SinglePytorchMilestonesConfig]
        | SchedulerMilestones
        | None
    ]
