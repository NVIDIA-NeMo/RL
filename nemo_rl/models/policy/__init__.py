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

from typing import Any, Literal, NotRequired, TypedDict, Union, cast

from pydantic import BaseModel, StrictBool, StrictInt

from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    need_top_k_or_top_p_filtering,
)
from nemo_rl.data.packing.shared_prefix_tensors import (
    resolve_shared_prefix_parallel_topology,
    resolve_shared_prefix_physical_padding_multiple,
)
from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.policy.draft_config import Eagle3DraftConfig
from nemo_rl.utils.checkpoint import PretrainedCheckpointConfig


def _patch_transformers_tokenizer_class_set():
    """Undo the transformers block on deepseek_v3 tokenizers.

    Root cause: transformers >=5.4 lists "deepseek_v3" in two internal
    registries -- MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS (a set) and
    TOKENIZER_MAPPING_NAMES (a dict pinning it to "TokenizersBackend"). Together
    they force the fast tokenizer backend and suppress trust_remote_code, so
    AutoTokenizer can only load via a local tokenizer.json. Models like
    Moonlight-16B-A3B ship no tokenizer.json (only tiktoken.model + a remote-code
    TikTokenTokenizer), so offline loading fails.

    Removing both entries restores the trust_remote_code / auto_map path.
    discard/pop-with-default are no-ops when the entries are absent, so this is
    safe on any transformers version in the currently-supported range.

    Placed here (nemo_rl/models/policy/__init__.py) so it fires exactly once
    per process the first time any policy code is imported -- covers the driver
    (via nemo_rl.algorithms.grpo) and every policy worker (Megatron / DTensor /
    DTensor v2 all import from nemo_rl.models.policy) without polluting nemo_rl
    consumers that don't touch tokenizers.
    """
    import transformers
    from packaging.version import Version as PkgVersion

    # Transformers 5.12.1 still ships both registry entries, so the patch remains
    # load-bearing across the currently supported backend environments.
    # TODO: remove this patch (and the assert below) once the deepseek_v3
    # entries actually disappear upstream.
    # https://github.com/NVIDIA-NeMo/RL/issues/2764
    assert PkgVersion(transformers.__version__) < PkgVersion("5.13.0"), (
        f"transformers {transformers.__version__} detected. "
        "The deepseek_v3 tokenizer-blocklist patch was verified against <5.13. "
        "Check if the upstream fix now applies and remove this patch if so."
    )

    from transformers import AutoTokenizer
    from transformers.models.auto.tokenization_auto import (
        MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS,
        TOKENIZER_MAPPING_NAMES,
    )

    _original_from_pretrained = AutoTokenizer.from_pretrained

    def _patched_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        try:
            # DSV3 goes here: the transformers blocklist routes its
            # tokenizer.json around LlamaTokenizerFast.__init__'s Llama-specific
            # post-processing, which would corrupt DSV3 special tokens.
            return _original_from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs
            )
        except Exception:
            # Moonlight goes here: it ships no tokenizer.json (only
            # tiktoken.model + remote-code TikTokenTokenizer), so the blocklist
            # prevents loading. Strip deepseek_v3 from the registries so
            # trust_remote_code / auto_map takes over.
            MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS.discard("deepseek_v3")
            TOKENIZER_MAPPING_NAMES.pop("deepseek_v3", None)
            return _original_from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs
            )

    AutoTokenizer.from_pretrained = _patched_from_pretrained


_patch_transformers_tokenizer_class_set()


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
    # Warm start: path to a PEFT adapter checkpoint (a directory containing
    # adapter_model.safetensors + adapter_config.json, e.g. a previous run's
    # step_*/policy/weights or step_*/policy/weights/model directory) whose
    # adapter weights initialize this run's LoRA modules. Ignored when resuming
    # from a NeMo RL training checkpoint (resumed weights take precedence for
    # the policy; the reference policy still anchors to the restored adapters).
    restore_from: NotRequired[str | None]


class AutomodelBackendConfig(TypedDict):
    """Configuration for custom MoE implementation backend in Automodel.

    Used when setting the backend in automodel_kwargs in your config.
    Alternatively, pass `force_hf: true` in automodel_kwargs to fall back
    to the HuggingFace implementation.
    """

    # Hydra target class path (e.g., "nemo_automodel.components.models.common.utils.BackendConfig")
    _target_: str
    # Attention implementation: "te" (Transformer Engine), "flex" (FlexAttention), etc.
    attn: NotRequired[str]
    # Linear layer implementation: "te" (Transformer Engine), etc.
    linear: NotRequired[str]
    # RMSNorm implementation: "te" (Transformer Engine), etc.
    rms_norm: NotRequired[str]
    # MoE expert GEMM backend: "torch" (per-expert loop), "te" (TE GroupedLinear),
    # "gmm" (grouped_gemm.ops.gmm), "torch_mm" (torch._grouped_mm).
    experts: NotRequired[str]
    # MoE token dispatcher: "torch" (DTensor all-gather/reduce-scatter) or "hybridep".
    dispatcher: NotRequired[str]
    # Use fake balanced gate for testing/debugging MoE
    fake_balanced_gate: NotRequired[bool]
    # Enable HuggingFace state dict adapter for checkpoint saving/loading plus refit support for RL
    # This should almost always be set to True when using a custom MoE implementation. Set to False only for specific use cases like debugging or performance testing.
    enable_hf_state_dict_adapter: NotRequired[bool]
    # Enable FSDP-specific optimizations
    enable_fsdp_optimizations: NotRequired[bool]
    # Precision for the MoE gate computation (e.g., "float64", "float32")
    gate_precision: NotRequired[str]


class AutomodelFreezeConfig(TypedDict):
    """Which sub-modules of a multi-modal Automodel to freeze during training.

    Used when setting freeze_config in automodel_kwargs in your config.
    """

    freeze_vision_tower: NotRequired[bool]
    freeze_audio_tower: NotRequired[bool]
    freeze_language_model: NotRequired[bool]


class AutomodelKwargs(TypedDict):
    # Whether to use Liger kernel optimizations (default: false)
    use_liger_kernel: NotRequired[bool]
    # Backend configuration for MoE models
    backend: NotRequired[AutomodelBackendConfig]
    # Freeze configuration for multi-modal models (vision/audio/language towers)
    freeze_config: NotRequired[AutomodelFreezeConfig]
    # Force the HuggingFace model implementation instead of the custom one.
    # Set to true if the custom model's state_dict_adapter doesn't implement
    # convert_single_tensor_to_hf (required for weight syncing). This is
    # auto-detected and set at runtime if not explicitly configured.
    # See: https://github.com/NVIDIA-NeMo/RL/issues/2072
    force_hf: NotRequired[bool]


class DTensorConfigDisabled(TypedDict):
    enabled: Literal[False]


class MoEParallelizerOptions(TypedDict):
    """MoE parallelizer config options (mirrors Automodel's MoEParallelizerConfig)."""

    ignore_router_for_ac: NotRequired[bool]
    reshard_after_forward: NotRequired[bool]
    lm_head_precision: NotRequired[str | None]
    wrap_outer_model: NotRequired[bool]


class AutomodelCheckpointConfig(TypedDict, total=False):
    """Automodel checkpoint settings used by DTensor v2."""

    model_save_format: Literal["torch_save", "safetensors"] | None
    save_consolidated: Literal["false", "final", "every"]
    single_rank_consolidation: bool
    consolidation_timeout_minutes: int


class DTensorConfig(TypedDict):
    enabled: Literal[True]
    env_vars: NotRequired[dict[str, str] | None]
    # Distributed parallelism sizes
    # data_parallel_size is derived from world_size / (tp * cp * ep)
    tensor_parallel_size: int
    context_parallel_size: int
    expert_parallel_size: NotRequired[int]
    # Size of the HSDP replicate dimension within the data-parallel axis.
    dp_replicate_size: NotRequired[int]
    # Distributed config options (mirrors Automodel's FSDP2Config)
    sequence_parallel: bool
    activation_checkpointing: bool
    cpu_offload: bool
    custom_parallel_plan: NotRequired[str | None]
    defer_fsdp_grad_sync: NotRequired[bool]
    # MoE parallelizer config
    moe_parallelizer: NotRequired[MoEParallelizerOptions]
    # Model config
    lora_cfg: NotRequired[LoRAConfig | LoRAConfigDisabled]
    automodel_kwargs: NotRequired[AutomodelKwargs]
    # Runtime
    clear_cache_every_n_steps: NotRequired[int | None]
    checkpoint: NotRequired[AutomodelCheckpointConfig]


class SequencePackingConfigDisabled(TypedDict):
    enabled: Literal[False]


class SequencePackingConfig(TypedDict):
    enabled: Literal[True]
    train_mb_tokens: int
    # Not required because some algorithms like SFT don't calculate log probs
    logprob_mb_tokens: NotRequired[int]
    algorithm: str
    # Preserve the packer's order (or omit for backward compatibility), or
    # execute each DP rank's assigned bins largest-first for allocator reuse.
    microbatch_order: NotRequired[Literal["packer", "largest_first"]]
    fuse_loss: NotRequired[bool]
    pair_grouping_key: NotRequired[Literal["pair_index"]]
    max_sequences_per_bin: NotRequired[int]


class SharedPrefixTrainingConfig(BaseModel, extra="allow"):
    """Controls prompt-prefix sharing in policy training forwards.

    ``disabled`` preserves the existing packing and model execution. ``observe``
    may report prefix-reuse opportunity metrics but must not alter execution.
    ``dense`` is an explicit comparison control: it uses conventional packing
    while permitting the same optional router arithmetic and evaluation MTP
    bypass as the shared arm. ``logprobs`` shares prefixes only for
    policy/reference logprob forwards;
    training APIs retain conventional packing and forward/backward execution.
    ``train`` shares prefixes in both stages. Both execution modes are
    experimental and guarded by :func:`validate_shared_prefix_training_config`.
    """

    mode: Literal["disabled", "observe", "dense", "logprobs", "train"] = "disabled"
    pack_groups: StrictBool = False
    """Experimental forest execution under each stage's token budget."""
    repack_groups: StrictBool = False
    """Rebuild local slots per group; experimental pending numerical validation."""
    pack_dense_fallbacks: StrictBool = False
    """Repack local dense fallback rows with the configured conventional packer."""
    merge_dense_fallbacks: StrictBool = False
    """Absorb whole fallback bins into training forests, preserving MTP groups."""
    evaluation_packing: StrictBool = False
    """Omit the expanded MTP budget during evaluation; changes forward packing."""

    align_data_parallel: StrictBool = False
    """Align packed forward counts across the model world using real-row splits."""
    match_logprob_training_layout: StrictBool = False
    """Use training's DP weights, token budget and execution plan for logprobs.

    Intended for same-weight consistency in streaming shared-prefix GRPO.
    Logprob execution remains forward-only and may still bypass auxiliary MTP.
    """
    training_dense_bins: StrictBool = False
    """Pack full training rows, align DP counts, then share prefixes within bins."""
    preserve_training_prefixes_during_alignment: StrictBool = False
    """Experimental shared reconstruction of expanded-budget training splits."""
    bypass_evaluation_mtp: StrictBool = False
    """Skip HybridModel MTP in logprobs; requires shared execution or dense control."""
    uniform_router_gating: StrictBool = False
    """Use MCore's fixed router row blocks in all policy forwards and backwards.

    Opt-in numerical reference for shared execution, dense fallbacks and MTP.
    Requires an execution or dense-control mode; disabled/observe must not
    change arithmetic.
    """

    shard_work_weights: tuple[StrictInt, StrictInt] | None = None
    """Optional (shared-backbone, expanded-token) weights for DP assignment.

    Real token lengths still determine packing. Disabled by default pending
    matched GPU timing and gradient validation of the configured weighting.
    """

    training_shard_work_weights: tuple[StrictInt, StrictInt] | None = None
    """Override DP assignment weights for training; logprobs keep common weights."""

    def token_budget_key_for(
        self, *, stage: Literal["train", "logprobs"]
    ) -> Literal["train_mb_tokens", "logprob_mb_tokens"]:
        """Select the packing budget without changing the actual autograd mode."""
        if stage == "train" or self.match_logprob_training_layout:
            return "train_mb_tokens"
        return "logprob_mb_tokens"

    def work_weights_for(
        self, *, stage: Literal["train", "logprobs"]
    ) -> tuple[int, int] | None:
        """Resolve weights only for an enabled shared-prefix execution stage."""
        if not self.enabled_for(stage=stage):
            return None
        if (
            stage == "train" or self.match_logprob_training_layout
        ) and self.training_shard_work_weights is not None:
            return self.training_shard_work_weights
        return self.shard_work_weights

    def enabled_for(self, *, stage: Literal["train", "logprobs"]) -> bool:
        """Whether a policy API stage should execute shared-prefix units."""
        return self.mode == "train" or (self.mode == "logprobs" and stage == "logprobs")


class RewardModelConfig(TypedDict):
    enabled: bool
    reward_model_type: str


class MegatronPeftConfigDisabled(TypedDict):
    enabled: Literal[False]


class MegatronPeftConfig(TypedDict):
    enabled: Literal[True]
    target_modules: list[str]
    exclude_modules: list[str]
    dim: int
    alpha: int
    dropout: float
    dropout_position: Literal["pre", "post"]
    lora_A_init_method: str
    lora_B_init_method: str
    a2a_experimental: bool
    lora_dtype: str | None
    # Share one adapter across grouped MoE experts on each EP rank. Set to
    # False to create one adapter per local expert.
    share_expert_adapters: NotRequired[bool]
    # Warm start: path to a native Megatron-Bridge PEFT checkpoint (an
    # iter_XXXXXXX directory, or a checkpoint root resolving to one) whose
    # adapter weights initialize this run's LoRA modules. The donor checkpoint
    # must have been saved with a matching peft configuration (dim and alpha
    # are validated against its run_config.yaml). Ignored when resuming from a
    # NeMo RL training checkpoint (resumed weights take precedence for the
    # policy; the reference policy still anchors to the restored adapters).
    restore_from: NotRequired[str | None]


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
    # knob to set the fraction of optimizer state and work to keep on CPU
    optimizer_offload_fraction: float
    # overlap optimizer state transfers with CPU optimizer updates
    overlap_cpu_optimizer_d2h_h2d: NotRequired[bool]
    # Precision-aware Adam moment / remainder dtypes (YAML strings resolved in setup).
    exp_avg_dtype: NotRequired[str]
    exp_avg_sq_dtype: NotRequired[str]
    store_param_remainders: NotRequired[bool]


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


class Fp8Config(TypedDict):
    # Master switch for FP8 training. When False, all other fields are ignored.
    enabled: bool
    # FP8 format used for the GEMMs (e.g. "e4m3").
    fp8: NotRequired[str]
    # FP8 scaling recipe (e.g. "blockwise").
    fp8_recipe: NotRequired[str]
    # When True, keep parameters in FP8. Can cause NaN token_mult_prob_error;
    # use with caution (see https://github.com/NVIDIA-NeMo/RL/issues/1164).
    fp8_param: NotRequired[bool]
    # Python import path for a Transformer Engine custom recipe quantizer factory.
    fp8_quantizer_factory: NotRequired[str]
    # When True, clear Transformer Engine's per-module _fp8_workspaces scratch
    # buffers in offload_before_refit (before weight transfer to the inference
    # engine). These FP8 workspace tensors anchor large CUDA segments and
    # aggravate allocator fragmentation across the train->offload->refit->generate
    # cycle. Useful for FP8 training runs that observe growing reserved GPU memory
    # after offload.
    force_clear_fp8_caches: NotRequired[bool]


class Fp4Config(BaseModel, extra="forbid"):
    """Transformer Engine NVFP4 training configuration."""

    enabled: bool
    fp4: str | None = None
    fp4_recipe: str = "nvfp4"
    fp4_param: bool = False


# Type exists to be lax if not specified
class MegatronConfigDisabled(TypedDict):
    enabled: Literal[False]


class MegatronCheckpointConfig(TypedDict, total=False):
    """Checkpoint knobs passed through to Megatron Bridge CheckpointConfig."""

    # Offload disk writes to a persistent background worker so save_checkpoint
    # returns after D2H staging.
    async_save: bool
    # Skip metadata recomputation after the first two saves when the sharded
    # state structure is constant across steps.
    ckpt_assume_constant_structure: bool
    # Field names match megatron.bridge CheckpointConfig (ckpt_ prefix).
    ckpt_fully_parallel_save_process_group: str  # "dp" | "ep_dp"
    ckpt_fully_parallel_load_process_group: str  # "dp" | "ep_dp"
    ckpt_fully_parallel_load_exchange_algo: str  # "broadcast" | "gather_rounds"


class MegatronConfig(TypedDict):
    enabled: Literal[True]
    env_vars: NotRequired[dict[str, str] | None]
    # Arbitrary model-provider attributes applied recursively to the Megatron
    # Bridge model config before model instantiation. Keys must match configurable
    # provider fields and must not duplicate first-class megatron_cfg fields.
    model_overrides: NotRequired[dict[str, Any]]
    # 1 is the minimum recommendation for RL since we almost always need to offload before beginning generation.
    # Setting to 0 is faster, but you are more likely to run out of GPU memory. In SFT/DPO, the default is 0.
    empty_unused_memory_level: int
    activation_checkpointing: bool
    # Recompute granularity: "full" recomputes all activations, "selective" recomputes
    # only specific modules (see recompute_modules). "selective" typically saves ~10-18GB
    # for MoE models while retaining higher throughput than "full".
    recompute_granularity: NotRequired[Literal["full", "selective"]]
    # Full recompute resolves to uniform chunks of one layer in Megatron setup.
    # Optional raw values are accepted only when they agree with that resolution.
    recompute_method: NotRequired[Literal["uniform"]]
    recompute_num_layers: NotRequired[int]
    # Modules to selectively recompute when recompute_granularity="selective".
    # MCore valid options: ["core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe", "shared_experts"].
    # Defaults to ["core_attn"] when None. Full list and per-module constraints:
    # https://github.com/NVIDIA/Megatron-LM/blob/d30c3ae5469fe3f6a64d4fd2e63b6e7f7844ea81/megatron/core/transformer/transformer_config.py#L483
    # when None. Use ["moe"] to recompute only expert activations (production-proven config).
    recompute_modules: NotRequired[list[str] | None]
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    num_layers_in_first_pipeline_stage: int | None
    num_layers_in_last_pipeline_stage: int | None
    context_parallel_size: int
    # Nemotron Omni RADIO/provider booleans. Omit any field to retain the model
    # provider's checkpoint/default value.
    radio_force_eval_mode: NotRequired[bool]
    radio_force_cpe_eval_mode: NotRequired[bool]
    # Nemotron Omni tower freeze booleans. Omit any field to retain the model
    # provider's checkpoint/default value.
    freeze_vision_model: NotRequired[bool]
    freeze_vision_projection: NotRequired[bool]
    freeze_sound_encoder: NotRequired[bool]
    freeze_sound_projection: NotRequired[bool]
    # Nemotron Omni vision-tower activation recomputation controls. Omit any
    # field to retain the model provider's checkpoint/default value.
    recompute_vision: NotRequired[bool]
    vision_recompute_granularity: NotRequired[str | None]
    vision_recompute_method: NotRequired[str | None]
    vision_recompute_num_layers: NotRequired[int | None]
    pipeline_dtype: str
    sequence_parallel: bool
    freeze_moe_router: bool
    # Optional multimodal provider controls. These map legacy Omni recipe
    # names onto the canonical NemotronOmniModel provider fields.
    freeze_vision_encoder: NotRequired[bool]
    freeze_vision_projector: NotRequired[bool]
    freeze_audio_encoder: NotRequired[bool]
    freeze_audio_projector: NotRequired[bool]
    moe_router_dtype: str | None
    moe_router_load_balancing_type: str | list[str]
    moe_router_bias_update_rate: float
    moe_permute_fusion: bool
    expert_tensor_parallel_size: int
    expert_model_parallel_size: int
    # If True, defer the casting of logits to float32 until the backward pass.
    # If you are using logprob_chunk_size, you must set this to True.
    defer_fp32_logits: NotRequired[bool]
    # gives ~20% training perf speedup with sequence packing
    apply_rope_fusion: bool
    # gives ~25% training perf speedup with sequence packing and apply_rope_fusion
    bias_activation_fusion: bool
    # Force reconvert from HF even if the checkpoint already exists (default: False)
    force_reconvert_from_hf: NotRequired[bool]
    # Attention backend available values:
    # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/enums.py#L60
    attention_backend: NotRequired[str]
    moe_per_layer_logging: bool
    # Set to true to enable DeepEP for expert parallel communication
    # Must set moe_token_dispatcher_type to 'flex'
    # Must set moe_shared_expert_overlap to False
    moe_enable_deepep: bool
    # The type of token dispatcher to use. The default is 'alltoall'.
    # Options are 'allgather','alltoall' and 'flex'
    # Use 'flex' when using DeepEP
    moe_token_dispatcher_type: str
    # Inference-only MoE dispatcher selection.
    # Options are 'nvls' (requires Hopper+ NVLink) and 'nccl' (fallback for non-NVLS systems).
    inference_moe_token_dispatcher_type: NotRequired[str]
    # Backend for grouped-GEMM during inference-optimized MoE forward.
    # Options: 'flashinfer', 'torch', 'vllm' (mcore default).
    inference_grouped_gemm_backend: NotRequired[str]
    # InferenceTopKRouter requires moe_router_num_groups=None
    # (used when transformer_impl='inference_optimized')
    moe_router_num_groups: NotRequired[int | None]
    moe_router_group_topk: NotRequired[int | None]
    # Transformer implementation backing the model. 'inference_optimized'
    # trains through the TE parent path and requires sequence_parallel with
    # TP>1 (enforced at setup).
    # Options are 'transformer_engine' and 'inference_optimized'.
    transformer_impl: NotRequired[str]
    # CUDA-graph implementation.
    # Options: 'none', 'local', 'transformer_engine', 'full_iteration'.
    cuda_graph_impl: NotRequired[str]
    # Training capture regions supported by Megatron-Core: attn, mlp, moe,
    # moe_router, moe_preprocess, and mamba. An empty list captures whole layers.
    # Scoped training capture requires the transformer_engine implementation.
    cuda_graph_modules: NotRequired[str | list[str]]
    # Number of training warmup steps before CUDA Graph capture. This is inactive
    # when cuda_graph_impl is 'none'; the Megatron-Core default is 3.
    cuda_graph_warmup_steps: NotRequired[int]
    # When True, each expert sees a fixed number of tokens for cuda-graph capture.
    # Required when cuda_graph_impl= 'local' with transformer_impl != 'inference_optimized'.
    moe_pad_experts_for_cuda_graph_inference: NotRequired[bool]
    # Can be used only with 'alltoall' token dispatcher
    moe_shared_expert_overlap: bool
    # Offload specific module activations to CPU to reduce peak GPU memory.
    # Works with both dense and MoE models. Different from
    # optimizer_cpu_offload which offloads optimizer states.
    # Requires transformer_engine. For TE >= 2.10.0 also requires
    # NVTE_CPU_OFFLOAD_V1=1 in the environment (validated by
    # Megatron-Bridge at runtime).
    fine_grained_activation_offloading: NotRequired[bool]
    # Modules to offload when fine_grained_activation_offloading is True.
    # Required (no default). Common examples: "core_attn", "attn_proj",
    # "expert_fc1", and "moe_act". Supported names depend on the pinned
    # Megatron-LM version and are validated by MCore. "attn_proj" requires
    # "core_attn". See the latest upstream module reference:
    # https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/fine_grained_activation_offloading.md#offloadable-modules
    offload_modules: NotRequired[list[str] | None]
    # Create gloo process groups during Megatron distributed init.
    # Omitted: use the Megatron Bridge default.
    use_gloo_process_groups: NotRequired[bool]
    # Enable grouped GEMM for MoE experts via CUTLASS. Significant throughput
    # gain when multiple experts are assigned per rank (num_local_experts > 1).
    # Requires TE >= 1.11.0 for FP8 and Ampere (sm_80) or newer.
    moe_grouped_gemm: NotRequired[bool]
    # HybridEP settings for MoE expert parallelism (requires moe_token_dispatcher_type='flex')
    # See: https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep
    moe_flex_dispatcher_backend: NotRequired[str]
    moe_hybridep_num_sms: NotRequired[int]
    # Align packed inputs once before the model forward instead of padding in every
    # MoE layer. Currently requires NeMo-owned packing, PP=1, and MTP disabled.
    moe_hybridep_prepad_packed_inputs: NotRequired[bool]
    # Number of HybridEP ranks per NVLink domain (default: min(expert_model_parallel_size, 64))
    hybridep_num_ranks_per_nvlink_domain: NotRequired[int]
    # Enable multi-node NVLink support (default: expert_model_parallel_size > 4)
    hybridep_use_mnnvl: NotRequired[bool]
    peft: NotRequired[MegatronPeftConfig | MegatronPeftConfigDisabled]
    optimizer: MegatronOptimizerConfig
    scheduler: MegatronSchedulerConfig
    distributed_data_parallel_config: MegatronDDPConfig
    # Megatron-specific checkpointing knobs (async save, parallel I/O, etc.)
    checkpoint: NotRequired[MegatronCheckpointConfig]
    gradient_accumulation_fusion: NotRequired[bool]
    # Enable fused weighted squared ReLU when the architecture supports it.
    use_fused_weighted_squared_relu: NotRequired[bool]
    # When True, computes per-token logprobs with a chunked linear cross-entropy
    # fusion kernel directly from hidden states, avoiding materialization of the
    # full [batch, seq_len, vocab_size] logit tensor. This significantly reduces
    # peak GPU memory, extending the maximum trainable sequence length (e.g. from
    # <65K to >100K tokens). Supported for SFT, DPO, and GRPO. Not compatible with
    # context parallelism, sequence packing, or top-k/top-p training-time filtering.
    use_fused_linear_logprobs: NotRequired[bool]
    # Number of tokens per chunk when computing fused linear logprobs.
    # Smaller values reduce peak memory further but may decrease throughput.
    fused_linear_logprobs_chunk_size: NotRequired[int]
    # Emit fp32 logits from Megatron-LM's LM head. Set the matching
    # generation.vllm_cfg.fp32_lm_head flag for vLLM generation: applying this
    # to only one engine makes the generation/training logprob mismatch worse.
    # No effect when use_fused_linear_logprobs is set, which bypasses the
    # output layer's logits path.
    fp32_lm_head: NotRequired[bool]
    # When mtp_num_layers=0, Multi-Token Prediction is disabled.
    mtp_num_layers: NotRequired[int]
    # MTP loss weight added to the main next-token loss (0.0 disables the MTP loss contribution).
    mtp_loss_scaling_factor: NotRequired[float]
    # Populated by the algorithm before Megatron setup to size the LR scheduler.
    train_iters: NotRequired[int]
    # When True, repeat a single MTP layer mtp_num_layers times instead of using distinct layers.
    mtp_use_repeated_layer: NotRequired[bool]
    # When True, detach MTP heads from the main model so MTP loss does not affect main-model gradients.
    mtp_detach_heads: NotRequired[bool]
    # When True, clear the RotaryEmbedding LRU cache and MoE token dispatcher
    # routing tensors in offload_before_refit (before weight transfer to the
    # inference engine). Useful when training and logprob runs use different
    # sequence lengths (rope cache) or for MoE models with activation recompute
    # (dispatcher reference cycles).
    clear_memory_caches_before_refit: NotRequired[bool]
    # FP8 quantization settings for the Megatron training backend.
    fp8_cfg: NotRequired[Fp8Config]
    # TE NVFP4 training settings. Unknown keys are rejected by Fp4Config so
    # misspelled precision controls cannot silently fall back to defaults.
    fp4_cfg: NotRequired[Fp4Config]
    # Keep the first/last N transformer blocks in BF16 under FP8/FP4 training.
    first_last_layers_bf16: NotRequired[bool]
    num_layers_at_start_in_bf16: NotRequired[int]
    num_layers_at_end_in_bf16: NotRequired[int]
    # Path to a per-module Transformer Engine precision recipe loaded into
    # Megatron quant_recipe.
    te_precision_config_file: NotRequired[str]
    # Passed through to the Megatron model's freeze() method.
    # Supported keys are model-specific, such as freeze_vision_model,
    # freeze_vision_projection, and freeze_language_model.
    freeze_config: NotRequired[dict[str, Any]]


class TokenizerConfig(TypedDict):
    name: str
    # None selects NeMo-RL's passthrough prompt/response template.
    chat_template: NotRequired[str | None]
    # Arguments to pass to tokenizer.apply_chat_template(...). This can be used to pass kwargs like enable_thinking=true
    chat_template_kwargs: NotRequired[dict[str, Any] | None]
    # Arguments forwarded to tokenizer loading via get_tokenizer.
    tokenizer_kwargs: NotRequired[dict[str, Any] | None]
    # Multimodal configs
    audio: NotRequired[dict[str, Any]]
    video: NotRequired[dict[str, Any]]
    use_processor: NotRequired[bool]
    # Opt-in fastokens Rust-backed BPE tokenizer for NeMo-RL tokenization.
    # Defaults to off when absent; NRL_USE_FASTOKENS overrides this and also sets VLLM_USE_FASTOKENS.
    use_fastokens: NotRequired[bool]


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


class RouterReplayConfigDisabled(TypedDict):
    enabled: Literal[False]


class RouterReplayConfig(TypedDict):
    enabled: Literal[True]


class OnPolicyDistillationFullTransport(TypedDict):
    """Resolved full-vocabulary MOPD settings carried to the policy workers.

    A ``model_dump`` of ``OnPolicyDistillationFullConfig`` plus the resolved
    ``payload_field`` and ``teacher_index_field``. That BaseModel remains the
    authoritative schema and the only place defaults are declared, so readers
    must take these keys as required rather than supplying their own fallbacks.
    """

    enabled: bool
    teacher_payload: Literal["hidden_states", "logits"]
    divergence: Literal["reverse_kl"]
    payload_dtype: Literal["bfloat16", "float16", "float32"]
    chunk_size: int | None
    teacher_lm_head_lifecycle: Literal["none", "offload", "evict"]
    validate_decomposition: bool
    payload_field: str
    # Per-sample teacher-identity column, routing each row's payload to the
    # right teacher LM-head shard. Only the hidden-state path needs it; the
    # logits payload ships an already-projected distribution, so it is None.
    teacher_index_field: str | None


class PolicyConfig(TypedDict):
    model_name: str
    tokenizer: TokenizerConfig
    train_global_batch_size: int
    train_micro_batch_size: int
    offload_optimizer_for_logprob: bool
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
    draft: NotRequired[Eagle3DraftConfig]
    pretrained_checkpoint: NotRequired[PretrainedCheckpointConfig]
    # Resolved once by the driver and carried to the student workers and (via
    # deepcopy) to the teacher group. Absent means full-vocabulary MOPD is off.
    on_policy_distillation_full: NotRequired[OnPolicyDistillationFullTransport]
    router_replay: NotRequired[RouterReplayConfig | RouterReplayConfigDisabled]
    hf_config_overrides: NotRequired[dict[str, Any]]
    dynamic_batching: DynamicBatchingConfig | DynamicBatchingConfigDisabled
    sequence_packing: NotRequired[SequencePackingConfig | SequencePackingConfigDisabled]
    shared_prefix_training: NotRequired[SharedPrefixTrainingConfig]
    make_sequence_length_divisible_by: int
    max_total_sequence_length: int
    # This sets the clipping norm for the DTensorPolicyWorkers (Megatron's is called clip_grad)
    max_grad_norm: NotRequired[float | int | None]
    refit_buffer_size_gb: NotRequired[float | int]
    optimizer: NotRequired[PytorchOptimizerConfig | None]
    scheduler: NotRequired[
        list[SinglePytorchSchedulerConfig | SinglePytorchMilestonesConfig]
        | SchedulerMilestones
        | None
    ]

    # quantization configs
    quant_cfg: NotRequired[str | None]
    quant_calib_data: NotRequired[str | None]
    quant_calib_size: NotRequired[int | None]
    quant_batch_size: NotRequired[int | None]
    quant_sequence_length: NotRequired[int | None]
    # If true, use standard Megatron layer specs while keeping ModelOpt
    # quantization enabled. Useful for faster QARL runs and logged in configs.
    disable_modelopt_layer_spec: NotRequired[bool]

    is_vlm: NotRequired[bool]

    # FQN of a worker extension class to use instead of the resolved default
    # policy worker. Must be a subclass of the resolved worker and cannot be
    # combined with quant_cfg. Its runtime environment must already be in
    # ACTOR_ENVIRONMENT_REGISTRY.
    worker_extension_cls_fqn: NotRequired[str | None]


def get_shared_prefix_training_config(
    config: PolicyConfig,
) -> SharedPrefixTrainingConfig:
    """Return the validated shared-prefix block, including its legacy default.

    ``PolicyConfig`` is still a legacy ``TypedDict``, so older configs may omit
    this newly introduced block. The default remains centralized on
    :class:`SharedPrefixTrainingConfig`; callers should use this accessor rather
    than inventing an absence fallback.
    """
    shared_prefix_config = config.get("shared_prefix_training")
    if shared_prefix_config is None:
        return SharedPrefixTrainingConfig()
    return SharedPrefixTrainingConfig.model_validate(shared_prefix_config)


def validate_shared_prefix_training_config(
    config: PolicyConfig,
) -> SharedPrefixTrainingConfig:
    """Validate backend-independent shared-prefix training requirements.

    Observation mode is deliberately backend-neutral and execution-neutral.
    The resolved TP/PP/CP topology and matching MCore capability are validated
    later, after Megatron Bridge resolves the concrete model provider. Accepting
    TP/SP or CP here does not advertise support: the run remains fail-closed
    unless MCore exports the exact topology and physical-layout capabilities.
    """
    shared_prefix_config = get_shared_prefix_training_config(config)
    if (
        shared_prefix_config.match_logprob_training_layout
        and shared_prefix_config.mode != "train"
    ):
        raise ValueError("match_logprob_training_layout requires shared train mode")
    if shared_prefix_config.training_dense_bins and not (
        shared_prefix_config.mode == "train"
        and shared_prefix_config.pack_groups
        and shared_prefix_config.repack_groups
        and shared_prefix_config.align_data_parallel
    ):
        raise ValueError(
            "training_dense_bins requires train mode, pack_groups, repack_groups, "
            "and align_data_parallel"
        )
    if shared_prefix_config.training_shard_work_weights is not None:
        weights = shared_prefix_config.training_shard_work_weights
        if min(weights) < 0 or sum(weights) <= 0:
            raise ValueError(
                "training_shard_work_weights must be nonnegative with a positive sum"
            )
        if shared_prefix_config.mode != "train":
            raise ValueError("training_shard_work_weights requires shared train mode")
    if shared_prefix_config.shard_work_weights is not None:
        weights = shared_prefix_config.shard_work_weights
        if min(weights) < 0 or sum(weights) <= 0:
            raise ValueError(
                "shard_work_weights must be nonnegative with a positive sum"
            )
        if not shared_prefix_config.enabled_for(stage="logprobs"):
            raise ValueError(
                "shard_work_weights requires shared logprobs or train mode"
            )
    if (
        shared_prefix_config.uniform_router_gating
        and shared_prefix_config.mode != "dense"
        and not shared_prefix_config.enabled_for(stage="logprobs")
    ):
        raise ValueError(
            "uniform_router_gating requires logprobs or train mode, or explicit dense control"
        )
    if (
        shared_prefix_config.bypass_evaluation_mtp
        and shared_prefix_config.mode != "dense"
        and not shared_prefix_config.enabled_for(stage="logprobs")
    ):
        raise ValueError(
            "bypass_evaluation_mtp requires logprobs or train mode, or explicit dense control"
        )
    if shared_prefix_config.mode == "dense":
        megatron_config = config.get("megatron_cfg")
        if megatron_config is None or not megatron_config["enabled"]:
            raise ValueError("dense comparison control requires the Megatron backend")
    if not shared_prefix_config.enabled_for(stage="logprobs"):
        return shared_prefix_config

    generation_config = config.get("generation")
    if generation_config is not None and need_top_k_or_top_p_filtering(
        TrainingSamplingParams(
            top_k=generation_config["top_k"],
            top_p=generation_config["top_p"],
            temperature=generation_config["temperature"],
        )
    ):
        raise ValueError(
            f"policy.shared_prefix_training.mode={shared_prefix_config.mode} "
            "does not support top-k/top-p filtering. Set "
            "policy.generation.top_p=1.0 and policy.generation.top_k to "
            "null, 0, or -1; temperature scaling remains supported."
        )

    if (
        shared_prefix_config.preserve_training_prefixes_during_alignment
        and not shared_prefix_config.align_data_parallel
    ):
        raise ValueError("Preserving training prefixes requires align_data_parallel")
    if shared_prefix_config.align_data_parallel:
        if not (
            shared_prefix_config.pack_groups and shared_prefix_config.repack_groups
        ):
            raise ValueError(
                "align_data_parallel requires pack_groups and repack_groups"
            )
        if shared_prefix_config.merge_dense_fallbacks:
            raise ValueError(
                "Distributed packing does not support merge_dense_fallbacks"
            )
        if (
            shared_prefix_config.evaluation_packing
            and not shared_prefix_config.bypass_evaluation_mtp
        ):
            raise ValueError(
                "Distributed evaluation packing requires uniform MTP bypass"
            )

    megatron_config = config.get("megatron_cfg")
    if megatron_config is None or megatron_config["enabled"] is not True:
        raise ValueError(
            f"policy.shared_prefix_training.mode={shared_prefix_config.mode} requires "
            "policy.megatron_cfg.enabled=true. Observation mode remains "
            "available with policy.shared_prefix_training.mode=observe."
        )
    megatron_config = cast(MegatronConfig, megatron_config)

    peft_config = megatron_config.get("peft")
    if peft_config is not None and peft_config["enabled"]:
        raise ValueError(
            f"policy.shared_prefix_training.mode={shared_prefix_config.mode} currently requires "
            "policy.megatron_cfg.peft.enabled=false; PEFT/LoRA adapter dropout "
            "and shared-prefix gradient semantics have not been validated."
        )

    sequence_packing_config = config.get("sequence_packing")
    if sequence_packing_config is None or not sequence_packing_config["enabled"]:
        raise ValueError(
            f"policy.shared_prefix_training.mode={shared_prefix_config.mode} requires "
            "policy.sequence_packing.enabled=true."
        )

    try:
        tp_size, cp_size, _sequence_parallel = resolve_shared_prefix_parallel_topology(
            tp_size=megatron_config["tensor_model_parallel_size"],
            cp_size=megatron_config["context_parallel_size"],
            sequence_parallel=megatron_config["sequence_parallel"],
        )
    except ValueError as error:
        raise ValueError(
            f"policy.shared_prefix_training.mode={shared_prefix_config.mode} requires positive integer "
            "policy.megatron_cfg.tensor_model_parallel_size/context_parallel_size "
            "and a boolean policy.megatron_cfg.sequence_parallel that is true "
            f"exactly when TP>1: {error}"
        ) from error

    try:
        padding_multiple = resolve_shared_prefix_physical_padding_multiple(
            tp_size=tp_size,
            cp_size=cp_size,
            padding_multiple=config.get("make_sequence_length_divisible_by"),
        )
    except ValueError as error:
        raise ValueError(
            "policy.make_sequence_length_divisible_by must be absent/None or "
            "a positive integer multiple of the shared-prefix TP/CP topology "
            f"alignment; got {config.get('make_sequence_length_divisible_by')!r}."
        ) from error
    for capacity_key in ("train_mb_tokens", "logprob_mb_tokens"):
        capacity = sequence_packing_config.get(capacity_key)
        if capacity is None:
            continue
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity < 1
            or capacity % padding_multiple
        ):
            raise ValueError(
                "shared-prefix physical padding requires "
                f"policy.sequence_packing.{capacity_key} to be a positive "
                f"multiple of resolved padding M={padding_multiple}; got {capacity}."
            )

    if megatron_config["pipeline_model_parallel_size"] != 1:
        raise ValueError(
            f"policy.shared_prefix_training.mode={shared_prefix_config.mode} currently requires "
            "policy.megatron_cfg.pipeline_model_parallel_size=1."
        )

    recompute_granularity = megatron_config.get("recompute_granularity")
    if megatron_config["activation_checkpointing"] and recompute_granularity in (
        None,
        "full",
    ):
        recompute_method = megatron_config.get("recompute_method")
        if recompute_method not in (None, "uniform"):
            raise ValueError(
                f"policy.shared_prefix_training.mode={shared_prefix_config.mode} full activation "
                "recomputation requires policy.megatron_cfg.recompute_method='uniform' "
                f"when supplied; got {recompute_method!r}."
            )
        recompute_num_layers = megatron_config.get("recompute_num_layers")
        if recompute_num_layers is not None and (
            isinstance(recompute_num_layers, bool) or recompute_num_layers != 1
        ):
            raise ValueError(
                f"policy.shared_prefix_training.mode={shared_prefix_config.mode} full activation "
                "recomputation requires policy.megatron_cfg.recompute_num_layers=1 "
                f"when supplied; got {recompute_num_layers!r}."
            )

    cuda_graph_impl = megatron_config.get("cuda_graph_impl")
    if cuda_graph_impl is not None and cuda_graph_impl != "none":
        raise ValueError(
            f"policy.shared_prefix_training.mode={shared_prefix_config.mode} currently requires training "
            "CUDA graphs to be disabled with policy.megatron_cfg.cuda_graph_impl='none'."
        )

    fp8_config = megatron_config.get("fp8_cfg")
    if fp8_config is not None and fp8_config["enabled"]:
        raise ValueError(
            f"policy.shared_prefix_training.mode={shared_prefix_config.mode} currently requires "
            "policy.megatron_cfg.fp8_cfg.enabled=false."
        )

    if config.get("quant_cfg") is not None:
        raise ValueError(
            f"policy.shared_prefix_training.mode={shared_prefix_config.mode} currently requires "
            "policy.quant_cfg=null; FP4 and other ModelOpt training quantization "
            "are not supported."
        )

    return shared_prefix_config
