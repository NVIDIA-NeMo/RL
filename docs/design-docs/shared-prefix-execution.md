# Shared-prefix execution for GRPO (experimental)

GRPO generates several completions for each prompt. Conventional policy execution reads that prompt again for every completion when calculating log probabilities and training. Shared-prefix packing stores a prompt once per packed group and maps each completion to its own suffix. A logical GRPO group may span several packed groups, so the prompt is not necessarily computed once across all completions globally.

This contribution is opt-in and currently targets text-only Megatron hybrid attention/Mamba models. The dense path remains the default (`mode: disabled`). It does not change the GRPO objective, increase the number of completions or accelerate rollout generation by itself.

## Implementation ownership

Reusable packing is implemented in `megatron.rl`: parent-linked multi-level tree layouts,
group subdivision and sharding, execution slots and plans, tensor
materialization, TP/CP geometry, real-row alignment, and reconstruction within
dense training bins. NeMo RL translates `BatchedDataDict` fields and policy
configuration into that API, selects the conventional length-only packer,
transports metadata, and coordinates worker execution. NeMo RL imports
`megatron.rl` lazily, only on shared-prefix paths, and does not re-export it:
`nemo_rl.data.packing` keeps only the row-metadata field names and
`with_prompt_length_tags`.

The matching Megatron package must be available in both the driver and model
worker environments when shared-prefix planning is enabled. Its pure packing
modules do not initialize the GPU model or depend on NeMo RL. Standard dense
NeMo imports do not require the optional Megatron backend. Portable packing
tests are owned by Megatron; NeMo tests cover configuration validation,
metadata transport, driver-side sharding and worker integration.

The canonical `PackedTreeLayout` stores physical token spans, logical lengths,
and parent node indices. Current packing emits stars/forests and uses that
descriptor for positions, predecessors, reference attention and model-input
lowering. The descriptor and reference mask support deeper trees; the current
fused attention/Mamba backend explicitly rejects them. Source-row and loss
mappings stay in the star/forest packing wrappers until generalized execution
is implemented.

This representation also accepts PPO rollout groups. A group with one answer
per prompt retains its real source row and uses ordinary dense execution.
Moving the packer does not change the RL objective or add arbitrary-depth
trajectory-tree execution.

## Execution contract

- Only rows with the same group identity and exactly identical prompt token IDs share a prefix. Invalid or incompatible rows use conventional packing.
- Attention permits each completion to read its shared prompt and its own causal history. Sibling completions cannot attend to each other.
- The Mamba backend forks recurrent state at an aligned prompt boundary. It replays the residual prompt tail and supplies each branch's convolution halo where required. Branch backward contributions accumulate into the shared prefix. This is not a claim that every kernel reads each prompt token only once.
- Logical-to-physical maps restore completion logits/logprobs to the original row and token positions, including the first completion token's prompt predecessor.
- DP alignment uses real-row splits, never dummy training examples. TP/CP layouts preserve causal positions and exclude physical padding from losses.
- `match_logprob_training_layout` makes logprob evaluation use training's sharding weights, packing budget and execution plan. Changing only one phase's layout can change numerical results even at identical weights.
- MTP receives dense expanded inputs and preserves its existing loss grouping. MTP prefix sharing is outside the current scope.

## Configuration fragment

Merge this fragment into an otherwise validated text-only Megatron hybrid GRPO recipe. It is not a complete launcher. The example budget is illustrative; size it for the model and hardware, then measure both dense and shared configurations at matched budgets.

```yaml
policy:
  shared_prefix_training:
    mode: train
    pack_groups: true
    repack_groups: true
    align_data_parallel: true
    training_dense_bins: true
    match_logprob_training_layout: true
    bypass_evaluation_mtp: true
    merge_dense_fallbacks: false
    evaluation_packing: false
  sequence_packing:
    enabled: true
    algorithm: modified_first_fit_decreasing
    train_mb_tokens: 32768
    logprob_mb_tokens: 32768
  make_sequence_length_divisible_by: 16  # a multiple of M = 2 * TP * CP
  megatron_cfg:
    enabled: true
    pipeline_model_parallel_size: 1
    tensor_model_parallel_size: 2
    context_parallel_size: 4
    sequence_parallel: true
    cuda_graph_impl: none
    fp8_cfg:
      enabled: false
  quant_cfg: null
```

Shared execution requires the single-controller GRPO launcher with `data_plane.enabled: true` and `token_capture.enabled: true` in its existing algorithm configuration. Standard GRPO setup rejects `train` and `logprobs` modes because it does not produce group identities or prompt boundaries. Prompt boundaries are captured from the rollout rather than inferred from padded tokens. Workers must use the matching MCore shared-prefix implementation and its capability checks.

Shared execution also requires the following. Validation rejects a violation before policy workers are allocated, except for the data-parallel condition, which the controller checks before the first training chunk.

- `policy.router_replay` disabled.
- `policy.megatron_cfg.moe_hybridep_prepad_packed_inputs` disabled with the flex HybridEP dispatcher.
- A deterministic `policy.sequence_packing.algorithm` (not `first_fit_shuffle`) and no `pair_grouping_key`.
- An explicit `policy.make_sequence_length_divisible_by` that is a multiple of the TP/CP alignment M: 1 at TP1/CP1, 2×CP at TP1 with CP>1, and 2×TP×CP at TP>1. Both `train_mb_tokens` and `logprob_mb_tokens` must also be multiples of it.
- `async_rl.rollout_failure.max_skipped_prompts: 0` and `max_consecutive_dropped_prompts: 0`, because a dropped prompt can leave a step that no longer splits into complete data-parallel groups.
- `grpo.num_prompts_per_step` a multiple of the policy data-parallel size, because complete prompt groups are assigned to DP ranks.

Unknown keys under `policy.shared_prefix_training` are rejected. The commented block in `examples/configs/grpo_math_1B_megatron_single_controller.yaml` lists every key at its default.

Set `policy.generation.top_p: 1.0` and disable `policy.generation.top_k` (`null`, `0`, or `-1`). Shared next-token logprob extraction does not support top-k/top-p filtering, and configuration validation rejects it before policy workers are allocated. Temperature scaling remains supported. `bypass_evaluation_mtp` is accepted only in `dense`, `logprobs`, or `train` mode; disable it when switching to `disabled`.

For the ragged Mamba implementation studied in the experiments, set `NRL_SP_MAMBA_IMPL=ragged_state_fork` in worker environments. The backend's default `state_fork` is a different implementation. Other `NRL_SP_*` switches are experimental/diagnostic controls, not a supported tuning API. Their presence does not mean those variants were measured or qualified; do not enable them when reproducing the default configuration.

## Supported scope and validation

The first scope is the guarded hybrid model path, PP=1, supported TP/SP and CP layouts, no training CUDA graphs, no FP8/FP4 training, and no PEFT. Model capability checks further constrain attention, MoE and recomputation settings. Multimodal support, arbitrary model/backends and newly introduced upstream execution modes need separate qualification even if an underlying dense implementation supports them.

The contribution carries planner, metadata, materialization, causal-mask and alignment tests. Before merge it also needs distributed GPU tests covering the dense default, shared logprob extraction, matched own-logprob/training forwards, Mamba state backward, router logical multiplicity and MTP normalization.

Within-shared forward consistency and long-run evaluation quality answer different questions from dense/shared backbone-gradient agreement. A historical comparison with matched rows/bins and a common output cotangent measured a 13.9305% sampled, size-weighted relative L2 gradient difference, versus roughly 1% repeat variation. Relative L2 measures the norm of the gradient difference relative to the dense gradient norm; it is not a measure of accuracy loss or a percentage of incorrect parameters.

This leaves cross-implementation numerical agreement unverified; it does not establish a backward bug or worse training quality. Reduced-precision execution and different packing shapes can change numerical results, and bitwise dense/shared identity is not the acceptance criterion. The intended tolerance and training-quality evidence need maintainer agreement. The current-main port still needs its own distributed regression qualification and remains experimental.
