# Shared-prefix execution for GRPO (experimental)

GRPO generates several completions for each prompt. Conventional policy execution reads that prompt again for every completion when calculating log probabilities and training. Shared-prefix packing stores a prompt once per packed group and maps each completion to its own suffix. A logical GRPO group may span several packed groups, so the prompt is not necessarily computed once across all completions globally.

This contribution is opt-in and currently targets text-only Megatron hybrid attention/Mamba models. The dense path remains the default (`mode: disabled`). It does not change the GRPO objective, increase the number of completions or accelerate rollout generation by itself.

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

Set `policy.generation.top_p: 1.0` and disable `policy.generation.top_k` (`null`, `0`, or `-1`). Shared next-token logprob extraction does not support top-k/top-p filtering, and configuration validation rejects it before policy workers are allocated. Temperature scaling remains supported. `bypass_evaluation_mtp` is accepted only in `dense`, `logprobs`, or `train` mode; disable it when switching to `disabled` or `observe`.

For the ragged Mamba implementation studied in the experiments, set `NRL_SP_MAMBA_IMPL=ragged_state_fork` in worker environments. The backend's default `state_fork` is a different implementation. Other `NRL_SP_*` switches are experimental/diagnostic controls, not a supported tuning API. Their presence does not mean those variants were measured or qualified; do not enable them when reproducing the default configuration.

## Supported scope and validation

The first scope is the guarded hybrid model path, PP=1, supported TP/SP and CP layouts, no training CUDA graphs, no FP8/FP4 training, and no PEFT. Model capability checks further constrain attention, MoE and recomputation settings. Multimodal support, arbitrary model/backends and newly introduced upstream execution modes need separate qualification even if an underlying dense implementation supports them.

The contribution carries planner, metadata, materialization, causal-mask and alignment tests. Before merge it also needs distributed GPU tests covering the dense default, shared logprob extraction, matched own-logprob/training forwards, Mamba state backward, router logical multiplicity and MTP normalization.

Within-shared forward consistency and long-run evaluation quality answer different questions from dense/shared backbone-gradient agreement. A historical comparison with matched rows/bins and a common output cotangent measured a 13.9305% sampled, size-weighted relative L2 gradient difference, versus roughly 1% repeat variation. Relative L2 measures the norm of the gradient difference relative to the dense gradient norm; it is not a measure of accuracy loss or a percentage of incorrect parameters.

This leaves cross-implementation numerical agreement unverified; it does not establish a backward bug or worse training quality. Reduced-precision execution and different packing shapes can change numerical results, and bitwise dense/shared identity is not the acceptance criterion. The intended tolerance and training-quality evidence need maintainer agreement. The current-main port still needs its own distributed regression qualification and remains experimental.
