# Shared-prefix execution for GRPO (experimental)

GRPO generates several completions for each prompt. Conventional policy execution reads that prompt again for every completion when it computes log probabilities and trains. Shared-prefix execution stores the prompt once per packed unit and gives each completion its own suffix. A logical GRPO group may span several packed units, so a prompt is not necessarily computed only once across all of its completions.

The feature is opt-in and targets text-only Megatron hybrid attention/Mamba models. The dense path remains the default (`mode: disabled`). The feature does not change the GRPO objective, add completions or speed up rollout generation.

## Terms

- A *group* is the set of rollout rows generated from one prompt. Its identity comes from the row's `GROUP_ID_TAG`.
- A *star* stores `[prompt, completion_1, ..., completion_G]` once for rows whose prompt token IDs are identical.
- A *forest* packs several independent stars into one forward.
- A *dense fallback* is a row that cannot share a prefix. It runs as an ordinary packed row.
- An *execution unit* is one model forward: a star, a forest or a dense bin.
- *M* is the TP/CP padding quantum: 1 at TP1/CP1, 2×CP at TP1 with CP>1, and 2×TP×CP at TP>1.

## Implementation ownership

Megatron-LM owns the reusable packing in `megatron.rl`. That covers tree layouts, group sharding, execution plans, tensor materialization, TP/CP geometry, real-row alignment and dense-bin reconstruction. Megatron-LM also owns model execution through `HybridModel.forward(..., shared_prefix_layout=...)`. Its `docs/developer/shared_prefix.md` describes the model contract.

NeMo RL translates `BatchedDataDict` fields and the policy configuration into that API. It transports row metadata and coordinates worker execution. `nemo_rl.data.packing.shared_prefix_metadata` keeps only the row-metadata field names and `with_prompt_length_tags`. Modules that plan or run shared units, such as driver-side sharding and the Megatron worker, import `megatron.rl` directly and lazily, only on shared-prefix paths. NeMo RL does not re-export it. The dense paths of `nemo_rl.models.megatron.data` and `nemo_rl.models.megatron.train` import no `megatron.rl` module.

The driver and the model workers both need a Megatron-LM build that contains `megatron.rl`. The driver needs it because configuration validation and data-parallel sharding run there. Launch with the Megatron extra, for example `uv run --extra mcore`. Without it, configuration validation raises an `ImportError` that names this requirement.

Prompt boundaries come from the rollout, not from padded tokens. The rollout reassembler writes each row's prompt length as a data-plane column, which is why token capture is required. `TQPolicy` turns that column into row tags for planning.

`PackedTreeLayout` in Megatron-LM stores physical token spans, logical lengths and parent node indices. Current packing emits only stars and forests. The descriptor can describe deeper trees, but the model backend rejects them. A group with one completion, such as a PPO rollout, keeps its source row and runs densely.

## Execution contract

- Rows share a prefix only when they have the same group identity and identical prompt token IDs. Other rows become dense fallbacks.
- Each completion attends to its prompt and its own causal history. Sibling completions cannot attend to each other.
- The Mamba backend forks the recurrent state at a scan-chunk boundary. It replays the remaining prompt tail per branch and supplies each branch's convolution halo. Branch gradients accumulate into the shared prefix. Not every kernel reads each prompt token only once.
- Megatron defaults to the ragged Mamba backend (`ragged_state_fork`) at every topology. No environment variable is needed. `NRL_SP_MAMBA_IMPL` remains a Megatron-side override for diagnostics; see the Megatron documentation for its values.
- A star holds at most 16 completions (`MAX_SHARED_PREFIX_BRANCHES`). A larger group is split evenly; for example, 17 rows become stars of 9 and 8.
- Logical-to-physical maps restore completion logprobs to their original rows and positions. That includes the first completion token, whose predecessor is the last prompt token.
- Data-parallel alignment splits real rows. It never adds dummy training examples. TP/CP layouts keep causal positions and exclude physical padding from losses.
- Dense fallback bins honor `sequence_packing.max_sequences_per_bin`. Star and forest units are not row-capped, because one forward holds every sibling.
- MoE expert-bias counts follow the dense convention. With the HybridEP flex dispatcher, per-branch padding is excluded. Otherwise it counts, as in a dense packed forward.
- MTP runs on the dense expanded rows with its existing heads. MTP prefix sharing is out of scope. Each loss group is normalized as a packed dense forward of that group would be, using CP-local token counts at CP>1.
- `bypass_evaluation_mtp` passes `compute_mtp_loss=False` to every logprob forward.
- `uniform_router_gating` runs the MoE router GEMM in MCore's fixed row blocks in every policy forward and backward. Router arithmetic then does not depend on how rows are packed.
- `match_logprob_training_layout` makes logprob forwards use training's sharding weights, token budget and execution plan. A different layout in one phase can change numerical results even at identical weights.
- Matched layouts alone do not make logprob and training forwards bit-identical. Exact equality also needs deterministic algorithms and the same MoE top-k ordering. MCore sorts top-k results only when gradients are enabled, so a no-grad logprob forward can combine experts in a different order than the training forward.

## Configuration

Merge this fragment into a validated text-only Megatron hybrid GRPO recipe. It is not a complete launcher. It is the configuration used for the reported measurements, and every flag in it is experimental. The token budget is illustrative; size it for the model and hardware.

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
  generation:
    top_p: 1.0
    top_k: null
  quant_cfg: null
```

Run it with the single-controller launcher, for example `uv run --extra mcore python examples/run_grpo_single_controller.py --config <recipe>.yaml`. The recipe needs `data_plane.enabled: true` and `token_capture.enabled: true`. Standard GRPO setup rejects the `logprobs` and `train` modes, because it does not produce group identities or prompt lengths.

The commented block in `examples/configs/grpo_math_1B_megatron_single_controller.yaml` lists every key at its default.

### Modes and flag rules

- `disabled` keeps the existing packing and model execution.
- `dense` is a comparison control. It keeps conventional packing but allows `uniform_router_gating` and `bypass_evaluation_mtp`.
- `logprobs` shares prefixes in policy and reference logprob forwards only.
- `train` shares prefixes in logprob forwards and in training.

These rules are checked when the configuration is loaded:

- Unknown keys under `policy.shared_prefix_training` are rejected.
- `disabled` and `dense` reject every execution flag that differs from its default. `uniform_router_gating` and `bypass_evaluation_mtp` are the exceptions in `dense`.
- `repack_groups`, `pack_dense_fallbacks` and `evaluation_packing` each require `pack_groups`.
- `merge_dense_fallbacks` requires `pack_dense_fallbacks`.
- `align_data_parallel` requires `pack_groups` and `repack_groups`, and rejects `merge_dense_fallbacks`. With it, `evaluation_packing` also requires `bypass_evaluation_mtp`.
- `preserve_training_prefixes_during_alignment` requires `align_data_parallel`.
- `training_dense_bins` requires `mode: train`, `pack_groups`, `repack_groups` and `align_data_parallel`.
- `match_logprob_training_layout` and `training_shard_work_weights` require `mode: train`. `evaluation_packing` has no effect with `match_logprob_training_layout` and is rejected.
- Work weights must be nonnegative with a positive sum.

`repack_groups` re-plans each complete group on the worker and ignores the execution slots that the driver assigned. `pack_groups` at data-parallel size greater than one requires `align_data_parallel`; `Policy` checks this before it allocates workers.

### Other requirements

The `logprobs` and `train` modes also require the following. Policy and single-controller setup check them before policy workers are allocated, except where noted.

- The Megatron backend with `megatron.rl` importable in the driver.
- `policy.sequence_packing.enabled` with a deterministic algorithm (not `first_fit_shuffle`) and no `pair_grouping_key`.
- An explicit `policy.make_sequence_length_divisible_by` that is a multiple of M. Both `train_mb_tokens` and `logprob_mb_tokens` must be multiples of it.
- `policy.generation.top_p: 1.0` and `top_k` disabled (`null`, `0` or `-1`). Shared logprob extraction does not support top-k/top-p filtering. Temperature scaling is supported.
- `token_capture.enabled: true`.
- `async_rl.rollout_failure.max_skipped_prompts: 0` and `max_consecutive_dropped_prompts: 0`. A dropped prompt can leave a step that no longer splits into complete data-parallel groups.
- `grpo.num_prompts_per_step` a multiple of the policy data-parallel size, because complete groups are assigned to data-parallel ranks. The controller checks this before the first training chunk.
- A `TQPolicy`. Other policies raise `NotImplementedError` at construction.

Samplers align their selection to the data-parallel size only when their own class body declares `supports_prompt_group_multiple = True`. The flag is not inherited. A subclass of a built-in sampler must redeclare it and accept the `prompt_group_multiple` keyword. Other samplers fall back to exact-minimum chunks, which are correct but smaller.

On-policy distillation teachers always run with `mode: disabled`, because teacher dispatches are not group-sharded.

### Rejected configurations

The policy configuration rejects these settings in shared modes:

- PEFT/LoRA, `policy.router_replay`, pipeline parallelism, training CUDA graphs, FP8, and `quant_cfg` (FP4 and other ModelOpt training quantization).
- `moe_hybridep_prepad_packed_inputs` with the HybridEP flex dispatcher.

Worker setup resolves the model provider and rejects these model settings:

- Any provider other than Megatron Bridge's `HybridModelProvider`.
- TP=1 with sequence parallelism, or TP>1 without it.
- Full recomputation other than `recompute_method: uniform` with one layer, and selective recomputation of `core_attn`.
- Nonzero attention or hidden dropout, sliding-window attention, multi-latent attention, non-vanilla softmax, fine-grained activation offloading, QK-clip and maximum-logit statistics.
- Position embeddings other than RoPE or none.
- MoE load balancing other than `none`, `aux_loss`, `seq_aux_loss` or `global_aux_loss`, and a nonzero auxiliary-loss coefficient.
- A non-null `moe_z_loss_coeff` or `moe_input_jitter_eps`. Set them through `policy.megatron_cfg.model_overrides`.
- Expert capacity or per-rank capacity token dropping, and forced load balancing or forced biased routing.
- A Megatron build without the capability tokens for the resolved TP/CP/SP topology, explicit physical padding, and, when used, full recomputation, MTP, expert bias or positionless attention. The tokens identify implemented code paths; they are not evidence that a topology was qualified.

Worker setup also checks `uniform_router_gating` and `bypass_evaluation_mtp` in every mode, including `dense`. The first needs MCore's fixed-row router blocks, and the second needs a PP1 `HybridModel`. The microbatch iterator rejects dynamic batching, prepacked inputs and models that own packing, MTP masking or CP slicing. The Megatron stack validator adds its own guards, for example hash routing and a contiguous CP layout for Mamba.

### Runtime behavior

Each worker plans its execution units, then all model-world ranks agree on forward counts in one collective. A planning failure on one rank travels in that collective, so every rank raises instead of hanging. The worker times this under the `shared_prefix_execution_planning` timer label, which includes the agreement wait.

## Scope and validation

The supported scope is the guarded hybrid model path with PP=1, TP with sequence parallelism, CP, no training CUDA graphs, no FP8/FP4 training and no PEFT. Multimodal models and other backends are out of scope.

NeMo RL tests in this repository:

- `tests/unit/models/megatron/test_shared_prefix_logprob_oracle.py` compares shared logprobs, loss and gradients with dense rows through the worker's entry points. It uses a table-lookup model whose logits depend only on causal history, so a correct consumer must match dense exactly. Cases cover stars, forests, split execution slots, dense bins and dense fallbacks, with and without temperature. The TP1/CP1 cases need one GPU. The TP1/CP2, TP2/CP1 and TP2/CP2 variants need up to four: `CUDA_VISIBLE_DEVICES=0,1,2,3 pytest --mcore-only tests/unit/models/megatron/test_shared_prefix_logprob_oracle.py -k distributed`.
- `tests/unit/models/megatron/test_megatron_imports_without_shared_prefix.py` checks that the default Megatron path imports without `megatron.rl`.
- `tests/unit/data_plane/test_shared_prefix_preshard.py`, `tests/unit/single_controller/test_shared_prefix_controller.py`, `tests/unit/models/policy/test_shared_prefix_config.py`, `test_shared_prefix_stage_weights.py`, `test_megatron_worker_shared_prefix.py` and `tests/unit/models/megatron/test_shared_prefix_setup_capability.py` cover sharding, the train pump, configuration rules, dispatch plumbing, worker gating and setup checks.

Megatron-LM tests compare the shared HybridModel forward and backward with the dense rows it replaces, at TP1/CP1, TP2/SP/CP2 and TP1/CP4. They use an FP32 dense reference with MoE top-k choices replayed from a fixed table. The shared BF16 error must stay within 1.25× of the dense BF16 error. At TP/CP, the shared-versus-dense gap must stay within 1.5× of the TP1/CP1 gap.

No functional single-controller GRPO test for this feature exists yet. Training quality has not been qualified on this revision.

### Dense-versus-shared gradient differences

Shared and dense execution are different but equally valid arithmetic, so their BF16 gradients differ. An earlier comparison reported a 13.9% relative L2 difference on a sampled subset of gradient entries, against about 1% repeat variation. A repeat measures only run-to-run nondeterminism, so it is the wrong baseline.

Single-GPU (TP1/CP1) measurements on small random-init hybrid models explain the difference:

- In FP32 with IEEE GEMMs, each shared layer matches dense to about 1e-7.
- With MoE routing replayed, shared and dense are equally close to an FP32 reference: 1.96% each for a 28-layer model.
- With routing replayed, the shared-versus-dense gap (2.34%) is close to the gap between two dense runs with different packing layouts (2.05%). The gap shrinks about 8× in FP16, which matches the ratio of the two formats' rounding steps.
- With natural routing, BF16 rounding flips MoE top-k choices, which inflates every comparison. Two dense runs with different packing layouts then differ by 6.4%, more than a dense repeat (5.4%).

The difference is therefore BF16 rounding amplified by MoE top-k flips. The right baseline is dense versus dense with a different packing layout, or distance to a high-precision reference, with routing replayed and measured over the whole gradient. This comparison has not been repeated on a full-size model at the production TP/CP topology.
