# TrainingEngine policy: pending work and unsupported cases

Tracks what the Megatron-Bridge TrainingEngine policy path (`policy.training_engine`) does not do
yet. Update this file in the same change that adds or removes a limitation.

Code covered:

- `workers/training_engine_policy_worker.py` (`TrainingEnginePolicyWorker`)
- `training_engine_adapter/` (`config.py`, `engine.py`, `worker_environment.py`)
- `lm_policy.py` (`Policy.__init__` builds the engine worker when `policy.training_engine` is set)

How the path is put together:

- There is no engine-specific policy class. `Policy` (sync GRPO) and `TQPolicy` (data plane and
  SingleController) drive the engine worker exactly as they drive the Megatron worker.
- NeMo RL decides the packing and the data-parallel split on the driver: it plans the packed
  microbatches, sends each data-parallel replica its own samples, and attaches the plan. This is
  the same code the Megatron policy uses, so both backends train on the same microbatches.
- The worker hands the engine the replica's NeMo RL batch as it is; nothing is converted to the
  engine's `Datum` type. `NeMoRLTrainingEngine` overrides the engine's data hooks and adds a
  loss method:
  - `preprocess` packs each planned microbatch with NeMo RL's own `process_microbatch`
    (`nemo_rl/models/megatron/data.py`), the function the Megatron worker packs with. That gives
    the packed input tokens, sequence boundaries, positions and padding mask for this
    context-parallel rank. The override adds the two things the engine needs beyond the model
    inputs, because it scores targets inside the model and counts loss tokens itself: each
    position's target token and loss mask, in the same layout.
  - `clipped_pg_loss` runs the loss the way the Megatron policy does: through NeMo RL's packed
    loss wrapper (`SequencePackingFusionLossWrapper`), which calls a prepare function and then
    the loss on NeMo RL's own microbatch. The Megatron policy's prepare function turns logits
    into whole-sequence `[B, S-1]` logprobs; the engine's model already returns logprobs, so the
    adapter's `prepare_packed_logprobs` only gathers each sequence across context-parallel
    ranks (NeMo RL's `allgather_cp_sharded_tensor`, which keeps gradients) and puts it on its
    row. Every context-parallel rank then computes the same loss and the gather adds their
    gradients up, so the loss and metrics are divided by the number of those ranks, as the
    Megatron policy's `cp_normalize` does. The per-token loss inputs (advantages, the logprob
    fields) are never packed.
  - `postprocess_forward_only` gathers the scores with the same `prepare_packed_logprobs` and
    returns them in NeMo RL's `[B, S]` layout (token `i`'s logprob at position `i`), in the
    order the samples came in. The engine's own reassembly blocks and its microbatch mapping
    type are not used; the only Megatron-Bridge helper left in the data path is
    `get_thd_cp_partition_indices`, which `preprocess` uses to put the targets and the loss
    mask in this rank's layout.
- NeMo RL's own microbatch (the `[B, S]` masks, advantages and logprob fields the loss reads)
  travels with the engine's microbatch in `sample_inputs`, the engine's slot for per-sample data
  that is passed through untouched and never packed or split. The engine documents one dict per
  sample there; the adapter stores one entry holding the whole batched microbatch. The sample
  offsets in the packed row come from the microbatch's `packed_seq_params`, as in NeMo RL's
  loss builder. The engine's `mapping` slot is left empty.
- Each sample is packed whole (`L` tokens, as the Megatron policy does), not as `L - 1`
  input/target pairs; a sample's last position has no target and is masked out.
- The worker includes `TQWorkerMixin`, which adds the data-plane entry points (`*_presharded`):
  fetch this replica's samples from the data plane, call the worker's own method, write scores back.

What works today: GRPO with `ClippedPGLossFn` (token-level loss), text input, sequence packing, a
vLLM generation engine refit over IPC (colocated) or by NCCL broadcast (own GPUs), a frozen
reference model, constant learning rate with linear warmup, checkpoint save and resume, and a
streaming train step. See "Verified" below for which of these a run has confirmed.

## 1. Gaps to feature parity with the Megatron policy

Things `MegatronPolicyWorker` does that this path does not. "Raises" means the code fails with a
clear error; "absent" means the worker has no such method, so the call fails by name.

### Training

| Gap | State | What is needed |
|---|---|---|
| Losses other than `ClippedPGLossFn` (NLL, DPO, distillation, ...) | Raises in `check_supported_loss` (worker `begin_train_step`) | A loss method on `NeMoRLTrainingEngine` per loss. |
| `ClippedPGLossFn` options: sequence-level loss, `sequence_level_importance_ratios`, `seq_logprob_error_in_loss`, `seq-mask-tis`, `score_centering`, `opd_full`, `positive_example_nll_weight > 0` | Raises in `check_supported_loss` | The loss now sees whole sequences, so what is left is normalization and logits. The adapter passes global counts of 1 and the engine divides the gradient once by its own token count; the Megatron policy passes the real global counts into the loss. Per option: `seq-mask-tis` keeps the token count as divisor, so the gradient needs only a verification run; its `is_oob_ratio` metric is a mean over sequences and needs the sequence count. Sequence-level loss and `sequence_level_importance_ratios` divide by the number of valid sequences: select the engine's per-sample normalization and report valid sequences from `postprocess_forward_backward`. `seq_logprob_error_in_loss` divides by the tokens that survive the in-loss filter (the Megatron worker rescales gradients in `_normalize_in_loss_seq_filter`): report the loss's survivor count as the token count from `postprocess_forward_backward`. `positive_example_nll_weight` adds a term with its own per-microbatch divisor, which one engine divisor cannot express: pass real global counts and call `optim_step(normalize_gradients=False)`. `score_centering` and `opd_full` read the model's logits, which the engine does not return yet (its `requires_logits` path). |
| `train(eval_mode=True)` | Raises in worker `train` | A forward with the loss and no gradient accumulation. |
| PPO critic (`TQValue`, `get_values_presharded`) | Absent: no engine-backed value worker in NeMo RL. The engine itself supports a value head | A value worker built on a value-head engine. |
| Router replay (routed experts from the rollout) | Not passed: `process_microbatch` packs `routed_experts` when the batch has it, but `preprocess` does not hand it to the engine, and the worker does not provide `_routed_experts_layout`. The engine itself can replay routing | Pass the packed field into the `PreparedMicrobatch` and build the engine from a recipe with `router_replay`. |
| SFT v2 (`train_placed_microbatches`) | Not tried | Needs a loss method for its loss first. |
| Weight decay schedule (`start_weight_decay != end_weight_decay`) | Raises in worker `__init__` | Compute the decay per `train()` call, as for the learning rate. |
| Weighted (non 0/1) token or sample masks | Raises in the engine's microbatch check (`loss_mask must be a binary float32 mask`) | A binary loss mask for the engine's token count, with the weights left to NeMo RL's loss. |
| Metrics `probs_ratio_min/max`, `probs_ratio_clamped_min/max`, `num_valid_samples`, `positive_nll_loss` | Dropped silently | Reduce extrema with min and max; give the others their own denominators. |
| Per-replica metric lists | Only the first data-parallel replica reports `all_mb_metrics`, one entry per update. The values are totals over all ranks; the Megatron worker reports one entry per microbatch per replica | Nothing for GRPO, which sums the lists. Anything that reads per-microbatch entries needs them. |

### Scoring

| Gap | State | What is needed |
|---|---|---|
| `get_logprobs_with_full_payload` | Absent | Return the extra per-token payload from `forward_only`. |
| `get_topk_logits` | Absent | Needs the engine's full-logits path, which is not implemented in the engine either. |
| `load_opd_full_teacher_lm_head` | Absent | Goes with `opd_full`. |
| Sequence packing off or dynamic batching | Raises in `align_policy_config_with_recipe` | A `preprocess` branch that pads instead of packing (the engine's dense layout is not implemented either). |

### Checkpointing

| Gap | State | What is needed |
|---|---|---|
| Weights-only checkpoint (`checkpointing.save_optimizer=false`) | Raises in worker `save_checkpoint` | An engine save without optimizer state. |
| Async save (`finalize_async_save`, `poll_async_save`) | Base-class defaults only; saves block (about 34 s for this model) | Build the engine with its save worker and pass `async_save`. |
| Resume message | `Checkpointer.get_resume_paths` warns "Optimizer will be freshly initialized" because it does not recognize the engine layout (`iter_<update count>`); the optimizer state is restored anyway | Teach `get_resume_paths` the engine layout. |
| Tokenizer copy in the checkpoint (`tokenizer_path`) | Ignored, as in the Megatron worker | |
| Data-plane state in the checkpoint (`checkpointing.save_data_plane`) | Not tried | |

### Refit (weights to the generation engine)

| Gap | State | What is needed |
|---|---|---|
| NCCL reshard refit (`refit_transport: nccl_reshard`, `prepare_nccl_reshard_refit_info`) | Base-class default only | Local parameter shards and their mapping to HF names. |
| SGLang, colocated and distributed (`update_weights_to_sglang_*`, `connect_sglang_rollout_engines_distributed`) | Absent | Same export stream, SGLang transport. |
| Remote sparse delta sync (`init_remote_sparse_delta_baseline`, `stream_remote_sparse_weights`, `finish_remote_sparse_delta_sync`) | Absent | |
| Pre-quantized refit (`enable_refit_prequantize`) | Absent | |
| FP8 KV-cache scales (`kv_scales`, `calibrate_qkv_fp8_scales`) | Raises / absent | Calibrate on the engine's model and append the scales to the stream. |
| Policy as a refit destination | Raises in worker `__init__` | An engine path that imports weights. |
| Megatron generation (the training worker also generates; the only colocated mode the SingleController accepts) | Absent | Generation on the engine's model. |

### Configuration and modes

| Gap | State | What is needed |
|---|---|---|
| Policy without an optimizer (`init_optimizer=false`) | Raises in worker `__init__` | Build one frozen engine. |
| Pipeline parallelism | Raises in `align_policy_config_with_recipe` | Engine support for PP. |
| Multimodal input (images) | Only token ids reach the engine | Pass image inputs through `preprocess`; the engine's model call does not take them yet either. |
| `worker_extension_cls_fqn` or `quant_cfg` together with `policy.training_engine` | Raises in `Policy.__init__` | |
| Entry points with other losses (SFT, DPO, distillation) | Not wired | Their loss methods. |
| Worker utilities: `finish_inference`, `check_tensor_parallel_attributes`, `validate_forest_execution_padding`, `move_model`, `move_optimizer` | Absent | Add each when a caller needs it. |

## 2. Remaining work

### Verified

Filled in from `workspace/TRAININGENGINE_TESTS.md` (run ledger). Qwen3.5-35B-A3B, 16 training
GPUs, engine recipe TP1 CP8 EP16 against the Megatron policy TP2 EP16.

| Workflow | Generation | Megatron baseline | Engine |
|---|---|---|---|
| Sync GRPO (`examples/run_grpo.py`, `Policy`) | colocated vLLM | passed | passed, 4 steps; step 1 equals the baseline's loss and reward |
| Sync GRPO | vLLM on 2 more nodes | passed | passed, 6 steps |
| Sync GRPO over the data plane (`data_plane.enabled=true`, `TQPolicy.train_from_meta`) | colocated vLLM | passed | passed, 3 steps; step 1 equals the colocated run's |
| SingleController (`examples/run_grpo_single_controller.py`, `TQPolicy` streaming train step) | vLLM on 2 more nodes | passed, 4 steps | passed, 4 steps, two chunks per update |
| SingleController | colocated vLLM | not possible: NeMo RL rejects it for both backends (`single_controller_utils/config.py`) | |

### Not verified

- **Legacy async GRPO** (`grpo.async_grpo.enabled`). Never run on this path. It calls whole-batch
  `policy.train`.
- **Learning-rate decay** (`linear`, `cosine`, `inverse-square-root`). Present in the worker's
  `_adam_params`; no run or unit test has exercised it. Runs so far used `constant`.
- **More than one update per `train()` call** (batch larger than the global batch size): unit
  test only.
- **`abort_train_step` and `pause_train_step_with_offloading` inside a workflow.** Exercised only
  by the fixed-batch test. Their callers are the SingleController's abort path and PPO.
- **Preemption.** Resume was tested by starting a second job on the same checkpoint directory,
  not by a Slurm requeue of one job.
- **Runs longer than 10 steps.**

### Environment

- **Workaround: borrowed worker environment.** `register_worker_environment` in
  `training_engine_adapter/worker_environment.py` registers the engine worker at runtime with the
  Megatron worker's interpreter (`<venv dir>/<MegatronPolicyWorker class path>/bin/python`).
  `Policy.__init__` calls it when `policy.training_engine` is set.
  - Why: the standard path builds an environment per actor class from the lock file, which pins
    Transformer Engine 2.18.0. That release has no Gated DeltaNet kernel, so the engine recipe
    cannot run on it.
  - What it depends on: an image whose Megatron worker environment exists and holds a rebuilt
    Transformer Engine. Today that is a hand-patched image, not one the Dockerfile produces.
  - Long-term fix: (1) raise the Transformer Engine pin in `pyproject.toml` and `uv.lock`;
    (2) add the worker to `nemo_rl/distributed/actor_environments.py` with the `mcore` extra;
    (3) rebuild the image so its environment is pre-built; (4) delete
    `register_worker_environment`.
- The run uses `model.gdn_kernel_backend: fla`. Transformer Engine's kernel needs cuDNN frontend
  1.29 or later; the image has 1.25.0.
- `model.gradient_accumulation_fusion: false`: the environment has no fused weight-gradient
  extension.

### Open investigations

- **Engine scoring is not repeatable per token.** **Critical; blocks trusting any per-token
  comparison and any ratio-based loss on this path.** Scoring one batch twice with the same
  packing differs by more than 0.01 on 1 to 2 percent of tokens (worst 9.8 to 20); the Megatron
  policy is bit-identical across passes. In each pass 1 to 4 of 48 samples get large differences
  on 10 to 50 percent of the tokens in their last part, never in their first half; which samples
  changes from pass to pass. The frozen reference engine shows it too. Bisection, four passes
  each: repeatable with CP1 on the engine and with TP1 CP8 on the Megatron policy; still not
  repeatable on the engine with CP8 when the dispatcher is all-to-all, router fusion is off, cross
  entropy fusion is off, or autotuning is pinned; much reduced, not gone, with Gated DeltaNet
  fusion off. Investigated in `workspace/worker-determinism/` (see its `LOG.md` for the current
  state).
- **Speed.** On the same microbatches (NeMo RL's plan, 16k sample limit) training and scoring
  are about 2 times slower than the Megatron policy (TP1 CP8 EP16 against TP2 EP16): sync GRPO
  over the data plane, step 3, training 130 s against 64 s and scoring 83 s against 39 s;
  generation is equal. With vLLM on its own nodes: training 128 to 138 s against 74 to 78 s,
  scoring 79 to 82 s against 46 to 49 s, whole step about 1.25 times. Three or four steps per
  run, so not steady state. The layout accounts for the gap: the Megatron policy itself at
  TP1 CP8 (job 5433647, otherwise the same run) took 265 / 239 / 178 s for training and
  157 / 137 / 102 s for scoring over steps 1 to 3, against the engine's 211 / 146 / 130 s and
  190 / 94 / 83 s. The adapter's own share (datum conversion, packing, score gathering) is
  not timed separately.
- **Reference model residency.** `prepare_for_training` moves the reference engine to host memory
  and `prepare_for_lp_inference` brings it back. The SingleController calls both once per chunk,
  so the reference engine moves twice per chunk. With generation on its own GPUs both engines
  could stay on the GPU.
- **Chunks move per-token logprobs.** Scoring the same samples per chunk instead of as one batch
  changed 12 percent of tokens by more than 0.01 when the engine planned its own packing. A
  streaming workflow scores each chunk with the plan it trains it with, so both passes see the
  same microbatches; whether the difference remains is the non-repeatability item above.
- **Step-1 KL to the reference** is 1.9e-6 where the Megatron policy gives about zero. Small next
  to the 0.0005 seen from step 2; cause not looked at.
- **Per-token logprobs depend on the parallel layout** for this model (about 0.4 mean absolute
  difference between any two layouts, native ones included; means agree within 0.003). Tracked in
  Megatron-Bridge `src/megatron/bridge/engine/TODO.md`.

### Checks specific to this path (intended, not gaps)

The policy config and the recipe are brought together in one place, on the driver, before
`Policy` lays out the workers: `align_policy_config_with_recipe` (`training_engine_adapter/config.py`).
The driver's environment cannot import the engine, so it runs a short Ray task in the engine
worker's environment that builds the recipe and returns its layout. Then it:

- overrides what the recipe decides, printing each value it replaces: `megatron_cfg`
  `tensor_model_parallel_size`, `context_parallel_size`, `pipeline_model_parallel_size`,
  `sequence_parallel`, and `sequence_packing.train_mb_tokens` / `logprob_mb_tokens`
  (set to `training_engine.seq_length`);
- raises ValueError for what it cannot decide: `megatron_cfg.enabled=false` and
  `sequence_packing.enabled=false` (both are read by the algorithm setup before the policy is
  built); `max_total_sequence_length > training_engine.seq_length`; `seq_length` inside
  `recipe_kwargs`; `make_sequence_length_divisible_by` not a multiple of what the recipe's layout
  needs every sample padded to (the same rule the Megatron worker asserts);
- raises NotImplementedError for a recipe with pipeline parallelism.

`megatron_cfg` still supplies the optimizer, the learning-rate schedule, `train_iters` and
`freeze_moe_router`; the recipe has no scheduler.

The worker checks nothing against the config. Like the Megatron worker, it relies on NeMo RL's
worker grid and Megatron's process groups numbering the ranks the same way (TP fastest, then CP,
then DP). The engine packs each sample to `make_sequence_length_divisible_by`, the multiple the
driver's plan rounds with.

Other intended errors on this path: a `check_dim_skip_keys` argument; a planned microbatch with
no samples; samples shorter than 2 tokens.
