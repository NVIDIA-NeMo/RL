# TrainingEngine policy: pending work and unsupported cases

Tracks what the Megatron-Bridge TrainingEngine policy path (`policy.training_engine`) does not do
yet. Update this file in the same change that adds or removes a limitation.

Code covered:

- `workers/training_engine_policy_worker.py` (`TrainingEnginePolicyWorker`)
- `training_engine_adapter/` (`config.py`, `engine.py`, `worker_environment.py`)
- `lm_policy.py` (when `policy.training_engine` is set, `Policy.__init__` aligns the policy
  config with the recipe and builds the engine worker)

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
| PPO critic (`TQValue`, `get_values_presharded`) | Absent: no engine-backed value worker in NeMo RL, and `Value.__init__` picks its worker from `megatron_cfg.enabled` only. The engine itself builds a value head (`from_recipe(output_head="value")`, the same `LinearForLastLayer` head NeMo RL installs); on the merged Megatron-Bridge that needs the head to accept `output_dtype`, a fix not merged yet | A value worker next to `MegatronValueWorker`, selected by `value.training_engine`; a loss method that runs NeMo RL's `MseValueLossFn` through the packed loss wrapper with a prepare step that gathers the values across context-parallel ranks and shifts them right by one, as `_value_loss_prepare_fn` does; `get_values` in the same `[B, S]` layout. On the policy side PPO needs nothing new for training; `finish_inference` is called only with colocated generation. There is no Megatron PPO recipe for Qwen3.5 to compare with yet. |
| Router replay (routed experts from the rollout) | Not passed: `process_microbatch` packs `routed_experts` when the batch has it, but `preprocess` does not hand it to the engine, and the worker does not provide `_routed_experts_layout`. The engine can replay routing, but with a stricter rule than NeMo RL: it accepts the missing-route value (-1) at padding only and replaces it with a placeholder route. NeMo RL has -1 at real tokens (a sample's last token, messages generation recorded no routes for) and routes those tokens natively, through a patch of Megatron's `RouterReplay.get_replay_topk` | Engine: route natively where a real token's route is missing (or the adapter installs NeMo RL's patch and the engine lets -1 through). Layer numbering is settled on the engine side: `engine.router_replay_layers` gives the HF layer of each MoE layer (the Qwen3.5 recipe's `router_replay_layer_index`), which is what vLLM records under. Adapter: pass `routed_experts_cp_sharded` into the `PreparedMicrobatch` for the policy's scoring and training, not for the reference; `_routed_experts_layout` from the recipe's model config; require `policy.router_replay.enabled` and the recipe's `router_replay` to agree in `align_policy_config_with_recipe`. |
| SFT v2 (`train_placed_microbatches`) | Not tried | Needs a loss method for its loss first. |
| Weighted (non 0/1) token or sample masks | Raises in the engine's microbatch check (`loss_mask must be a binary float32 mask`) | A binary loss mask for the engine's token count, with the weights left to NeMo RL's loss. |
| Metrics `num_valid_samples`, `positive_nll_loss` | Dropped silently (`_NOT_ADDITIVE` in `engine.py`, a hand-written list). The ratio extrema are reported: every replica reports its own and the driver takes the min or max | Give them their own denominators. The loss's own `metric_normalizations` table says how each metric is normalized and could drive this. |
| One microbatch on the GPU at a time | All of a replica's microbatches are moved to the GPU before the first forward, because the engine's `preprocess` returns the whole list; the Megatron worker moves one at a time. Small for text | Keep NeMo RL's microbatch on the CPU in `preprocess` and move it in the loss. Matters once batches carry images. |
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
| Polling an async save (`poll_async_save`, called by the SingleController with `defer_async_checkpoint_finalize=true`) | Raises when `megatron_cfg.checkpoint.async_save` is set. Async save itself works: the engine is built with its save worker, `save_checkpoint` passes `async_save`, and `finalize_async_save` waits. With colocated generation the write does not overlap anything, because the engine waits for it before moving to host memory. Not run on the real model yet. The other `megatron_cfg.checkpoint` fields are not passed | Engine: a check on its pending save that does not block. |
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
| Pipeline parallelism | Raises in `align_policy_config_with_recipe` | Engine support for PP. |
| Multimodal input (images) | Only token ids reach the engine | Pass image inputs through `preprocess`; the engine's model call does not take them yet either. |
| `worker_extension_cls_fqn` or `quant_cfg` together with `policy.training_engine` | Raises in `Policy.__init__` | |
| `megatron_cfg` fields other than the layout, the Adam values, the schedule and `freeze_moe_router` | Ignored silently: the recipe decides the model, the precision and the rest of the optimizer. Today the two sides differ in `grad_reduce_in_fp32` (recipe true, baseline false), `use_precision_aware_optimizer` (baseline true), `overlap_grad_reduce` / `overlap_param_gather` (baseline true), recompute (recipe selective, baseline full), the MoE dispatcher (recipe flex with the HybridEP backend, baseline all-to-all) and the fused router and Gated DeltaNet kernels (recipe on, baseline off). `clip_grad` is 1.0 on both sides only by coincidence: `policy.max_grad_norm` does not reach the engine | Either override these from the recipe and print them, as for the layout, or reject values that differ. The dispatcher and the fusions change per-token scores (see "Numeric differences between the backends"); gradient precision moves the gradient norm by 0.01 to 0.09 percent only. |
| Multi-token-prediction head and MoE auxiliary loss | Not wired: the run turns both off in the recipe overrides, as the baseline does. The Megatron worker sets `mtp_grad_scale_func` / `moe_grad_scale_func` per update and reports their metrics; this worker does not | Set the scale functions from the worker and report the metrics. |
| Entry points with other losses (SFT, DPO, distillation) | Not wired | Their loss methods. |
| Worker utilities: `finish_inference`, `check_tensor_parallel_attributes`, `validate_forest_execution_padding`, `move_model`, `move_optimizer` | Absent | Add each when a caller needs it. |

## 2. Remaining work

What has been run and passed is in the appendix ("Verified").

### Blocked

- **The Megatron baseline does not run on the current Megatron-Bridge.**
  `nemo_rl/models/megatron/community_import.py` builds the model through Bridge's provider
  interface, and Bridge's checkpoint save then rejects the Qwen3.5 model (job 5578688). The
  baseline numbers in the appendix are from before the Bridge merge. Needed: port the Megatron
  policy to Bridge's model config and builder interface (NeMo RL `main` does not have it,
  checked 2026-11-17), or a Bridge fix that keeps the model config on a provider-built model.
- **Bridge fixes not merged yet**, mounted from a working tree in the runs: the value head
  accepting `output_dtype`, and the router replay layer mapping on the recipe.

### Not verified

- **`train(eval_mode=True)` in a workflow.** The worker scores the batch with `forward_only`,
  runs NeMo RL's loss on the scores without gradients and adds the replicas' sums up. Covered by
  a unit test and by the fixed-batch training test (the loss must equal the training pass's). No
  algorithm calls it with `ClippedPGLossFn`: its callers are SFT, DPO, the reward model and
  distillation, whose losses this path does not have.
- **A policy without an optimizer** (`init_optimizer=false`): the worker builds a frozen engine
  and refuses to train. Unit test for the refusal only; never built in a run.
- **Weight decay schedules** (`weight_decay_incr_style` `linear`, `cosine`): unit test only.
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
  `align_policy_config_with_recipe` calls it, because its Ray task and the workers both run in
  that environment.
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
- **Workaround: Triton 3.8.0 from outside the image.** The image has Triton 3.7.1. The test
  runs put a `pip install --target` copy of 3.8.0 on `PYTHONPATH` for the policy workers of both
  backends, through `policy.megatron_cfg.env_vars` in the run YAML; the vLLM workers keep the
  image's. Long-term fix: Triton 3.8 or later in the image.
- **Megatron-Bridge and the checkpoint.** The runs mount Megatron-Bridge `training_engine`
  (`662ee744a`) and Megatron-LM `5b069b1f7` over the image's copies. This Bridge builds Qwen3.5
  as two Megatron layers per Hugging Face layer, so the checkpoint was converted again and the
  run passes it as `policy.training_engine.checkpoint_path`; a checkpoint converted before that
  layout does not load. The Megatron policy does not run on this tree (see "Verified").

### Open investigations

- **Scores depend on which samples share a pack (cause known, see the appendix).** Open:
  - Remove it: make the row padding its own empty segment in the packer
    (`_pack_sequences_for_megatron`), or `model.cp_comm_type: all_gather` in the recipe
    overrides. Both verified for scoring; decided 2026-11-17: low priority, neither applied.
  - Until then, a check that refuses `logprob_mb_tokens != train_mb_tokens` under context
    parallelism, the case where a run would be affected.
- **Cross-backend comparison (background in the appendix).** Open:
  - The engine on the baseline's numeric settings matched the baseline's fixed-batch loss to
    every printed digit before the Bridge merge and is 0.05 percent off after it. Not traced;
    needs the baseline to run on the merged tree first.
  - Context parallelism, HybridEP and the fused kernels are compared only at the 0.5 percent
    level. A small error in them would pass; closing that needs per-component tests (a fused
    kernel against its unfused version on one input).
- **Speed.** On the same microbatches (NeMo RL's plan, 16k sample limit) training and scoring
  are about 2 times slower than the Megatron policy (TP1 CP8 EP16 against TP2 EP16): sync GRPO
  over the data plane, step 3, training 130 s against 64 s and scoring 83 s against 39 s;
  generation is equal. With vLLM on its own nodes: training 128 to 138 s against 74 to 78 s,
  scoring 79 to 82 s against 46 to 49 s, whole step about 1.25 times. Three or four steps per
  run, so not steady state. The layout accounts for the gap: the Megatron policy itself at
  TP1 CP8 (job 5433647, otherwise the same run) took 265 / 239 / 178 s for training and
  157 / 137 / 102 s for scoring over steps 1 to 3, against the engine's 211 / 146 / 130 s and
  190 / 94 / 83 s. Step times did not change when the adapter moved to NeMo RL's packing and
  to the gather before the loss. The adapter's own share (packing, gathering) is not timed
  separately.
- **Reference model residency.** `prepare_for_training` moves the reference engine to host memory
  and `prepare_for_lp_inference` brings it back. The SingleController calls both once per chunk,
  so the reference engine moves twice per chunk. With generation on its own GPUs both engines
  could stay on the GPU.

## Appendix: finished work and background

Nothing in this part is pending.

### Verified

Job ids and numbers per run are in `workspace/TRAININGENGINE_TESTS.md` (run ledger).
Qwen3.5-35B-A3B, 16 training GPUs, engine recipe TP1 CP8 EP16, Megatron policy TP2 EP16, Triton
3.8.0 on the policy workers. Engine results are from Megatron-Bridge `training_engine`
`662ee744a` (rerun 2026-11-17); Megatron baseline results are from before the Bridge merge.

| Workflow | Generation | Megatron baseline | Engine |
|---|---|---|---|
| Sync GRPO (`examples/run_grpo.py`, `Policy`) | colocated vLLM | passed, 3 steps (job 5445213) | passed, 3 steps; step 1 equals the baseline's loss and reward (job 5590264) |
| Sync GRPO | vLLM on 2 more nodes | not run on Triton 3.8.0 | passed, 6 steps (job 5591089) |
| Sync GRPO over the data plane (`TQPolicy.train_from_meta`) | colocated vLLM | not run on Triton 3.8.0 | passed, 3 steps; step 1 equals the colocated run's (job 5591068) |
| SingleController (`TQPolicy` streaming train step) | vLLM on 2 more nodes | passed, 4 steps (job 5445239) | passed, 4 steps (job 5591094) |
| SingleController | colocated vLLM | not possible: NeMo RL rejects it for both backends | |
| Fixed-batch scoring | none | passes identical; mean NLL 2.0523 (job 5444690) | passes identical; mean NLL 2.0471 (job 5590262) |
| Fixed-batch training and streaming train step | none | loss 0.0488003, gradient norm 11.93 (job 5444699) | every repeat identical; loss 0.051521, gradient norm 12.99 (job 5591058) |
| Fixed batch, engine on the baseline's numeric settings (`variants/tp2-alltoall-match.yaml`) | none | 0.0488003 / 11.9342 | 0.0488258 / 11.9148, 0.05% and 0.16% apart (job 5619211) |
| Fixed batch, production recipe, every advantage 1 (the loss is the mean NLL) | none | 2.0523 / 101.851 | 2.04708 / 101.743, 0.25% and 0.11% apart (job 5619210) |
| Checkpoint save and resume | colocated vLLM | | save run 4 steps (job 5591078); resume continues at step 5 with the expected learning rate (job 5591086) |
| Adapter unit tests | none | | passed (job 5590256) |

### Closed investigations

- **Repeatability: fixed by Triton 3.8.0.** With the image's Triton 3.7.1, scoring one batch
  twice on the engine differed by more than 0.01 on 0.1 to 2 percent of tokens: a race in the
  output stage of the `fla` Gated DeltaNet Triton kernel on Blackwell
  (`workspace/worker-determinism/LOG.md`). With 3.8.0 every repeat is exact on both backends.
  The values also changed between the versions (fixed-batch gradient norms down about 20
  percent, per-token scores moved by 0.40 on average with the same mean), which is the size any
  change of arithmetic produces on this model. No independent reference was checked for either
  version. Decided 2026-11-16: stay on 3.8.0.
- **Step-1 KL penalty to the reference** was 1.9e-6 on the engine with Triton 3.7.1. With 3.8.0
  it is exactly 0 on both backends, so it was the non-repeatable kernel.
- **Scores depend on which samples share a pack: cause found**
  (`workspace/worker-packdep/REPORT.md`). Scoring the same samples per chunk instead of as one
  batch changes 9 percent of tokens by more than 0.01. A row is padded to full width by
  lengthening its last sample, which moves where context parallelism cuts that sample;
  chunk-wise attention then rounds differently, and the model amplifies it. No sample reads
  another's tokens; without context parallelism every grouping is bit-identical. A run is not
  affected while scoring and training pack the same samples with the same row size, which the
  engine recipe does. The two ways to remove it were verified in jobs 5553201 (packer) and
  5576645, 5576942 (`all_gather`).

### Numeric differences between the backends (explained, not a bug)

From `workspace/worker-gradgap/REPORT.md` and `workspace/worker-packdep/REPORT.md`.

- **This model amplifies last-bit differences.** Any two evaluations that differ in arithmetic
  (layout, expert dispatcher, a fused kernel, precision) are about 0.4 apart per token in
  logprob, with half the tokens moving by more than 0.01, while the mean NLL agrees within 0.3
  percent. The Hugging Face implementation shows the same for a 1e-6 input perturbation. Per-token
  comparison across backends, or against vLLM, is therefore not a usable pass criterion.
- **The 5.6 percent loss gap and 9 percent gradient norm gap on the fixed batch** (engine 0.0515 /
  13.00, Megatron policy 0.0488 / 11.93) come from the recipes' numeric settings, magnified by
  the test: its random advantages make per-sample terms cancel, so the loss is about 3 percent
  of the sum of their absolute values. With every advantage set to 1 the gap is 0.25 and 0.11
  percent. With the engine on the baseline's settings (TP2 CP1, sequence parallelism, all-to-all
  dispatcher, router and Gated DeltaNet fusion off) the loss matched to every printed digit
  before the merge and is 0.05 percent off after it (table under "Verified").
- Sync GRPO step 1 reports gradient norm 0.0528 on the engine against 0.0581; same cause, not
  measured separately.

### Design decisions

Decided for this proof of concept, each with what it leaves behind.

- **Reuse NeMo RL's code wherever it exists.** Packing, the gather across context-parallel ranks
  before the loss, the loss wrapper and the loss are NeMo RL's. One Megatron-Bridge helper stays
  in the data path, `get_thd_cp_partition_indices`, for the targets and the loss mask: NeMo RL's
  own split helpers for this are private functions.
- **NeMo RL's microbatch travels in `sample_inputs`**, as one entry for the whole microbatch,
  not in `mapping`. The engine documents one dict per sample there, so the adapter deviates from
  that wording.
- **The recipe decides the parallel layout and the token budget; `megatron_cfg` is overridden to
  match.** The override runs when the policy is built, so code that reads the layout fields
  earlier (telemetry, for example) still sees the YAML's values.
- **`megatron_cfg` keeps the training hyperparameters**: optimizer, learning-rate schedule,
  `train_iters` and `freeze_moe_router`. The recipe has an optimizer config but no scheduler, and
  the engine takes the learning rate per update from its caller.
- **No rank-position check in the worker**, as in the Megatron worker (see the checks below).

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
