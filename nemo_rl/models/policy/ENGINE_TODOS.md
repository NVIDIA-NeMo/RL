# TrainingEngine policy: pending work and unsupported cases

Tracks what the Megatron-Bridge TrainingEngine policy path (`policy.training_engine`) does not do
yet. Update this file in the same change that adds or removes a limitation.

Code covered:

- `training_engine_policy.py` (`TrainingEnginePolicy`)
- `workers/training_engine_policy_worker.py` (`TrainingEnginePolicyWorker`)
- `training_engine_adapter/` (`config.py`, `data.py`, `losses.py`)
- `examples/run_grpo.py` (selects the policy when `policy.training_engine` is set)

What works today: synchronous GRPO with `ClippedPGLossFn` (token-level loss), text input, sequence
packing, a colocated vLLM generation engine refit over IPC, a frozen reference model, and constant
learning rate with linear warmup. Verified on Qwen3.5-35B-A3B, 16 GPUs, for 10 steps against the
Megatron policy.

## 1. Gaps to feature parity with the Megatron policy

Things `MegatronPolicyWorker` does that this path does not. "Raises" means the code fails with a
clear error; "absent" means the worker has no such method, so the call fails by name.

### Training

| Gap | State | What is needed |
|---|---|---|
| Losses other than `ClippedPGLossFn` (NLL, DPO, distillation, ...) | Raises in `_check_supported_loss` | A loss method on `NeMoRLTrainingEngine` per loss. |
| `ClippedPGLossFn` options: sequence-level loss, `sequence_level_importance_ratios`, `seq_logprob_error_in_loss`, `seq-mask-tis`, `score_centering`, `opd_full`, `positive_example_nll_weight > 0` | Raises in `_check_supported_loss` | These need whole sequences or their own denominators; the engine hands the loss one context-parallel slice of each sequence. |
| `train(eval_mode=True)` | Raises in policy `train` | A forward with the loss and no gradient accumulation. |
| Streaming train step: `begin_train_step`, `train_microbatch`, `finish_train_step`, `abort_train_step`, `pause_train_step_with_offloading` | Absent | Map onto repeated `engine.forward_backward` calls, one `optim_step`, and `zero_grad` for abort. The engine already accumulates across calls. |
| Weight decay schedule (`start_weight_decay != end_weight_decay`) | Raises in worker `__init__` | Compute the decay per `train()` call, as for the learning rate. |
| Weighted (non 0/1) token or sample masks | Raises in `training_datums` | Carry the weight as a per-token loss field. |
| Metrics `probs_ratio_min/max`, `probs_ratio_clamped_min/max`, `num_valid_samples`, `positive_nll_loss` | Dropped silently | Reduce extrema with min and max; give the others their own denominators. |

### Scoring

| Gap | State | What is needed |
|---|---|---|
| `get_logprobs_with_full_payload` | Absent | Return the extra per-token payload from `forward_only`. |
| `get_topk_logits` | Absent | Needs the engine's full-logits path, which is not implemented in the engine either. |
| `load_opd_full_teacher_lm_head` | Absent | Goes with `opd_full`. |
| `micro_batch_size` argument | Raises in policy | The engine packs by `training_engine.seq_length`; decide whether to honor or keep rejecting. |

### Checkpointing

| Gap | State | What is needed |
|---|---|---|
| `save_checkpoint` | Raises | `engine.save_state` plus the `train()` call count. Run with `checkpointing.enabled=false` until then. |
| Resume from `weights_path` / `optimizer_path`, `load_checkpoint` | Raises in worker `__init__` / absent | `engine.load_state`, and restoring the `train()` call count for the schedule. |
| Async save (`finalize_async_save`, `poll_async_save`) | Base-class defaults only | The engine is built with `spawn_async_save_worker=False`. |

### Refit (weights to the generation engine)

| Gap | State | What is needed |
|---|---|---|
| Non-colocated vLLM (`broadcast_weights_for_collective`) | Absent | Feed `engine.export(streaming=True)` to the collective broadcast. |
| NCCL reshard refit (`prepare_nccl_reshard_refit_info`) | Base-class default only | Local parameter shards and their mapping to HF names. |
| SGLang, colocated and distributed (`update_weights_to_sglang_*`, `connect_sglang_rollout_engines_distributed`) | Absent | Same export stream, SGLang transport. |
| Remote sparse delta sync (`init_remote_sparse_delta_baseline`, `stream_remote_sparse_weights`, `finish_remote_sparse_delta_sync`) | Absent | |
| Pre-quantized refit (`enable_refit_prequantize`) | Absent | |
| FP8 KV-cache scales (`kv_scales`, `calibrate_qkv_fp8_scales`) | Raises / absent | Calibrate on the engine's model and append the scales to the stream. |
| Policy as a refit destination | Raises in worker `__init__` | An engine path that imports weights. |

### Configuration and modes

| Gap | State | What is needed |
|---|---|---|
| Policy without an optimizer (`init_optimizer=false`) | Raises in worker `__init__` | Build one frozen engine. |
| Pipeline parallelism | Raises in `validate_training_engine_config` | Engine support for PP. |
| `sequence_packing.enabled=false` | Raises in `validate_training_engine_config` | The engine always packs. |
| Multimodal input (images) | Only token ids reach the engine | Pass image inputs through the Datum. |
| Data-plane policy (`data_plane` factory) | Raises in `examples/run_grpo.py` | An engine variant of the data-plane policy. |
| Entry points other than `examples/run_grpo.py` (SFT, DPO, distillation) | Not wired | Select the policy there once their losses exist. |
| Worker utilities: `finish_inference`, `check_tensor_parallel_attributes`, `validate_forest_execution_padding`, `move_model`, `move_optimizer` | Absent | Add each when a caller needs it. |

## 2. Remaining work

### Not verified

- **Async GRPO** (`grpo.async_grpo.enabled`). Never run on this path. Trace its calls into the
  policy first; it may need the streaming train step above.
- **Learning-rate decay** (`linear`, `cosine`, `inverse-square-root`). Present in the worker's
  `_adam_params`; no run or unit test has exercised it. Runs so far used `constant`.
- **More than one update per `train()` call** (batch larger than the global batch size).
- **Preemption.** With no checkpointing, a requeued job starts over.
- **Runs longer than 10 steps.**

### Environment

- **Workaround: borrowed worker environment.** `_register_worker_environment` in
  `training_engine_policy.py` registers the engine worker at runtime with the Megatron worker's
  interpreter (`<venv dir>/<MegatronPolicyWorker class path>/bin/python`).
  - Why: the standard path builds an environment per actor class from the lock file, which pins
    Transformer Engine 2.18.0. That release has no Gated DeltaNet kernel, so the engine recipe
    cannot run on it.
  - What it depends on: an image whose Megatron worker environment exists and holds a rebuilt
    Transformer Engine. Today that is a hand-patched image, not one the Dockerfile produces.
  - Long-term fix: (1) raise the Transformer Engine pin in `pyproject.toml` and `uv.lock`;
    (2) add the worker to `nemo_rl/distributed/actor_environments.py` with the `mcore` extra;
    (3) rebuild the image so its environment is pre-built; (4) delete
    `_register_worker_environment`.
- The run uses `model.gdn_kernel_backend: fla`. Transformer Engine's kernel needs cuDNN frontend
  1.29 or later; the image has 1.25.0.
- `model.gradient_accumulation_fusion: false`: the environment has no fused weight-gradient
  extension.

### Open investigations

- **Speed.** Scoring and training are about 1.7 times slower than the Megatron policy (TP1 CP8
  EP16 against TP2 EP16); steps take 344 to 434 s against 247 to 357 s. Not broken down yet.
- **Step-1 KL to the reference** is 1.9e-6 where the Megatron policy gives about zero. Small next
  to the 0.0005 seen from step 2; cause not looked at.
- **Per-token logprobs depend on the parallel layout** for this model (about 0.4 mean absolute
  difference between any two layouts, native ones included; means agree within 0.003). Tracked in
  Megatron-Bridge `src/megatron/bridge/engine/TODO.md`.

### Checks specific to this path (intended, not gaps)

These raise ValueError by design: `megatron_cfg.enabled=false`; `train_mb_tokens` or
`logprob_mb_tokens` different from `training_engine.seq_length`;
`max_total_sequence_length > training_engine.seq_length`; `seq_length` inside `recipe_kwargs`;
`mbs` or `check_dim_skip_keys` arguments; a batch that is not a multiple of the global batch
size; samples shorter than 2 tokens; a worker layout that differs from the engine's TP, CP, PP
and DP sizes.
